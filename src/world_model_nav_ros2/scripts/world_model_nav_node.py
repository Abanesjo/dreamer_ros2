#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSHistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from world_model_nav_ros2.core import (
    ControllerConfigValues,
    NavigationConfig,
    ObstacleObservation,
    PlanResult,
    StepResult,
    WorldModelNavigator,
)


PACKAGE_NAME = "world_model_nav_ros2"


def default_policy_path() -> str:
    try:
        return str(Path(get_package_share_directory(PACKAGE_NAME)) / "model" / "best.pt")
    except PackageNotFoundError:
        return str(Path(__file__).resolve().parents[1] / "model" / "best.pt")


def quaternion_from_yaw(yaw: float) -> Quaternion:
    quat = Quaternion()
    quat.z = math.sin(float(yaw) * 0.5)
    quat.w = math.cos(float(yaw) * 0.5)
    return quat


def yaw_from_quaternion(quat: Quaternion) -> float:
    siny_cosp = 2.0 * (float(quat.w) * float(quat.z) + float(quat.x) * float(quat.y))
    cosy_cosp = 1.0 - 2.0 * (float(quat.y) * float(quat.y) + float(quat.z) * float(quat.z))
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class WorldModelNavNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dynamic_obstacles_topic = str(self.get_parameter("dynamic_obstacles_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.subgoal_topic = str(self.get_parameter("subgoal_topic").value)
        self.control_frequency = float(self.get_parameter("control_frequency").value)
        configured_policy_path = str(self.get_parameter("policy_path").value)

        nav_config = NavigationConfig(
            policy_path=configured_policy_path if configured_policy_path else default_policy_path(),
            device=str(self.get_parameter("device").value),
            policy_seed=int(self.get_parameter("policy_seed").value),
            control_frequency=self.control_frequency,
            lookahead_distance=float(self.get_parameter("lookahead_distance").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
            robot_radius=float(self.get_parameter("robot_radius").value),
            inflation_margin=float(self.get_parameter("inflation_margin").value),
            map_occupied_threshold=int(self.get_parameter("map_occupied_threshold").value),
            treat_unknown_as_occupied=bool(self.get_parameter("treat_unknown_as_occupied").value),
            dynamic_obstacle_radius=float(self.get_parameter("dynamic_obstacle_radius").value),
            expected_dynamic_obstacles=int(self.get_parameter("expected_dynamic_obstacles").value),
            min_obstacle_dt=float(self.get_parameter("min_obstacle_dt").value),
        )
        controller_config = ControllerConfigValues(
            horizon=int(self.get_parameter("controller.horizon").value),
            w_progress=float(self.get_parameter("controller.w_progress").value),
            w_heading=float(self.get_parameter("controller.w_heading").value),
            w_path=float(self.get_parameter("controller.w_path").value),
            w_velocity=float(self.get_parameter("controller.w_velocity").value),
            w_static=float(self.get_parameter("controller.w_static").value),
            w_dynamic=float(self.get_parameter("controller.w_dynamic").value),
            static_collision_penalty=float(
                self.get_parameter("controller.static_collision_penalty").value
            ),
            dynamic_collision_penalty=float(
                self.get_parameter("controller.dynamic_collision_penalty").value
            ),
            dynamic_risk_tau=float(self.get_parameter("controller.dynamic_risk_tau").value),
            min_clearance_clip=float(self.get_parameter("controller.min_clearance_clip").value),
            backward_penalty_enabled=bool(
                self.get_parameter("controller.backward_penalty_enabled").value
            ),
            backward_action_name=str(self.get_parameter("controller.backward_action_name").value),
            backward_penalty=float(self.get_parameter("controller.backward_penalty").value),
            backward_gate_enabled=bool(self.get_parameter("controller.backward_gate_enabled").value),
            dynamic_stop_clearance_threshold=float(
                self.get_parameter("controller.dynamic_stop_clearance_threshold").value
            ),
            backward_dynamic_margin=float(
                self.get_parameter("controller.backward_dynamic_margin").value
            ),
            action_indices=tuple(int(value) for value in self.get_parameter("controller.action_indices").value),
        )

        self.navigator = WorldModelNavigator(nav_config, controller_config)
        self._warn_last_time: dict[str, float] = {}
        self._reported_map_yaw = False

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        default_qos = QoSProfile(depth=10)
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, default_qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, default_qos)
        self.obstacle_sub = self.create_subscription(
            MarkerArray,
            self.dynamic_obstacles_topic,
            self._on_dynamic_obstacles,
            default_qos,
        )
        self.path_pub = self.create_publisher(NavPath, self.path_topic, path_qos)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, default_qos)
        self.subgoal_pub = self.create_publisher(PoseStamped, self.subgoal_topic, default_qos)

        self.timer = self.create_timer(1.0 / max(self.control_frequency, 1e-6), self._on_timer)
        self.get_logger().info(
            f"World-Model-Nav ROS node ready: policy={nav_config.policy_path}, "
            f"control_frequency={self.control_frequency:.2f} Hz"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("dynamic_obstacles_topic", "/dynamic_obstacles")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("subgoal_topic", "/world_model_nav/subgoal")
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("policy_path", default_policy_path())
        self.declare_parameter("device", "auto")
        self.declare_parameter("policy_seed", 0)

        self.declare_parameter("lookahead_distance", 1.0)
        self.declare_parameter("goal_tolerance", 0.4)
        self.declare_parameter("robot_radius", 0.19)
        self.declare_parameter("inflation_margin", 0.05)
        self.declare_parameter("map_occupied_threshold", 50)
        self.declare_parameter("treat_unknown_as_occupied", True)
        self.declare_parameter("dynamic_obstacle_radius", 0.5)
        self.declare_parameter("expected_dynamic_obstacles", 4)
        self.declare_parameter("min_obstacle_dt", 1e-3)

        self.declare_parameter("controller.horizon", 10)
        self.declare_parameter("controller.w_progress", 2.0)
        self.declare_parameter("controller.w_heading", 0.8)
        self.declare_parameter("controller.w_path", 1.1)
        self.declare_parameter("controller.w_velocity", 0.3)
        self.declare_parameter("controller.w_static", 0.5)
        self.declare_parameter("controller.w_dynamic", 2.0)
        self.declare_parameter("controller.static_collision_penalty", 1e6)
        self.declare_parameter("controller.dynamic_collision_penalty", 1e6)
        self.declare_parameter("controller.dynamic_risk_tau", 0.5)
        self.declare_parameter("controller.min_clearance_clip", 0.05)
        self.declare_parameter("controller.backward_penalty_enabled", True)
        self.declare_parameter("controller.backward_action_name", "backward")
        self.declare_parameter("controller.backward_penalty", 0.25)
        self.declare_parameter("controller.backward_gate_enabled", True)
        self.declare_parameter("controller.dynamic_stop_clearance_threshold", 0.20)
        self.declare_parameter("controller.backward_dynamic_margin", 0.10)
        self.declare_parameter("controller.action_indices", [0, 1, 2, 3, 4, 5, 6])

    def _on_map(self, msg: OccupancyGrid) -> None:
        origin_yaw = yaw_from_quaternion(msg.info.origin.orientation)
        if abs(origin_yaw) > 1e-6 and not self._reported_map_yaw:
            self.get_logger().warning(
                "Map origin yaw is non-zero; this node assumes axis-aligned OccupancyGrid data."
            )
            self._reported_map_yaw = True

        self.navigator.set_map(
            msg.data,
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin=(float(msg.info.origin.position.x), float(msg.info.origin.position.y)),
        )
        replan = self.navigator.replan_to_current_goal()
        if replan is not None:
            self._handle_plan_result(replan)

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.navigator.set_robot_pose(
            np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_from_quaternion(pose.orientation),
                ],
                dtype=float,
            )
        )

    def _on_dynamic_obstacles(self, msg: MarkerArray) -> None:
        observations: list[ObstacleObservation] = []
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        for marker in msg.markers:
            if marker.action in (Marker.DELETEALL, Marker.DELETE):
                continue
            center = self._center_from_marker(marker)
            if center is None:
                continue
            stamp = marker.header.stamp
            stamp_sec = stamp_to_seconds(stamp) if stamp.sec or stamp.nanosec else now_sec
            observations.append(
                ObstacleObservation(
                    marker_id=int(marker.id),
                    position=center,
                    stamp_sec=float(stamp_sec),
                )
            )
        self.navigator.set_obstacle_observations(observations)

    def _center_from_marker(self, marker: Marker) -> np.ndarray | None:
        if marker.points:
            points = marker.points[:-1] if len(marker.points) > 1 else marker.points
            coords = np.asarray([[float(point.x), float(point.y)] for point in points], dtype=float)
            if coords.size:
                return np.mean(coords, axis=0)
        return np.array([float(marker.pose.position.x), float(marker.pose.position.y)], dtype=float)

    def _on_goal(self, msg: PoseStamped) -> None:
        result = self.navigator.plan_to_goal(
            np.array([float(msg.pose.position.x), float(msg.pose.position.y)], dtype=float)
        )
        self._handle_plan_result(result)

    def _on_timer(self) -> None:
        result = self.navigator.step()
        self._publish_cmd(result.command)
        if result.subgoal_world is not None:
            self._publish_subgoal(result.subgoal_world)
        self._log_step_message(result)

    def _handle_plan_result(self, result: PlanResult) -> None:
        if result.success and result.path_world is not None:
            self._publish_path(result.path_world)
            self.get_logger().info(result.message)
            return
        self._publish_empty_path()
        self._publish_cmd(np.zeros((2,), dtype=np.float32))
        self.get_logger().warning(result.message)

    def _publish_cmd(self, command: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x = float(command[0])
        msg.angular.z = float(command[1])
        self.cmd_pub.publish(msg)

    def _publish_path(self, path_world: np.ndarray) -> None:
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        for index, point in enumerate(path_world):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation = quaternion_from_yaw(self._path_yaw_at(path_world, index))
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_empty_path(self) -> None:
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.path_pub.publish(msg)

    def _path_yaw_at(self, path_world: np.ndarray, index: int) -> float:
        if len(path_world) < 2:
            return 0.0
        if index < len(path_world) - 1:
            delta = path_world[index + 1] - path_world[index]
        else:
            delta = path_world[index] - path_world[index - 1]
        return math.atan2(float(delta[1]), float(delta[0]))

    def _publish_subgoal(self, subgoal_world: np.ndarray) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(subgoal_world[0])
        msg.pose.position.y = float(subgoal_world[1])
        msg.pose.orientation.w = 1.0
        self.subgoal_pub.publish(msg)

    def _log_step_message(self, result: StepResult) -> None:
        if result.message is None:
            return
        if result.message_level == "error":
            self.get_logger().error(result.message)
        elif result.message_level == "warning":
            self._warn_throttled(result.message)
        else:
            self.get_logger().info(result.message)

    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._warn_last_time.get(message, -float("inf"))
        if now - last >= period:
            self.get_logger().warning(message)
            self._warn_last_time[message] = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorldModelNavNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
