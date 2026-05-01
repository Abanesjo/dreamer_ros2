#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from world_model_nav_ros2.core import (
    ControllerConfigValues,
    NavigationConfig,
    ObstacleObservation,
    StepResult,
    WorldModelPolicyController,
)
from world_model_nav_ros2.ros_utils import (
    default_policy_path,
    default_qos,
    marker_center,
    path_from_msg,
    stamp_to_seconds,
    transient_local_qos,
    yaw_from_quaternion,
)


class PolicyControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_policy_controller")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dynamic_obstacles_topic = str(self.get_parameter("dynamic_obstacles_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.tracked_waypoint_topic = str(self.get_parameter("tracked_waypoint_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.robot_marker_topic = str(self.get_parameter("robot_marker_topic").value)
        self.policy_debug_topic = str(self.get_parameter("policy_debug_topic").value)
        self.control_frequency = float(self.get_parameter("control_frequency").value)
        configured_policy_path = str(self.get_parameter("policy_path").value)
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.robot_marker_segments = max(8, int(self.get_parameter("robot_marker_segments").value))
        self.robot_marker_line_width = float(self.get_parameter("robot_marker_line_width").value)

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
            dynamic_obstacle_stale_timeout=float(
                self.get_parameter("dynamic_obstacle_stale_timeout").value
            ),
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
            action_indices=tuple(
                int(value) for value in self.get_parameter("controller.action_indices").value
            ),
        )

        self.controller = WorldModelPolicyController(nav_config, controller_config)
        self._warn_last_time: dict[str, float] = {}
        self._reported_map_yaw = False

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._on_map,
            transient_local_qos(),
        )
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, default_qos())
        self.obstacle_sub = self.create_subscription(
            MarkerArray,
            self.dynamic_obstacles_topic,
            self._on_dynamic_obstacles,
            default_qos(),
        )
        self.path_sub = self.create_subscription(
            NavPath,
            self.path_topic,
            self._on_path,
            transient_local_qos(),
        )
        self.tracked_waypoint_sub = self.create_subscription(
            PoseStamped,
            self.tracked_waypoint_topic,
            self._on_tracked_waypoint,
            default_qos(),
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, default_qos())
        self.robot_marker_pub = self.create_publisher(Marker, self.robot_marker_topic, default_qos())
        self.policy_debug_pub = self.create_publisher(String, self.policy_debug_topic, default_qos())

        self.timer = self.create_timer(1.0 / max(self.control_frequency, 1e-6), self._on_timer)
        self.get_logger().info(
            "Policy controller ready: "
            f"policy={nav_config.policy_path}, path={self.path_topic}, "
            f"tracked={self.tracked_waypoint_topic}, cmd={self.cmd_vel_topic}, "
            f"debug={self.policy_debug_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("dynamic_obstacles_topic", "/dynamic_obstacles")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("tracked_waypoint_topic", "/world_model_nav/tracked_waypoint")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("robot_marker_topic", "/world_model_nav/robot_footprint")
        self.declare_parameter("policy_debug_topic", "/world_model_nav/policy_debug")
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("policy_path", default_policy_path())
        self.declare_parameter("device", "auto")
        self.declare_parameter("policy_seed", 0)

        self.declare_parameter("lookahead_distance", 1.0)
        self.declare_parameter("goal_tolerance", 0.4)
        self.declare_parameter("robot_radius", 0.4)
        self.declare_parameter("robot_marker_segments", 64)
        self.declare_parameter("robot_marker_line_width", 0.04)
        self.declare_parameter("inflation_margin", 0.05)
        self.declare_parameter("map_occupied_threshold", 50)
        self.declare_parameter("treat_unknown_as_occupied", True)
        self.declare_parameter("dynamic_obstacle_radius", 0.5)
        self.declare_parameter("expected_dynamic_obstacles", 4)
        self.declare_parameter("min_obstacle_dt", 1e-3)
        self.declare_parameter("dynamic_obstacle_stale_timeout", 1.0)

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
        self.declare_parameter("controller.backward_penalty", 10.0)
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

        self.controller.set_map(
            msg.data,
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin=(float(msg.info.origin.position.x), float(msg.info.origin.position.y)),
        )

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.controller.set_robot_pose(
            np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_from_quaternion(pose.orientation),
                ],
                dtype=float,
            )
        )

    def _on_path(self, msg: NavPath) -> None:
        changed = self.controller.set_path(path_from_msg(msg))
        if changed and not msg.poses:
            self._publish_cmd(np.zeros((2,), dtype=np.float32))

    def _on_tracked_waypoint(self, msg: PoseStamped) -> None:
        self.controller.set_tracked_waypoint(
            np.array([float(msg.pose.position.x), float(msg.pose.position.y)], dtype=float)
        )

    def _on_dynamic_obstacles(self, msg: MarkerArray) -> None:
        observations: list[ObstacleObservation] = []
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        for marker in msg.markers:
            if marker.action in (Marker.DELETEALL, Marker.DELETE):
                continue
            center = marker_center(marker)
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
        self.controller.set_obstacle_observations(observations, stamp_sec=now_sec)

    def _on_timer(self) -> None:
        self._publish_robot_marker()
        result = self.controller.step()
        self._publish_cmd(result.command)
        self._log_step_message(result)
        self._publish_policy_debug(result)

    def _publish_cmd(self, command: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x = float(command[0])
        msg.angular.z = float(command[1])
        self.cmd_pub.publish(msg)

    def _publish_robot_marker(self) -> None:
        if self.controller.robot_pose is None:
            return

        robot_pose = self.controller.robot_pose
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = "robot_footprint"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.robot_marker_line_width
        marker.color.r = 0.1
        marker.color.g = 0.45
        marker.color.b = 1.0
        marker.color.a = 1.0

        for index in range(self.robot_marker_segments + 1):
            angle = (2.0 * math.pi * index) / float(self.robot_marker_segments)
            point = Point()
            point.x = float(robot_pose[0] + self.robot_radius * math.cos(angle))
            point.y = float(robot_pose[1] + self.robot_radius * math.sin(angle))
            point.z = 0.03
            marker.points.append(point)

        self.robot_marker_pub.publish(marker)

    def _log_step_message(self, result: StepResult) -> None:
        if result.message is None:
            return
        if result.message_level == "error":
            self.get_logger().error(result.message)
        elif result.message_level == "warning":
            self._warn_throttled(result.message)
        else:
            self.get_logger().info(result.message)

    def _publish_policy_debug(self, result: StepResult) -> None:
        if result.selected_action_name is None:
            return
        msg = String()
        msg.data = json.dumps(
            {
                "action": str(result.selected_action_name),
                "v": float(result.command[0]),
                "omega": float(result.command[1]),
                "min_clearance": (
                    None
                    if result.chosen_action_min_clearance is None
                    else float(result.chosen_action_min_clearance)
                ),
                "pose_estimate_error": (
                    None if result.pose_estimate_error is None else float(result.pose_estimate_error)
                ),
                "selection_mode": result.selection_mode,
                "num_feasible_non_stop": result.num_feasible_non_stop,
                "num_feasible_all": result.num_feasible_all,
                "obstacle_count": result.obstacle_count,
                "reasons": list(result.chosen_infeasible_reasons),
            },
            sort_keys=True,
        )
        self.policy_debug_pub.publish(msg)

    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._warn_last_time.get(message, -float("inf"))
        if now - last >= period:
            self.get_logger().warning(message)
            self._warn_last_time[message] = now

    def _info_throttled(self, key: str, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._warn_last_time.get(key, -float("inf"))
        if now - last >= period:
            self.get_logger().info(message)
            self._warn_last_time[key] = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PolicyControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
