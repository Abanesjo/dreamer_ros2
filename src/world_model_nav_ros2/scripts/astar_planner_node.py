#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.node import Node

from world_model_nav_ros2.core import AStarPathPlanner, NavigationConfig, PlanResult
from world_model_nav_ros2.ros_utils import (
    default_qos,
    empty_path_msg,
    path_to_msg,
    transient_local_qos,
    yaw_from_quaternion,
)


class AStarPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_astar_planner")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.path_republish_frequency = float(self.get_parameter("path_republish_frequency").value)

        nav_config = NavigationConfig(
            policy_path="",
            robot_radius=float(self.get_parameter("robot_radius").value),
            inflation_margin=float(self.get_parameter("inflation_margin").value),
            map_occupied_threshold=int(self.get_parameter("map_occupied_threshold").value),
            treat_unknown_as_occupied=bool(self.get_parameter("treat_unknown_as_occupied").value),
        )
        self.planner = AStarPathPlanner(nav_config)
        self._pending_goal_xy: np.ndarray | None = None
        self._path_world: np.ndarray | None = None
        self._reported_map_yaw = False
        self._warn_last_time: dict[str, float] = {}

        map_qos = transient_local_qos()
        path_qos = transient_local_qos()
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, default_qos())
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, default_qos())
        self.path_pub = self.create_publisher(NavPath, self.path_topic, path_qos)

        self.republish_timer = None
        if self.path_republish_frequency > 0.0:
            self.republish_timer = self.create_timer(
                1.0 / self.path_republish_frequency,
                self._on_republish_timer,
            )

        self.get_logger().info(
            "A* planner ready: "
            f"map={self.map_topic}, goal={self.goal_topic}, odom={self.odom_topic}, path={self.path_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("path_republish_frequency", 1.0)
        self.declare_parameter("robot_radius", 0.19)
        self.declare_parameter("inflation_margin", 0.05)
        self.declare_parameter("map_occupied_threshold", 50)
        self.declare_parameter("treat_unknown_as_occupied", True)

    def _on_map(self, msg: OccupancyGrid) -> None:
        origin_yaw = yaw_from_quaternion(msg.info.origin.orientation)
        if abs(origin_yaw) > 1e-6 and not self._reported_map_yaw:
            self.get_logger().warning(
                "Map origin yaw is non-zero; this node assumes axis-aligned OccupancyGrid data."
            )
            self._reported_map_yaw = True

        self.planner.set_map(
            msg.data,
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin=(float(msg.info.origin.position.x), float(msg.info.origin.position.y)),
        )
        self._try_plan_pending_goal()

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.planner.set_robot_pose(
            np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_from_quaternion(pose.orientation),
                ],
                dtype=float,
            )
        )
        self._try_plan_pending_goal()

    def _on_goal(self, msg: PoseStamped) -> None:
        self._pending_goal_xy = np.array(
            [float(msg.pose.position.x), float(msg.pose.position.y)],
            dtype=float,
        )
        if self.planner.map_state is None or self.planner.robot_pose is None:
            self._path_world = None
            self._publish_empty_path()
            self._warn_throttled("Goal received; waiting for map and odom before one-shot A* planning.")
            return
        self._try_plan_pending_goal()

    def _on_republish_timer(self) -> None:
        self._try_plan_pending_goal()
        if self._path_world is not None:
            self._publish_path(self._path_world)

    def _try_plan_pending_goal(self) -> None:
        if self._pending_goal_xy is None:
            return
        if self.planner.map_state is None or self.planner.robot_pose is None:
            return

        goal_xy = self._pending_goal_xy.copy()
        self._pending_goal_xy = None
        self._handle_plan_result(self.planner.plan_to_goal(goal_xy))

    def _handle_plan_result(self, result: PlanResult) -> None:
        if result.success and result.path_world is not None:
            self._path_world = result.path_world.copy()
            self._publish_path(self._path_world)
            self.get_logger().info(result.message)
            return

        self._path_world = None
        self._publish_empty_path()
        self.get_logger().warning(result.message)

    def _publish_path(self, path_world: np.ndarray) -> None:
        self.path_pub.publish(
            path_to_msg(path_world, frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        )

    def _publish_empty_path(self) -> None:
        self.path_pub.publish(
            empty_path_msg(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        )

    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._warn_last_time.get(message, -float("inf"))
        if now - last >= period:
            self.get_logger().warning(message)
            self._warn_last_time[message] = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
