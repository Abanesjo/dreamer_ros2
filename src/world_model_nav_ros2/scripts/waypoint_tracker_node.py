#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node

from world_model_nav_ros2.core import PathWaypointTracker, WaypointResult
from world_model_nav_ros2.ros_utils import (
    default_qos,
    path_from_msg,
    transient_local_qos,
    waypoint_to_pose_msg,
    yaw_from_quaternion,
)


class WaypointTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_waypoint_tracker")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.subgoal_topic = str(self.get_parameter("subgoal_topic").value)
        self.tracked_waypoint_topic = str(self.get_parameter("tracked_waypoint_topic").value)
        self.waypoint_frequency = float(self.get_parameter("waypoint_frequency").value)

        self.tracker = PathWaypointTracker(
            lookahead_distance=float(self.get_parameter("lookahead_distance").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
        )
        self._warn_last_time: dict[str, float] = {}

        self.path_sub = self.create_subscription(
            NavPath,
            self.path_topic,
            self._on_path,
            transient_local_qos(),
        )
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, default_qos())
        self.subgoal_pub = self.create_publisher(PoseStamped, self.subgoal_topic, default_qos())
        self.tracked_waypoint_pub = self.create_publisher(
            PoseStamped,
            self.tracked_waypoint_topic,
            default_qos(),
        )
        self.timer = self.create_timer(1.0 / max(self.waypoint_frequency, 1e-6), self._on_timer)

        self.get_logger().info(
            "Waypoint tracker ready: "
            f"path={self.path_topic}, odom={self.odom_topic}, tracked={self.tracked_waypoint_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("subgoal_topic", "/world_model_nav/subgoal")
        self.declare_parameter("tracked_waypoint_topic", "/world_model_nav/tracked_waypoint")
        self.declare_parameter("waypoint_frequency", 10.0)
        self.declare_parameter("lookahead_distance", 1.0)
        self.declare_parameter("goal_tolerance", 0.4)

    def _on_path(self, msg: NavPath) -> None:
        self.tracker.set_path(path_from_msg(msg))

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.tracker.set_robot_pose(
            np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_from_quaternion(pose.orientation),
                ],
                dtype=float,
            )
        )

    def _on_timer(self) -> None:
        result = self.tracker.step()
        if result.waypoint_world is not None:
            self._publish_tracked_waypoint(result.waypoint_world)
        self._log_step_message(result)

    def _publish_tracked_waypoint(self, waypoint: np.ndarray) -> None:
        msg = waypoint_to_pose_msg(
            waypoint,
            frame_id=self.frame_id,
            stamp=self.get_clock().now().to_msg(),
        )
        self.subgoal_pub.publish(msg)
        self.tracked_waypoint_pub.publish(msg)

    def _log_step_message(self, result: WaypointResult) -> None:
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
    node = WaypointTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
