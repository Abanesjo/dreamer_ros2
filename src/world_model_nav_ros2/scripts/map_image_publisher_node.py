#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from world_model_nav_ros2.ros_utils import (
    default_qos,
    marker_center,
    path_from_msg,
    transient_local_qos,
    yaw_from_quaternion,
)


@dataclass(frozen=True)
class RenderedMap:
    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    frame_id: str


@dataclass(frozen=True)
class RenderedObstacle:
    position: np.ndarray
    radius: float


class MapImagePublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_map_image_publisher")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dynamic_obstacles_topic = str(self.get_parameter("dynamic_obstacles_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.tracked_waypoint_topic = str(self.get_parameter("tracked_waypoint_topic").value)
        self.map_image_topic = str(self.get_parameter("map_image_topic").value)
        self.publish_frequency = float(self.get_parameter("map_image_frequency").value)
        self.figure_size = self._figure_size_parameter("map_image_figure_size")
        self.dpi = int(self.get_parameter("map_image_dpi").value)
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.dynamic_obstacle_radius = float(self.get_parameter("dynamic_obstacle_radius").value)
        self.map_occupied_threshold = int(self.get_parameter("map_occupied_threshold").value)
        self.treat_unknown_as_occupied = bool(self.get_parameter("treat_unknown_as_occupied").value)

        self._map: RenderedMap | None = None
        self._robot_pose: np.ndarray | None = None
        self._path_world: np.ndarray | None = None
        self._tracked_waypoint: np.ndarray | None = None
        self._obstacles: list[RenderedObstacle] = []
        self._warned_frames: set[str] = set()
        self._warn_last_time: dict[str, float] = {}

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
        self.image_pub = self.create_publisher(Image, self.map_image_topic, default_qos(depth=1))
        self.timer = self.create_timer(1.0 / max(self.publish_frequency, 1e-6), self._on_timer)

        self.get_logger().info(
            "Map image publisher ready: "
            f"map={self.map_topic}, odom={self.odom_topic}, obstacles={self.dynamic_obstacles_topic}, "
            f"path={self.path_topic}, tracked={self.tracked_waypoint_topic}, image={self.map_image_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("dynamic_obstacles_topic", "/dynamic_obstacles")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("tracked_waypoint_topic", "/world_model_nav/tracked_waypoint")
        self.declare_parameter("map_image_topic", "/map_image")
        self.declare_parameter("map_image_frequency", 5.0)
        self.declare_parameter("map_image_figure_size", [8.0, 8.0])
        self.declare_parameter("map_image_dpi", 120)
        self.declare_parameter("robot_radius", 0.4)
        self.declare_parameter("dynamic_obstacle_radius", 0.5)
        self.declare_parameter("map_occupied_threshold", 50)
        self.declare_parameter("treat_unknown_as_occupied", True)

    def _figure_size_parameter(self, name: str) -> tuple[float, float]:
        raw_value = self.get_parameter(name).value
        if isinstance(raw_value, Sequence) and len(raw_value) >= 2:
            return (float(raw_value[0]), float(raw_value[1]))
        self.get_logger().warning(f"Invalid {name}; using default 8.0 x 8.0 inches.")
        return (8.0, 8.0)

    def _on_map(self, msg: OccupancyGrid) -> None:
        origin_yaw = yaw_from_quaternion(msg.info.origin.orientation)
        if abs(origin_yaw) > 1e-6:
            self._warn_once(
                "map_origin_yaw",
                "Map image publisher assumes axis-aligned OccupancyGrid data; origin yaw is non-zero.",
            )

        width = int(msg.info.width)
        height = int(msg.info.height)
        raw = np.asarray(msg.data, dtype=np.int16).reshape((height, width))
        occupancy = np.zeros(raw.shape, dtype=np.float32)
        occupancy[raw >= self.map_occupied_threshold] = 1.0
        if self.treat_unknown_as_occupied:
            occupancy[raw < 0] = 1.0
        else:
            occupancy[raw < 0] = 0.5

        self._map = RenderedMap(
            occupancy=occupancy,
            resolution=float(msg.info.resolution),
            origin=(float(msg.info.origin.position.x), float(msg.info.origin.position.y)),
            frame_id=str(msg.header.frame_id) if msg.header.frame_id else self.frame_id,
        )
        self._warn_if_frame_mismatch("map", self._map.frame_id)

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._robot_pose = np.array(
            [
                float(pose.position.x),
                float(pose.position.y),
                yaw_from_quaternion(pose.orientation),
            ],
            dtype=float,
        )
        self._warn_if_frame_mismatch("odom", str(msg.header.frame_id))

    def _on_path(self, msg: NavPath) -> None:
        self._path_world = path_from_msg(msg)
        self._warn_if_frame_mismatch("path", str(msg.header.frame_id))

    def _on_tracked_waypoint(self, msg: PoseStamped) -> None:
        self._tracked_waypoint = np.array(
            [float(msg.pose.position.x), float(msg.pose.position.y)],
            dtype=float,
        )
        self._warn_if_frame_mismatch("tracked_waypoint", str(msg.header.frame_id))

    def _on_dynamic_obstacles(self, msg: MarkerArray) -> None:
        obstacles: list[RenderedObstacle] = []
        saw_delete_all = False
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                saw_delete_all = True
                continue
            if marker.action == Marker.DELETE:
                continue
            center = marker_center(marker)
            if center is None:
                continue
            obstacles.append(
                RenderedObstacle(
                    position=np.asarray(center, dtype=float).reshape(2),
                    radius=self._marker_radius(marker, center),
                )
            )
            self._warn_if_frame_mismatch("dynamic_obstacles", str(marker.header.frame_id))

        if obstacles or saw_delete_all:
            self._obstacles = obstacles

    def _on_timer(self) -> None:
        if self._map is None:
            self._warn_throttled(f"Waiting for {self.map_topic} before publishing {self.map_image_topic}.")
            return

        rgb = self._render_rgb()
        self.image_pub.publish(self._image_msg(rgb, frame_id=self._map.frame_id))

    def _render_rgb(self) -> np.ndarray:
        assert self._map is not None

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        try:
            self._draw_map(ax, self._map)
            if self._path_world is not None and len(self._path_world):
                ax.plot(self._path_world[:, 0], self._path_world[:, 1], color="tab:cyan", linewidth=1.2)
            if self._robot_pose is not None:
                self._draw_robot(ax, self._robot_pose)
            if self._tracked_waypoint is not None:
                ax.scatter(
                    self._tracked_waypoint[0],
                    self._tracked_waypoint[1],
                    c="orange",
                    marker="x",
                    s=45,
                )
            for obstacle in self._obstacles:
                self._draw_dynamic_obstacle(ax, obstacle)

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            return frame[:, :, :3].copy()
        finally:
            plt.close(fig)

    def _draw_map(self, ax: plt.Axes, rendered_map: RenderedMap) -> None:
        occupancy = rendered_map.occupancy
        resolution = rendered_map.resolution
        origin = rendered_map.origin
        extent = [
            origin[0],
            origin[0] + occupancy.shape[1] * resolution,
            origin[1],
            origin[1] + occupancy.shape[0] * resolution,
        ]
        ax.imshow(occupancy, cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")

    def _draw_robot(self, ax: plt.Axes, robot_pose: np.ndarray) -> None:
        circle = plt.Circle(
            (robot_pose[0], robot_pose[1]),
            radius=self.robot_radius,
            color="tab:red",
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(circle)
        heading_end = robot_pose[:2] + 1.1 * np.array(
            [np.cos(robot_pose[2]), np.sin(robot_pose[2])],
            dtype=float,
        )
        ax.plot(
            [robot_pose[0], heading_end[0]],
            [robot_pose[1], heading_end[1]],
            color="tab:red",
            linewidth=1.4,
        )

    def _draw_dynamic_obstacle(self, ax: plt.Axes, obstacle: RenderedObstacle) -> None:
        circle = plt.Circle(
            (obstacle.position[0], obstacle.position[1]),
            radius=obstacle.radius,
            color="tab:blue",
            alpha=0.45,
        )
        ax.add_patch(circle)

    def _marker_radius(self, marker: Marker, center: np.ndarray) -> float:
        if marker.points:
            points = marker.points[:-1] if len(marker.points) > 1 else marker.points
            coords = np.asarray([[float(point.x), float(point.y)] for point in points], dtype=float)
            if coords.size:
                distances = np.linalg.norm(coords - np.asarray(center, dtype=float).reshape(1, 2), axis=1)
                radius = float(np.mean(distances))
                if radius > 0.0:
                    return radius
        if marker.scale.x > 0.0:
            return float(marker.scale.x)
        return self.dynamic_obstacle_radius

    def _image_msg(self, rgb: np.ndarray, *, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(rgb.shape[1] * 3)
        msg.data = rgb.tobytes()
        return msg

    def _warn_if_frame_mismatch(self, source: str, frame_id: str) -> None:
        if not frame_id or frame_id == self.frame_id:
            return
        key = f"{source}:{frame_id}"
        self._warn_once(
            key,
            f"{source} frame '{frame_id}' differs from configured frame '{self.frame_id}'; rendering assumes they are aligned.",
        )

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_frames:
            return
        self._warned_frames.add(key)
        self.get_logger().warning(message)

    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._warn_last_time.get(message, -float("inf"))
        if now - last >= period:
            self.get_logger().warning(message)
            self._warn_last_time[message] = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MapImagePublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
