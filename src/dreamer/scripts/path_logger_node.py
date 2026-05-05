#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


CSV_COLUMNS = (
    "run_id",
    "record_type",
    "stamp_sec",
    "frame_id",
    "path_snapshot_id",
    "path_pose_index",
    "x",
    "y",
    "yaw",
)


def stamp_to_seconds(stamp: object) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(quat: object) -> float:
    siny_cosp = 2.0 * (float(quat.w) * float(quat.z) + float(quat.x) * float(quat.y))
    cosy_cosp = 1.0 - 2.0 * (float(quat.y) * float(quat.y) + float(quat.z) * float(quat.z))
    return math.atan2(siny_cosp, cosy_cosp)


class PathLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("dreamer_path_logger")
        self._declare_parameters()

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.service_name = str(self.get_parameter("path_logger_service").value)
        self.data_dir = Path(str(self.get_parameter("path_log_data_dir").value)).expanduser()

        self.recording = False
        self.run_id = ""
        self.csv_path: Path | None = None
        self.csv_file = None
        self.writer: csv.DictWriter | None = None
        self.path_snapshot_id = 0
        self.latest_path: NavPath | None = None

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._on_odom,
            QoSProfile(depth=100),
        )
        self.path_sub = self.create_subscription(
            NavPath,
            self.path_topic,
            self._on_path,
            self._path_qos(),
        )
        self.trigger_service = self.create_service(Trigger, self.service_name, self._on_trigger)

        self.get_logger().info(
            f"Path logger ready: service={self.service_name}, odom={self.odom_topic}, "
            f"path={self.path_topic}, data_dir={self.data_dir}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("path_logger_service", "/path_logger")
        self.declare_parameter("path_log_data_dir", "data")

    def _path_qos(self) -> QoSProfile:
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        return qos

    def _on_trigger(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.recording:
            stopped_path = self._stop_recording()
            response.success = True
            response.message = f"Stopped path logging: {stopped_path}"
            return response

        started_path = self._start_recording()
        response.success = True
        response.message = f"Started path logging: {started_path}"
        return response

    def _start_recording(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.csv_path = (self.data_dir / f"path_log_{self.run_id}.csv").resolve()
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.recording = True
        self.path_snapshot_id = 0

        if self.latest_path is not None and self.latest_path.poses:
            self._write_path_snapshot(self.latest_path)
        self._flush()
        self.get_logger().info(f"Started path logging: {self.csv_path}")
        return self.csv_path

    def _stop_recording(self) -> Path | None:
        path = self.csv_path
        self.recording = False
        self._flush()
        if self.csv_file is not None:
            self.csv_file.close()
        self.csv_file = None
        self.writer = None
        self.csv_path = None
        self.get_logger().info(f"Stopped path logging: {path}")
        return path

    def _on_odom(self, msg: Odometry) -> None:
        if not self.recording or self.writer is None:
            return
        pose = msg.pose.pose
        stamp_sec = stamp_to_seconds(msg.header.stamp)
        self.writer.writerow(
            {
                "run_id": self.run_id,
                "record_type": "odom",
                "stamp_sec": stamp_sec,
                "frame_id": str(msg.header.frame_id),
                "path_snapshot_id": "",
                "path_pose_index": "",
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "yaw": yaw_from_quaternion(pose.orientation),
            }
        )

    def _on_path(self, msg: NavPath) -> None:
        if msg.poses:
            self.latest_path = msg
        if not self.recording or self.writer is None or not msg.poses:
            return
        self._write_path_snapshot(msg)
        self._flush()

    def _write_path_snapshot(self, msg: NavPath) -> None:
        if self.writer is None:
            return
        self.path_snapshot_id += 1
        stamp_sec = stamp_to_seconds(msg.header.stamp)
        for index, pose_stamped in enumerate(msg.poses):
            pose = pose_stamped.pose
            pose_stamp = pose_stamped.header.stamp
            pose_stamp_sec = stamp_to_seconds(pose_stamp) if pose_stamp.sec or pose_stamp.nanosec else stamp_sec
            frame_id = str(pose_stamped.header.frame_id or msg.header.frame_id)
            self.writer.writerow(
                {
                    "run_id": self.run_id,
                    "record_type": "path",
                    "stamp_sec": pose_stamp_sec,
                    "frame_id": frame_id,
                    "path_snapshot_id": self.path_snapshot_id,
                    "path_pose_index": index,
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "yaw": yaw_from_quaternion(pose.orientation),
                }
            )

    def _flush(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()

    def destroy_node(self) -> bool:
        if self.recording:
            self._stop_recording()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PathLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
