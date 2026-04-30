from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path as NavPath
from rclpy.qos import DurabilityPolicy, QoSHistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker


PACKAGE_NAME = "world_model_nav_ros2"


def default_policy_path() -> str:
    try:
        return str(Path(get_package_share_directory(PACKAGE_NAME)) / "model" / "best.pt")
    except PackageNotFoundError:
        return str(Path(__file__).resolve().parents[1] / "model" / "best.pt")


def default_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(depth=int(depth))


def transient_local_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=int(depth),
    )


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


def path_from_msg(msg: NavPath) -> np.ndarray | None:
    if not msg.poses:
        return None
    return np.asarray(
        [[float(pose.pose.position.x), float(pose.pose.position.y)] for pose in msg.poses],
        dtype=float,
    )


def path_to_msg(path_world: np.ndarray, *, frame_id: str, stamp: Any) -> NavPath:
    msg = NavPath()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    for index, point in enumerate(np.asarray(path_world, dtype=float).reshape((-1, 2))):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.orientation = quaternion_from_yaw(path_yaw_at(path_world, index))
        msg.poses.append(pose)
    return msg


def empty_path_msg(*, frame_id: str, stamp: Any) -> NavPath:
    msg = NavPath()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    return msg


def path_yaw_at(path_world: np.ndarray, index: int) -> float:
    path = np.asarray(path_world, dtype=float).reshape((-1, 2))
    if len(path) < 2:
        return 0.0
    if index < len(path) - 1:
        delta = path[index + 1] - path[index]
    else:
        delta = path[index] - path[index - 1]
    return math.atan2(float(delta[1]), float(delta[0]))


def waypoint_to_pose_msg(waypoint: np.ndarray, *, frame_id: str, stamp: Any) -> PoseStamped:
    waypoint_xy = np.asarray(waypoint, dtype=float).reshape(2)
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    msg.pose.position.x = float(waypoint_xy[0])
    msg.pose.position.y = float(waypoint_xy[1])
    msg.pose.orientation.w = 1.0
    return msg


def marker_center(marker: Marker) -> np.ndarray | None:
    if marker.points:
        points = marker.points[:-1] if len(marker.points) > 1 else marker.points
        coords = np.asarray([[float(point.x), float(point.y)] for point in points], dtype=float)
        if coords.size:
            return np.mean(coords, axis=0)
    return np.array([float(marker.pose.position.x), float(marker.pose.position.y)], dtype=float)
