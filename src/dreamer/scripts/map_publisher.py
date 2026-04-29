#!/usr/bin/env python3

import math
from pathlib import Path

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def quaternion_from_yaw(yaw):
    quat = Quaternion()
    quat.z = math.sin(yaw * 0.5)
    quat.w = math.cos(yaw * 0.5)
    return quat


class MapPublisher(Node):
    def __init__(self):
        super().__init__("dreamer_map_publisher")

        default_map_yaml = str(Path(get_package_share_directory("dreamer")) / "map" / "map.yaml")
        self.declare_parameter("map_yaml", default_map_yaml)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("topic", "/map")

        self._map_yaml = Path(self.get_parameter("map_yaml").value)
        self._frame_id = self.get_parameter("frame_id").value
        topic = self.get_parameter("topic").value

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._publisher = self.create_publisher(OccupancyGrid, topic, qos)

        self._map_msg = self._load_map()
        self._publish_map()
        self._timer = self.create_timer(1.0, self._publish_map)

        self.get_logger().info(
            f"Publishing occupancy grid from {self._map_yaml} on {topic} in frame {self._frame_id}"
        )

    def _publish_map(self):
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._map_msg)

    def _load_map(self):
        with self._map_yaml.open("r", encoding="utf-8") as map_file:
            metadata = yaml.safe_load(map_file)

        image_path = Path(metadata["image"])
        if not image_path.is_absolute():
            image_path = self._map_yaml.parent / image_path

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to load map image: {image_path}")

        if image.ndim == 2:
            gray = image
            alpha = None
        else:
            channels = image.shape[2]
            if channels == 4:
                bgr = image[:, :, :3]
                alpha = image[:, :, 3]
            else:
                bgr = image[:, :, :3]
                alpha = None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        resolution = float(metadata["resolution"])
        origin = metadata.get("origin", [0.0, 0.0, 0.0])
        negate = int(metadata.get("negate", 0))
        occupied_thresh = float(metadata.get("occupied_thresh", 0.65))
        free_thresh = float(metadata.get("free_thresh", 0.196))

        height, width = gray.shape
        msg = OccupancyGrid()
        msg.header.frame_id = self._frame_id
        msg.info.resolution = resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = quaternion_from_yaw(float(origin[2]) if len(origin) > 2 else 0.0)
        msg.data = self._image_to_occupancy_data(gray, alpha, negate, occupied_thresh, free_thresh)
        return msg

    def _image_to_occupancy_data(self, gray, alpha, negate, occupied_thresh, free_thresh):
        height, width = gray.shape
        data = []

        for map_y in range(height):
            image_y = height - map_y - 1
            for x in range(width):
                if alpha is not None and alpha[image_y, x] == 0:
                    data.append(-1)
                    continue

                pixel = float(gray[image_y, x])
                occupancy = pixel / 255.0 if negate else (255.0 - pixel) / 255.0

                if occupancy > occupied_thresh:
                    data.append(100)
                elif occupancy < free_thresh:
                    data.append(0)
                else:
                    data.append(-1)

        return data


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
