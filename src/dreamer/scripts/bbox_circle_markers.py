#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


class BboxCircleMarkers(Node):
    def __init__(self):
        super().__init__("dreamer_bbox_circle_markers")

        self.declare_parameter("input_topic", "/bbox_3d")
        self.declare_parameter("marker_topic", "/dynamic_obstacles")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("radius", 1.0)
        self.declare_parameter("segments", 64)
        self.declare_parameter("line_width", 0.04)

        self._input_topic = self.get_parameter("input_topic").value
        self._marker_topic = self.get_parameter("marker_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._radius = float(self.get_parameter("radius").value)
        self._segments = max(8, int(self.get_parameter("segments").value))
        self._line_width = float(self.get_parameter("line_width").value)

        self._publisher = self.create_publisher(MarkerArray, self._marker_topic, QoSProfile(depth=10))
        self._subscription = self.create_subscription(
            Detection3DArray,
            self._input_topic,
            self._detections_callback,
            QoSProfile(depth=10),
        )

        self.get_logger().info(
            f"Publishing {self._radius:.2f} m bbox circles from {self._input_topic} "
            f"to {self._marker_topic} in frame {self._frame_id}"
        )

    def _detections_callback(self, msg):
        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec else self.get_clock().now().to_msg()

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self._frame_id
        delete_all.header.stamp = stamp
        delete_all.ns = "dynamic_obstacles_clear"
        delete_all.id = 0
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for marker_id, detection in enumerate(msg.detections):
            center = detection.bbox.center.position
            markers.markers.append(self._make_circle_marker(marker_id, center, stamp))

        self._publisher.publish(markers)

    def _make_circle_marker(self, marker_id, center, stamp):
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "dynamic_obstacles"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self._line_width
        marker.color = ColorRGBA(r=1.0, g=0.18, b=0.05, a=1.0)
        marker.points = self._circle_points(center)
        return marker

    def _circle_points(self, center):
        points = []
        for index in range(self._segments + 1):
            angle = (2.0 * math.pi * index) / self._segments
            point = Point()
            point.x = center.x + self._radius * math.cos(angle)
            point.y = center.y + self._radius * math.sin(angle)
            point.z = center.z
            points.append(point)
        return points


def main(args=None):
    rclpy.init(args=args)
    node = BboxCircleMarkers()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
