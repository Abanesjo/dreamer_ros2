#!/usr/bin/env python3

import math
from typing import Iterable, List, Sequence, Tuple

import rclpy
from dwb_msgs.msg import LocalPlanEvaluation, TrajectoryScore
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class DwbTrajectoryVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("dwb_trajectory_visualizer")

        self.declare_parameter("evaluation_topic", "/FollowPath/evaluation,/evaluation")
        self.declare_parameter("trajectory_topic", "/trajectories")
        self.declare_parameter("fallback_frame_id", "odom")
        self.declare_parameter("max_trajectories", 150)
        self.declare_parameter("publish_frequency", 5.0)
        self.declare_parameter("trajectory_point_step", 1)
        self.declare_parameter("line_width", 0.018)
        self.declare_parameter("best_line_width", 0.035)
        self.declare_parameter("marker_lifetime", 0.4)

        self._evaluation_topics = self._split_topics(
            self._get_string_parameter("evaluation_topic")
        )
        self._trajectory_topic = self._get_string_parameter("trajectory_topic")
        self._fallback_frame_id = self._get_string_parameter("fallback_frame_id")
        self._max_trajectories = max(1, self._get_int_parameter("max_trajectories"))
        publish_frequency = max(0.1, self._get_float_parameter("publish_frequency"))
        self._trajectory_point_step = max(1, self._get_int_parameter("trajectory_point_step"))
        self._line_width = max(0.001, self._get_float_parameter("line_width"))
        self._best_line_width = max(
            self._line_width, self._get_float_parameter("best_line_width")
        )
        self._marker_lifetime = max(0.0, self._get_float_parameter("marker_lifetime"))

        self._latest_evaluation = None
        self._has_unpublished_evaluation = False

        self._evaluation_subs = [
            self.create_subscription(
                LocalPlanEvaluation,
                topic,
                self._on_evaluation,
                10,
            )
            for topic in self._evaluation_topics
        ]
        self._trajectory_pub = self.create_publisher(
            MarkerArray,
            self._trajectory_topic,
            10,
        )
        self._publish_timer = self.create_timer(1.0 / publish_frequency, self._on_timer)

        self.get_logger().info(
            "Publishing downsampled DWB rollouts from "
            f"{', '.join(self._evaluation_topics)} to {self._trajectory_topic}"
        )

    def _on_evaluation(self, msg: LocalPlanEvaluation) -> None:
        self._latest_evaluation = msg
        self._has_unpublished_evaluation = True

    def _on_timer(self) -> None:
        if self._latest_evaluation is None or not self._has_unpublished_evaluation:
            return

        marker_array = self._build_markers(self._latest_evaluation)
        self._trajectory_pub.publish(marker_array)
        self._has_unpublished_evaluation = False

    def _build_markers(self, evaluation: LocalPlanEvaluation) -> MarkerArray:
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_marker(evaluation))

        selected = self._select_trajectories(evaluation.twists, evaluation.best_index)
        finite_scores = [score.total for _, score in selected if math.isfinite(score.total)]
        min_score = min(finite_scores, default=0.0)
        max_score = max(finite_scores, default=min_score)

        for marker_id, (trajectory_index, score) in enumerate(selected, start=1):
            marker = Marker()
            marker.header = evaluation.header
            if not marker.header.frame_id:
                marker.header.frame_id = self._fallback_frame_id
            marker.ns = "dwb_rollouts"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = (
                self._best_line_width
                if trajectory_index == evaluation.best_index
                else self._line_width
            )
            marker.color.r, marker.color.g, marker.color.b = self._score_color(
                score.total, min_score, max_score
            )
            marker.color.a = 0.9 if trajectory_index == evaluation.best_index else 0.38
            marker.lifetime = self._duration(self._marker_lifetime)
            marker.points = list(self._trajectory_points(score))

            if len(marker.points) >= 2:
                marker_array.markers.append(marker)

        return marker_array

    def _delete_all_marker(self, evaluation: LocalPlanEvaluation) -> Marker:
        marker = Marker()
        marker.header = evaluation.header
        if not marker.header.frame_id:
            marker.header.frame_id = self._fallback_frame_id
        marker.ns = "dwb_rollouts"
        marker.action = Marker.DELETEALL
        return marker

    def _select_trajectories(
        self, scores: Sequence[TrajectoryScore], best_index: int
    ) -> List[Tuple[int, TrajectoryScore]]:
        valid = [
            (index, score)
            for index, score in enumerate(scores)
            if self._is_valid_score(score)
        ]
        if not valid:
            return []

        valid.sort(key=lambda item: item[1].total)
        if len(valid) <= self._max_trajectories:
            return valid

        selected_indices = set()
        if 0 <= best_index < len(scores) and self._is_valid_score(scores[best_index]):
            selected_indices.add(best_index)

        slots = max(0, self._max_trajectories - len(selected_indices))
        if slots > 0:
            ranked_without_best = [
                item for item in valid if item[0] not in selected_indices
            ]
            for item in self._rank_quantiles(ranked_without_best, slots):
                selected_indices.add(item[0])

        selected = [item for item in valid if item[0] in selected_indices]
        selected.sort(key=lambda item: item[1].total)
        return selected[: self._max_trajectories]

    def _rank_quantiles(
        self, ranked: Sequence[Tuple[int, TrajectoryScore]], slots: int
    ) -> List[Tuple[int, TrajectoryScore]]:
        if slots <= 0 or not ranked:
            return []
        if len(ranked) <= slots:
            return list(ranked)
        if slots == 1:
            return [ranked[0]]

        selected = []
        last_index = len(ranked) - 1
        used = set()
        for slot in range(slots):
            rank = int(round(slot * last_index / float(slots - 1)))
            while rank in used and rank < last_index:
                rank += 1
            while rank in used and rank > 0:
                rank -= 1
            used.add(rank)
            selected.append(ranked[rank])
        return selected

    def _is_valid_score(self, score: TrajectoryScore) -> bool:
        if not math.isfinite(score.total) or len(score.traj.poses) < 2:
            return False
        return all(
            math.isfinite(pose.x) and math.isfinite(pose.y)
            for pose in score.traj.poses
        )

    def _trajectory_points(self, score: TrajectoryScore) -> Iterable[Point]:
        poses = score.traj.poses
        last_index = len(poses) - 1
        emitted_last = False

        for index, pose in enumerate(poses):
            if index % self._trajectory_point_step != 0 and index != last_index:
                continue
            point = Point()
            point.x = pose.x
            point.y = pose.y
            point.z = 0.0
            emitted_last = emitted_last or index == last_index
            yield point

        if not emitted_last and poses:
            point = Point()
            point.x = poses[-1].x
            point.y = poses[-1].y
            point.z = 0.0
            yield point

    def _score_color(self, score: float, min_score: float, max_score: float) -> Tuple[float, float, float]:
        if not math.isfinite(score) or max_score <= min_score:
            return (0.0, 0.85, 0.15)

        ratio = max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))
        return (ratio, 1.0 - ratio, 0.12)

    def _duration(self, seconds: float):
        duration = Duration(seconds=seconds).to_msg()
        return duration

    def _get_string_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _get_int_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _get_float_parameter(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _split_topics(self, topics: str) -> List[str]:
        split_topics = [topic.strip() for topic in topics.split(",") if topic.strip()]
        if not split_topics:
            raise ValueError("At least one DWB evaluation topic must be provided")
        return split_topics


def main() -> None:
    rclpy.init()
    node = DwbTrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
