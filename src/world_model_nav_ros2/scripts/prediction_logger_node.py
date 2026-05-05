#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from world_model_nav_ros2.ros_utils import default_qos, marker_center, stamp_to_seconds


CSV_COLUMNS = (
    "run_id",
    "policy_mode",
    "prediction_id",
    "prediction_stamp_sec",
    "prediction_dt_sec",
    "target_stamp_sec",
    "horizon_step",
    "horizon_time_sec",
    "action_index",
    "action_name",
    "obstacle_id",
    "baseline_x",
    "baseline_y",
    "baseline_before_stamp_sec",
    "baseline_after_stamp_sec",
    "pred_x",
    "pred_y",
    "actual_x",
    "actual_y",
    "pred_displacement_m",
    "actual_displacement_m",
    "signed_displacement_error_m",
    "pred_heading_deg",
    "actual_heading_deg",
    "signed_heading_error_deg",
    "error_m",
    "actual_before_stamp_sec",
    "actual_after_stamp_sec",
    "status",
)


@dataclass(frozen=True)
class ActualSample:
    stamp_sec: float
    x: float
    y: float


@dataclass(frozen=True)
class PendingPrediction:
    row: dict[str, object]
    prediction_stamp_sec: float
    target_stamp_sec: float
    obstacle_id: str
    pred_x: float
    pred_y: float


class PredictionLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_prediction_logger")
        self._declare_parameters()

        self.policy_debug_topic = str(self.get_parameter("policy_debug_topic").value)
        self.dynamic_obstacles_topic = str(self.get_parameter("dynamic_obstacles_topic").value)
        self.policy_mode = str(self.get_parameter("policy_mode").value)
        self.data_dir = Path(str(self.get_parameter("prediction_log_data_dir").value)).expanduser()
        self.actual_match_timeout_sec = float(self.get_parameter("actual_match_timeout_sec").value)
        self.actual_history_sec = float(self.get_parameter("actual_history_sec").value)
        self.flush_period_sec = float(self.get_parameter("flush_period_sec").value)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.prediction_id = 0
        self.actual_history: dict[str, list[ActualSample]] = {}
        self.pending: list[PendingPrediction] = []

        self.output_dir = (self.data_dir / self.policy_mode).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"prediction_log_{self.policy_mode}_{self.run_id}.csv"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.csv_file.flush()

        self.policy_debug_sub = self.create_subscription(
            String,
            self.policy_debug_topic,
            self._on_policy_debug,
            default_qos(),
        )
        self.obstacles_sub = self.create_subscription(
            MarkerArray,
            self.dynamic_obstacles_topic,
            self._on_dynamic_obstacles,
            default_qos(),
        )
        self.timer = self.create_timer(max(self.flush_period_sec, 1e-3), self._on_timer)

        self.get_logger().info(
            f"Prediction logger writing {self.csv_path} from debug={self.policy_debug_topic}, "
            f"actual={self.dynamic_obstacles_topic}, mode={self.policy_mode}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("policy_debug_topic", "/world_model_nav/policy_debug")
        self.declare_parameter("dynamic_obstacles_topic", "/dynamic_obstacles")
        self.declare_parameter("policy_mode", "quadruped")
        self.declare_parameter("prediction_log_data_dir", "data")
        self.declare_parameter("actual_match_timeout_sec", 2.0)
        self.declare_parameter("actual_history_sec", 15.0)
        self.declare_parameter("flush_period_sec", 0.25)

    def _on_dynamic_obstacles(self, msg: MarkerArray) -> None:
        now_sec = self._now_sec()
        for marker in msg.markers:
            if marker.action in (Marker.DELETEALL, Marker.DELETE):
                continue
            center = marker_center(marker)
            if center is None:
                continue
            stamp = marker.header.stamp
            stamp_sec = stamp_to_seconds(stamp) if stamp.sec or stamp.nanosec else now_sec
            obstacle_id = self._obstacle_id_from_marker_id(int(marker.id))
            sample = ActualSample(
                stamp_sec=float(stamp_sec),
                x=float(center[0]),
                y=float(center[1]),
            )
            samples = self.actual_history.setdefault(obstacle_id, [])
            samples.append(sample)
            samples.sort(key=lambda item: item.stamp_sec)
            self.actual_history[obstacle_id] = self._trim_history(samples, now_sec)
        self._process_pending(now_sec=now_sec)

    def _on_policy_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Could not parse policy debug JSON: {exc}")
            return

        visualization = payload.get("visualization", {})
        if not isinstance(visualization, dict):
            return
        predictions = visualization.get("selected_dynamic_obstacle_predictions", {})
        if not isinstance(predictions, dict):
            return
        raw_positions = predictions.get("positions", [])
        if not isinstance(raw_positions, list) or not raw_positions:
            return

        obstacle_ids = predictions.get("obstacle_ids", [])
        if not isinstance(obstacle_ids, list):
            obstacle_ids = []

        prediction_stamp_sec = self._float_or_default(payload.get("prediction_stamp_sec"), self._now_sec())
        prediction_dt_sec = self._float_or_default(payload.get("prediction_dt_sec"), 0.0)
        action_index = payload.get("action_index", "")
        action_name = str(payload.get("action", payload.get("action_name", "")))
        payload_policy_mode = str(payload.get("policy_mode", self.policy_mode))
        self.prediction_id += 1

        obstacle_count = self._prediction_obstacle_count(raw_positions, obstacle_ids)
        for obstacle_index in range(obstacle_count):
            obstacle_id = (
                str(obstacle_ids[obstacle_index])
                if obstacle_index < len(obstacle_ids)
                else self._obstacle_id_from_marker_id(obstacle_index)
            )
            row = self._base_prediction_row(
                policy_mode=payload_policy_mode,
                prediction_stamp_sec=prediction_stamp_sec,
                prediction_dt_sec=prediction_dt_sec,
                target_stamp_sec=prediction_stamp_sec,
                horizon_step=0,
                horizon_time_sec=0.0,
                action_index=action_index,
                action_name=action_name,
                obstacle_id=obstacle_id,
                pred_x="",
                pred_y="",
            )
            self.pending.append(
                PendingPrediction(
                    row=row,
                    prediction_stamp_sec=prediction_stamp_sec,
                    target_stamp_sec=prediction_stamp_sec,
                    obstacle_id=obstacle_id,
                    pred_x=float("nan"),
                    pred_y=float("nan"),
                )
            )

        for step_index, raw_step in enumerate(raw_positions, start=1):
            if not isinstance(raw_step, list):
                continue
            horizon_time_sec = float(step_index) * float(prediction_dt_sec)
            target_stamp_sec = float(prediction_stamp_sec) + horizon_time_sec
            for obstacle_index, raw_position in enumerate(raw_step):
                xy = self._xy(raw_position)
                if xy is None:
                    continue
                obstacle_id = (
                    str(obstacle_ids[obstacle_index])
                    if obstacle_index < len(obstacle_ids)
                    else self._obstacle_id_from_marker_id(obstacle_index)
                )
                row = self._base_prediction_row(
                    policy_mode=payload_policy_mode,
                    prediction_stamp_sec=prediction_stamp_sec,
                    prediction_dt_sec=prediction_dt_sec,
                    target_stamp_sec=target_stamp_sec,
                    horizon_step=step_index,
                    horizon_time_sec=horizon_time_sec,
                    action_index=action_index,
                    action_name=action_name,
                    obstacle_id=obstacle_id,
                    pred_x=xy[0],
                    pred_y=xy[1],
                )
                self.pending.append(
                    PendingPrediction(
                        row=row,
                        prediction_stamp_sec=prediction_stamp_sec,
                        target_stamp_sec=target_stamp_sec,
                        obstacle_id=obstacle_id,
                        pred_x=xy[0],
                        pred_y=xy[1],
                    )
                )
        self._process_pending(now_sec=self._now_sec())

    def _on_timer(self) -> None:
        self._process_pending(now_sec=self._now_sec())

    def _process_pending(self, *, now_sec: float) -> None:
        if not self.pending:
            return
        still_pending: list[PendingPrediction] = []
        wrote_any = False
        for pending in self.pending:
            baseline_match = self._interpolated_actual(
                pending.obstacle_id,
                pending.prediction_stamp_sec,
            )
            actual_match = self._interpolated_actual(pending.obstacle_id, pending.target_stamp_sec)
            if baseline_match is not None and actual_match is not None:
                row = self._matched_row(
                    pending,
                    baseline_match=baseline_match,
                    actual_match=actual_match,
                )
                row["status"] = "matched"
                self.writer.writerow(row)
                wrote_any = True
                continue
            if now_sec >= pending.target_stamp_sec + self.actual_match_timeout_sec:
                row = dict(pending.row)
                row["status"] = "missing_actual"
                self.writer.writerow(row)
                wrote_any = True
                continue
            still_pending.append(pending)
        self.pending = still_pending
        if wrote_any:
            self.csv_file.flush()

    def _interpolated_actual(
        self,
        obstacle_id: str,
        target_stamp_sec: float,
    ) -> tuple[float, float, float, float] | None:
        samples = self.actual_history.get(obstacle_id, [])
        if not samples:
            return None
        before: ActualSample | None = None
        after: ActualSample | None = None
        for sample in samples:
            if sample.stamp_sec <= target_stamp_sec:
                before = sample
            if sample.stamp_sec >= target_stamp_sec:
                after = sample
                break
        if before is None or after is None:
            return None
        if after.stamp_sec <= before.stamp_sec:
            return (before.x, before.y, before.stamp_sec, after.stamp_sec)
        ratio = (target_stamp_sec - before.stamp_sec) / (after.stamp_sec - before.stamp_sec)
        x = before.x + ratio * (after.x - before.x)
        y = before.y + ratio * (after.y - before.y)
        return (float(x), float(y), float(before.stamp_sec), float(after.stamp_sec))

    def _base_prediction_row(
        self,
        *,
        policy_mode: str,
        prediction_stamp_sec: float,
        prediction_dt_sec: float,
        target_stamp_sec: float,
        horizon_step: int,
        horizon_time_sec: float,
        action_index: object,
        action_name: str,
        obstacle_id: str,
        pred_x: object,
        pred_y: object,
    ) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "policy_mode": policy_mode,
            "prediction_id": self.prediction_id,
            "prediction_stamp_sec": prediction_stamp_sec,
            "prediction_dt_sec": prediction_dt_sec,
            "target_stamp_sec": target_stamp_sec,
            "horizon_step": horizon_step,
            "horizon_time_sec": horizon_time_sec,
            "action_index": action_index,
            "action_name": action_name,
            "obstacle_id": obstacle_id,
            "baseline_x": "",
            "baseline_y": "",
            "baseline_before_stamp_sec": "",
            "baseline_after_stamp_sec": "",
            "pred_x": pred_x,
            "pred_y": pred_y,
            "actual_x": "",
            "actual_y": "",
            "pred_displacement_m": "",
            "actual_displacement_m": "",
            "signed_displacement_error_m": "",
            "pred_heading_deg": "",
            "actual_heading_deg": "",
            "signed_heading_error_deg": "",
            "error_m": "",
            "actual_before_stamp_sec": "",
            "actual_after_stamp_sec": "",
            "status": "pending",
        }

    def _matched_row(
        self,
        pending: PendingPrediction,
        *,
        baseline_match: tuple[float, float, float, float],
        actual_match: tuple[float, float, float, float],
    ) -> dict[str, object]:
        baseline_x, baseline_y, baseline_before_stamp, baseline_after_stamp = baseline_match
        actual_x, actual_y, actual_before_stamp, actual_after_stamp = actual_match

        row = dict(pending.row)
        if int(row["horizon_step"]) == 0:
            pred_x = baseline_x
            pred_y = baseline_y
            actual_x = baseline_x
            actual_y = baseline_y
            actual_before_stamp = baseline_before_stamp
            actual_after_stamp = baseline_after_stamp
        else:
            pred_x = pending.pred_x
            pred_y = pending.pred_y

        pred_dx = float(pred_x) - baseline_x
        pred_dy = float(pred_y) - baseline_y
        actual_dx = actual_x - baseline_x
        actual_dy = actual_y - baseline_y
        pred_displacement = float(np.hypot(pred_dx, pred_dy))
        actual_displacement = float(np.hypot(actual_dx, actual_dy))
        pred_heading = self._heading_deg(pred_dx, pred_dy)
        actual_heading = self._heading_deg(actual_dx, actual_dy)

        row["baseline_x"] = baseline_x
        row["baseline_y"] = baseline_y
        row["baseline_before_stamp_sec"] = baseline_before_stamp
        row["baseline_after_stamp_sec"] = baseline_after_stamp
        row["pred_x"] = pred_x
        row["pred_y"] = pred_y
        row["actual_x"] = actual_x
        row["actual_y"] = actual_y
        row["pred_displacement_m"] = pred_displacement
        row["actual_displacement_m"] = actual_displacement
        row["signed_displacement_error_m"] = pred_displacement - actual_displacement
        row["pred_heading_deg"] = pred_heading
        row["actual_heading_deg"] = actual_heading
        row["signed_heading_error_deg"] = self._wrap_angle_deg(pred_heading - actual_heading)
        row["error_m"] = float(np.hypot(float(pred_x) - actual_x, float(pred_y) - actual_y))
        row["actual_before_stamp_sec"] = actual_before_stamp
        row["actual_after_stamp_sec"] = actual_after_stamp
        return row

    def _prediction_obstacle_count(self, raw_positions: list[object], obstacle_ids: list[object]) -> int:
        if obstacle_ids:
            return len(obstacle_ids)
        count = 0
        for raw_step in raw_positions:
            if isinstance(raw_step, list):
                count = max(count, len(raw_step))
        return count

    def _trim_history(self, samples: list[ActualSample], now_sec: float) -> list[ActualSample]:
        cutoff = now_sec - max(self.actual_history_sec, self.actual_match_timeout_sec)
        return [sample for sample in samples if sample.stamp_sec >= cutoff]

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _xy(self, value: object) -> tuple[float, float] | None:
        try:
            xy = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
            return None
        return (float(xy[0]), float(xy[1]))

    def _float_or_default(self, value: object, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return number if np.isfinite(number) else float(default)

    def _heading_deg(self, dx: float, dy: float) -> float:
        return float(np.degrees(np.arctan2(float(dy), float(dx))))

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        return float((float(angle_deg) + 180.0) % 360.0 - 180.0)

    def _obstacle_id_from_marker_id(self, marker_id: int) -> str:
        return f"dyn_{int(marker_id):02d}"

    def destroy_node(self) -> bool:
        self._process_pending(now_sec=float("inf"))
        self.csv_file.flush()
        self.csv_file.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PredictionLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
