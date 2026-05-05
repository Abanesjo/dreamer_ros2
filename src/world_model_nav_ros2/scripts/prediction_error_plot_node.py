#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rclpy.node import Node


# Optional hardcoded defaults for quick local use. ROS parameters override these.
HARDCODED_CSV_PATH = "/home/john/workspaces/navigation_ws/src/dreamer_ros2/src/world_model_nav_ros2/data/quadruped/prediction_log_quadruped_20260505_034527_628229.csv"
HARDCODED_PLOT_ROOT = ""
HARDCODED_POLICY_MODE = ""
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_FIG_SIZE = (10.0, 8.2)
PLOT_TITLE_FONT_SIZE = 24
PLOT_SUBTITLE_FONT_SIZE = 18
PLOT_AXIS_FONT_SIZE = 20
PLOT_TICK_FONT_SIZE = 20
PLOT_GRID_LINEWIDTH = 1.35


@dataclass(frozen=True)
class ActualTeleportEvent:
    obstacle_id: str
    start_sec: float
    end_sec: float
    distance_m: float
    speed_mps: float


class PredictionErrorPlotNode(Node):
    def __init__(self) -> None:
        super().__init__("world_model_nav_prediction_error_plot")
        self.declare_parameter("csv_path", HARDCODED_CSV_PATH)
        self.declare_parameter("plot_root", HARDCODED_PLOT_ROOT)
        self.declare_parameter("policy_mode", HARDCODED_POLICY_MODE)
        self.declare_parameter("outlier_filter_enabled", True)
        self.declare_parameter("outlier_max_actual_jump_m", 1.0)
        self.declare_parameter("outlier_max_actual_speed_mps", 3.0)
        self.declare_parameter("outlier_guard_time_sec", 0.0)
        self.declare_parameter("outlier_error_iqr_multiplier", 3.0)
        self.declare_parameter("outlier_min_error_threshold_m", 1.5)
        self.declare_parameter("outlier_max_error_m", 0.0)

    def run(self) -> None:
        csv_path = self._configured_csv_path()
        if csv_path is None:
            self.get_logger().error("No csv_path provided. Set the csv_path ROS parameter or HARDCODED_CSV_PATH.")
            return
        if not csv_path.exists():
            self.get_logger().error(f"CSV file does not exist: {csv_path}")
            return

        rows = self._read_matched_rows(csv_path)
        if not rows:
            self.get_logger().error(f"No matched prediction rows found in {csv_path}")
            return

        policy_mode = self._configured_policy_mode(rows, csv_path)
        plot_root = self._configured_plot_root(csv_path)
        output_dir = (plot_root / policy_mode).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        filtered_rows, filtered_counts, filter_stats = self._filter_outlier_rows(rows)
        if not filtered_rows:
            self.get_logger().error(f"Outlier filtering removed all matched rows from {csv_path}")
            return
        total_filtered = sum(filtered_counts.values())
        if total_filtered:
            self.get_logger().info(
                f"Filtered {total_filtered} outlier rows from {len(rows)} matched rows "
                f"({filter_stats['teleport_rows']} teleport-window, "
                f"{filter_stats['error_rows']} robust-error, "
                f"{filter_stats['teleport_events']} detected actual-state jumps)"
            )

        metrics = self._compute_metrics(filtered_rows, filtered_counts)
        metrics_path = output_dir / f"{csv_path.stem}_metrics.csv"
        unsigned_plot_path = output_dir / f"{csv_path.stem}_unsigned.png"

        self._write_metrics(metrics_path, metrics)
        self._write_plot(
            unsigned_plot_path,
            filtered_rows,
        )
        self.get_logger().info(f"Wrote metrics: {metrics_path}")
        self.get_logger().info(f"Wrote plot: {unsigned_plot_path}")

    def _configured_csv_path(self) -> Path | None:
        raw_path = str(self.get_parameter("csv_path").value).strip()
        if not raw_path:
            raw_path = HARDCODED_CSV_PATH.strip()
        if not raw_path:
            return None
        return Path(raw_path).expanduser().resolve()

    def _configured_plot_root(self, csv_path: Path) -> Path:
        raw_root = str(self.get_parameter("plot_root").value).strip()
        if not raw_root:
            raw_root = HARDCODED_PLOT_ROOT.strip()
        if raw_root:
            return Path(raw_root).expanduser().resolve()
        if csv_path.parent.parent.name == "data":
            return (csv_path.parent.parent.parent / "plots").resolve()
        return (csv_path.parent / "plots").resolve()

    def _configured_policy_mode(self, rows: list[dict[str, object]], csv_path: Path) -> str:
        raw_mode = str(self.get_parameter("policy_mode").value).strip() or HARDCODED_POLICY_MODE.strip()
        if raw_mode:
            return raw_mode
        for row in rows:
            policy_mode = str(row.get("policy_mode", "")).strip()
            if policy_mode:
                return policy_mode
        if csv_path.parent.name in {"quadruped", "unicycle"}:
            return csv_path.parent.name
        return "unknown"

    def _read_matched_rows(self, csv_path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if str(row.get("status", "")).strip() != "matched":
                    continue
                signed_displacement_error = self._finite_float(row.get("signed_displacement_error_m"))
                signed_heading_error = self._finite_float(row.get("signed_heading_error_deg"))
                horizon_step = self._finite_float(row.get("horizon_step"))
                horizon_time = self._finite_float(row.get("horizon_time_sec"))
                prediction_stamp = self._finite_float(row.get("prediction_stamp_sec"))
                target_stamp = self._finite_float(row.get("target_stamp_sec"))
                actual_x = self._finite_float(row.get("actual_x"))
                actual_y = self._finite_float(row.get("actual_y"))
                if (
                    signed_displacement_error is None
                    or signed_heading_error is None
                    or horizon_step is None
                    or horizon_time is None
                    or prediction_stamp is None
                    or target_stamp is None
                    or actual_x is None
                    or actual_y is None
                ):
                    continue
                rows.append(
                    {
                        "policy_mode": str(row.get("policy_mode", "")),
                        "horizon_step": int(horizon_step),
                        "horizon_time_sec": float(horizon_time),
                        "prediction_stamp_sec": float(prediction_stamp),
                        "target_stamp_sec": float(target_stamp),
                        "obstacle_id": str(row.get("obstacle_id", "")),
                        "actual_x": float(actual_x),
                        "actual_y": float(actual_y),
                        "signed_displacement_error_m": float(signed_displacement_error),
                        "signed_heading_error_deg": float(signed_heading_error),
                    }
                )
        return rows

    def _filter_outlier_rows(
        self,
        rows: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[int, int], dict[str, int]]:
        if not bool(self.get_parameter("outlier_filter_enabled").value):
            return rows, {}, {"teleport_events": 0, "teleport_rows": 0, "error_rows": 0}

        max_jump_m = max(0.0, float(self.get_parameter("outlier_max_actual_jump_m").value))
        max_speed_mps = max(0.0, float(self.get_parameter("outlier_max_actual_speed_mps").value))
        guard_time_sec = max(0.0, float(self.get_parameter("outlier_guard_time_sec").value))
        error_iqr_multiplier = max(0.0, float(self.get_parameter("outlier_error_iqr_multiplier").value))
        min_error_threshold_m = max(0.0, float(self.get_parameter("outlier_min_error_threshold_m").value))
        max_error_m = max(0.0, float(self.get_parameter("outlier_max_error_m").value))
        teleport_events = self._detect_actual_teleports(
            rows,
            max_jump_m=max_jump_m,
            max_speed_mps=max_speed_mps,
        )
        events_by_obstacle: dict[str, list[ActualTeleportEvent]] = defaultdict(list)
        for event in teleport_events:
            events_by_obstacle[event.obstacle_id].append(event)

        filtered_counts: dict[int, int] = defaultdict(int)
        after_teleport_filter: list[dict[str, object]] = []
        teleport_rows = 0
        for row in rows:
            obstacle_id = str(row["obstacle_id"])
            row_events = events_by_obstacle.get(obstacle_id, [])
            if row_events and self._row_crosses_teleport(
                row,
                row_events,
                guard_time_sec=guard_time_sec,
            ):
                filtered_counts[int(row["horizon_step"])] += 1
                teleport_rows += 1
                continue
            after_teleport_filter.append(row)

        error_thresholds = self._error_outlier_thresholds(
            after_teleport_filter,
            iqr_multiplier=error_iqr_multiplier,
            min_threshold_m=min_error_threshold_m,
            max_error_m=max_error_m,
        )
        filtered_rows: list[dict[str, object]] = []
        error_rows = 0
        for row in after_teleport_filter:
            horizon_step = int(row["horizon_step"])
            threshold = error_thresholds.get(horizon_step)
            if threshold is not None and abs(float(row["signed_displacement_error_m"])) > threshold:
                filtered_counts[horizon_step] += 1
                error_rows += 1
                continue
            filtered_rows.append(row)

        filter_stats = {
            "teleport_events": len(teleport_events),
            "teleport_rows": teleport_rows,
            "error_rows": error_rows,
        }
        return filtered_rows, dict(filtered_counts), filter_stats

    def _detect_actual_teleports(
        self,
        rows: list[dict[str, object]],
        *,
        max_jump_m: float,
        max_speed_mps: float,
    ) -> list[ActualTeleportEvent]:
        samples_by_obstacle: dict[str, dict[tuple[int, int, int], tuple[float, float, float]]] = defaultdict(dict)
        for row in rows:
            obstacle_id = str(row["obstacle_id"])
            stamp_sec = float(row["target_stamp_sec"])
            actual_x = float(row["actual_x"])
            actual_y = float(row["actual_y"])
            key = (
                int(round(stamp_sec * 1_000_000.0)),
                int(round(actual_x * 1_000_000.0)),
                int(round(actual_y * 1_000_000.0)),
            )
            samples_by_obstacle[obstacle_id][key] = (stamp_sec, actual_x, actual_y)

        teleport_events: list[ActualTeleportEvent] = []
        for obstacle_id, sample_map in samples_by_obstacle.items():
            samples = sorted(sample_map.values(), key=lambda item: item[0])
            for before, after in zip(samples, samples[1:]):
                before_t, before_x, before_y = before
                after_t, after_x, after_y = after
                dt = after_t - before_t
                if dt <= 1e-6:
                    continue
                distance_m = float(np.hypot(after_x - before_x, after_y - before_y))
                speed_mps = distance_m / dt
                jump_exceeded = max_jump_m <= 0.0 or distance_m > max_jump_m
                speed_exceeded = max_speed_mps <= 0.0 or speed_mps > max_speed_mps
                if jump_exceeded and speed_exceeded:
                    teleport_events.append(
                        ActualTeleportEvent(
                            obstacle_id=obstacle_id,
                            start_sec=float(before_t),
                            end_sec=float(after_t),
                            distance_m=distance_m,
                            speed_mps=float(speed_mps),
                        )
                    )
        return teleport_events

    def _row_crosses_teleport(
        self,
        row: dict[str, object],
        teleport_events: list[ActualTeleportEvent],
        *,
        guard_time_sec: float,
    ) -> bool:
        prediction_stamp = float(row["prediction_stamp_sec"])
        target_stamp = float(row["target_stamp_sec"])
        window_start = min(prediction_stamp, target_stamp)
        window_end = max(prediction_stamp, target_stamp)
        for event in teleport_events:
            event_start = event.start_sec - guard_time_sec
            event_end = event.end_sec + guard_time_sec
            if event_start <= window_end and event_end >= window_start:
                return True
        return False

    def _error_outlier_thresholds(
        self,
        rows: list[dict[str, object]],
        *,
        iqr_multiplier: float,
        min_threshold_m: float,
        max_error_m: float,
    ) -> dict[int, float]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            grouped[int(row["horizon_step"])].append(abs(float(row["signed_displacement_error_m"])))

        thresholds: dict[int, float] = {}
        for horizon_step, errors in grouped.items():
            if not errors:
                continue
            values = np.asarray(errors, dtype=float)
            q1 = float(np.percentile(values, 25.0))
            q3 = float(np.percentile(values, 75.0))
            iqr = max(0.0, q3 - q1)
            threshold = max(min_threshold_m, q3 + iqr_multiplier * iqr)
            if max_error_m > 0.0:
                threshold = min(threshold, max_error_m)
            thresholds[horizon_step] = threshold
        return thresholds

    def _compute_metrics(
        self,
        rows: list[dict[str, object]],
        filtered_counts: dict[int, int] | None = None,
    ) -> list[dict[str, float | int]]:
        grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["horizon_step"])].append(
                (
                    float(row["horizon_time_sec"]),
                    float(row["signed_displacement_error_m"]),
                    float(row["signed_heading_error_deg"]),
                )
            )

        metrics: list[dict[str, float | int]] = []
        for horizon_step in sorted(grouped):
            pairs = grouped[horizon_step]
            times = np.asarray([pair[0] for pair in pairs], dtype=float)
            displacement_errors = np.asarray([pair[1] for pair in pairs], dtype=float)
            heading_errors = np.asarray([pair[2] for pair in pairs], dtype=float)
            metrics.append(
                {
                    "horizon_step": int(horizon_step),
                    "horizon_time_sec": float(np.mean(times)),
                    "count": int(displacement_errors.size),
                    "filtered_outlier_count": int((filtered_counts or {}).get(horizon_step, 0)),
                    "mean_signed_displacement_error_m": float(np.mean(displacement_errors)),
                    "median_signed_displacement_error_m": float(np.median(displacement_errors)),
                    "std_signed_displacement_error_m": float(np.std(displacement_errors)),
                    "rmse_signed_displacement_error_m": float(
                        np.sqrt(np.mean(np.square(displacement_errors)))
                    ),
                    "mean_signed_heading_error_deg": float(np.mean(heading_errors)),
                    "median_signed_heading_error_deg": float(np.median(heading_errors)),
                    "std_signed_heading_error_deg": float(np.std(heading_errors)),
                    "rmse_signed_heading_error_deg": float(
                        np.sqrt(np.mean(np.square(heading_errors)))
                    ),
                }
            )
        return metrics

    def _write_metrics(self, metrics_path: Path, metrics: list[dict[str, float | int]]) -> None:
        columns = (
            "horizon_step",
            "horizon_time_sec",
            "count",
            "filtered_outlier_count",
            "mean_signed_displacement_error_m",
            "median_signed_displacement_error_m",
            "std_signed_displacement_error_m",
            "rmse_signed_displacement_error_m",
            "mean_signed_heading_error_deg",
            "median_signed_heading_error_deg",
            "std_signed_heading_error_deg",
            "rmse_signed_heading_error_deg",
        )
        with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(metrics)

    def _write_plot(
        self,
        plot_path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=PLOT_FIG_SIZE, dpi=140)
            try:
                self._plot_metric(
                    axes[0],
                    rows,
                    row_key="signed_displacement_error_m",
                    ylabel="Displacement Error [m]",
                )
                self._plot_metric(
                    axes[1],
                    rows,
                    row_key="signed_heading_error_deg",
                    ylabel="Heading Error [deg]",
                )
                axes[1].set_xlabel("Prediction horizon [s]", fontsize=PLOT_AXIS_FONT_SIZE)
                fig.suptitle(
                    "Dynamic Obstacle Prediction Error",
                    y=0.985,
                    fontsize=PLOT_TITLE_FONT_SIZE,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.925,
                    "Error based on world-frame coordinate convention.\nHeading error based on angle wr.t. horizon origin",
                    ha="center",
                    va="top",
                    fontsize=PLOT_SUBTITLE_FONT_SIZE,
                    linespacing=1.25,
                )
                fig.legend(
                    handles=self._mean_std_legend_handles(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.87),
                    ncol=2,
                    frameon=False,
                )
                fig.subplots_adjust(left=0.115, right=0.985, bottom=0.115, top=0.805, hspace=0.12)
                fig.savefig(plot_path)
            finally:
                plt.close(fig)

    def _plot_metric(
        self,
        ax: plt.Axes,
        rows: list[dict[str, object]],
        *,
        row_key: str,
        ylabel: str,
    ) -> None:
        x_values, mean_values, std_values = self._metric_summary(rows, row_key=row_key)
        lower_values = [
            max(0.0, mean_value - std_value)
            for mean_value, std_value in zip(mean_values, std_values)
        ]
        upper_values = [
            mean_value + std_value
            for mean_value, std_value in zip(mean_values, std_values)
        ]

        ax.fill_between(
            x_values,
            lower_values,
            upper_values,
            color="tab:blue",
            alpha=0.18,
        )
        ax.plot(
            x_values,
            mean_values,
            color="tab:orange",
            marker="o",
            linewidth=2.0,
            markersize=4.5,
        )
        ax.set_ylabel(ylabel, fontsize=PLOT_AXIS_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
        ax.minorticks_on()
        ax.grid(True, which="major", color="black", alpha=0.5, linewidth=PLOT_GRID_LINEWIDTH)
        ax.grid(True, which="minor", color="black", alpha=0.28, linewidth=PLOT_GRID_LINEWIDTH * 0.7)

    def _mean_std_legend_handles(self) -> list[Patch | Line2D]:
        return [
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.18, label="Mean +/- 1 std"),
            Line2D(
                [0],
                [0],
                color="tab:orange",
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label="Mean",
            ),
        ]

    def _metric_summary(
        self,
        rows: list[dict[str, object]],
        *,
        row_key: str,
    ) -> tuple[list[float], list[float], list[float]]:
        grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            value = abs(float(row[row_key]))
            grouped[int(row["horizon_step"])].append((float(row["horizon_time_sec"]), value))

        x_values: list[float] = []
        mean_values: list[float] = []
        std_values: list[float] = []
        for horizon_step in sorted(grouped):
            pairs = grouped[horizon_step]
            times = np.asarray([pair[0] for pair in pairs], dtype=float)
            values = np.asarray([pair[1] for pair in pairs], dtype=float)
            x_values.append(float(np.mean(times)))
            mean_values.append(float(np.mean(values)))
            std_values.append(float(np.std(values)))
        return x_values, mean_values, std_values

    def _finite_float(self, value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PredictionErrorPlotNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
