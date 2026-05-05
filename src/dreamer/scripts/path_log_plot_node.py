#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class PathLogPlotNode(Node):
    def __init__(self) -> None:
        super().__init__("dreamer_path_log_plot")
        default_map_yaml = str(Path(get_package_share_directory("dreamer")) / "map" / "map.yaml")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("plot_dir", "")
        self.declare_parameter("map_yaml", default_map_yaml)

    def run(self) -> None:
        csv_path = self._configured_csv_path()
        if csv_path is None:
            self.get_logger().error("No csv_path provided.")
            return
        if not csv_path.exists():
            self.get_logger().error(f"CSV file does not exist: {csv_path}")
            return

        path_points, odom_points = self._read_log(csv_path)
        if path_points.size == 0:
            self.get_logger().error(f"No path rows found in {csv_path}")
            return
        if odom_points.size == 0:
            self.get_logger().error(f"No odom rows found in {csv_path}")
            return

        map_image, extent = self._load_map(Path(str(self.get_parameter("map_yaml").value)).expanduser())
        rmse_m = self._closest_path_rmse(odom_points, path_points)

        output_dir = self._configured_plot_dir(csv_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{csv_path.stem}.png"
        self._write_plot(
            plot_path,
            map_image=map_image,
            extent=extent,
            path_points=path_points,
            odom_points=odom_points,
            rmse_m=rmse_m,
        )
        self.get_logger().info(f"Wrote path plot: {plot_path}")

    def _configured_csv_path(self) -> Path | None:
        raw_path = str(self.get_parameter("csv_path").value).strip()
        if not raw_path:
            return None
        return Path(raw_path).expanduser().resolve()

    def _configured_plot_dir(self, csv_path: Path) -> Path:
        raw_path = str(self.get_parameter("plot_dir").value).strip()
        if raw_path:
            return Path(raw_path).expanduser().resolve()
        if csv_path.parent.name == "data":
            return (csv_path.parent.parent / "plots").resolve()
        return (csv_path.parent / "plots").resolve()

    def _read_log(self, csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
        path_snapshots: dict[int, list[tuple[int, float, float]]] = {}
        odom_points: list[tuple[float, float]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                record_type = str(row.get("record_type", "")).strip()
                x = self._finite_float(row.get("x"))
                y = self._finite_float(row.get("y"))
                if x is None or y is None:
                    continue
                if record_type == "odom":
                    odom_points.append((x, y))
                    continue
                if record_type != "path":
                    continue
                snapshot_id = self._int_or_default(row.get("path_snapshot_id"), 0)
                pose_index = self._int_or_default(row.get("path_pose_index"), len(path_snapshots))
                path_snapshots.setdefault(snapshot_id, []).append((pose_index, x, y))

        path_points: list[tuple[float, float]] = []
        for snapshot_id in sorted(path_snapshots):
            snapshot = sorted(path_snapshots[snapshot_id], key=lambda item: item[0])
            if snapshot:
                path_points = [(point[1], point[2]) for point in snapshot]
                break

        return (
            np.asarray(path_points, dtype=float).reshape((-1, 2)) if path_points else np.empty((0, 2), dtype=float),
            np.asarray(odom_points, dtype=float).reshape((-1, 2)) if odom_points else np.empty((0, 2), dtype=float),
        )

    def _load_map(self, map_yaml: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        with map_yaml.open("r", encoding="utf-8") as map_file:
            metadata = yaml.safe_load(map_file)

        image_path = Path(metadata["image"])
        if not image_path.is_absolute():
            image_path = map_yaml.parent / image_path

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
        height, width = gray.shape
        map_image = np.flipud(gray)
        if alpha is not None:
            alpha = np.flipud(alpha)
            map_image = np.ma.array(map_image, mask=(alpha == 0))

        x_min = float(origin[0])
        y_min = float(origin[1])
        x_max = x_min + float(width) * resolution
        y_max = y_min + float(height) * resolution
        return map_image, (x_min, x_max, y_min, y_max)

    def _write_plot(
        self,
        plot_path: Path,
        *,
        map_image: np.ndarray,
        extent: tuple[float, float, float, float],
        path_points: np.ndarray,
        odom_points: np.ndarray,
        rmse_m: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=150)
        try:
            ax.imshow(map_image, cmap="gray", origin="lower", extent=extent)
            ax.plot(path_points[:, 0], path_points[:, 1], color="tab:blue", linewidth=2.0, label="A* path")
            ax.plot(odom_points[:, 0], odom_points[:, 1], color="tab:orange", linewidth=2.0, label="Actual path")
            ax.scatter(
                path_points[0, 0],
                path_points[0, 1],
                marker="o",
                s=52,
                color="tab:blue",
                edgecolors="white",
                linewidths=0.8,
                label="A* start",
            )
            ax.scatter(
                path_points[-1, 0],
                path_points[-1, 1],
                marker="*",
                s=110,
                color="tab:blue",
                edgecolors="white",
                linewidths=0.8,
                label="A* goal",
            )
            ax.scatter(
                odom_points[0, 0],
                odom_points[0, 1],
                marker="o",
                s=52,
                color="tab:orange",
                edgecolors="black",
                linewidths=0.8,
                label="Actual start",
            )
            ax.scatter(
                odom_points[-1, 0],
                odom_points[-1, 1],
                marker="X",
                s=76,
                color="tab:orange",
                edgecolors="black",
                linewidths=0.8,
                label="Actual end",
            )
            ax.set_title(f"Path Tracking: RMSE={rmse_m:.3f} m")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_path)
        finally:
            plt.close(fig)

    def _closest_path_rmse(self, odom_points: np.ndarray, path_points: np.ndarray) -> float:
        distances = [self._point_to_polyline_distance(point, path_points) for point in odom_points]
        if not distances:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(np.asarray(distances, dtype=float)))))

    def _point_to_polyline_distance(self, point: np.ndarray, polyline: np.ndarray) -> float:
        if len(polyline) == 1:
            return float(np.linalg.norm(point - polyline[0]))
        start = polyline[:-1]
        end = polyline[1:]
        segment = end - start
        segment_norm_sq = np.sum(segment * segment, axis=1)
        raw_t = np.sum((point - start) * segment, axis=1) / np.maximum(segment_norm_sq, 1e-12)
        t = np.clip(raw_t, 0.0, 1.0)
        projection = start + t[:, None] * segment
        distances = np.linalg.norm(point - projection, axis=1)
        return float(np.min(distances))

    def _finite_float(self, value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _int_or_default(self, value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PathLogPlotNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
