from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model_nav_ros2.vendor.controllers.baseline_structured_controller import (
    StructuredControllerConfig,
)
from world_model_nav_ros2.vendor.controllers.learned_structured_controller import (
    LearnedStructuredController,
)
from world_model_nav_ros2.vendor.sim2d.astar import astar_search, path_to_world
from world_model_nav_ros2.vendor.sim2d.config import DatasetConfig, MapConfig, RobotConfig
from world_model_nav_ros2.vendor.sim2d.utils import (
    inflate_occupancy_grid,
    occupied_cell_centers,
    world_to_grid,
)
from world_model_nav_ros2.vendor.sim2d.waypoint import compute_local_subgoal


@dataclass(frozen=True)
class NavigationConfig:
    policy_path: str
    device: str = "auto"
    policy_seed: int = 0
    control_frequency: float = 10.0
    lookahead_distance: float = 1.0
    goal_tolerance: float = 0.4
    robot_radius: float = 0.4
    inflation_margin: float = 0.05
    map_occupied_threshold: int = 50
    treat_unknown_as_occupied: bool = True
    dynamic_obstacle_radius: float = 0.5
    expected_dynamic_obstacles: int = 4
    min_obstacle_dt: float = 1e-3


@dataclass(frozen=True)
class ControllerConfigValues:
    horizon: int = 10
    w_progress: float = 2.0
    w_heading: float = 0.8
    w_path: float = 1.1
    w_velocity: float = 0.3
    w_static: float = 0.5
    w_dynamic: float = 2.0
    static_collision_penalty: float = 1e6
    dynamic_collision_penalty: float = 1e6
    dynamic_risk_tau: float = 0.5
    min_clearance_clip: float = 0.05
    backward_penalty_enabled: bool = True
    backward_action_name: str = "backward"
    backward_penalty: float = 0.25
    backward_gate_enabled: bool = True
    dynamic_stop_clearance_threshold: float = 0.20
    backward_dynamic_margin: float = 0.10
    action_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    def to_vendor_config(self) -> StructuredControllerConfig:
        return StructuredControllerConfig(
            horizon=int(self.horizon),
            w_progress=float(self.w_progress),
            w_heading=float(self.w_heading),
            w_path=float(self.w_path),
            w_velocity=float(self.w_velocity),
            w_static=float(self.w_static),
            w_dynamic=float(self.w_dynamic),
            static_collision_penalty=float(self.static_collision_penalty),
            dynamic_collision_penalty=float(self.dynamic_collision_penalty),
            dynamic_risk_tau=float(self.dynamic_risk_tau),
            min_clearance_clip=float(self.min_clearance_clip),
            backward_penalty_enabled=bool(self.backward_penalty_enabled),
            backward_action_name=str(self.backward_action_name),
            backward_penalty=float(self.backward_penalty),
            backward_gate_enabled=bool(self.backward_gate_enabled),
            dynamic_stop_clearance_threshold=float(self.dynamic_stop_clearance_threshold),
            backward_dynamic_margin=float(self.backward_dynamic_margin),
            action_indices=tuple(int(value) for value in self.action_indices),
        )


@dataclass(frozen=True)
class ObstacleObservation:
    marker_id: int
    position: np.ndarray
    stamp_sec: float


@dataclass
class RuntimeDynamicObstacle:
    obstacle_id: str
    position: np.ndarray
    radius: float
    velocity_vector: np.ndarray

    @property
    def velocity(self) -> np.ndarray:
        return self.velocity_vector.copy()

    def clone(self) -> "RuntimeDynamicObstacle":
        return RuntimeDynamicObstacle(
            obstacle_id=str(self.obstacle_id),
            position=self.position.copy(),
            radius=float(self.radius),
            velocity_vector=self.velocity_vector.copy(),
        )

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity_vector * float(dt)

    def snapshot(self) -> dict[str, object]:
        return {
            "obstacle_id": str(self.obstacle_id),
            "position": self.position.copy(),
            "velocity": self.velocity_vector.copy(),
            "radius": float(self.radius),
        }


@dataclass(frozen=True)
class MapState:
    true_occupancy: np.ndarray
    planning_occupancy: np.ndarray
    occupied_points: np.ndarray
    resolution: float
    origin: tuple[float, float]


@dataclass(frozen=True)
class PlanResult:
    success: bool
    message: str
    path_world: np.ndarray | None = None


@dataclass(frozen=True)
class StepResult:
    command: np.ndarray
    subgoal_world: np.ndarray | None = None
    message: str | None = None
    message_level: str = "info"
    goal_reached: bool = False
    tracking_active: bool = False
    selected_action_index: int | None = None
    selected_action_name: str | None = None
    chosen_action_min_clearance: float | None = None
    pose_estimate_error: float | None = None


@dataclass(frozen=True)
class WaypointResult:
    waypoint_world: np.ndarray | None = None
    message: str | None = None
    message_level: str = "info"
    goal_reached: bool = False
    tracking_active: bool = False


def zero_command() -> np.ndarray:
    return np.zeros((2,), dtype=np.float32)


def occupancy_from_ros_data(
    data: list[int] | tuple[int, ...] | np.ndarray,
    *,
    width: int,
    height: int,
    occupied_threshold: int,
    treat_unknown_as_occupied: bool,
) -> np.ndarray:
    raw = np.asarray(data, dtype=np.int16).reshape((int(height), int(width)))
    occupancy = np.zeros(raw.shape, dtype=np.uint8)
    occupancy[raw >= int(occupied_threshold)] = 1
    if treat_unknown_as_occupied:
        occupancy[raw < 0] = 1
    return occupancy


def build_map_state(
    data: list[int] | tuple[int, ...] | np.ndarray,
    *,
    width: int,
    height: int,
    resolution: float,
    origin: tuple[float, float],
    robot_radius: float,
    inflation_margin: float,
    occupied_threshold: int,
    treat_unknown_as_occupied: bool,
) -> MapState:
    true_occupancy = occupancy_from_ros_data(
        data,
        width=width,
        height=height,
        occupied_threshold=occupied_threshold,
        treat_unknown_as_occupied=treat_unknown_as_occupied,
    )
    planning_occupancy = inflate_occupancy_grid(
        true_occupancy,
        inflation_radius_m=float(robot_radius) + float(inflation_margin),
        resolution=float(resolution),
    )
    return MapState(
        true_occupancy=true_occupancy,
        planning_occupancy=planning_occupancy,
        occupied_points=occupied_cell_centers(true_occupancy, float(resolution), origin),
        resolution=float(resolution),
        origin=(float(origin[0]), float(origin[1])),
    )


class DynamicObstacleTracker:
    def __init__(self, *, radius: float, min_dt: float) -> None:
        self.radius = float(radius)
        self.min_dt = float(min_dt)
        self._previous: dict[str, tuple[np.ndarray, float]] = {}
        self.obstacles: list[RuntimeDynamicObstacle] = []

    def update(self, observations: list[ObstacleObservation]) -> list[RuntimeDynamicObstacle]:
        observations = sorted(observations, key=lambda item: int(item.marker_id))
        next_previous: dict[str, tuple[np.ndarray, float]] = {}
        obstacles: list[RuntimeDynamicObstacle] = []

        for observation in observations:
            obstacle_id = f"dyn_{int(observation.marker_id):02d}"
            position = np.asarray(observation.position, dtype=float).reshape(2)
            previous = self._previous.get(obstacle_id)
            if previous is None:
                velocity = np.zeros((2,), dtype=float)
            else:
                previous_position, previous_stamp = previous
                dt = float(observation.stamp_sec) - float(previous_stamp)
                velocity = (
                    (position - previous_position) / dt
                    if dt > self.min_dt
                    else np.zeros((2,), dtype=float)
                )
            obstacles.append(
                RuntimeDynamicObstacle(
                    obstacle_id=obstacle_id,
                    position=position.copy(),
                    radius=self.radius,
                    velocity_vector=velocity.astype(float),
                )
            )
            next_previous[obstacle_id] = (position.copy(), float(observation.stamp_sec))

        self._previous = next_previous
        self.obstacles = obstacles
        return [obstacle.clone() for obstacle in self.obstacles]


class AStarPathPlanner:
    def __init__(self, nav_config: NavigationConfig) -> None:
        self.nav_config = nav_config
        self.map_state: MapState | None = None
        self.robot_pose: np.ndarray | None = None
        self.path_world: np.ndarray | None = None
        self.goal_xy: np.ndarray | None = None

    def set_map(
        self,
        data: list[int] | tuple[int, ...] | np.ndarray,
        *,
        width: int,
        height: int,
        resolution: float,
        origin: tuple[float, float],
    ) -> None:
        self.map_state = build_map_state(
            data,
            width=width,
            height=height,
            resolution=resolution,
            origin=origin,
            robot_radius=float(self.nav_config.robot_radius),
            inflation_margin=float(self.nav_config.inflation_margin),
            occupied_threshold=int(self.nav_config.map_occupied_threshold),
            treat_unknown_as_occupied=bool(self.nav_config.treat_unknown_as_occupied),
        )

    def set_robot_pose(self, pose: np.ndarray) -> None:
        self.robot_pose = np.asarray(pose, dtype=float).reshape(3)

    def plan_to_goal(self, goal_xy: np.ndarray) -> PlanResult:
        if self.map_state is None or self.robot_pose is None:
            return PlanResult(False, "Cannot plan yet; waiting for map and odom.")

        goal = np.asarray(goal_xy, dtype=float).reshape(2)
        start_rc = world_to_grid(self.robot_pose[:2], self.map_state.resolution, self.map_state.origin)
        goal_rc = world_to_grid(goal, self.map_state.resolution, self.map_state.origin)

        if not self._cell_is_free(start_rc):
            return self._clear_plan("Start pose is outside the planning map or inside inflated occupancy.")
        if not self._cell_is_free(goal_rc):
            return self._clear_plan("Goal pose is outside the planning map or inside inflated occupancy.")

        path_rc = astar_search(self.map_state.planning_occupancy, start_rc, goal_rc)
        if not path_rc:
            return self._clear_plan("A* could not find a path to the requested goal.")

        self.goal_xy = goal.copy()
        self.path_world = path_to_world(path_rc, self.map_state.resolution, self.map_state.origin)
        return PlanResult(True, f"Planned path with {len(self.path_world)} poses.", self.path_world.copy())

    def _cell_is_free(self, cell: tuple[int, int]) -> bool:
        assert self.map_state is not None
        row, col = cell
        occupancy = self.map_state.planning_occupancy
        return 0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1] and occupancy[row, col] == 0

    def _clear_plan(self, message: str) -> PlanResult:
        self.path_world = None
        self.goal_xy = None
        return PlanResult(False, message)


class PathWaypointTracker:
    def __init__(self, *, lookahead_distance: float, goal_tolerance: float) -> None:
        self.lookahead_distance = float(lookahead_distance)
        self.goal_tolerance = float(goal_tolerance)
        self.robot_pose: np.ndarray | None = None
        self.path_world: np.ndarray | None = None
        self.tracking_active = False

    def set_robot_pose(self, pose: np.ndarray) -> None:
        self.robot_pose = np.asarray(pose, dtype=float).reshape(3)

    def set_path(self, path_world: np.ndarray | None) -> None:
        if path_world is None or len(path_world) == 0:
            self.path_world = None
            self.tracking_active = False
            return
        next_path = np.asarray(path_world, dtype=float).reshape((-1, 2))
        if (
            self.path_world is not None
            and self.path_world.shape == next_path.shape
            and np.allclose(self.path_world, next_path)
        ):
            return
        self.path_world = next_path.copy()
        self.tracking_active = True

    def step(self) -> WaypointResult:
        if not self.tracking_active or self.path_world is None or len(self.path_world) == 0:
            return WaypointResult(tracking_active=False)
        if self.robot_pose is None:
            return WaypointResult(
                message="Waiting for odom before publishing tracked waypoint.",
                message_level="warning",
                tracking_active=True,
            )

        if np.linalg.norm(self.robot_pose[:2] - self.path_world[-1]) <= self.goal_tolerance:
            self.tracking_active = False
            return WaypointResult(
                message="Goal reached; stopping waypoint tracking.",
                message_level="info",
                goal_reached=True,
                tracking_active=False,
            )

        waypoint = compute_local_subgoal(
            self.robot_pose,
            self.path_world,
            self.lookahead_distance,
        )
        return WaypointResult(
            waypoint_world=np.asarray(waypoint["selected_subgoal_world"], dtype=float),
            tracking_active=True,
        )


class WorldModelPolicyController:
    def __init__(
        self,
        nav_config: NavigationConfig,
        controller_config: ControllerConfigValues,
    ) -> None:
        self.nav_config = nav_config
        self.controller_config = controller_config
        self.control_dt = 1.0 / max(float(nav_config.control_frequency), 1e-6)
        dataset_cfg = DatasetConfig(
            max_steps=1000000,
            lookahead_distance=float(nav_config.lookahead_distance),
            map_config=MapConfig(inflation_margin=float(nav_config.inflation_margin)),
            robot_config=RobotConfig(
                radius=float(nav_config.robot_radius),
                dt=float(self.control_dt),
                goal_tolerance=float(nav_config.goal_tolerance),
            ),
        )
        self.policy = LearnedStructuredController(
            nav_config.policy_path,
            config=dataset_cfg,
            controller_cfg=controller_config.to_vendor_config(),
            device=nav_config.device,
        )
        self.obstacle_tracker = DynamicObstacleTracker(
            radius=float(nav_config.dynamic_obstacle_radius),
            min_dt=float(nav_config.min_obstacle_dt),
        )
        self.map_state: MapState | None = None
        self.robot_pose: np.ndarray | None = None
        self.path_world: np.ndarray | None = None
        self.tracked_waypoint: np.ndarray | None = None
        self.goal_xy: np.ndarray | None = None
        self.tracking_active = False
        self._pending_commit: dict[str, object] | None = None
        self._reset_policy_pending = False

    def set_map(
        self,
        data: list[int] | tuple[int, ...] | np.ndarray,
        *,
        width: int,
        height: int,
        resolution: float,
        origin: tuple[float, float],
    ) -> None:
        self.map_state = build_map_state(
            data,
            width=width,
            height=height,
            resolution=resolution,
            origin=origin,
            robot_radius=float(self.nav_config.robot_radius),
            inflation_margin=float(self.nav_config.inflation_margin),
            occupied_threshold=int(self.nav_config.map_occupied_threshold),
            treat_unknown_as_occupied=bool(self.nav_config.treat_unknown_as_occupied),
        )

    def set_robot_pose(self, pose: np.ndarray) -> None:
        self.robot_pose = np.asarray(pose, dtype=float).reshape(3)

    def set_path(self, path_world: np.ndarray | None) -> bool:
        if path_world is None or len(path_world) == 0:
            self._clear_path()
            return True

        next_path = np.asarray(path_world, dtype=float).reshape((-1, 2))
        if (
            self.path_world is not None
            and self.path_world.shape == next_path.shape
            and np.allclose(self.path_world, next_path)
        ):
            return False

        self.path_world = next_path.copy()
        self.goal_xy = self.path_world[-1].copy()
        self.tracked_waypoint = None
        self.tracking_active = True
        self._pending_commit = None
        self._reset_policy_pending = True
        return True

    def set_tracked_waypoint(self, waypoint: np.ndarray) -> None:
        self.tracked_waypoint = np.asarray(waypoint, dtype=float).reshape(2)

    def set_obstacle_observations(
        self,
        observations: list[ObstacleObservation],
    ) -> list[RuntimeDynamicObstacle]:
        return self.obstacle_tracker.update(observations)

    @property
    def obstacles(self) -> list[RuntimeDynamicObstacle]:
        return [obstacle.clone() for obstacle in self.obstacle_tracker.obstacles]

    def step(self) -> StepResult:
        if not self.tracking_active or self.path_world is None or len(self.path_world) == 0:
            return StepResult(command=zero_command(), tracking_active=False)

        ready_message = self._readiness_error()
        if ready_message is not None:
            return StepResult(
                command=zero_command(),
                message=ready_message,
                message_level="warning",
                tracking_active=self.tracking_active,
            )

        assert self.robot_pose is not None
        assert self.map_state is not None
        assert self.path_world is not None

        if self._reset_policy_pending:
            self.policy.reset(initial_pose=self.robot_pose.copy(), seed=int(self.nav_config.policy_seed))
            self._reset_policy_pending = False

        dynamic_obstacles = self.obstacles
        if self._pending_commit is not None:
            try:
                self._commit_previous_step(self.robot_pose, dynamic_obstacles)
            except Exception as exc:  # noqa: BLE001 - controller state must fail closed.
                self.policy.reset(initial_pose=self.robot_pose.copy(), seed=int(self.nav_config.policy_seed))
                self._pending_commit = None
                return StepResult(
                    command=zero_command(),
                    message=f"Policy commit failed; controller state was reset: {exc}",
                    message_level="error",
                    tracking_active=self.tracking_active,
                )

        if np.linalg.norm(self.robot_pose[:2] - self.path_world[-1]) <= float(self.nav_config.goal_tolerance):
            self.tracking_active = False
            self.tracked_waypoint = None
            self._pending_commit = None
            self._reset_policy_pending = False
            return StepResult(
                command=zero_command(),
                message="Goal reached; stopping policy controller.",
                message_level="info",
                goal_reached=True,
                tracking_active=False,
            )

        robot_pose_for_policy = self.robot_pose.copy()
        waypoint = compute_local_subgoal(
            robot_pose_for_policy,
            self.path_world,
            float(self.nav_config.lookahead_distance),
        )
        self.tracked_waypoint = np.asarray(waypoint["selected_subgoal_world"], dtype=float).reshape(2)
        pose_estimate = np.asarray(self.policy.current_pose_estimate(), dtype=float)
        pose_estimate_error = float(np.linalg.norm(pose_estimate[:2] - robot_pose_for_policy[:2]))
        try:
            decision = self.policy.select_action(
                robot_pose=robot_pose_for_policy,
                current_subgoal_world=self.tracked_waypoint.copy(),
                path_world=self.path_world,
                planning_occupancy=self.map_state.planning_occupancy,
                true_occupancy=self.map_state.true_occupancy,
                resolution=self.map_state.resolution,
                origin=self.map_state.origin,
                occupied_points=self.map_state.occupied_points,
                dynamic_obstacles=dynamic_obstacles,
            )
        except Exception as exc:  # noqa: BLE001 - command output must fail closed.
            self._pending_commit = None
            return StepResult(
                command=zero_command(),
                message=f"Policy action selection failed: {exc}",
                message_level="error",
                tracking_active=self.tracking_active,
            )

        command = np.array([float(decision["v"]), float(decision["omega"])], dtype=np.float32)
        self._pending_commit = {
            "robot_pose_t": self.robot_pose.copy(),
            "robot_pose_est_t": robot_pose_for_policy.copy(),
            "dynamic_obstacles_t": self._snapshot_obstacles(dynamic_obstacles),
            "action_index": int(decision["action_index"]),
            "action_cont": command.copy(),
            "path_world": self.path_world.copy(),
        }
        return StepResult(
            command=command,
            subgoal_world=self.tracked_waypoint.copy(),
            tracking_active=True,
            selected_action_index=int(decision["action_index"]),
            selected_action_name=str(decision["action_name"]),
            chosen_action_min_clearance=float(decision["chosen_action_min_clearance"]),
            pose_estimate_error=pose_estimate_error,
        )

    def _clear_path(self) -> None:
        self.tracking_active = False
        self.path_world = None
        self.tracked_waypoint = None
        self.goal_xy = None
        self._pending_commit = None
        self._reset_policy_pending = False

    def _readiness_error(self) -> str | None:
        if self.map_state is None or self.robot_pose is None:
            return "Waiting for map and odom before controlling."
        if len(self.obstacle_tracker.obstacles) != int(self.nav_config.expected_dynamic_obstacles):
            return (
                f"Waiting for {int(self.nav_config.expected_dynamic_obstacles)} dynamic obstacles; "
                f"currently have {len(self.obstacle_tracker.obstacles)}."
            )
        return None

    def _commit_previous_step(
        self,
        robot_pose_t1: np.ndarray,
        dynamic_obstacles_t1: list[RuntimeDynamicObstacle],
    ) -> None:
        assert self._pending_commit is not None
        self.policy.commit_step(
            robot_pose_t=np.asarray(self._pending_commit["robot_pose_t"], dtype=float),
            robot_pose_t1=np.asarray(robot_pose_t1, dtype=float),
            robot_pose_est_t=np.asarray(self._pending_commit["robot_pose_est_t"], dtype=float),
            pose_observation=np.asarray(robot_pose_t1, dtype=float),
            dynamic_obstacles_t=list(self._pending_commit["dynamic_obstacles_t"]),
            dynamic_obstacles_t1=self._snapshot_obstacles(dynamic_obstacles_t1),
            action_index=int(self._pending_commit["action_index"]),
            action_cont=np.asarray(self._pending_commit["action_cont"], dtype=np.float32),
            path_world=np.asarray(self._pending_commit["path_world"], dtype=float),
        )
        self._pending_commit = None

    def _snapshot_obstacles(
        self,
        obstacles: list[RuntimeDynamicObstacle],
    ) -> list[dict[str, object]]:
        return [obstacle.snapshot() for obstacle in obstacles]
