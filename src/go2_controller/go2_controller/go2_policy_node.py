from __future__ import annotations

import math
from pathlib import Path

import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, JointState


OBS_DIM = 47
ACTION_DIM = 12
ACTION_SCALE = 0.25
POLICY_DT = 1.0 / 50.0
PHASE_PERIOD = 0.6
CMD_STAND_THRESHOLD = 0.1

TOPIC_JOINT_STATES = "/joint_states"
TOPIC_IMU = "/imu"
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_JOINT_COMMANDS = "/joint_commands"

POLICY_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

DEFAULT_JOINT_POS = torch.tensor(
    [[-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8]],
    dtype=torch.float32,
)

QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def default_policy_path() -> str:
    return str(Path(get_package_share_directory("go2_controller")) / "model" / "policy.pt")


def projected_gravity_from_imu(imu: Imu) -> tuple[float, float, float]:
    qx = float(imu.orientation.x)
    qy = float(imu.orientation.y)
    qz = float(imu.orientation.z)
    qw = float(imu.orientation.w)

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-6:
        return 0.0, 0.0, -1.0

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    return (
        2.0 * (qw * qy - qx * qz),
        -2.0 * (qw * qx + qy * qz),
        -1.0 + 2.0 * (qx * qx + qy * qy),
    )


def read_policy_joints(joint_state: JointState) -> tuple[torch.Tensor, torch.Tensor] | None:
    name_to_index = {name: i for i, name in enumerate(joint_state.name)}
    try:
        joint_ids = [name_to_index[name] for name in POLICY_JOINT_NAMES]
    except KeyError:
        return None

    if max(joint_ids) >= len(joint_state.position):
        return None

    positions = [float(joint_state.position[i]) for i in joint_ids]
    velocities = [float(joint_state.velocity[i]) if i < len(joint_state.velocity) else 0.0 for i in joint_ids]
    return torch.tensor([positions], dtype=torch.float32), torch.tensor([velocities], dtype=torch.float32)


class Go2PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_policy_ros")

        self.declare_parameter("policy_path", default_policy_path())
        policy_path = self.get_parameter("policy_path").get_parameter_value().string_value
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.latest_joint_state: JointState | None = None
        self.latest_imu: Imu | None = None
        self.latest_cmd = Twist()
        self.last_action = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        self.policy_start_time: float | None = None

        self.command_pub = self.create_publisher(JointState, TOPIC_JOINT_COMMANDS, QOS)
        self.subscription_refs = [
            self.create_subscription(JointState, TOPIC_JOINT_STATES, self.on_joint_state, QOS),
            self.create_subscription(Imu, TOPIC_IMU, self.on_imu, QOS),
            self.create_subscription(Twist, TOPIC_CMD_VEL, self.on_cmd_vel, QOS),
        ]
        self.timer = self.create_timer(POLICY_DT, self.on_timer)

    def on_joint_state(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def on_imu(self, msg: Imu) -> None:
        self.latest_imu = msg

    def on_cmd_vel(self, msg: Twist) -> None:
        self.latest_cmd = msg

    def on_timer(self) -> None:
        if self.latest_joint_state is None or self.latest_imu is None:
            return

        now = self.get_clock().now()
        policy_time = now.nanoseconds * 1e-9
        if self.policy_start_time is None:
            self.policy_start_time = policy_time

        obs = self.build_observation(policy_time)
        if obs is None:
            return

        with torch.inference_mode():
            action = self.policy(obs).detach().cpu().reshape(1, -1)[:, :ACTION_DIM]

        self.last_action = action
        self.publish_joint_command(now.to_msg(), DEFAULT_JOINT_POS + ACTION_SCALE * action)

    def build_observation(self, policy_time: float) -> torch.Tensor | None:
        assert self.latest_joint_state is not None
        assert self.latest_imu is not None
        assert self.policy_start_time is not None

        joint_data = read_policy_joints(self.latest_joint_state)
        if joint_data is None:
            return None
        joint_pos, joint_vel = joint_data

        cmd_x = float(self.latest_cmd.linear.x)
        cmd_y = float(self.latest_cmd.linear.y)
        cmd_yaw = float(self.latest_cmd.angular.z)
        gravity_x, gravity_y, gravity_z = projected_gravity_from_imu(self.latest_imu)

        obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
        obs[0, 0] = float(self.latest_imu.angular_velocity.x)
        obs[0, 1] = float(self.latest_imu.angular_velocity.y)
        obs[0, 2] = float(self.latest_imu.angular_velocity.z)
        obs[0, 3:6] = torch.tensor([gravity_x, gravity_y, gravity_z], dtype=torch.float32)
        obs[0, 6:9] = torch.tensor([cmd_x, cmd_y, cmd_yaw], dtype=torch.float32)

        cmd_norm = math.sqrt(cmd_x * cmd_x + cmd_y * cmd_y + cmd_yaw * cmd_yaw)
        if cmd_norm >= CMD_STAND_THRESHOLD:
            phase = math.fmod(policy_time - self.policy_start_time, PHASE_PERIOD) / PHASE_PERIOD
            obs[0, 9] = math.sin(2.0 * math.pi * phase)
            obs[0, 10] = math.cos(2.0 * math.pi * phase)

        obs[:, 11:23] = joint_pos - DEFAULT_JOINT_POS
        obs[:, 23:35] = joint_vel
        obs[:, 35:47] = self.last_action
        return obs

    def publish_joint_command(self, stamp: Time, target: torch.Tensor) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(POLICY_JOINT_NAMES)
        msg.position = target.reshape(-1).tolist()
        self.command_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = Go2PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
