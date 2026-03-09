from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import yaw_quat
from isaaclab.managers import ManagerTermBase
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power

    计算关节功率的绝对值之和，用于惩罚过大的功率输出。
    功率 = 关节速度 × 应用力矩

    Args:
        env: The reinforcement learning environment.
        asset_cfg: The asset configuration specifying which robot to compute power for.

    Returns:
        The sum of absolute joint power for each environment.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute the reward
    # Power = |joint_velocity × applied_torque|
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )

    return reward


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Additional reward functions from msz006_go2w framework.
"""


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation.

    根据运动状态调整关节位置偏差惩罚：
    - 运动时：正常惩罚
    - 静止时：放大惩罚（stand_still_scale倍）

    Args:
        env: The reinforcement learning environment.
        command_name: Command name to check.
        asset_cfg: Asset configuration.
        stand_still_scale: Scale factor for standing still penalty.
        velocity_threshold: Velocity threshold for determining if moving.
        command_threshold: Command threshold for determining if moving.

    Returns:
        The penalty reward.
    """
    # 获取机器人对象
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算命令速度的模长
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    # 计算实际基座速度的模长
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # 计算关节位置偏差（L2范数）
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    # 根据运动状态选择惩罚力度
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # 应用姿态因子
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other.

    惩罚足部空中时间和接触时间的方差，鼓励所有足部保持一致的运动模式。

    Args:
        env: The reinforcement learning environment.
        sensor_cfg: Contact sensor configuration.

    Returns:
        The variance penalty.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    足部接触奖励函数 - 用于运动状态

    惩罚足部接触数量不等于期望数量的情况。

    Args:
        env: 强化学习环境实例
        command_name: 命令名称（如 "base_velocity"）
        expect_contact_num: 期望的足部接触数量（2、3或4）
        sensor_cfg: 接触传感器配置

    Returns:
        奖励值（0或1），不等于期望数量时为1（惩罚）
    """
    # 获取接触传感器实例
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 计算接触状态
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # 统计接触数量
    contact_num = torch.sum(contact, dim=1)
    # 计算奖励（不等于期望数量时惩罚）
    reward = (contact_num != expect_contact_num).float()
    # 只在有命令时应用
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    奖励摆动的足部达到特定高度离地。

    Args:
        env: The reinforcement learning environment.
        command_name: Command name.
        asset_cfg: Asset configuration.
        target_height: Target height off ground.
        tanh_mult: Tanh multiplier for velocity.

    Returns:
        The height reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


class GaitReward(ManagerTermBase):
    """步态强制奖励项 - 用于四足机器人

    该奖励通过惩罚选定足对之间的接触时间差异来引导策略采用期望的步态，
    即小跑、跳跃或溜蹄。注意，此奖励仅适用于具有两对同步足的四足步态。

    物理意义：
    1. 同步足对：同一对的两条腿应该同时接触地面或同时腾空
    2. 异步足对：不同对的腿应该一个接触地面时另一个腾空
    3. 步态稳定性：正确的步态可以提高机器人的稳定性和能效
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """初始化奖励项

        Args:
            cfg: 奖励配置对象
            env: 强化学习环境实例
        """
        super().__init__(cfg, env)

        # 从配置中读取参数
        self.std: float = cfg.params["std"]    # 高斯函数的标准差，控制奖励的敏感度
        self.command_name: str = cfg.params["command_name"] # 命令名称（如 "base_velocity"）
        self.max_err: float = cfg.params["max_err"] # 最大允许误差，用于截断平方误差
        self.velocity_threshold: float = cfg.params["velocity_threshold"] # 速度阈值，用于判断机器人是否在移动
        self.command_threshold: float = cfg.params["command_threshold"] # 命令阈值，用于判断是否有移动指令

        # 获取接触传感器和机器人实例
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # 将足部身体名称与对应的身体ID匹配
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]

        # 验证配置：必须是两对同步足，每对包含两条腿
        # 例如：[["LF", "RH"], ["RF", "LH"]] 表示小跑步态
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")

        # 查找每条腿对应的身体ID
        # find_bodies() 将身体名称转换为传感器中的索引
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]

        # 存储同步足对：[[腿0_0, 腿0_1], [腿1_0, 腿1_1]]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """计算奖励值

        该奖励定义为六个项的乘积，其中两个项强制配对的足保持同步，
        另外四个项奖励所有其他剩余配对不同步。

        物理意义：
        - 同步奖励：同一对的两条腿应该有相同的接触时间和腾空时间
        - 异步奖励：不同对的腿应该一个接触时另一个腾空
        - 总奖励 = 同步奖励 × 异步奖励

        Args:
            env: 强化学习环境实例
            其他参数：配置参数（通过函数签名传递，实际使用类属性）

        Returns:
            奖励值（0到1之间的标量）
        """
        # ========== 同步奖励计算 ==========
        # 对于同步的足对，两条腿的接触（腾空）时间应该匹配

        # 计算第一对同步足的同步奖励
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        # 计算第二对同步足的同步奖励
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])

        # 同步奖励是两对同步足奖励的乘积
        # 只有当两对都同步时，同步奖励才高
        sync_reward = sync_reward_0 * sync_reward_1

        # ========== 异步奖励计算 ==========
        # 对于异步的足对，一条腿的接触时间应该与另一条腿的腾空时间匹配

        # 计算第一对的第一条腿与第二对的第一条腿的异步奖励
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        # 计算第一对的第二条腿与第二对的第二条腿的异步奖励
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        # 计算第一对的第一条腿与第二对的第二条腿的异步奖励
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        # 计算第二对的第一条腿与第一对的第二条腿的异步奖励
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])

        # 异步奖励是所有异步配对奖励的乘积
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

        # ========== 条件激活 ==========
        # 只在有移动命令时强制步态

        # 计算目标命令的范数（期望速度大小）
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)

        # 计算机器人实际线速度的范数（x-y平面上的速度大小）
        # root_com_lin_vel_b 是质心线速度在机体坐标系中的表示
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)

        # 根据运动状态选择是否应用步态奖励
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,     # 运动时：应用步态奖励
            0.0,                            # 静止时：不应用步态奖励
        )

        # ========== 姿态加权 ==========
        # 根据机器人的姿态（倾斜程度）对奖励进行加权
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """计算两条腿的同步奖励

        物理意义：
        - 同步的两条腿应该同时接触地面，同时腾空
        - 因此它们的接触时间应该相等，腾空时间也应该相等
        - 使用高斯函数将时间差转换为奖励：时间差越小，奖励越高

        Args:
            foot_0: 第一条腿的索引
            foot_1: 第二条腿的索引

        Returns:
            同步奖励值（0到1之间的张量）
        """
        # 获取当前腾空时间和接触时间
        air_time = self.contact_sensor.data.current_air_time    # 每条腿当前的腾空时间
        contact_time = self.contact_sensor.data.current_contact_time    # 每条腿当前的接触时间

        # 计算两条腿腾空时间的平方误差
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)

        # 计算两条腿接触时间的平方误差
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)

        # 使用高斯函数将误差转换为奖励
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """计算两条腿的异步奖励（反同步奖励）

        物理意义：
        - 异步的两条腿应该交替接触地面
        - 当一条腿接触地面时，另一条腿应该腾空
        - 因此：腿0的接触时间 ≈ 腿1的腾空时间，腿0的腾空时间 ≈ 腿1的接触时间

        Args:
            foot_0: 第一条腿的索引
            foot_1: 第二条腿的索引

        Returns:
            异步奖励值（0到1之间的张量）
        """
        # 获取当前腾空时间和接触时间
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time

        # 计算腿0的腾空时间与腿1的接触时间的平方误差
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)

        # 计算腿0的接触时间与腿1的腾空时间的平方误差
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)

        # 使用高斯函数将误差转换为奖励
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    轮速惩罚奖励函数 - 惩罚不必要的轮子转动以节省能量

    物理意义：
    1. 空中状态：当机器人腾空时，轮子转动无法产生推进力，属于能量浪费，应给予惩罚
    2. 静止状态：当机器人未收到移动命令且实际静止时，轮子不应空转，否则浪费能量
    3. 运动状态：当机器人正在移动时，只惩罚在空中的轮子速度，允许地面轮子正常工作

    Args:
        env: 强化学习环境
        sensor_cfg: 接触传感器配置，用于检测是否在空中
        command_name: 命令名称（如 "base_velocity"）
        velocity_threshold: 实际速度阈值，用于判断机器人是否在移动
        command_threshold: 命令阈值，用于判断是否有移动指令
        asset_cfg: 机器人资产配置

    Returns:
        惩罚值（负数），轮速越大惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 目标命令的速度大小（期望速度）
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    # 机器人实际的线速度大小
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # 各关节（轮子）的角速度
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    # 获取接触传感器实例，用于检测机器人与地面的接触状态
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断每个关节（轮子）是否在空中
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    # 运动场景的惩罚：只惩罚在空中的轮子速度
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    # 静止场景的惩罚：惩罚所有轮子的速度
    standing_reward = torch.sum(joint_vel, dim=1)
    # 根据运动状态选择惩罚
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward
