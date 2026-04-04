from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
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


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """关节速度惩罚 - 限制关节速度超过阈值

    物理意义：
    1. 运动平滑：防止关节速度过大导致动作不流畅
    2. 安全保护：避免高速运动造成的机械应力
    3. 精度控制：提高轨迹跟踪精度

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        threshold: 速度阈值 (rad/s)，超过此值的速度将被惩罚

    Returns:
        关节速度惩罚值（正数），速度越大惩罚越大
    """
    # 提取机器人数据
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节速度
    joint_velocities = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])

    # 计算超过阈值的部分（只惩罚超过阈值的速度）
    excess_velocity = torch.clamp(joint_velocities - threshold, min=0.0)

    # 平方惩罚：速度越大惩罚呈指数增长
    penalty = torch.sum(torch.square(excess_velocity), dim=-1)

    return penalty


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

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """初始化奖励项

        Args:
            cfg: 奖励配置对象
            env: 强化学习环境实例
        """
        super().__init__(cfg, env)

        # 从配置中读取参数
        self.std: float = cfg.params["std"]  # 高斯函数的标准差，控制奖励的敏感度
        self.command_name: str = cfg.params["command_name"]  # 命令名称（如 "base_velocity"）
        self.max_err: float = cfg.params["max_err"]  # 最大允许误差，用于截断平方误差
        self.velocity_threshold: float = cfg.params["velocity_threshold"]  # 速度阈值，用于判断机器人是否在移动
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


def arm_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="arm_joint.*"),
    stability_window: int = 100
) -> torch.Tensor:
    """机械臂稳定性奖励

    鼓励机械臂保持稳定姿态，避免干扰腿部运动

    Args:
        env: 强化学习环境
        asset_cfg: 机械臂关节配置
        stability_window: 稳定性计算窗口

    Returns:
        机械臂稳定性奖励值
    """
    # 获取机械臂关节数据
    asset: Articulation = env.scene[asset_cfg.name]
    arm_joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
    arm_velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 计算关节位置方差（越小说明越稳定）
    joint_variance = torch.var(arm_joints, dim=-1)
    stability_reward = torch.exp(-joint_variance * 10.0)  # 指数衰减

    # 考虑运动强度（运动时稳定性应该更好）
    arm_velocity = torch.linalg.norm(arm_velocities, dim=-1)
    motion_bonus = torch.clamp(arm_velocity / 5.0, 0.0, 1.0)  # 速度适中时给予奖励

    return stability_reward * (1.0 + motion_bonus)


def stand_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = None,  # 高度传感器配置（可选）
    target_pitch: float = 0.0,
    target_roll: float = 0.0,
    max_tilt: float = 0.5,
    min_height: float = 0.3,
    max_height: float = 0.6
) -> torch.Tensor:
    """站立恢复奖励

    基于姿态和高度的综合奖励，鼓励机器人从倒下状态恢复到直立状态。

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        sensor_cfg: 高度传感器配置（可选，如果None则直接使用Z坐标）
        target_pitch: 目标俯仰角
        target_roll: 目标翻滚角
        max_tilt: 最大允许倾斜角度
        min_height: 最小站立高度
        max_height: 最大站立高度

    Returns:
        站立恢复奖励值
    """
    # 提取机器人状态
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前姿态角度
    # 简单的俯仰角和翻滚角计算
    pitch = asset.data.projected_gravity_b[:, 0]  # X分量对应俯仰
    roll = asset.data.projected_gravity_b[:, 1]   # Y分量对应翻滚

    # 获取当前高度（优先使用传感器，否则直接使用Z坐标）
    if sensor_cfg is not None and hasattr(env.scene, sensor_cfg.name) and env.scene[sensor_cfg.name] is not None:
        height_sensor = env.scene[sensor_cfg.name]
        current_height = height_sensor.data.ray_hits[..., 2].squeeze(-1)
    else:
        # 直接使用机器人基座的Z坐标
        current_height = asset.data.root_state_w[:, 2]

    # 计算姿态奖励（越接近直立越好）
    tilt_reward = 1.0 - (torch.abs(pitch) + torch.abs(roll)) / max_tilt
    tilt_reward = torch.clamp(tilt_reward, 0.0, 1.0)

    # 计算高度奖励（越接近目标高度越好）
    height_target = (min_height + max_height) / 2.0
    height_error = torch.abs(current_height - height_target)
    height_reward = torch.exp(-height_error * 5.0)  # 指数衰减

    # 综合奖励
    return tilt_reward * height_reward


def balance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0
) -> torch.Tensor:
    """平衡控制奖励

    基于足端接触和姿态的平衡奖励，鼓励保持稳定平衡。

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        contact_threshold: 最小接触力阈值

    Returns:
        平衡控制奖励值
    """
    # 提取机器人状态
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene["contact_forces"]

    # 获取足端接触力
    contact_forces = contact_sensor.data.net_forces_w
    # 只考虑足端
    foot_indices = [i for i, name in enumerate(contact_sensor.body_names) if "foot" in name.lower()]
    if foot_indices:
        foot_contacts = torch.norm(contact_forces[:, foot_indices], dim=-1)
        active_feet = (foot_contacts > contact_threshold).float().sum(dim=-1)

    # 计算稳定性：4个足端应该都有适当接触
    # 考虑步态周期，允许部分足端离地
    stability_score = active_feet / 4.0  # 归一化到0-1

    # 结合姿态信息
    tilt_magnitude = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    upright_bonus = torch.exp(-tilt_magnitude * 5.0)

    return stability_score * upright_bonus


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 5.0
) -> torch.Tensor:
    """足端接触奖励

    鼓励机器人保持适当的足端接触，这对于平衡和站立很重要。

    Args:
        env: 强化学习环境
        sensor_cfg: 接触传感器配置
        threshold: 接触力阈值

    Returns:
        足端接触奖励值
    """
    # 提取接触传感器数据
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # 获取足端接触力
    contact_forces = contact_sensor.data.net_forces_w
    foot_contacts = torch.norm(contact_forces, dim=-1)

    # 计算接触奖励：每个足端都有适当接触时给予奖励
    is_in_contact = (foot_contacts > threshold).float()
    contact_reward = is_in_contact.mean(dim=-1)

    return contact_reward


def torso_upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """躯干直立奖励

    专门针对躯干直立姿态的奖励，强调保持正确的身体姿态。

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置

    Returns:
        躯干直立奖励值
    """
    # 提取机器人状态
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取重力投影
    projected_gravity = asset.data.projected_gravity_b

    # 计算直立程度（重力向量的Z分量应该接近1.0）
    upright_score = projected_gravity[:, 2]  # Z分量

    # 归一化到0-1范围
    upright_reward = torch.clamp(upright_score, 0.0, 1.0)

    return upright_reward


def upright_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.2
) -> torch.Tensor:
    """直立状态额外奖励

    当机器人接近完全直立状态时给予额外奖励。

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        tolerance: 姿容许误差

    Returns:
        直立状态奖励值
    """
    # 提取机器人状态
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前倾斜程度
    projected_gravity = asset.data.projected_gravity_b
    tilt_magnitude = torch.norm(projected_gravity[:, :2], dim=-1)

    # 计算直立程度
    upright_score = torch.exp(-tilt_magnitude * 10.0)  # 指数衰减

    return upright_score


def torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sustained_window: float = 2.0,  # 持续超出时间窗口（秒）
    burst_threshold: float = 1.5,  # 爆发扭矩阈值（额定扭矩的倍数）
    decay_rate: float = 0.9,  # 衰减率
    rated_torque: float = 23.5,  # 额定扭矩
) -> torch.Tensor:
    """扭矩惩罚函数

    惩罚持续超出额定扭矩的行为，但允许瞬时高扭矩
    用于防止电机过热建模，同时允许起跳时的爆发力

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        sustained_window: 持续超出时间窗口（秒）
        burst_threshold: 爆发扭矩阈值（额定扭矩的倍数）
        decay_rate: 衰减率
        rated_torque: 额定扭矩

    Returns:
        扭矩惩罚值
    """
    # 提取机器人数据
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节扭矩
    joint_torques = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    # 获取关节速度（用于判断瞬时行为）
    joint_velocities = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])

    # 判断是否为瞬时高扭矩（起跳动作）
    is_burst = (joint_velocities > 5.0) & (joint_torques > burst_threshold * rated_torque)

    # 计算持续性高扭矩（持续超出）
    sustained_high = (joint_torques > rated_torque) & (~is_burst)

    # 计算惩罚
    penalty = torch.zeros(env.num_envs, device=env.device)

    # 持续性惩罚
    sustained_mask = sustained_high.float()
    sustained_count = torch.clamp(
        torch.full(env.num_envs, device=env.device, dtype=torch.float32),
        0.0,
        1.0,
        min=0.0,
        max=10.0 / sustained_window,  # 最大10步（假设30Hz）
    ).sum(dim=-1)

    # 爆发性惩罚（衰减）
    burst_penalty = torch.where(
        sustained_mask,
        sustained_count * decay_rate ** sustained_count,
        torch.zeros(env.num_envs, device=env.device)
    )

    # 组合惩罚
    penalty = torch.where(
        sustained_mask,
        sustained_count / sustained_window * 0.01,  # 归一化到0-10范围
        burst_penalty  # 爆发性惩罚已包含衰减
    )

    # 总惩罚范围：0-0.01
    penalty = torch.clamp(penalty, min=0.0, max=0.01)

    return -penalty  # 负奖励


def joint_regularization(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.95,  # 软系数
) -> torch.Tensor:
    """关节正则化惩罚

    预留缓冲空间，防止因达到限位导致的"卡死"状态

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        soft_ratio: 软系数（接近极值95%时开始惩罚）

    Returns:
        正则化惩罚值
    """
    # 提取机器人数据
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节位置
    joint_positions = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 获取关节限制（需要从关节限制计算）
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_lower_limits = joint_limits - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]

    # 计算每个关节的接近程度
    joint_upper_limits = joint_lower_limits + 0.001  # 留一点容差
    joint_lower_limits_normalized = (joint_positions - joint_lower_limits) / (joint_upper_limits + 1e-6)

    # 计算接近指数
    closeness = joint_lower_limits_normalized ** 10.0

    # 计算惩罚
    penalty = torch.sum(torch.where(
        closeness > soft_ratio,  # 超过95%接近极限
        closeness * 0.5,  # 中等惩罚
        torch.zeros(env.num_envs, device=env.device),
    ), dim=-1)

    # 归一化到0-0.5
    penalty = torch.clamp(penalty, min=0.0, max=0.5)

    return -penalty  # 负奖励


def joint_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    reward_threshold: float = 0.5,  # 奖励阈值（力值）
) -> torch.Tensor:
    """关节接触奖励

    奖励非足端部位（如膝盖、机械臂）离开地面的行为
    促使机器人利用轮子或肘部支撑借力

    Args:
        env: 强化学习环境
        asset_cfg: 机器人配置
        sensor_cfg: 接触力传感器配置
        reward_threshold: 奖励阈值

    Returns:
        接触奖励值
    """
    # 提取机器人数据
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取接触力（绝对值）
    contact_forces = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids])

    # 获取所有身体名称并处理 sensor_cfg.body_ids
    body_names = contact_sensor.body_names
    if isinstance(sensor_cfg.body_ids, slice):
        num_bodies = len(body_names)
        valid_indices = list(range(*sensor_cfg.body_ids.indices(num_bodies)))
    else:
        valid_indices = list(sensor_cfg.body_ids)

    # 获取基座接触力（查找名为"base"的链接）
    base_index = None
    for i, name in enumerate(body_names):
        if name == "base" and i in valid_indices:
            base_index = i
            break

    if base_index is not None:
        base_contact_forces = contact_forces[:, base_index].sum(dim=-1)  # (num_envs,)
    else:
        # 如果找不到base，使用第一个链接作为基座
        base_contact_forces = contact_forces[:, 0].sum(dim=-1)  # (num_envs,)

    # 过滤出非轮子身体（排除基座）
    non_wheel_body_indices = [
        i for i, name in enumerate(body_names)
        if "wheel" not in name.lower() and i in valid_indices and i != base_index
    ]

    # 计算非轮子身体的接触力（减去基座接触力）
    joint_contact_forces = contact_forces[:, non_wheel_body_indices] - base_contact_forces.unsqueeze(1).unsqueeze(2)

    # 计算关节接触总力（沿xyz维度求和）
    joint_contact_sum = joint_contact_forces.sum(dim=-1)  # (num_envs, num_non_wheel_bodies)

    # 对每个环境，判断是否有任何非轮子身体有显著接触
    has_joint_contact = (joint_contact_sum > reward_threshold).any(dim=-1)  # (num_envs,)

    # 计算奖励（归一化到0-0.5）
    reward = torch.where(
        has_joint_contact,
        0.5,  # 有接触时的奖励
        torch.zeros(env.num_envs, device=env.device),
    )

    return reward


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """身体线加速度惩罚 - 惩罚过大的身体线加速度

    物理意义：
    1. 平滑性：减少身体的突然加速，提高运动平滑度
    2. 稳定性：降低加速度对平衡的影响，减少摔倒风险
    3. 能量效率：避免不必要的能量消耗，提高续航能力
    4. 舒适性：减少冲击和振动，提高运动舒适度

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        身体线加速度惩罚值（负值）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取上一时刻的线速度（如果存在）
    if hasattr(asset, 'last_root_lin_vel') and asset.last_root_lin_vel is not None:
        last_lin_vel = asset.last_root_lin_vel
        current_lin_vel = asset.data.root_lin_vel_w

        # 计算加速度 (v2 - v1) / dt
        dt = env.step_dt
        lin_acc = (current_lin_vel - last_lin_vel) / dt

        # 存储当前速度为下一时刻使用
        asset.last_root_lin_vel = current_lin_vel.clone()
    else:
        # 如果没有上一时刻的速度，使用当前速度作为初始化
        asset.last_root_lin_vel = asset.data.root_lin_vel_w.clone()
        lin_acc = torch.zeros_like(asset.data.root_lin_vel_w)

    # 计算线加速度的L2范数（xyz三个方向）
    acc_norm = torch.norm(lin_acc[:, :3], dim=1)

    # 惩罚过大的加速度
    reward = -torch.square(acc_norm)

    return reward


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """动作变化率惩罚 - 惩罚动作的快速变化

    物理意义：
    1. 平滑控制：鼓励平滑的动作过渡，避免突变
    2. 能量效率：减少动作变化带来的能量浪费
    3. 稳定性：降低因快速动作导致的平衡失调
    4. 硬件保护：保护执行器免受频繁切换的冲击

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        动作变化率惩罚值（负值）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取上一时刻的动作（如果存在）
    if hasattr(asset, 'last_action') and asset.last_action is not None:
        last_action = asset.last_action
        current_action = env.action_manager.data.actions_raw[env.action_manager.actions_idx].flatten()

        # 计算动作变化率
        action_delta = torch.abs(current_action - last_action)
        action_rate = torch.norm(action_delta, dim=0)

        # 存储当前动作为下一时刻使用
        asset.last_action = current_action.clone()
    else:
        # 如果没有上一时刻的动作，使用当前动作作为初始化
        current_action = env.action_manager.data.actions_raw[env.action_manager.actions_idx].flatten()
        asset.last_action = current_action.clone()
        action_rate = torch.zeros(env.num_envs, device=env.device)

    # 惩罚过大的动作变化率
    reward = -torch.square(action_rate)

    return reward


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, soft_ratio: float = 1.0) -> torch.Tensor:
    """关节位置限制惩罚 - 惩罚超出软限制的关节位置

    物理意义：
    1. 安全性：防止关节运动到机械限位附近，避免硬件损坏
    2. 运动学约束：确保关节在有效工作范围内，避免奇异位形
    3. 寿命保护：延长机械部件使用寿命，减少磨损
    4. 控制精度：保持在关节的高效工作区间内

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        soft_ratio: 软比例因子，1.0表示使用默认软限位

    Returns:
        关节位置限制惩罚值（负值）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节位置和软限制
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_limits_min = asset.data.soft_joint_pos_limits_min[:, asset_cfg.joint_ids] * soft_ratio
    joint_pos_limits_max = asset.data.soft_joint_pos_limits_max[:, asset_cfg.joint_ids] * soft_ratio

    # 计算超出限位的程度
    pos_above_max = torch.max(joint_pos - joint_pos_limits_max, torch.zeros_like(joint_pos))
    pos_below_min = torch.max(joint_pos_limits_min - joint_pos, torch.zeros_like(joint_pos))

    # 计算惩罚（平方惩罚）
    pos_penalty = torch.square(pos_above_max) + torch.square(pos_below_min)

    # 对所有关节求和并返回平均惩罚
    reward = -torch.sum(pos_penalty, dim=1)

    return reward


def joint_vel_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, soft_ratio: float = 1.0) -> torch.Tensor:
    """关节速度限制惩罚 - 惩罚超出软限制的关节速度

    物理意义：
    1. 安全性：防止关节速度过快，避免失控和机械损坏
    2. 平滑性：鼓励平滑的运动，减少冲击和振动
    3. 能量效率：减少高速运动带来的不必要的能量消耗
    4. 精度控制：提高轨迹跟踪精度，避免过冲

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        soft_ratio: 软比例因子，1.0表示使用默认软限位

    Returns:
        关节速度限制惩罚值（负值）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节速度和软限制
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel_limits_max = asset.data.soft_joint_vel_limits_max[:, asset_cfg.joint_ids] * soft_ratio
    joint_vel_limits_min = asset.data.soft_joint_vel_limits_min[:, asset_cfg.joint_ids] * soft_ratio

    # 计算超出限位的程度
    vel_above_max = torch.max(joint_vel - joint_vel_limits_max, torch.zeros_like(joint_vel))
    vel_below_min = torch.max(joint_vel_limits_min - joint_vel, torch.zeros_like(joint_vel))

    # 计算惩罚（平方惩罚）
    vel_penalty = torch.square(vel_above_max) + torch.square(vel_below_min)

    # 对所有关节求和并返回平均惩罚
    reward = -torch.sum(vel_penalty, dim=1)

    return reward