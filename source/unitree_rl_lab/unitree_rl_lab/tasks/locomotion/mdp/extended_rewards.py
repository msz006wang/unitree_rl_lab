from __future__ import annotations

"""
扩展的Reward函数用于长时间行走和摔倒恢复
参考了FRASA、HoST等项目的设计
"""

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """
    动作镜像奖励 - 鼓励左右对称关节采取相似的绝对动作值

    物理意义：
    1. 对称性：对于对称的机器人结构，左右侧应该执行对称的动作
    2. 简化策略：减少策略需要学习的动作空间复杂度
    3. 稳定性：对称动作有助于保持机器人平衡

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        mirror_joints: 镜像关节对列表，每个元素是一对关节名称

    Returns:
        惩罚值（负数），动作差异越大惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    """
    动作同步奖励 - 鼓励同一组内的关节采取相似的动作

    物理意义：
    1. 协调性：某些关节组需要协调工作，如四足机器人的髋关节
    2. 一致性：减少不必要的关节差异，提高运动效率
    3. 简化控制：降低策略需要学习的复杂度

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        joint_groups: 关节组列表，每个元素是一组需要同步的关节名称

    Returns:
        惩罚值（负数），动作方差越大惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


"""
==============================
长时间行走相关的Reward
==============================
"""


def survival_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    生存奖励 - 每个时间步给予正奖励，鼓励长时间行走
    参考：WalkingSpider_OpenAI_PyBullet项目

    这是最简单但最有效的长时间行走奖励
    """
    return torch.ones(env.num_envs, device=env.device)


def distance_traveled_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    行走距离奖励 - 奖励机器人向前行进的距离
    鼓励机器人探索更远的距离
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取机器人的前进速度（在身体坐标系中）
    forward_velocity = asset.data.root_lin_vel_b[:, 0]  # x方向速度

    # 只在前进命令时奖励
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0]  # 前进命令

    # 只奖励前进方向
    reward = forward_velocity * torch.clamp(forward_cmd, min=0)

    return reward


def energy_efficiency_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_weight: float = 1.0,
    energy_weight: float = -0.1
) -> torch.Tensor:
    """
    能量效率奖励 - 奖励单位能耗下的行进速度
    鼓励机器人以更节能的方式移动
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算前进速度
    forward_velocity = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)

    # 计算能量消耗
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    energy = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

    # 效率 = 速度 / (能量 + 小常数)
    efficiency = forward_velocity / (energy + 1e-6)

    return efficiency


def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
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


def consistent_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    std: float = 0.5
) -> torch.Tensor:
    """
    速度一致性奖励 - 惩罚速度波动，鼓励稳定行走
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "past_velocity"):
        env.past_velocity = torch.zeros_like(asset.data.root_lin_vel_b)

    # 计算速度变化
    velocity_diff = torch.norm(
        asset.data.root_lin_vel_b - env.past_velocity,
        dim=-1
    )

    # 更新历史速度
    env.past_velocity = asset.data.root_lin_vel_b.clone()

    # 使用指数惩罚
    reward = torch.exp(-velocity_diff / std)

    return reward


"""
==============================
摔倒恢复相关的Reward
==============================
"""


def is_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fallback_angle: float = 0.8,  # ~45度
    fallback_height: float = 0.3
) -> torch.Tensor:
    """
    检测机器人是否摔倒
    参考：FRASA论文的摔倒检测方法
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 方法1：检查身体倾斜角度
    # 使用投影重力的z分量来判断是否倾斜
    # projected_gravity_b[:, 2] 在直立时接近1.0，摔倒时接近0
    tilt_angle = torch.acos(
        torch.clamp(asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    is_tilted = tilt_angle > fallback_angle

    # 方法2：检查身体高度
    is_low = asset.data.root_pos_w[:, 2] < fallback_height

    # 结合两种方法
    fallen = torch.logical_or(is_tilted, is_low)

    return fallen.float()


def fall_recovery_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_bonus: float = 10.0,
    recovery_bonus: float = 50.0
) -> torch.Tensor:
    """
    摔倒恢复奖励
    参考：FRASA的奖励设计

    给予以下奖励：
    1. 从摔倒状态恢复到直立状态的大额奖励
    2. 保持直立状态的小额奖励
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 检查当前是否摔倒
    is_currently_fallen = is_fallen(env, asset_cfg)

    # 检查上一帧是否摔倒
    if not hasattr(env, "past_fallen_state"):
        env.past_fallen_state = is_currently_fallen.clone()

    # 检测是否刚刚恢复
    just_recovered = torch.logical_and(
        env.past_fallen_state > 0.5,
        is_currently_fallen < 0.5
    ).float()

    # 更新状态
    env.past_fallen_state = is_currently_fallen.clone()

    # 计算奖励
    reward = (
        just_recovered * recovery_bonus +  # 恢复奖励
        (1 - is_currently_fallen) * upright_bonus * 0.01  # 保持直立的小奖励
    )

    return reward


def stand_up_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.78,
    std: float = 0.2
) -> torch.Tensor:
    """
    站起进度奖励 - 奖励向目标高度靠近
    参考：HoST的站立奖励设计
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 只在摔倒状态下给予此奖励
    fallen = is_fallen(env, asset_cfg)

    # 计算当前高度与目标高度的差距
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(current_height - target_height)

    # 使用高斯形状的奖励
    reward = torch.exp(-height_error / std) * fallen

    return reward


def upright_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.3
) -> torch.Tensor:
    """
    直立姿态奖励 - 奖励保持身体直立
    参考：FRASA的方向奖励
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 使用投影重力的z分量（1.0表示完全直立）
    uprightness = asset.data.projected_gravity_b[:, 2]

    # 使用高斯形状的奖励
    reward = torch.exp(-torch.abs(1.0 - uprightness) / std)

    return reward


def ground_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    body_names: list[str] = [".*ankle.*", ".*_contact.*"],
    penalty_weight: float = -1.0
) -> torch.Tensor:
    """
    非脚部接触惩罚 - 惩罚膝盖、手臂等部位接触地面
    参考：FRASA的不期望接触惩罚
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取所有身体的接触状态
    all_contacts = contact_sensor.data.current_contact_time > 0

    # 获取脚部接触（我们希望这些）
    # 这里假设body_names中包含了脚部名称
    foot_ids = []
    for pattern in body_names:
        matching_bodies = env.scene[asset_cfg.name].find_bodies(pattern)
        foot_ids.extend(matching_bodies)

    # 惩罚非脚部接触
    # 这需要更细致的实现，取决于具体的身体命名
    # 简化版本：惩罚除脚部外的所有接触
    reward = torch.zeros(env.num_envs, device=env.device)
    # TODO: 实现详细的非期望接触检测

    return reward * penalty_weight


def stable_base_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_std: float = 0.2,
    ang_vel_std: float = 0.1
) -> torch.Tensor:
    """
    稳定基座奖励 - 惩罚过快的线速度和角速度变化
    有助于机器人保持平稳
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "past_base_lin_vel"):
        env.past_base_lin_vel = asset.data.root_lin_vel_b.clone()
        env.past_base_ang_vel = asset.data.root_ang_vel_b.clone()

    # 计算加速度
    lin_acc = torch.norm(
        asset.data.root_lin_vel_b - env.past_base_lin_vel,
        dim=-1
    )
    ang_acc = torch.norm(
        asset.data.root_ang_vel_b - env.past_base_ang_vel,
        dim=-1
    )

    # 更新历史
    env.past_base_lin_vel = asset.data.root_lin_vel_b.clone()
    env.past_base_ang_vel = asset.data.root_ang_vel_b.clone()

    # 指数惩罚
    reward = torch.exp(-lin_acc / lin_vel_std) * torch.exp(-ang_acc / ang_vel_std)

    return reward


"""
==============================
组合Reward函数
==============================
"""


def locomotion_bonus_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    运动奖励组合 - 结合多个奖励项
    """
    return (
        1.0 * survival_reward(env, asset_cfg) +
        0.5 * distance_traveled_reward(env, asset_cfg, command_name) +
        0.1 * energy_efficiency_reward(env, asset_cfg) +
        0.3 * consistent_velocity_reward(env, asset_cfg, command_name)
    )


def recovery_bonus_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    恢复奖励组合 - 结合摔倒恢复相关奖励
    """
    return (
        1.0 * fall_recovery_reward(env, asset_cfg) +
        0.3 * stand_up_progress_reward(env, asset_cfg) +
        0.5 * upright_orientation_reward(env, asset_cfg) +
        0.2 * stable_base_reward(env, asset_cfg)
    )
