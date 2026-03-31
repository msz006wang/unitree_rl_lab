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
GO2W ARM 专用Reward函数
==============================
"""


def upward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    向上速度奖励 - 鼓励身体Z轴向上速度，促进快速蹬地起跳

    物理意义：
    1. 爆发力：鼓励机器人产生向上的爆发力，从地面弹起
    2. 站立恢复：当机器人从侧卧状态恢复时，向上的速度有助于重新站立
    3. 动量利用：利用向上动量完成从不倒翁式的恢复动作

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        向上速度奖励值（正数），向上速度越大奖励越高
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取身体Z轴向上的线速度（在世界坐标系中）
    upward_velocity = asset.data.root_lin_vel_w[:, 2]

    # 只奖励向上的速度（正数）
    reward = torch.clamp(upward_velocity, min=0.0)

    return reward


def orientation_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    姿态跟踪奖励 - 奖励身体Z轴与世界坐标系Z轴重合

    物理意义：
    1. 直立稳定性：鼓励机器人保持直立姿态
    2. 重心控制：正确的姿态有助于控制重心位置
    3. 恢复导向：当机器人倾斜时，此奖励引导其恢复到直立状态

    计算方法：
    使用投影重力的Z分量作为衡量指标：
    - 完全直立时：projected_gravity_b[:, 2] = 1.0
    - 完全倒下时：projected_gravity_b[:, 2] = 0.0

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        姿态跟踪奖励值（0到1之间），越直立奖励越高
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 使用投影重力的Z分量（1.0表示完全直立，0.0表示完全倒下）
    uprightness = asset.data.projected_gravity_b[:, 2]

    # 归一化到0-1范围
    reward = torch.clamp(uprightness, 0.0, 1.0)

    return reward


def torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sustained_window: float = 2.0,
    burst_threshold: float = 1.5,
    decay_rate: float = 0.9,
    rated_torque: float = 23.5,
) -> torch.Tensor:
    """
    扭矩惩罚 - 惩罚持续超出额定扭矩，允许瞬时高扭矩

    物理意义：
    1. 过热保护：防止电机长时间在高负载下工作导致过热
    2. 爆发力允许：允许起跳瞬间的爆发扭矩，这是恢复站立所必需的
    3. 持续性惩罚：只有持续超出额定扭矩才给予惩罚，瞬时峰值是允许的

    实现方法：
    - 维护每个关节的扭矩历史（指数移动平均）
    - 只有当持续超出额定扭矩超过指定时间窗口时才惩罚
    - 使用衰减率控制历史的影响

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        sustained_window: 持续超出时间窗口（秒）
        burst_threshold: 瞬发扭矩阈值（额定扭矩的倍数）
        decay_rate: 衰减率（0-1之间，越大历史影响越小）
        rated_torque: 额定扭矩值

    Returns:
        扭矩惩罚值（负数），持续超标越大惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取当前扭矩
    current_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])

    # 初始化或更新扭矩历史
    if not hasattr(env, "torque_history"):
        env.torque_history = torch.zeros_like(current_torque)
        env.torque_counter = torch.zeros_like(current_torque)

    # 指数移动平均
    env.torque_history = decay_rate * env.torque_history + (1 - decay_rate) * current_torque

    # 计算超出持续时间（帧数）
    is_over_limit = env.torque_history > (rated_torque * burst_threshold)
    env.torque_counter = torch.where(is_over_limit, env.torque_counter + 1, 0)

    # 计算持续超出时间（秒）
    sustained_time = env.torque_counter * env.step_dt

    # 只惩罚持续超出时间窗口的情况
    over_window = sustained_time > sustained_window

    # 惩罚值 = 超出幅度 * 持续时间
    exceed_amount = env.torque_history - rated_torque
    penalty = over_window.float() * exceed_amount * sustained_time

    # 求和所有关节的惩罚
    reward = -torch.sum(penalty, dim=-1)

    return reward


def joint_regularization(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.95,
) -> torch.Tensor:
    """
    关节正则化奖励 - 惩罚关节位置接近极值

    物理意义：
    1. 避免卡死：预留缓冲空间，防止因达到限位导致的"卡死"状态
    2. 运动灵活性：保持在关节限位内有一定余量，提高运动灵活性
    3. 安全性：避免在极端位置工作，减少关节磨损

    实现方法：
    - 计算每个关节距离限位的百分比
    - 当接近限位时（超过soft_ratio）开始惩罚
    - 使用指数函数增强接近限位时的惩罚

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        soft_ratio: 软系数（0-1之间），距离极值的百分比阈值

    Returns:
        关节正则化惩罚值（负数），越接近极值惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节位置和限位
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 获取关节限位（直接从asset获取，避免缓存问题）
    joint_limits = asset.data.joint_pos_limits
    if joint_limits is not None:
        # joint_limits的形状可能是 (2, num_joints) 或 (num_joints, 2)
        # 第一种情况：第一行是下限，第二行是上限
        # 第二种情况：第一列是下限，第二列是上限
        if joint_limits.shape[0] == 2 and len(joint_limits.shape) == 2:
            # 形状 (2, num_joints): [lower_limits, upper_limits]
            # 使用index_select确保正确的维度
            # 转换asset_cfg.joint_ids为tensor
            if isinstance(asset_cfg.joint_ids, (list, tuple)):
                joint_ids_tensor = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
            elif hasattr(asset_cfg.joint_ids, '__len__'):
                joint_ids_tensor = torch.arange(len(asset_cfg.joint_ids), device=env.device, dtype=torch.long)
            else:
                joint_ids_tensor = asset_cfg.joint_ids
            limits_lower = torch.index_select(joint_limits[0], 0, joint_ids_tensor)
            limits_upper = torch.index_select(joint_limits[1], 0, joint_ids_tensor)
        else:
            # 形状 (num_joints, 2): [[lower1, upper1], [lower2, upper2], ...]
            limits = joint_limits[asset_cfg.joint_ids, :]  # shape: (num_selected_joints, 2)
            limits_lower = limits[:, 0]
            limits_upper = limits[:, 1]
    else:
        # 默认限位
        limits_lower = torch.ones(len(asset_cfg.joint_ids), device=env.device) * -1.0
        limits_upper = torch.ones(len(asset_cfg.joint_ids), device=env.device)

    # 计算每个关节在限位范围内的位置百分比（0表示下限，1表示上限）
    range_size = limits_upper - limits_lower
    normalized_pos = (joint_pos - limits_lower) / range_size.unsqueeze(0)

    # 计算距离最近限位的最小距离
    dist_to_lower = normalized_pos
    dist_to_upper = 1.0 - normalized_pos
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)

    # 软限位：只惩罚小于soft_ratio的距离
    # 例如soft_ratio=0.95表示距离限位小于5%时开始惩罚
    safe_zone = soft_ratio
    penalty_zone = torch.clamp(safe_zone - min_dist, min=0.0)

    # 使用指数函数增强惩罚
    penalty = torch.sum(torch.exp(penalty_zone * 10.0), dim=-1)

    return -penalty


def contact_management(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_body_names: list = None,
) -> torch.Tensor:
    """
    接触管理奖励 - 奖励非足端部位离开地面

    物理意义：
    1. 接触简化：鼓励机器人仅通过足端与地面接触，简化控制
    2. 避免干扰：减少膝盖、机械臂等部位接触地面，避免干扰运动
    3. 策略引导：引导机器人学习如何正确使用身体部位
    4. 借力策略：允许轮子和肘部在特定情况下借力

    实现方法：
    - 监测非足端部位的接触状态
    - 奖励这些部位离开地面的行为
    - 根据身体部位的重要性给予不同权重

    Args:
        env: 强化学习环境
        sensor_cfg: 接触传感器配置
        foot_body_names: 足端身体名称列表，这些部位允许接触

    Returns:
        接触管理奖励值（负数），非期望接触越大惩罚越大
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # 如果没有提供足端名称，使用默认模式
    if foot_body_names is None:
        foot_body_names = [".*_foot"]

    # 获取所有接触的身体
    contact_forces = contact_sensor.data.net_forces_w  # shape: (num_envs, num_bodies, 3)
    contact_norm = torch.norm(contact_forces, dim=-1)  # shape: (num_envs, num_bodies)

    # 找出足端身体索引
    foot_body_indices = []
    for pattern in foot_body_names:
        indices = contact_sensor.find_bodies([pattern])
        foot_body_indices.extend(indices[0] if len(indices) > 0 else [])

    # 创建足端掩码
    foot_mask = torch.zeros(contact_norm.shape[1], device=env.device, dtype=torch.bool)
    if foot_body_indices:
        foot_mask[foot_body_indices] = True

    # 非足端部位的接触力
    non_foot_contacts = torch.where(foot_mask.unsqueeze(0), torch.zeros_like(contact_norm), contact_norm)

    # 只考虑超过阈值的接触（避免噪声）
    contact_threshold = 1.0  # N
    significant_contacts = (non_foot_contacts > contact_threshold).float()

    # 惩罚：所有显著的非足端接触的总和
    penalty = torch.sum(significant_contacts, dim=-1)

    return -penalty


def wheel_assisted_recovery(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joint_names: list = None,
) -> torch.Tensor:
    """
    轮子辅助恢复奖励 - 鼓励在侧卧时使用轮子辅助改变姿态

    物理意义：
    1. 轮足协同：利用轮子产生地面摩擦力，辅助改变机身朝向
    2. 姿态转换：将"侧向推起"转化为"前后撑起"
    3. 借力策略：轮子转动可以作为额外的支撑点和动力源

    实现方法：
    - 检测机器人是否处于侧卧状态（倾角较大）
    - 计算轮子产生的扭矩/速度
    - 根据机器人倾斜方向和轮子动作的协同性给予奖励

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        wheel_joint_names: 轮子关节名称列表

    Returns:
        轮子辅助恢复奖励值（正数），轮足协同越好奖励越高
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 如果没有提供轮子名称，使用默认
    if wheel_joint_names is None:
        wheel_joint_names = [".*_foot_joint"]

    # 找出轮子关节索引
    import re
    wheel_indices = []

    # 获取所有关节名称
    all_joint_names = asset.data.joint_names

    for pattern in wheel_joint_names:
        # 使用正则表达式匹配关节名称
        for idx, joint_name in enumerate(all_joint_names):
            if re.match(pattern, joint_name):
                wheel_indices.append(idx)

    # 转换为tensor索引
    wheel_indices = torch.tensor(wheel_indices, device=env.device, dtype=torch.long)

    # 检测是否处于侧卧状态（倾角超过阈值）
    # 使用投影重力：侧卧时projected_gravity_b[:, 2] < 0.3
    tilt_severity = torch.clamp(0.3 - asset.data.projected_gravity_b[:, 2], min=0.0)
    is_side_lying = tilt_severity > 0.0

    # 获取轮子速度
    wheel_velocities = torch.abs(asset.data.joint_vel[:, wheel_indices])

    # 计算轮子产生的有效扭矩（假设：速度×刚度）
    # 这需要知道轮子的刚度，这里简化处理
    wheel_torques = wheel_velocities * 10.0  # 简化假设

    # 计算角速度（用于判断转向）
    ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])  # yaw角速度

    # 协同性：轮子动作应该有助于改变姿态（产生角速度）
    # 奖励轮子扭矩和角速度的正相关性
    synergy = torch.mean(wheel_torques, dim=-1) * ang_vel

    # 只在侧卧时给予奖励
    reward = synergy * is_side_lying.float() * tilt_severity

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
