from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions (without the wheel joints).

    This function computes the relative joint positions for all selected joints,
    but sets the wheel joint positions to zero. This is useful for wheel-legged
    robots where we want to observe only the leg joint positions.

    Args:
        env: The reinforcement learning environment.
        asset_cfg: Asset configuration for the joints to return.
        wheel_asset_cfg: Asset configuration for the wheel joints to exclude.

    Returns:
        The relative joint positions without wheel joints.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute relative positions for ALL joints first
    all_joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    # Set wheel joint positions to zero
    all_joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    # Return only the selected joints (leg joints)
    return all_joint_pos_rel[:, asset_cfg.joint_ids]


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Compute the gait phase as a sine and cosine tensor (alias for gait_phase).

    Args:
        env: The reinforcement learning environment.
        cycle_time: The duration of one complete gait cycle.

    Returns:
        A tensor containing [sin(phase), cos(phase)] for each environment.
    """
    return gait_phase(env, cycle_time)


def history_buffer(
    env: ManagerBasedRLEnv,
    obs_term_func,
    buffer_length: int = 10,
) -> torch.Tensor:
    """Create a history buffer for observation terms.

    This function maintains a sliding window of past observations for a given
    observation term, allowing the policy to perceive temporal trends.

    Args:
        env: The reinforcement learning environment.
        obs_term_func: The observation function to buffer (e.g., mdp.base_lin_vel).
        buffer_length: Number of history frames to store (default: 10).

    Returns:
        A flattened tensor containing the buffered observations for each environment.
        Shape: (num_envs, obs_dim * buffer_length)
    """
    # Generate a unique cache key for this observation term
    cache_key = f"history_buffer_{obs_term_func.__name__}_{buffer_length}"

    # Initialize buffer if not exists
    if not hasattr(env, cache_key):
        # Get current observation to determine dimension
        current_obs = obs_term_func(env)

        # Initialize buffer with zeros
        env.__dict__[cache_key] = torch.zeros(
            env.num_envs, buffer_length, current_obs.shape[-1],
            device=env.device, dtype=current_obs.dtype
        )
        env.__dict__[f"{cache_key}_index"] = 0

    # Get current observation
    current_obs = obs_term_func(env)

    # Get buffer and index
    buffer = env.__dict__[cache_key]
    buffer_idx = env.__dict__[f"{cache_key}_index"]

    # Update buffer (circular buffer)
    buffer[:, buffer_idx, :] = current_obs
    buffer_idx = (buffer_idx + 1) % buffer_length

    # Store updated index
    env.__dict__[f"{cache_key}_index"] = buffer_idx

    # Return flattened buffer
    return buffer.reshape(env.num_envs, -1)


def joint_pos_history(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    buffer_length: int = 10,
) -> torch.Tensor:
    """History buffer for joint positions.

    This provides the network with past 10 frames of joint position data,
    helping it perceive movement trends and momentum.

    Args:
        env: The reinforcement learning environment.
        asset_cfg: Asset configuration for joints.
        buffer_length: Number of history frames (default: 10).

    Returns:
        Flattened joint position history for each environment.
    """
    return history_buffer(env, lambda e: mdp.joint_pos_rel(e, asset_cfg), buffer_length)


def body_vel_history(
    env: ManagerBasedRLEnv,
    buffer_length: int = 10,
) -> torch.Tensor:
    """History buffer for body linear velocity.

    This provides the network with past 10 frames of body velocity data,
    helping it perceive momentum and acceleration trends.

    Args:
        env: The reinforcement learning environment.
        buffer_length: Number of history frames (default: 10).

    Returns:
        Flattened body velocity history for each environment.
    """
    return history_buffer(env, mdp.base_lin_vel, buffer_length)


def history_joint_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    buffer_length: int = 10,
) -> torch.Tensor:
    """Joint position history L2 penalty.

    This provides a penalty based on the difference between current and previous
    joint positions, encouraging smooth and continuous motion.

    Args:
        env: The reinforcement learning environment.
        asset_cfg: Asset configuration for joints.
        buffer_length: Number of history frames to consider (default: 10).

    Returns:
        Negative penalty value based on joint position changes.
    """
    # Get joint position history
    joint_pos_history = joint_pos_history(env, asset_cfg, buffer_length)

    # Calculate differences between consecutive frames
    # Reshape to (num_envs, buffer_length-1, num_joints)
    joint_pos_reshaped = joint_pos_history.view(env.num_envs, buffer_length, -1)
    pos_diff = joint_pos_reshaped[:, 1:] - joint_pos_reshaped[:, :-1]

    # Calculate L2 norm of differences
    diff_l2 = torch.sum(pos_diff ** 2, dim=(-1, -2))

    return -diff_l2


def body_state_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """身体状态观测 - 提供当前身体状态的详细信息

    这为策略提供了以下关键信息：
    - 身体高度：判断当前处于趴伏、侧卧还是站立状态
    - 倾斜角度：提供身体姿态的详细信息
    - 重心位置：判断重心是否在支撑基础内
    - 角速度：提供当前旋转状态的信息

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        身体状态观测张量，包含[高度, 倾斜角度, 重心x, 重心y, 角速度x, 角速度y, 角速度z]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 身体高度
    body_height = asset.data.root_pos_w[:, 2:3]  # 只取Z坐标

    # 身体倾斜角度（使用投影重力计算）
    projected_gravity = asset.data.projected_gravity_b  # shape: (num_envs, 3)
    print(f"[DEBUG] projected_gravity shape: {projected_gravity.shape}")

    # 分别提取各轴的倾斜角度
    gravity_x = projected_gravity[:, 0:1]  # x component
    gravity_y = projected_gravity[:, 1:2]  # y component
    print(f"[DEBUG] gravity_x shape: {gravity_x.shape}, gravity_y shape: {gravity_y.shape}")

    # 计算倾斜角度（使用acos，确保是2D）
    tilt_angle_x = torch.acos(torch.clamp(gravity_x, -1.0, 1.0)).unsqueeze(-1)  # pitch (x-axis tilt)
    tilt_angle_y = torch.acos(torch.clamp(gravity_y, -1.0, 1.0)).unsqueeze(-1)  # roll (y-axis tilt)
    print(f"[DEBUG] tilt_angle_x shape: {tilt_angle_x.shape}, tilt_angle_y shape: {tilt_angle_y.shape}")

    # 重心位置在世界坐标系中的投影
    com_x = asset.data.root_pos_w[:, 0:1]
    com_y = asset.data.root_pos_w[:, 1:1]

    # 角速度
    ang_vel = asset.data.root_ang_vel_w  # 世界坐标系角速度
    ang_vel_x = ang_vel[:, 0:1]
    ang_vel_y = ang_vel[:, 1:1]
    ang_vel_z = ang_vel[:, 2:3]

    # 组合所有观测
    state_obs = torch.cat([
        body_height.view(-1, 1),
        tilt_angle_x.view(-1, 1),
        tilt_angle_y.view(-1, 1),
        com_x.view(-1, 1),
        com_y.view(-1, 1),
        ang_vel_x.view(-1, 1),
        ang_vel_y.view(-1, 1),
        ang_vel_z.view(-1, 1)
    ], dim=1)

    return state_obs


def contact_state_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """接触状态观测 - 提供详细的接触信息

    这为策略提供了以下关键信息：
    - 足端接触：哪个足端在地面上
    - 接触力大小：每个接触点的力度
    - 接触时间：接触持续时间
    - 非足端接触：膝盖、机械臂等部位的接触情况

    Args:
        env: 强化学习环境
        sensor_cfg: 接触传感器配置
        asset_cfg: 机器人资产配置

    Returns:
        接触状态观测张量，包含[接触数量, 总接触力, 左右接触差异, 前后接触差异, 非足端接触标志]
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # 获取所有接触力
    contact_forces = contact_sensor.data.net_forces_w  # shape: (num_envs, num_bodies, 3)
    contact_norm = torch.norm(contact_forces, dim=-1)  # shape: (num_envs, num_bodies)

    # 找出足端身体
    foot_body_names = [".*_foot"]
    foot_body_indices = []
    for pattern in foot_body_names:
        indices = contact_sensor.find_bodies([pattern])
        if indices and len(indices[0]) > 0:
            foot_body_indices.extend(indices[0])

    # 创建足端掩码
    foot_mask = torch.zeros(contact_norm.shape[1], device=env.device, dtype=torch.bool)
    if foot_body_indices:
        foot_mask[foot_body_indices] = True

    # 计算足端接触信息
    foot_contacts = torch.where(foot_mask.unsqueeze(0), contact_norm, torch.zeros_like(contact_norm))
    num_foot_contacts = torch.sum((foot_contacts > 1.0).float(), dim=1)  # 足端接触数量
    total_foot_force = torch.sum(foot_contacts, dim=1)  # 总足端接触力

    # 左右差异
    FR_contact = foot_contacts[:, 0] if foot_contacts.shape[1] > 0 else torch.zeros(env.num_envs, device=env.device)
    FL_contact = foot_contacts[:, 1] if foot_contacts.shape[1] > 1 else torch.zeros(env.num_envs, device=env.device)
    RL_contact = foot_contacts[:, 2] if foot_contacts.shape[1] > 2 else torch.zeros(env.num_envs, device=env.device)
    RR_contact = foot_contacts[:, 3] if foot_contacts.shape[1] > 3 else torch.zeros(env.num_envs, device=env.device)

    left_right_diff = torch.abs((FR_contact + RR_contact) - (FL_contact + RL_contact))
    front_back_diff = torch.abs((FR_contact + FL_contact) - (RL_contact + RR_contact))

    # 非足端接触
    non_foot_contacts = torch.where(foot_mask.unsqueeze(0), torch.zeros_like(contact_norm), contact_norm)
    significant_non_foot = torch.sum((non_foot_contacts > 1.0).float(), dim=1)  # 非足端显著接触数量

    # 组合接触观测
    contact_obs = torch.cat([
        num_foot_contacts.unsqueeze(1),
        total_foot_force.unsqueeze(1),
        left_right_diff.unsqueeze(1),
        front_back_diff.unsqueeze(1),
        significant_non_foot.unsqueeze(1)
    ], dim=1)

    return contact_obs


def phase_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """阶段观测 - 提供当前阶段的编码信息

    这为策略提供了以下信息：
    - 当前阶段编码：0=趴伏，1=侧卧，2=站立
    - 阶段转换信号：是否刚刚完成阶段转换
    - 阶段置信度：对当前阶段判断的置信度

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置

    Returns:
        阶段观测张量，包含[阶段编码, 阶段置信度, 刚刚转换到趴伏, 刚刚转换到侧卧, 刚刚转换到站立]
    """
    # 导入phase_detection函数
    from ..extended_rewards import phase_detection

    # 获取当前阶段和置信度
    phase, phase_conf = phase_detection(env, asset_cfg)

    # 如果没有历史阶段，初始化
    if not hasattr(env, "past_phase"):
        env.past_phase = phase.clone()

    # 检测阶段转换
    just_to_belly = torch.logical_and(env.past_phase == 1, phase == 0).float()  # 侧卧→趴伏
    just_to_side = torch.logical_and(env.past_phase == 0, phase == 1).float()   # 趴伏→侧卧
    just_to_stand = torch.logical_and(env.past_phase == 0, phase == 2).float()  # 趴伏→站立

    # 更新历史
    env.past_phase = phase.clone()

    # 创建one-hot编码的阶段
    phase_onehot = torch.zeros(env.num_envs, 3, device=env.device)
    phase_onehot.scatter_(1, phase.unsqueeze(1), 1.0)

    # 组合阶段观测
    phase_obs = torch.cat([
        phase_onehot,
        phase_conf.unsqueeze(1),
        just_to_belly.unsqueeze(1),
        just_to_side.unsqueeze(1),
        just_to_stand.unsqueeze(1)
    ], dim=1)

    return phase_obs


def two_stage_state_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """两段式状态观测 - 综合所有状态信息

    这是一个综合观测函数，将身体状态、接触状态和阶段信息组合在一起，
    为策略提供完整的两段式起身任务所需的全部信息。

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        sensor_cfg: 接触传感器配置

    Returns:
        综合状态观测张量，包含身体状态、接触状态和阶段信息
    """
    # 获取各项观测
    body_state = body_state_obs(env, asset_cfg)
    contact_state = contact_state_obs(env, sensor_cfg, asset_cfg)
    phase_state = phase_obs(env, asset_cfg)

    # 组合所有观测
    combined_obs = torch.cat([body_state, contact_state, phase_state], dim=1)

    return combined_obs


def body_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
        身体线加速度L2范数的负惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取身体线加速度
    lin_acc = asset.data.root_lin_acc_w

    # 计算L2范数平方
    acc_l2 = torch.sum(lin_acc ** 2, dim=1)

    return -acc_l2


def action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
        动作变化率L2范数的负惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 如果没有历史动作，初始化
    if not hasattr(env, "past_action"):
        env.past_action = torch.zeros_like(asset.data.joint_pos)

    # 计算动作变化
    action_change = asset.data.joint_pos - env.past_action

    # 更新历史动作
    env.past_action = asset.data.joint_pos.clone()

    # 计算L2范数平方
    rate_l2 = torch.sum(action_change ** 2, dim=1)

    return -rate_l2


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """关节位置限制惩罚 - 惩罚超出软限制的关节位置

    参考：IsaacLab 官方实现
    使用 soft_joint_pos_limits，形状为 (num_envs, num_joints, 2)

    物理意义：
    1. 安全性：防止关节运动到机械限位附近，避免硬件损坏
    2. 运动学约束：确保关节在有效工作范围内，避免奇异位形
    3. 寿命保护：延长机械部件使用寿命，减少磨损
    4. 控制精度：保持在关节的高效工作区间内

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        soft_ratio: 软系数（0-1之间），控制惩罚的严格程度

    Returns:
        关节位置超出限制的惩罚值（负数）
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """关节速度限制惩罚 - 惩罚超出软限制的关节速度

    参考：IsaacLab 官方实现
    使用 soft_joint_vel_limits，形状为 (num_envs, num_joints)

    物理意义：
    1. 安全性：防止关节速度过快，避免失控和机械损坏
    2. 平滑性：鼓励平滑的运动，减少冲击和振动
    3. 能量效率：减少高速运动带来的不必要的能量消耗
    4. 精度控制：提高轨迹跟踪精度，避免过冲

    Args:
        env: 强化学习环境
        asset_cfg: 机器人资产配置
        soft_ratio: 软系数（0-1之间），控制惩罚的严格程度

    Returns:
        关节速度超出限制的惩罚值（负数）
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)
