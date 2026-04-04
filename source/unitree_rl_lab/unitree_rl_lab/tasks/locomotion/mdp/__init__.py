"""
Terminations for MDP.
"""

import torch
from isaaclab.envs.mdp import *  # noqa: F401, F403

# 导出扩展奖励函数
from .extended_rewards import (  # noqa: F401
    action_mirror,
    action_sync,
    wheel_vel_penalty,
    feet_air_time,
    survival_reward,
    distance_traveled_reward,
    energy_efficiency_reward,
    fall_recovery_reward,
    is_fallen,
    upright_orientation_reward,
    # 新增GO2W ARM专用奖励函数
    upward_velocity,
    upward,  # Alias for upward_velocity
    upward_orientation,  # 基于姿态的向上奖励
    orientation_tracking,
    torque_penalty,
    joint_regularization,
    contact_management,
    wheel_assisted_recovery,
    # 参考robot_lab_locomanip的奖励函数
    feet_height,
    feet_height_body,
    feet_slide,
    joint_pos_penalty,
    feet_contact,
    feet_contact_without_cmd,
    feet_stumble,
    # 两段式恢复专用自适应奖励函数
    action_rate_adaptive,  # 自适应动作变化率惩罚
    torque_adaptive,  # 自适应扭矩惩罚
    contact_adaptive,  # 自适应非期望接触惩罚
)

# 导出基础奖励函数（来自rewards.py）
from .rewards import (  # noqa: F401
    joint_mirror,
    joint_vel_penalty,
    joint_power,
)

# 两段式起身策略专用奖励函数（待实现）
def phase_detection(env, asset_cfg):
    """阶段检测函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def tuck_and_roll_reward(env, asset_cfg):
    """蜷缩滚动奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def wheel_braking_reward(env, asset_cfg):
    """轮子刹车奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def asymmetric_kick_reward(env, asset_cfg):
    """非对称踢腿奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def explode_to_stand_reward(env, asset_cfg):
    """爆发行走奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def transition_reward(env, asset_cfg):
    """过渡奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

def two_stage_standing_reward(env, asset_cfg):
    """两段式站立奖励函数 - 待实现"""
    return torch.zeros(env.num_envs, device=env.device)

# 导出扩展观测函数
from .observations import (  # noqa: F401
    joint_pos_rel_without_wheel,
    gait_phase,
    phase,
    # 新增历史观测函数
    history_buffer,
    joint_pos_history,
    body_vel_history,
    history_joint_pos_l2,
    # 两段式状态观测函数
    body_state_obs,
    contact_state_obs,
    phase_obs,
    two_stage_state_obs,
    # 其他观测函数
    body_lin_acc_l2,
    action_rate_l2,
    joint_pos_limits,
    joint_vel_limits,
)

# 导出两段式恢复专用奖励函数
from .extended_rewards import (  # noqa: F401
    wheel_angular_momentum_reward,
)

# 导出课程学习函数
from .curriculums import (  # noqa: F401
    lin_vel_cmd_levels,
    ang_vel_cmd_levels,
    terrain_levels_vel,
    command_levels_vel,
    difficulty_levels_two_stage,
)

# 导出终止函数
from .terminations import (  # noqa: F401
    terrain_out_of_bounds,
    is_success_stand,
)
