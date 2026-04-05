from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """基于机器人在期望速度命令下行走距离的课程学习。
    该术语用于在机器人行走足够远时增加地形难度并在机器人行走距离小于命令速度要求距离的一半时降低难度。

    注意::
        该术语只能与地形类型 generator 一起使用。
        有关不同地形类型的更多信息，请查看 :class:`isaaclab.terrains.TerrainImporter` 类。

    返回值:
        给定环境ID的平均地形级别。
    """
    # 获取机器人和地形实例
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 获取运动命令
    command = env.command_manager.get_command("base_velocity")

    # 计算机器人行走的距离
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # 判断是否升级到更难地形
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    # 判断是否降级到简单地形
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # 更新地形级别
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 返回平均地形级别
    return torch.mean(terrain.terrain_levels.float())


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """
    命令速度课程学习函数

    物理意义：
    1. 自适应难度：根据机器人跟踪表现动态调整速度命令范围
    2. 循序渐进：从低速开始，逐渐增加到高速
    3. 性能驱动：表现好时增加难度，表现差时保持
    4. 泛化能力：最终学会在各种速度下稳定行走

    工作原理：
    - 初始化时设置初始和最终的速度范围
    - 每个 episode 结束后检查跟踪奖励
    - 如果跟踪奖励 > 80%，增加速度范围
    - 速度范围不会超过最终设定的范围

    速度范围调整：
    - 初始范围 = 原始范围 × range_multiplier[0]
    - 最终范围 = 原始范围 × range_multiplier[1]
    - 每次调整增加 0.1 m/s

    Args:
        env: 强化学习环境实例
        env_ids: 需要更新的环境索引列表
        reward_term_name: 用于评估表现的奖励项名称（通常是跟踪奖励）
        range_multiplier: 范围乘数，(初始乘数, 最终乘数)
                       默认 (0.1, 1.0) 表示从10%到100%的原始范围

    Returns:
        当前最大速度值（用于监控）
    """

    # 获取基础速度命令的配置
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # 初始化：只在第一个 episode 执行
    if env.common_step_counter == 0:
        # 保存原始速度范围
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

        # 计算初始速度范围
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]

        # 计算最终速度范围
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # 初始化命令范围为初始值
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # 每个 episode 结束后更新课程
    if env.common_step_counter % env.max_episode_length == 0:

        # 获取 episode 累积奖励
        episode_sums = env.reward_manager._episode_sums[reward_term_name]

        # 获取奖励项配置
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

        # 定义速度范围调整量
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # 判断是否需要增加难度
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:

            # 计算新的速度范围
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # 限制速度范围不超过最终范围
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # 更新速度范围
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    # 返回当前最大速度
    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def difficulty_levels_two_stage(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "two_stage_standing",
    range_multiplier: Sequence[float] = (0.3, 1.0),
) -> torch.Tensor:
    """
    两段式恢复难度课程学习函数

    物理意义：
    1. 渐进式恢复：从简单的侧卧状态开始，逐渐增加到复杂状态
    2. 自适应难度：根据恢复表现动态调整初始状态难度
    3. 能力建设：确保策略在简单任务上表现良好后再增加难度
    4. 泛化能力：最终在各种初始状态下都能成功恢复

    工作原理：
    - 初始化时设置初始和最终的状态难度范围
    - 每个 episode 结束后检查恢复奖励
    - 如果恢复奖励 > 80%，增加初始状态的随机性（增加难度）
    - 难度范围不会超过最终设定的范围

    难度调整：
    - 初始难度 = 原始难度 × range_multiplier[0]
    - 最终难度 = 原始难度 × range_multiplier[1]
    - 每次调整增加初始随机性的范围

    Args:
        env: 强化学习环境实例
        env_ids: 需要更新的环境索引列表
        reward_term_name: 用于评估表现的奖励项名称（通常是两段式站立奖励）
        range_multiplier: 难度乘数，(初始乘数, 最终乘数)
                     默认 (0.3, 1.0) 表示从30%到100%的原始难度

    Returns:
        当前难度级别（用于监控）
    """

    # 获取事件配置中的重置参数
    reset_event = env.cfg.events.randomize_reset_base
    if reset_event is None:
        # 如果没有找到重置事件，返回默认难度
        return torch.ones(len(env_ids), device=env.device)

    pose_range = reset_event.params["pose_range"]

    # 初始化：只在第一个 episode 执行
    if env.common_step_counter == 0:
        # 保存原始pose范围
        env._original_pose_range = {
            "roll": pose_range["roll"],
            "pitch": pose_range["pitch"],
            "yaw": pose_range["yaw"],
        }

        # 计算初始和最终的范围
        env._initial_pose_range = {
            "roll": (env._original_pose_range["roll"][0] * range_multiplier[0],
                    env._original_pose_range["roll"][1] * range_multiplier[0]),
            "pitch": (env._original_pose_range["pitch"][0] * range_multiplier[0],
                     env._original_pose_range["pitch"][1] * range_multiplier[0]),
            "yaw": (env._original_pose_range["yaw"][0] * range_multiplier[0],
                   env._original_pose_range["yaw"][1] * range_multiplier[0]),
        }

        env._final_pose_range = {
            "roll": (env._original_pose_range["roll"][0] * range_multiplier[1],
                    env._original_pose_range["roll"][1] * range_multiplier[1]),
            "pitch": (env._original_pose_range["pitch"][0] * range_multiplier[1],
                     env._original_pose_range["pitch"][1] * range_multiplier[1]),
            "yaw": (env._original_pose_range["yaw"][0] * range_multiplier[1],
                   env._original_pose_range["yaw"][1] * range_multiplier[1]),
        }

        # 初始化pose范围为初始值
        pose_range.update(env._initial_pose_range)

    # 每个 episode 结束后更新课程
    if env.common_step_counter % env.max_episode_length == 0:

        # 获取 episode 累积奖励
        episode_sums = env.reward_manager._episode_sums[reward_term_name]

        # 获取奖励项配置
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

        # 定义难度调整量
        delta_difficulty = 0.1  # 每次增加10%的随机性

        # 判断是否需要增加难度
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:

            # 计算新的pose范围
            new_roll_range = (
                pose_range["roll"][0] - delta_difficulty,
                pose_range["roll"][1] + delta_difficulty
            )
            new_pitch_range = (
                pose_range["pitch"][0] - delta_difficulty,
                pose_range["pitch"][1] + delta_difficulty
            )
            new_yaw_range = (
                pose_range["yaw"][0] - delta_difficulty,
                pose_range["yaw"][1] + delta_difficulty
            )

            # 限制难度不超过最终范围
            new_roll_range = (
                max(new_roll_range[0], env._final_pose_range["roll"][0]),
                min(new_roll_range[1], env._final_pose_range["roll"][1])
            )
            new_pitch_range = (
                max(new_pitch_range[0], env._final_pose_range["pitch"][0]),
                min(new_pitch_range[1], env._final_pose_range["pitch"][1])
            )
            new_yaw_range = (
                max(new_yaw_range[0], env._final_pose_range["yaw"][0]),
                min(new_yaw_range[1], env._final_pose_range["yaw"][1])
            )

            # 更新pose范围
            pose_range.update({
                "roll": new_roll_range,
                "pitch": new_pitch_range,
                "yaw": new_yaw_range
            })

    # 返回当前难度级别（使用roll角度范围的宽度作为指标）
    current_difficulty = pose_range["roll"][1] - pose_range["roll"][0]
    max_difficulty = env._final_pose_range["roll"][1] - env._final_pose_range["roll"][0]
    difficulty_level = current_difficulty / max_difficulty if max_difficulty > 0 else 1.0

    return torch.ones(len(env_ids), device=env.device) * difficulty_level


# =============================================================================
# Multi-Level Posture Curriculum Configuration
# =============================================================================

POSTURE_CURRICULUM_LEVELS = {
    0: {
        "name": "Upright Small Tilt (Phase 1 Baseline)",
        "description": "Basic balance maintenance from near-upright position",
        "pose_range": {
            "roll": (-0.2, 0.2),  # ±11.5° (Phase 1: 从直立小倾斜开始)
            "pitch": (-0.1, 0.1),  # ±5.7°
            "yaw": (-3.14, 3.14),  # Full range
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.45, 0.5)  # 提高初始高度，更接近站立状态
        },
        "velocity_range": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.1, 0.1),
        },
        "success_threshold": 0.70,  # 降低到 70%，更容易升级
        "min_episodes": 200,  # 增加到 200，给足够时间学习
        "focus": "Learn to stand still from near-upright position"
    },
    1: {
        "name": "Small Posture Variation",
        "description": "Recovery from small tilt with leg coordination",
        "pose_range": {
            "roll": (-0.4, 0.4),  # ±23°
            "pitch": (-0.2, 0.2),  # ±11.5°
            "yaw": (-3.14, 3.14),
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (0.35, 0.5)
        },
        "velocity_range": {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        },
        "success_threshold": 0.60,  # 降低到 60%
        "min_episodes": 200,
        "focus": "Learn vertical push-up from small tilt"
    },
    2: {
        "name": "Moderate Posture Variation",
        "description": "Recovery from moderate tilt with momentum",
        "pose_range": {
            "roll": (-0.6, 0.6),  # ±34°
            "pitch": (-0.3, 0.3),  # ±17°
            "yaw": (-3.14, 3.14),
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.3, 0.45)
        },
        "velocity_range": {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        },
        "success_threshold": 0.50,  # 降低到 50%
        "min_episodes": 200,
        "focus": "Learn torque distribution with momentum"
    },
    3: {
        "name": "Large Posture Variation (Target)",
        "description": "Full recovery from large tilt - final target",
        "pose_range": {
            "roll": (-0.8, 0.8),  # ±45° (目标姿态，保持不变)
            "pitch": (-0.3, 0.3),  # ±17°
            "yaw": (-3.14, 3.14),
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (0.4, 0.5)
        },
        "velocity_range": {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        },
        "success_threshold": 0.40,  # 降低到 40%
        "min_episodes": 200,
        "focus": "Complete recovery from large tilt (±45°)"
    }
}


def posture_curriculum_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    check_interval: int = 100,
    enable_backward: bool = True,
    hysteresis: float = 0.1
) -> torch.Tensor:
    """
    Multi-level posture recovery curriculum learning function.

    This function implements a 4-level progressive curriculum for the GO2W-ARM robot's
    two-stage recovery task, starting from simple balance and gradually increasing to
    extreme orientation recovery.

    Physical Principles:
    1. Progressive Skill Building: Each level builds on skills learned in previous levels
    2. Difficulty Gradient: Smooth transition from easy to hard initial conditions
    3. Adaptive Progression: Automatic advancement based on performance metrics
    4. Backward Recovery: Reduces difficulty when performance degrades

    Level Progression:
    - Level 0 (±5°): Learn basic balance with arm weight
    - Level 1 (±30°): Learn vertical push-up from low height
    - Level 2 (±60°): Learn recovery from moderate tilt
    - Level 3 (±180°): Learn complete flip recovery

    Args:
        env: ManagerBasedRLEnv instance
        env_ids: Environment indices to update
        asset_cfg: Robot asset configuration
        check_interval: Episodes between curriculum checks (default: 100)
        enable_backward: Whether to allow backward level recovery (default: True)
        hysteresis: Hysteresis factor to prevent level oscillation (default: 0.1)

    Returns:
        Success flags tensor (all zeros, function mainly updates state)

    Note:
        This function maintains curriculum state in the environment:
        - env._posture_curriculum_level: Current level for each environment (0-3)
        - env._posture_curriculum_episode_count: Episodes at current level
        - env._posture_curriculum_success_count: Successful episodes
        - env._posture_curriculum_timeout_count: Timeout episodes
        - env._posture_curriculum_last_check: Last check step
        - env._posture_curriculum_frozen: Manual freeze flag
        - env._posture_curriculum_history: List of check results
    """
    # Initialize curriculum state (first call only)
    if not hasattr(env, "_posture_curriculum_level"):
        env._posture_curriculum_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._posture_curriculum_episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._posture_curriculum_success_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._posture_curriculum_timeout_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._posture_curriculum_last_check = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._posture_curriculum_frozen = False
        env._posture_curriculum_history = []

        print("[Posture Curriculum] Initialized 4-level posture curriculum")
        print(f"  Level 0: {POSTURE_CURRICULUM_LEVELS[0]['name']}")
        print(f"  Level 1: {POSTURE_CURRICULUM_LEVELS[1]['name']}")
        print(f"  Level 2: {POSTURE_CURRICULUM_LEVELS[2]['name']}")
        print(f"  Level 3: {POSTURE_CURRICULUM_LEVELS[3]['name']}")

    # Return immediately if frozen
    if env._posture_curriculum_frozen:
        return torch.zeros(len(env_ids), device=env.device)

    # Update episode count
    env._posture_curriculum_episode_count[env_ids] += 1

    # Get reset event to update parameters
    reset_event = env.cfg.events.randomize_reset_base
    if reset_event is None:
        return torch.mean(env._posture_curriculum_level[env_ids].float())

    # Check if we need to evaluate curriculum progression
    should_check = (env._posture_curriculum_episode_count[env_ids] % check_interval == 0) & \
                   (env._posture_curriculum_episode_count[env_ids] > 0)

    if should_check.any():
        check_ids = env_ids[should_check]

        # Check for episode terminations to count successes/timeout
        # This requires access to termination manager state
        if hasattr(env, 'termination_manager'):
            # Get termination states from last reset
            # Note: This depends on Isaac Lab's internal state tracking
            # We'll use a simplified approach based on episode length
            pass

        for env_id in check_ids:
            current_level = env._posture_curriculum_level[env_id].item()
            episode_count = env._posture_curriculum_episode_count[env_id].item()
            success_count = env._posture_curriculum_success_count[env_id].item()
            timeout_count = env._posture_curriculum_timeout_count[env_id].item()

            # Calculate success rate (use timeout count as proxy for survival)
            # If not timed out, robot survived the episode
            survival_rate = 1.0 - (timeout_count / episode_count) if episode_count > 0 else 0.0

            # Get current level's threshold
            level_config = POSTURE_CURRICULUM_LEVELS.get(current_level)
            if level_config is None:
                continue

            threshold = level_config["success_threshold"]
            min_episodes = level_config["min_episodes"]

            # Check if we should advance to next level
            if episode_count >= min_episodes:
                if survival_rate >= threshold:
                    # Advance to next level
                    if current_level < 3:
                        new_level = current_level + 1
                        env._posture_curriculum_level[env_id] = new_level
                        env._posture_curriculum_episode_count[env_id] = 0
                        env._posture_curriculum_success_count[env_id] = 0
                        env._posture_curriculum_timeout_count[env_id] = 0

                        # Update reset parameters
                        _update_reset_parameters(env, env_id, new_level)

                        log_entry = {
                            "step": env.common_step_counter,
                            "env_id": env_id.item() if torch.is_tensor(env_id) else env_id,
                            "transition": f"{current_level} -> {new_level}",
                            "survival_rate": survival_rate,
                            "level_name": POSTURE_CURRICULUM_LEVELS[new_level]["name"]
                        }
                        env._posture_curriculum_history.append(log_entry)

                        print(f"[Posture Curriculum] Env {env_id}: Level {current_level} -> Level {new_level} "
                              f"(survival_rate={survival_rate:.2f}, threshold={threshold:.2f})")

                elif enable_backward and survival_rate < threshold * (1.0 - hysteresis):
                    # Backward recovery
                    if current_level > 0:
                        new_level = current_level - 1
                        env._posture_curriculum_level[env_id] = new_level
                        env._posture_curriculum_episode_count[env_id] = 0
                        env._posture_curriculum_success_count[env_id] = 0
                        env._posture_curriculum_timeout_count[env_id] = 0

                        # Update reset parameters
                        _update_reset_parameters(env, env_id, new_level)

                        log_entry = {
                            "step": env.common_step_counter,
                            "env_id": env_id.item() if torch.is_tensor(env_id) else env_id,
                            "transition": f"{current_level} -> {new_level} (backward)",
                            "survival_rate": survival_rate,
                            "level_name": POSTURE_CURRICULUM_LEVELS[new_level]["name"]
                        }
                        env._posture_curriculum_history.append(log_entry)

                        print(f"[Posture Curriculum] Env {env_id}: Level {current_level} -> Level {new_level} "
                              f"(backward recovery, survival_rate={survival_rate:.2f})")

    # 返回当前环境的平均难度级别（标量），供 CurriculumManager 使用
    return torch.mean(env._posture_curriculum_level[env_ids].float())


def _update_reset_parameters(env: ManagerBasedRLEnv, env_id: int, level: int):
    """
    Update reset parameters based on curriculum level.

    Args:
        env: ManagerBasedRLEnv instance
        env_id: Environment ID to update
        level: Curriculum level (0-3)
    """
    # Get level configuration
    level_config = POSTURE_CURRICULUM_LEVELS.get(level)
    if level_config is None:
        print(f"[Posture Curriculum] Warning: Invalid level {level}")
        return

    pose_range = level_config["pose_range"]
    velocity_range = level_config["velocity_range"]

    # Get reset event
    reset_event = env.cfg.events.randomize_reset_base
    if reset_event is None:
        print(f"[Posture Curriculum] Warning: reset_event not found")
        return

    # Update pose range
    if hasattr(reset_event, 'params'):
        reset_event.params["pose_range"] = pose_range
        reset_event.params["velocity_range"] = velocity_range

        print(f"[Posture Curriculum] Env {env_id}: Updated to Level {level} - {level_config['name']}")
        print(f"  Focus: {level_config['focus']}")
        print(f"  New pose range: roll={pose_range['roll']}, pitch={pose_range['pitch']}, z={pose_range['z']}")
        print(f"  Threshold: {level_config['success_threshold']*100:.0f}%, Min episodes: {level_config['min_episodes']}")
    else:
        print(f"[Posture Curriculum] Warning: reset_event.params not accessible")
