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
    reset_event = env.events.get_term("randomize_reset_base")
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
