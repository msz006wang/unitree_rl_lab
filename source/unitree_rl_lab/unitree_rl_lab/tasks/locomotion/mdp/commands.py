"""
两段式恢复专用命令函数
"""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PhaseCommandCfg:
    """Phase command configuration for two-stage recovery."""

    def __init__(
        self,
        asset_name: str,
        resampling_time_range: tuple[float, float] = (10.0, 10.0),
        ranges: 'Ranges' = None
    ):
        self.asset_name = asset_name
        self.resampling_time_range = resampling_time_range
        self.ranges = ranges or Ranges()

    class Ranges:
        def __init__(
            self,
            phase: tuple[float, float] = (0.0, 2.0)
        ):
            self.phase = phase


def phase_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: dict,
    ranges: dict
) -> torch.Tensor:
    """
    阶段命令生成器 - 生成当前应该进入的阶段

    这个命令会根据当前环境状态生成建议的阶段：
    - 0: 趴伏状态
    - 1: 侧卧状态
    - 2: 站立状态

    Args:
        env: 强化学习环境
        command_name: 命令名称
        asset_cfg: 资产配置
        ranges: 命令范围

    Returns:
        阶段命令张量
    """
    # 生成一个固定的阶段命令，用于指导训练
    # 这里可以根据实际需求实现更复杂的逻辑

    # 初始化命令缓冲区
    if not hasattr(env, f"_{command_name}_command"):
        env.__dict__[f"_{command_name}_command"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )

    # 递增阶段（简单的线性递增）
    if not hasattr(env, f"_{command_name}_time"):
        env.__dict__[f"_{command_name}_time"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )

    env.__dict__[f"_{command_name}_time"] += env.step_dt

    # 每5秒切换一个阶段
    phase_duration = 5.0
    current_phase = (env.__dict__[f"_{command_name}_time"] / phase_duration).floor() % 3

    # 确保在指定范围内
    min_phase, max_phase = ranges.get('phase', (0.0, 2.0))
    current_phase = torch.clamp(current_phase, min_phase, max_phase)

    env.__dict__[f"_{command_name}_command"] = current_phase

    return env.__dict__[f"_{command_name}_command"]