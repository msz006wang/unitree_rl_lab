from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
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
