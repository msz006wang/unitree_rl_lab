from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
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
