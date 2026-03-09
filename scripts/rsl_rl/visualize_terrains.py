#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化G1机器人在16个难度等级地形上的训练脚本
Visual Training Script for G1 Robot on 16 Difficulty Level Terrains

这个脚本专门用于可视化G1机器人在不同难度等级地形上的训练表现。
This script is specifically designed to visualize the training performance of G1 robot on different difficulty level terrains.

使用方法 Usage:
    python scripts/rsl_rl/visualize_terrains.py --task Isaac-Velocity-v1 --num_envs 16 --video

特性 Features:
    - 可视化16个不同难度等级的地形
    - 实时显示机器人在不同地形上的表现
    - 支持视频录制
    - 地形难度渐进式展示
"""

import argparse
import gymnasium as gym
import os
import sys
import time
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, f"{os.path.dirname(__file__)}/../../")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Visualize G1 robot training on 16 terrain difficulty levels.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-v1",
                   help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16,
                   help="Number of environments to simulate (one for each difficulty level).")
parser.add_argument("--video", action="store_true", default=False,
                   help="Record videos during visualization.")
parser.add_argument("--video_length", type=int, default=500,
                   help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False,
                   help="Run in real-time, if possible.")
parser.add_argument("--headless", action="store_true", default=False,
                   help="Run in headless mode (no rendering).")
parser.add_argument("--use_pretrained_checkpoint", action="store_true",
                   help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--checkpoint", type=str, default=None,
                   help="Path to the checkpoint to load.")
parser.add_argument("--show_info", action="store_true", default=True,
                   help="Show terrain difficulty information.")
# AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launcher
import cli_args
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def print_terrain_info(env):
    """打印地形信息 Print terrain information"""
    if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'terrain'):
        terrain_cfg = env.unwrapped.scene.terrain.terrain_generator
        print("\n" + "="*80)
        print("地形配置信息 / Terrain Configuration Information")
        print("="*80)
        print(f"地形行数 (难度等级数) / Terrain Rows (Difficulty Levels): {terrain_cfg.num_rows}")
        print(f"地形列数 / Terrain Columns: {terrain_cfg.num_cols}")
        print(f"难度范围 / Difficulty Range: {terrain_cfg.difficulty_range}")
        print(f"课程学习 / Curriculum Learning: {terrain_cfg.curriculum}")
        print(f"\n子地形类型 / Sub-terrain Types:")
        for name, cfg in terrain_cfg.sub_terrains.items():
            print(f"  - {name}: {cfg.proportion*100:.1f}%")
        print("="*80 + "\n")


def main():
    """主函数 Main function"""
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Set specific number of environments for visualization
    env_cfg.scene.num_envs = args_cli.num_envs

    # Force 16 terrain levels for visualization
    if hasattr(env_cfg.scene.terrain, 'terrain_generator'):
        env_cfg.scene.terrain.terrain_generator.num_rows = min(16, args_cli.num_envs)
        env_cfg.scene.terrain.terrain_generator.num_cols = 21

    # Try to load agent configuration
    try:
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    except:
        agent_cfg = None

    # Load checkpoint if available
    resume_path = None
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[WARNING] No pre-trained checkpoint found. Running with random policy.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        try:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        except:
            print("[WARNING] No checkpoint found. Running with random policy.")

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Print terrain information
    if args_cli.show_info:
        print_terrain_info(env)

    # Wrap for video recording
    if args_cli.video:
        video_dir = os.path.join("logs", "videos", "terrain_visualization")
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for rsl-rl
    if agent_cfg:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy if checkpoint is available
    policy = None
    if resume_path and agent_cfg:
        try:
            from rsl_rl.runners import OnPolicyRunner
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print("[INFO] Successfully loaded trained policy.")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print("[INFO] Falling back to random policy.")
            policy = None
    elif agent_cfg:
        print("[INFO] No checkpoint specified. Running with random policy.")

    # Get environment time step
    dt = env.unwrapped.step_dt

    # Reset environment
    if hasattr(env, 'get_observations'):
        obs = env.get_observations()
    else:
        obs, _ = env.reset()

    # Statistics
    total_steps = 0
    start_time = time.time()

    print("\n" + "="*80)
    print("开始可视化训练 / Starting Visualization Training")
    print("="*80)
    print(f"环境数量 / Number of Environments: {args_cli.num_envs}")
    print(f"地形难度等级数 / Terrain Difficulty Levels: {min(16, args_cli.num_envs)}")
    print(f"每步时间 / Time Step: {dt:.4f}s")
    print(f"策略 / Policy: {'Trained' if policy else 'Random'}")
    print("="*80 + "\n")

    # Simulate environment
    try:
        while simulation_app.is_running():
            step_start = time.time()

            # Run everything in inference mode
            with torch.inference_mode():
                # Agent stepping
                if policy:
                    actions = policy(obs)
                else:
                    # Random actions
                    actions = torch.rand(env.unwrapped.num_envs, env.unwrapped.action_space.shape[0],
                                       device=env.unwrapped.device)

                # Env stepping
                if hasattr(env, 'step'):
                    obs, reward, done, info = env.step(actions)
                else:
                    obs, reward, done, truncated, info = env.step(actions)

            total_steps += 1

            # Print progress every 100 steps
            if total_steps % 100 == 0:
                elapsed = time.time() - start_time
                fps = total_steps / elapsed
                print(f"[INFO] Steps: {total_steps}, FPS: {fps:.1f}, "
                      f"Avg Reward: {reward.mean().item():.3f}")

            # Video recording exit
            if args_cli.video and total_steps >= args_cli.video_length:
                print("[INFO] Video recording completed.")
                break

            # Time delay for real-time evaluation
            if args_cli.real_time:
                sleep_time = dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user.")

    # Print final statistics
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("可视化完成 / Visualization Completed")
    print("="*80)
    print(f"总步数 / Total Steps: {total_steps}")
    print(f"总时间 / Total Time: {elapsed:.2f}s")
    print(f"平均FPS / Average FPS: {total_steps / elapsed:.2f}")
    print("="*80 + "\n")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
