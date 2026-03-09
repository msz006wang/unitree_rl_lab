#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Unitree GO2W Wheel-Legged Robot

This script provides a comprehensive training interface for the GO2W robot
with support for both Flat and Rough terrain configurations.

Usage:
    python scripts/train_go2w.py --mode flat --num_envs 4096
    python scripts/train_go2w.py --mode rough --num_envs 8192
    python scripts/train_go2w.py --mode play-flat --resume

Author: Unitree RL Lab
Date: 2025
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root directory
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# Import tasks
import unitree_rl_lab.tasks  # noqa: F401

# Configure torch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GO2W wheel-legged robot on Flat or Rough terrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on flat terrain with default settings
  python train_go2w.py --mode flat

  # Train on rough terrain with custom settings
  python train_go2w.py --mode rough --num_envs 8192 --max_iterations 20000

  # Resume training from checkpoint
  python train_go2w.py --mode flat --resume --load_run recent

  # Play with trained policy
  python train_go2w.py --mode play-flat --resume

  # Train with video recording
  python train_go2w.py --mode flat --video --video_interval 2000

  # Train with GUI visualization
  python train_go2w.py --mode flat --gui
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["flat", "rough", "play-flat", "play-rough"],
        help="Training mode: flat (plane terrain), rough (generated terrain), or play mode"
    )

    # Environment parameters
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training parameters
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--load_run", type=str, default=None, help="Run name to load from (default: recent)")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint name to load")

    # Visualization
    parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--video", action="store_true", help="Record videos during training")
    parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval (steps)")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps)")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")

    # Multi-GPU
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")

    # AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Handle GUI vs headless
    if args.gui:
        args.headless = False

    return args


def main():
    """Main training function."""
    args = parse_args()

    # Determine task and configuration based on mode
    mode_map = {
        "flat": ("Unitree-Go2W-Velocity-Flat-v0", "Training on FLAT terrain"),
        "rough": ("Unitree-Go2W-Velocity-Rough-v0", "Training on ROUGH terrain"),
        "play-flat": ("Unitree-Go2W-Velocity-Flat-v0", "Playing on FLAT terrain"),
        "play-rough": ("Unitree-Go2W-Velocity-Rough-v0", "Playing on ROUGH terrain"),
    }

    if args.mode not in mode_map:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

    task_name, description = mode_map[args.mode]

    # Print configuration
    print("=" * 60)
    print(f"  GO2W Training - {description}")
    print("=" * 60)
    print(f"Task:              {task_name}")
    print(f"Environments:      {args.num_envs}")
    print(f"Device:            {args.device}")
    print(f"Max Iterations:    {args.max_iterations}")
    print(f"Seed:              {args.seed}")
    print(f"Video:             {args.video}")
    print(f"Headless:          {args.headless}")
    print(f"Resume:            {args.resume}")
    print(f"Distributed:       {args.distributed}")
    print("=" * 60)
    print()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import logging
        import time
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import get_checkpoint_path
        from isaaclab_tasks.utils.hydra import hydra_task_config

        # Import CLI args
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
        import cli_args

        # Import local utilities
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from list_envs import import_packages  # noqa: F401
        sys.path.pop(0)

        # Setup logging
        logger = logging.getLogger(__name__)

        # Clear sys.argv for Hydra
        sys.argv = [sys.argv[0]]

        @hydra_task_config(task_name, "rsl_rl_cfg_entry_point")
        def train_main(env_cfg, agent_cfg):
            """Main training function."""
            # Override configurations with CLI arguments
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
            env_cfg.scene.num_envs = args.num_envs
            agent_cfg.max_iterations = args.max_iterations

            # Set environment seed
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = args.device

            # Multi-GPU configuration
            if args.distributed:
                env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
                agent_cfg.device = f"cuda:{app_launcher.local_rank}"
                seed = agent_cfg.seed + app_launcher.local_rank
                env_cfg.seed = seed
                agent_cfg.seed = seed

            # Setup logging directory
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")

            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, log_dir)

            # Create environment
            env = gym.make(
                task_name,
                cfg=env_cfg,
                render_mode="rgb_array" if args.video else None
            )

            # Save resume path
            if args.resume:
                resume_path = get_checkpoint_path(
                    log_root_path,
                    args.load_run if args.load_run else agent_cfg.load_run,
                    args.load_checkpoint if args.load_checkpoint else agent_cfg.load_checkpoint
                )

            # Wrap for video recording
            if args.video:
                video_kwargs = {
                    "video_folder": os.path.join(log_dir, "videos", "train"),
                    "step_trigger": lambda step: step % args.video_interval == 0,
                    "video_length": args.video_length,
                    "disable_logger": True,
                }
                print("[INFO] Recording videos during training.")
                print_dict(video_kwargs, nesting=4)
                env = gym.wrappers.RecordVideo(env, **video_kwargs)

            start_time = time.time()

            # Wrap for RSL-RL
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            # Create runner
            runner = OnPolicyRunner(
                env,
                agent_cfg.to_dict(),
                log_dir=log_dir,
                device=agent_cfg.device
            )

            # Add git info
            runner.add_git_repo_to_log(__file__)

            # Load checkpoint
            if args.resume:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                runner.load(resume_path)

            # Save configurations
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

            # Run training
            print("[INFO] Starting training...")
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True
            )

            print(f"Training completed in {time.time() - start_time:.2f} seconds")

            # Close environment
            env.close()

        # Run training
        train_main()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Close simulator
        simulation_app.close()


if __name__ == "__main__":
    main()
