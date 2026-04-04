#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified training script for GO2W ARM two-stage recovery
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import after changing directory
import gymnasium as gym
import torch

# Configure torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Import AppLauncher (this handles omni module loading automatically)
from isaaclab.app import AppLauncher

# Create minimal parser - only add custom arguments, not AppLauncher arguments
parser = argparse.ArgumentParser(description="Train GO2W ARM two-stage recovery")

# Custom arguments (not provided by AppLauncher)
parser.add_argument("--task", type=str, required=True, help="Task name to train")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Add AppLauncher arguments (includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# Now we can use args.task, args.num_envs, args.max_iterations, args.seed
# and args.headless, args.device from AppLauncher

# CRITICAL: Launch Isaac Sim app BEFORE importing tasks
# This ensures SimulationApp is instantiated before any Omniverse imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import tasks AFTER SimulationApp is instantiated
import unitree_rl_lab.tasks  # noqa: F401

# Import training utilities
import time
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import CLI args utility
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
import cli_args
sys.path.pop(0)

# Import local utilities
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

# Print configuration
print("=" * 60)
print(f"GO2W ARM Two-Stage Recovery Training")
print("=" * 60)
print(f"Task:              {args.task}")
print(f"Environments:      {args.num_envs}")
print(f"Device:            {args.device}")
print(f"Max Iterations:    {args.max_iterations}")
print(f"Seed:              {args.seed}")
print("=" * 60)
print()

# Launch Isaac Sim app (SimulationApp instantiated here)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    # Clear sys.argv for Hydra
    sys.argv = [sys.argv[0]]

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
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
        env = gym.make(args.task, cfg=env_cfg)

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
            resume_path = get_checkpoint_path(
                log_root_path,
                args.load_run if args.load_run else agent_cfg.load_run,
                args.load_checkpoint if args.load_checkpoint else agent_cfg.load_checkpoint
            )
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
