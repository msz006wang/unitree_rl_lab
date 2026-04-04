#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Training Script for GO2W ARM Two-Stage Recovery
解决了 omni.timeline 模块导入问题

Usage:
    python scripts/train_fixed.py --task Unitree-Go2WArm-TwoStage-Recovery-v0 --headless --num_envs 4096
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root directory
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import torch

# Configure torch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

print("Initializing Isaac Sim environment...")
print(f"Project root: {PROJECT_ROOT}")

# CRITICAL: Import AppLauncher and instantiate SimulationApp BEFORE importing any isaaclab modules
# This is required by Carbonite framework's extension/runtime plugin system
from isaaclab.app import AppLauncher

# Import CLI args utility BEFORE parsing arguments
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
import cli_args
sys.path.pop(0)

# Create minimal parser - only add custom arguments, let AppLauncher add its own arguments
parser = argparse.ArgumentParser(
    description="Train GO2W ARM with two-stage recovery",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Train with default settings
  python train_fixed.py --task Unitree-Go2WArm-TwoStage-Recovery-v0 --headless --num_envs 4096

  # Train with fewer environments for testing
  python train_fixed.py --task Unitree-Go2WArm-TwoStage-Recovery-v0 --headless --num_envs 64
        """
)

# Custom arguments (not provided by AppLauncher)
parser.add_argument("--task", type=str, required=True, help="Task name to train")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

# Add AppLauncher arguments (this includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

# Add RSL-RL arguments (resume, load_run, checkpoint, logger, etc.)
cli_args.add_rsl_rl_args(parser)

# Parse arguments
args = parser.parse_args()

# Print configuration
print("=" * 60)
print("GO2W ARM Two-Stage Recovery Training")
print("=" * 60)
print(f"Task:              {args.task}")
print(f"Environments:      {args.num_envs}")
print(f"Max Iterations:    {args.max_iterations}")
print(f"Seed:              {args.seed}")
print("=" * 60)
print()

# CRITICAL: Launch Isaac Sim app BEFORE importing tasks
# This ensures SimulationApp is instantiated before any Omniverse imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("✅ Isaac Sim SimulationApp instantiated successfully!")
print("✅ omni.timeline module is now available")

# Import tasks AFTER SimulationApp is instantiated
import unitree_rl_lab.tasks  # noqa: F401

# Import training utilities
import time
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import local utilities
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

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

        # Handle deprecated RSL-RL configurations
        import importlib.metadata as metadata
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

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
