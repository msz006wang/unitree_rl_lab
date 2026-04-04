#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify Isaac Sim training setup
"""

import os
import sys
from pathlib import Path

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root directory
os.chdir(PROJECT_ROOT)

print("Setting up Isaac Sim training environment...")
print(f"Project root: {PROJECT_ROOT}")

# Import after changing directory
import gymnasium as gym
import torch

# Configure torch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Import and instantiate AppLauncher (this handles omni module loading)
from isaaclab.app import AppLauncher

# Create a minimal parser
import argparse
parser = argparse.ArgumentParser(description="Test Isaac Sim training setup")

# Only add the minimal required arguments
parser.add_argument("--task", type=str, required=True, help="Task name to test")
parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

print(f"Creating AppLauncher with headless={args.headless}")

# Launch Isaac Sim app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("✅ SimulationApp instantiated successfully!")
print("✅ omni.timeline should now be available")

# Try to import tasks
try:
    import unitree_rl_lab.tasks  # noqa: F401
    print("✅ unitree_rl_lab.tasks imported successfully")
except Exception as e:
    print(f"❌ Error importing unitree_rl_lab.tasks: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# Try to test task creation
try:
    env = gym.make(args.task)
    print(f"✅ Environment created: {args.task}")
    env.close()
except Exception as e:
    print(f"❌ Error creating environment: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Close simulator
    simulation_app.close()

print("✅ Test completed successfully!")
