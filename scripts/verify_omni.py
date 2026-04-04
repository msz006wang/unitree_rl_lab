#!/usr/bin/env python3
"""
Minimal test to verify omni.timeline is available after SimulationApp instantiation
"""

import sys
import os
from pathlib import Path

# Add IsaacLab to path
sys.path.insert(0, "/home/jay/IsaacLab/source")
sys.path.insert(0, "/home/jay/unitree_rl_lab/source")

# Change to project root
os.chdir("/home/jay/unitree_rl_lab")

print("Testing Isaac Sim omni.timeline module...")

# Import AppLauncher and instantiate SimulationApp
from isaaclab.app import AppLauncher

# Create minimal args
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)

# Parse empty args (will use defaults)
args = parser.parse_args([])

# Instantiate SimulationApp
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("✅ SimulationApp instantiated")

# Now test importing omni.timeline
try:
    import omni.timeline
    print("✅ SUCCESS: omni.timeline module is available!")
    print(f"   Module location: {omni.timeline.__file__}")
except Exception as e:
    print(f"❌ FAILED: Cannot import omni.timeline: {e}")
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
