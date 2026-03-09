#!/usr/bin/env python3
"""Test script to understand Articulation initialization timing."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project source to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

# Create argument parser
import argparse
parser = argparse.ArgumentParser(description="Test Articulation initialization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

try:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2W_CFG
except ImportError as e:
    print(f"Error importing UNITREE_GO2W_CFG: {e}")
    simulation_app.close()
    sys.exit(1)


def main():
    """Test Articulation initialization."""

    # Initialize simulation
    sim = SimulationContext()

    # Setup scene
    from isaaclab.sim import spawn_ground_plane, GroundPlaneCfg
    spawn_ground_plane(prim_path="/World/defaultGroundPlane", cfg=GroundPlaneCfg())

    # Add lights
    from pxr import UsdLux
    distant_light = UsdLux.DistantLight.Define(sim.stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(1000)

    print("=" * 80)
    print("Test 1: Before sim.reset()")
    print("=" * 80)

    # Spawn robot
    sim_utils.spawn_from_urdf("/World/Robot", UNITREE_GO2W_CFG.spawn)
    robot = Articulation(UNITREE_GO2W_CFG)

    print(f"Robot created: {robot}")
    print(f"Robot is_initialized: {robot.is_initialized}")
    print(f"Robot root_physx_view: {robot.root_physx_view}")

    print("\n" + "=" * 80)
    print("Test 2: After sim.reset()")
    print("=" * 80)

    sim.reset()

    print(f"Robot is_initialized: {robot.is_initialized}")
    print(f"Robot root_physx_view: {robot.root_physx_view}")

    if robot.is_initialized and robot.root_physx_view is not None:
        try:
            print(f"Robot num_joints: {robot.num_joints}")
            print(f"Robot num_bodies: {robot.num_bodies}")
            print("✓ Successfully accessed robot properties!")
        except Exception as e:
            print(f"✗ Error accessing robot properties: {e}")
    else:
        print("✗ Robot not initialized yet")

    print("\n" + "=" * 80)
    print("Test 3: After sim.step()")
    print("=" * 80)

    sim.step()

    print(f"Robot is_initialized: {robot.is_initialized}")
    print(f"Robot root_physx_view: {robot.root_physx_view}")

    if robot.is_initialized and robot.root_physx_view is not None:
        try:
            print(f"Robot num_joints: {robot.num_joints}")
            print(f"Robot num_bodies: {robot.num_bodies}")
            print("✓ Successfully accessed robot properties!")
        except Exception as e:
            print(f"✗ Error accessing robot properties: {e}")
    else:
        print("✗ Robot still not initialized")

    print("\n" + "=" * 80)
    print("Test complete. Press Ctrl+C to exit.")
    print("=" * 80)

    # Keep simulation running
    try:
        while simulation_app.is_running():
            sim.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
