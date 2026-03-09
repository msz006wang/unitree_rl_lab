#!/usr/bin/env python3
"""Launch IsaacLab Sim and spawn Go2W robot.

This script launches IsaacLab and spawns the Go2W wheel-legged robot
using the UNITREE_GO2W_CFG configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Add the project source to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Spawn Go2W robot in IsaacLab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app to avoid import errors
import torch
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

try:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2W_CFG
except ImportError as e:
    print(f"Error importing UNITREE_GO2W_CFG: {e}")
    print("Please ensure the unitree_rl_lab package is properly installed.")
    simulation_app.close()
    sys.exit(1)


def setup_scene(sim: SimulationContext):
    """Setup the scene with ground plane, lights, and environment."""
    from isaaclab.sim import spawn_ground_plane, GroundPlaneCfg

    # Spawn ground plane
    spawn_ground_plane(
        prim_path="/World/defaultGroundPlane",
        cfg=GroundPlaneCfg(),
    )

    # Add lights
    from pxr import UsdLux
    stage = sim.stage

    # Add distant light (sun)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(1000)

    # Add dome light for ambient lighting
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(500)


def spawn_go2w_robot(sim: SimulationContext, offset=(0, 0, 0)) -> Articulation:
    """Spawn the Go2W robot in the scene."""
    print("=" * 80)
    print("Spawning Go2W Robot")
    print("=" * 80)

    # Import required modules
    import isaaclab.sim as sim_utils

    # Set prim path
    prim_path = "/World/Robot"

    # Spawn the robot from URDF using IsaacLab's spawner
    sim_utils.spawn_from_urdf(
        prim_path,
        UNITREE_GO2W_CFG.spawn,
    )

    print("✓ Go2W robot spawned successfully")
    print(f"  - Prim path: {prim_path}")
    print()

    # Create articulation object (will be initialized after sim.reset())
    robot = Articulation(UNITREE_GO2W_CFG)

    return robot


def main():
    """Main function to spawn and visualize Go2W robot."""
    # Get number of environments from args
    num_envs = args_cli.num_envs

    # Initialize simulation (using default parameters)
    sim = SimulationContext()

    # Setup scene
    print("=" * 80)
    print("Setting up IsaacLab Scene")
    print("=" * 80)
    setup_scene(sim)
    print("✓ Scene setup complete")
    print()

    # Reset simulation
    sim.reset()

    # Spawn Go2W robot(s)
    robots = []
    for i in range(num_envs):
        # Set offset for each robot
        if i > 0:
            # Offset additional robots
            x_offset = i * 2.0
            UNITREE_GO2W_CFG.init_state.pos = (x_offset, 0.0, 0.4)
            # Also update prim_path for additional robots
            UNITREE_GO2W_CFG.prim_path = f"/World/Robot_{i}"

        robot = spawn_go2w_robot(sim)
        robots.append(robot)

    # Reset simulation after spawning (this initializes the articulation)
    sim.reset()

    # Now we can access robot properties
    if len(robots) > 0:
        print("Configuration Details:")
        print(f"  - Robot Type: Go2W (Wheel-Legged Robot)")
        print(f"  - Prim Path: {UNITREE_GO2W_CFG.prim_path}")
        print(f"  - Initial Position: {UNITREE_GO2W_CFG.init_state.pos}")
        print(f"  - Actuator Groups: {list(UNITREE_GO2W_CFG.actuators.keys())}")
        print(f"  - URDF Path: {UNITREE_GO2W_CFG.spawn.asset_path}")
        print(f"  - Number of joints: {robots[0].num_joints}")
        print(f"  - Number of bodies: {robots[0].num_bodies}")
        print(f"  - Joint names: {robots[0].joint_names}")
        print()

    # Print control instructions
    print("=" * 80)
    print("Simulation Running")
    print("=" * 80)
    print()
    print("Control Instructions:")
    print("  - Mouse Left: Rotate view")
    print("  - Mouse Wheel: Zoom in/out")
    print("  - Mouse Middle: Pan")
    print("  - Press 'Ctrl+C' or close window to exit simulation")
    print()
    print("Robot Information:")
    print(f"  - Total robots spawned: {len(robots)}")
    print()
    print("Close the Isaac Sim window or press Ctrl+C to stop the simulation.")
    print()

    # Simulation loop
    frame_count = 0
    sim_dt = 1.0 / 60.0  # 60 Hz timestep
    try:
        while simulation_app.is_running():
            # Step simulation
            sim.step()

            # Update robot states
            for robot in robots:
                robot.write_data_to_sim()
                robot.update(sim_dt)

            # Print status every 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Simulation frame: {frame_count}", end="\r")

    except KeyboardInterrupt:
        print("\n")
        print("Simulation stopped by user")
    finally:
        # Cleanup
        print()
        print("=" * 80)
        print("Cleaning up...")
        print("=" * 80)
        simulation_app.close()
        print("✓ Simulation closed")


if __name__ == "__main__":
    main()
