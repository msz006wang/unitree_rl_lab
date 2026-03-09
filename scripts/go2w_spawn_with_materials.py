#!/usr/bin/env python3
"""Launch IsaacLab Sim and spawn Go2W robot with materials.

This script demonstrates how to apply Omniverse PBR materials to Go2W robot
after spawning from URDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Add the project source to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Spawn Go2W robot with materials in IsaacLab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn")
parser.add_argument("--use-mdl", action="store_true", help="Use MDL materials instead of PreviewSurface")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import torch
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom

try:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2W_CFG
except ImportError as e:
    print(f"Error importing UNITREE_GO2W_CFG: {e}")
    print("Please ensure the unitree_rl_lab package is properly installed.")
    simulation_app.close()
    sys.exit(1)


def apply_go2w_materials(
    robot_prim_path: str,
    use_mdl: bool = False,
):
    """Apply materials to Go2W robot.

    Args:
        robot_prim_path: Prim path of the robot
        use_mdl: If True, use MDL materials from NVIDIA Nucleus.
                 If False, use PreviewSurface materials.
    """
    print("=" * 80)
    print("Applying Materials to Go2W Robot")
    print("=" * 80)

    # Get current stage
    from pxr import Usd
    stage = sim_utils.get_current_stage()

    # Define materials based on type
    if use_mdl:
        from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

        print(f"Using MDL materials from: {NVIDIA_NUCLEUS_DIR}")
        print()

        materials_cfg = {
            "base": sim_utils.materials.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Plastics/Carbon_Fiber.mdl",
                project_uvw=True,
                albedo_brightness=0.8,
                texture_scale=(1.0, 1.0),
            ),
            "_hip": sim_utils.materials.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
                project_uvw=True,
                albedo_brightness=0.8,
            ),
            "thigh": sim_utils.materials.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless.mdl",
                project_uvw=True,
                albedo_brightness=0.9,
            ),
            "calf": sim_utils.materials.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless.mdl",
                project_uvw=True,
                albedo_brightness=0.9,
            ),
            "foot": sim_utils.materials.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Plastics/Rubber.mdl",
                project_uvw=True,
                albedo_brightness=1.0,
            ),
        }
    else:
        print("Using PreviewSurface materials (PBR)")
        print()

        materials_cfg = {
            "base": sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18),
                roughness=0.7,
                metallic=0.0,
            ),
            "_hip": sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
                roughness=0.3,
                metallic=0.8,
            ),
            "thigh": sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.6),
                roughness=0.4,
                metallic=0.7,
            ),
            "calf": sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55),
                roughness=0.4,
                metallic=0.7,
            ),
            "foot": sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1),
                roughness=0.8,
                metallic=0.0,
            ),
        }

    # Create and bind materials
    from isaaclab.sim.utils import bind_visual_material

    applied_count = 0
    print()
    print("Binding materials to robot parts...")
    print()

    # Collect all prim paths first, then apply materials
    # This avoids modifying the stage while iterating
    prim_paths_to_bind = []

    for part_name, material_cfg in materials_cfg.items():
        # Create material path
        material_path = f"/Looks/{part_name}_material"

        # Create material
        try:
            material_cfg.func(material_path, material_cfg)
            print(f"✓ Created material: {part_name}")
        except Exception as e:
            print(f"✗ Failed to create material for {part_name}: {e}")
            continue

        # Collect prims to bind (don't bind yet)
        for prim in stage.Traverse():
            if not prim.GetPath().pathString.startswith(robot_prim_path):
                continue

            # Check if prim path contains part name
            if part_name in prim.GetPath().pathString:
                # Only bind to geometry prims
                if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Imageable):
                    prim_paths_to_bind.append((str(prim.GetPath()), material_path))

    # Now bind materials in a separate pass
    # This reduces the number of stage modifications
    for prim_path, material_path in prim_paths_to_bind:
        try:
            bind_visual_material(
                prim_path=prim_path,
                material_path=material_path,
            )
            applied_count += 1
        except Exception as e:
            # Some prims may not support material binding
            pass

    print()
    print(f"✓ Applied materials to {applied_count} prims")
    print()


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
    """Main function to spawn and visualize Go2W robot with materials."""
    # Get number of environments from args
    num_envs = args_cli.num_envs

    # Initialize simulation
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
    robot_prim_paths = []
    for i in range(num_envs):
        # Set offset for each robot
        if i > 0:
            x_offset = i * 2.0
            UNITREE_GO2W_CFG.init_state.pos = (x_offset, 0.0, 0.4)
            UNITREE_GO2W_CFG.prim_path = f"/World/Robot_{i}"

        robot = spawn_go2w_robot(sim)
        robots.append(robot)
        robot_prim_paths.append(UNITREE_GO2W_CFG.prim_path)

    # IMPORTANT: Read robot information from USD BEFORE any simulation steps
    # This avoids the physics view invalidation issue entirely
    robot_info = {}
    if len(robots) > 0:
        print()
        print("Configuration Details:")
        print(f"  - Robot Type: Go2W (Wheel-Legged Robot)")
        print(f"  - Prim Path: {UNITREE_GO2W_CFG.prim_path}")
        print(f"  - Initial Position: {UNITREE_GO2W_CFG.init_state.pos}")
        print(f"  - Actuator Groups: {list(UNITREE_GO2W_CFG.actuators.keys())}")
        print(f"  - URDF Path: {UNITREE_GO2W_CFG.spawn.asset_path}")

        # Read robot info directly from USD - no physics view needed
        print(f"  - Reading robot structure from USD...")
        try:
            from pxr import UsdPhysics
            prim = robots[0].stage.GetPrimAtPath(UNITREE_GO2W_CFG.prim_path)
            if prim.IsValid():
                # Collect all joints
                joints = []
                for child in prim.GetAllChildren():
                    if child.IsA(UsdPhysics.Joint):
                        joints.append(child.GetName())

                # Collect all bodies (non-joint prims)
                bodies = []
                for child in prim.GetAllChildren():
                    if not child.IsA(UsdPhysics.Joint):
                        bodies.append(child.GetName())

                robot_info['joint_names'] = joints
                robot_info['num_joints'] = len(joints)
                robot_info['num_bodies'] = len(bodies)

                print(f"  - Number of joints: {robot_info['num_joints']}")
                print(f"  - Number of bodies: {robot_info['num_bodies']}")
                print(f"  - Joint names: {robot_info['joint_names']}")
            else:
                print(f"  - ⚠️  Could not find robot prim at: {UNITREE_GO2W_CFG.prim_path}")
        except Exception as e:
            print(f"  - ✗ Failed to read from USD: {e}")
        print()

    # Now apply materials
    for robot_prim_path in robot_prim_paths:
        apply_go2w_materials(
            robot_prim_path=robot_prim_path,
            use_mdl=args_cli.use_mdl,
        )

    # Print material configuration
    if len(robots) > 0:
        print("Material Configuration:")
        print(f"  - Material Type: {'MDL (High Quality)' if args_cli.use_mdl else 'PreviewSurface (PBR)'}")
        if args_cli.use_mdl:
            from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
            print(f"  - MDL Source: {NVIDIA_NUCLEUS_DIR}")
        print()

    # Now reset and step the simulation for the actual physics
    print("Preparing physics simulation...")
    sim.reset()
    for _ in range(5):
        sim.step()
    print("✓ Physics simulation ready")
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
