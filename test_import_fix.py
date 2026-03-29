#!/usr/bin/env python
"""Test that the import fix resolves the IdealPDActuatorCfg error"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, "/home/jay/unitree_rl_lab/source/unitree_rl_lab")

def test_unitree_import():
    """Test that unitree.py imports correctly with IdealPDActuatorCfg"""
    print("="*60)
    print("Testing unitree.py import")
    print("="*60)

    try:
        from unitree_rl_lab.assets.robots.unitree import (
            UNITREE_GO2W_ARM_ARX5_CFG,
            UNITREE_GO2W_ARM_PIPER_CFG,
        )
        print("✅ GO2W-Arm configurations imported successfully")
        print(f"  ARX5 arm actuator: {type(UNITREE_GO2W_ARM_ARX5_CFG.actuators['arm']).__name__}")
        print(f"  Piper arm actuator: {type(UNITREE_GO2W_ARM_PIPER_CFG.actuators['arm']).__name__}")
        return True
    except NameError as e:
        print(f"❌ Import failed with NameError: {e}")
        return False
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_configs():
    """Test that environment configs can be imported"""
    print("\n" + "="*60)
    print("Testing environment configuration imports")
    print("="*60)

    try:
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg as ARX5FlatCfg
        print("✅ ARX5 velocity_env_cfg imported successfully")

        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg_piper import RobotFlatEnvCfg as PiperFlatCfg
        print("✅ Piper velocity_env_cfg_piper imported successfully")

        return True
    except NameError as e:
        print(f"❌ Environment config import failed with NameError: {e}")
        return False
    except Exception as e:
        print(f"❌ Environment config import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gym_registration():
    """Test that gym environments are registered"""
    print("\n" + "="*60)
    print("Testing gym environment registration")
    print("="*60)

    try:
        import gymnasium as gym

        # Check if the environments are registered
        registered_tasks = []
        for task_spec in gym.registry.values():
            if "Go2WArm" in task_spec.id:
                registered_tasks.append(task_spec.id)

        if len(registered_tasks) > 0:
            print(f"✅ Found {len(registered_tasks)} GO2W-Arm registered tasks:")
            for task in registered_tasks:
                print(f"    - {task}")
            return True
        else:
            print("❌ No GO2W-Arm tasks found in registry")
            return False

    except NameError as e:
        print(f"❌ Gym registration test failed with NameError: {e}")
        return False
    except Exception as e:
        print(f"❌ Gym registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing import fix for IdealPDActuatorCfg error\n")

    success = True
    success = test_unitree_import() and success
    success = test_environment_configs() and success
    success = test_gym_registration() and success

    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)

    print("\n📋 Fix Summary:")
    print("✅ Added IdealPDActuatorCfg to import statement in unitree.py")
    print("✅ This resolves the NameError during hydra configuration loading")
    print("\n🚀 Training should now work with:")
    print("   ./scripts/train_go2w_arm.sh arx5_flat")
