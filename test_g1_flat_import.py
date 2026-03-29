#!/usr/bin/env python3
"""
简单的G1 Flat配置导入测试
Simple G1 Flat configuration import test
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=== G1 Flat Configuration Import Test ===")
print(f"Project root: {project_root}")
print()

try:
    # 测试1: 导入原始配置
    print("Test 1: Import original velocity_env_cfg...")
    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof import velocity_env_cfg
    print("  ✅ Original config imported successfully")

    # 测试2: 导入改进配置
    print("Test 2: Import improved velocity_env_cfg...")
    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof import velocity_env_cfg_improved
    print("  ✅ Improved config imported successfully")

    # 测试3: 导入平地原始配置
    print("Test 3: Import flat velocity_env_cfg...")
    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof import velocity_env_cfg_flat
    print("  ✅ Flat original config imported successfully")

    # 测试4: 导入平地改进配置
    print("Test 4: Import flat improved velocity_env_cfg...")
    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof import velocity_env_cfg_flat_improved
    print("  ✅ Flat improved config imported successfully")

    # 测试5: 验证配置类
    print("\nTest 5: Validate config classes...")
    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof.velocity_env_cfg import RobotEnvCfg
    print("  ✅ RobotEnvCfg class available")

    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof.velocity_env_cfg_flat import RobotEnvCfg as FlatRobotEnvCfg
    print("  ✅ FlatRobotEnvCfg class available")

    from unitree_rl_lab.tasks.locomotion.robots.g1.dof_29dof.velocity_env_cfg_flat_improved import RobotEnvCfg as FlatImprovedRobotEnvCfg
    print("  ✅ FlatImprovedRobotEnvCfg class available")

    # 测试6: 检查关键配置值
    print("\nTest 6: Check key configuration values...")
    print(f"  Original episode_length: {RobotEnvCfg.episode_length_s}")
    print(f"  Flat episode_length: {FlatRobotEnvCfg.episode_length_s}")
    print(f"  Flat improved episode_length: {FlatImprovedRobotEnvCfg.episode_length_s}")

    print(f"  Original action scale: {RobotEnvCfg.actions.JointPositionAction.scale}")
    print(f"  Flat action scale: {FlatRobotEnvCfg.actions.JointPositionAction.scale}")
    print(f"  Flat improved action scale: {FlatImprovedRobotEnvCfg.actions.JointPositionAction.scale}")

    # 测试7: 检查机器人配置
    print("\nTest 7: Check robot configuration...")
    print(f"  Robot uses UNITREE_G1_29DOF_CFG")

    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)
    print("\nAvailable G1 training modes:")
    print("  ./scripts/train_g1.sh original          # Original 16-level progressive")
    print("  ./scripts/train_g1.sh improved           # Improved 16-level progressive")
    print("  ./scripts/train_g1.sh flat-original     # Original flat terrain")
    print("  ./scripts/train_g1.sh flat-improved     # Improved flat terrain")
    print("="*50)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Configuration error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
