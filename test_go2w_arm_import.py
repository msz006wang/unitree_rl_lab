#!/usr/bin/env python
"""测试GO2W-Arm配置导入"""

import gym
from isaaclab.envs import ManagerBasedRLEnv
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_GO2W_ARM_ARX5_CFG,
    UNITREE_GO2W_ARM_PIPER_CFG
)

def test_robot_configs():
    """测试机器人配置"""
    print("="*60)
    print("测试GO2W-Arm机器人配置")
    print("="*60)

    # 测试ARX5配置
    print("\n🤖 测试ARX5配置:")
    print(f"  Robot Path: {UNITREE_GO2W_ARM_ARX5_CFG.spawn.asset_path}")
    print(f"  Prim Path: {UNITREE_GO2W_ARM_ARX5_CFG.prim_path}")
    print(f"  Actuator Groups: {list(UNITREE_GO2W_ARM_ARX5_CFG.actuators.keys())}")
    for name, cfg in UNITREE_GO2W_ARM_ARX5_CFG.actuators.items():
        print(f"    {name}: {cfg.joint_names_expr}")

    # 测试Piper配置
    print("\n🤖 测试Piper配置:")
    print(f"  Robot Path: {UNITREE_GO2W_ARM_PIPER_CFG.spawn.asset_path}")
    print(f"  Prim Path: {UNITREE_GO2W_ARM_PIPER_CFG.prim_path}")
    print(f"  Actuator Groups: {list(UNITREE_GO2W_ARM_PIPER_CFG.actuators.keys())}")
    for name, cfg in UNITREE_GO2W_ARM_PIPER_CFG.actuators.items():
        print(f"    {name}: {cfg.joint_names_expr}")

def test_env_registrations():
    """测试环境注册"""
    print("\n" + "="*60)
    print("测试GO2W-Arm环境注册")
    print("="*60)

    # 检查环境是否注册成功
    env_ids = [
        "Unitree-Go2WArm-Velocity-Flat-v0",
        "Unitree-Go2WArm-Velocity-Rough-v0",
        "Unitree-Go2WArm-Velocity",
    ]

    for env_id in env_ids:
        if env_id in gym.envs.registry:
            print(f"✅ {env_id} - 已注册")
            spec = gym.envs.registry[env_id]
            print(f"   Entry Point: {spec.entry_point}")
        else:
            print(f"❌ {env_id} - 未注册")

def test_import_velocity_env_cfg():
    """测试velocity_env_cfg导入"""
    print("\n" + "="*60)
    print("测试velocity_env_cfg导入")
    print("="*60)

    try:
        # 测试ARX5版本
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg as ARX5FlatCfg
        print("✅ ARX5 velocity_env_cfg导入成功")
        print(f"   Robot: {ARX5FlatCfg.scene.robot.spawn.asset_path}")
        print(f"   Joint names: {ARX5FlatCfg.joint_names}")

    except Exception as e:
        print(f"❌ ARX5 velocity_env_cfg导入失败: {e}")

    try:
        # 测试Piper版本
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg_piper import RobotFlatEnvCfg as PiperFlatCfg
        print("✅ Piper velocity_env_cfg导入成功")
        print(f"   Robot: {PiperFlatCfg.scene.robot.spawn.asset_path}")
        print(f"   Joint names: {PiperFlatCfg.joint_names}")

    except Exception as e:
        print(f"❌ Piper velocity_env_cfg导入失败: {e}")

if __name__ == "__main__":
    try:
        test_robot_configs()
        test_env_registrations()
        test_import_velocity_env_cfg()
        print("\n" + "="*60)
        print("🎉 所有测试完成！")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
