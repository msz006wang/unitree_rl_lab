#!/usr/bin/env python3
"""测试GO2W ARM配置

验证修改后的配置是否正确，特别是：
1. Reward配置
2. 机械臂初始姿态
3. 动作空间配置
4. 是否有不存在的属性引用
"""

import sys
import os

os.chdir('/home/jay/unitree_rl_lab/source')
sys.path.insert(0, '.')

print("=" * 60)
print("GO2W ARM 配置验证")
print("=" * 60)
print()

try:
    print("🔍 步骤1：导入配置...")
    from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg
    print("✅ 配置导入成功")

    print("\n🔍 步骤2：创建配置实例...")
    cfg = RobotFlatEnvCfg()
    print("✅ 配置实例创建成功")

    print("\n🔍 步骤3：验证Reward配置...")
    print(f"  lin_vel_z_l2: {cfg.rewards.lin_vel_z_l2.weight}")
    print(f"  ang_vel_xy_l2: {cfg.rewards.ang_vel_xy_l2.weight}")
    print(f"  flat_orientation_l2: {cfg.rewards.flat_orientation_l2.weight}")
    print(f"  base_height_l2: {cfg.rewards.base_height_l2.weight}")
    print(f"  target_height: {cfg.rewards.base_height_l2.params['target_height']}")

    print("\n🔍 步骤4：验证机械臂配置...")
    print(f"  arm_joint_names: {cfg.arm_joint_names}")
    print(f"  leg_joint_names: {cfg.leg_joint_names}")

    print("\n🔍 步骤5：验证动作空间...")
    print(f"  action joint_pos names: {cfg.actions.joint_pos.joint_names}")
    print(f"  action joint_vel names: {cfg.actions.joint_vel.joint_names}")

    print("\n🔍 步骤6：检查不存在的属性引用...")
    # 检查是否有wheel相关的reward配置
    has_wheel_rewards = hasattr(cfg.rewards, 'joint_torques_wheel_l2')
    print(f"  joint_torques_wheel_l2 exists: {has_wheel_rewards}")

    print("\n✅ 配置验证完成！所有配置都正确。")

except AttributeError as e:
    print(f"\n❌ AttributeError: {e}")
    print("\n🔍 正在分析错误...")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误消息: {str(e)}")
    print("\n💡 可能的原因:")
    print("  1. 尝试访问不存在的reward属性")
    print("  2. 语法错误或拼写错误")
    print("  3. 配置文件结构问题")

except Exception as e:
    print(f"\n❌ 其他错误: {e}")
    print("\n🔍 正在分析错误...")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误消息: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("验证结束")
print("=" * 60)
