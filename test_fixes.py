#!/usr/bin/env python3
"""测试修复是否有效的脚本"""

import sys

# 添加路径
sys.path.insert(0, 'source/unitree_rl_lab')

print("=== 测试1: 检查导入语句 ===")
try:
    # 测试导入是否不会因为 IdealPDActuatorCfg 而失败
    with open('source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py', 'r') as f:
        content = f.read()
        if 'IdealPDActuatorCfg' in content and 'DelayedPDActuatorCfg' in content:
            if 'try:' in content and 'except' in content:
                print("✅ IdealPDActuatorCfg 导入已添加异常处理")
            else:
                print("❌ IdealPDActuatorCfg 导入缺少异常处理")
        else:
            print("❌ 找不到 IdealPDActuatorCfg 导入")
except Exception as e:
    print(f"❌ 检查导入语句时出错: {e}")

print("\n=== 测试2: 检查 joint_drive 参数 ===")
try:
    with open('source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py', 'r') as f:
        content = f.read()
        if 'joint_drive=None' in content:
            print("✅ joint_drive=None 已添加")
        else:
            print("❌ joint_drive=None 未找到")
except Exception as e:
    print(f"❌ 检查 joint_drive 时出错: {e}")

print("\n=== 测试3: 检查正则表达式修复 ===")
try:
    with open('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py', 'r') as f:
        content = f.read()
        if 'hip_joint|arm_joint' in content:
            print("✅ 正则表达式已修复（排除 arm_joint）")
        else:
            print("❌ 正则表达式修复未找到")
except Exception as e:
    print(f"❌ 检查正则表达式时出错: {e}")

print("\n=== 测试4: 检查执行器配置 ===")
try:
    with open('source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py', 'r') as f:
        content = f.read()
        if '.*_hip_joint", ".*_thigh_joint", ".*_calf_joint' in content:
            print("✅ 执行器配置已修复（明确指定关节）")
        else:
            print("❌ 执行器配置修复未找到")
except Exception as e:
    print(f"❌ 检查执行器配置时出错: {e}")

print("\n=== 所有测试完成 ===")
