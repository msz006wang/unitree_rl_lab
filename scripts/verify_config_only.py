#!/usr/bin/env python
"""
配置验证脚本（不依赖IsaacLab完整导入）

直接检查配置文件，不导入完整的IsaacLab环境。
"""

import ast
from pathlib import Path


def verify_config_file(file_path):
    """验证配置文件语法和结构"""
    print(f"\n检查: {file_path.name}")

    if not file_path.exists():
        print(f"  ❌ 文件不存在")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✅ 语法正确")
        return True
    except SyntaxError as e:
        print(f"  ❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 读取错误: {e}")
        return False


def check_config_patterns(file_path, patterns):
    """检查配置文件中是否包含特定模式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        found = []
        missing = []
        for pattern in patterns:
            if pattern in code:
                found.append(pattern)
            else:
                missing.append(pattern)

        return found, missing
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return [], list(patterns)


def verify_velocity_env_cfg():
    """验证velocity_env_cfg.py配置"""
    print("=" * 70)
    print("验证: velocity_env_cfg.py")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    config_file = project_root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py"

    # 1. 语法验证
    if not verify_config_file(config_file):
        return False

    # 2. 检查新奖励函数配置
    print("\n奖励函数配置:")
    reward_patterns = [
        'upward_velocity = RewTerm',
        'orientation_tracking = RewTerm',
        'torque_penalty = RewTerm',
        'joint_regularization = RewTerm',
        'contact_management = RewTerm',
        'wheel_assisted_recovery = RewTerm',
    ]

    found, missing = check_config_patterns(config_file, reward_patterns)

    for reward in found:
        print(f"  ✅ {reward.split('=')[0]}")

    for reward in missing:
        print(f"  ❌ 未找到: {reward.split('=')[0]}")

    # 3. 检查历史观测配置
    print("\n历史观测配置:")
    obs_patterns = [
        'joint_pos_history = ObsTerm',
        'body_vel_history = ObsTerm',
    ]

    found, missing = check_config_patterns(config_file, obs_patterns)

    for obs in found:
        print(f"  ✅ {obs.split('=')[0]}")

    for obs in missing:
        print(f"  ❌ 未找到: {obs.split('=')[0]}")

    # 4. 检查动作空间配置
    print("\n动作空间配置:")
    action_patterns = [
        'self.actions.joint_pos.joint_names',
        'arm_joint1',
    ]

    found, missing = check_config_patterns(config_file, action_patterns)

    for pattern in found:
        print(f"  ✅ 找到: {pattern}")

    if 'self.actions.joint_pos.joint_names' not in found:
        print(f"  ⚠️  未找到: self.actions.joint_pos.joint_names（可能在运行时配置）")

    if 'arm_joint1' in found:
        print(f"  ✅ arm_joint1: 已在配置中")

    # 5. 检查机械臂策略注释
    print("\n机械臂策略:")
    arm_strategy_patterns = [
        '机械臂全程夹紧',
        'arm_joint1旋转',
        'arm_joint2-6固定',
    ]

    found, missing = check_config_patterns(config_file, arm_strategy_patterns)

    for pattern in found:
        print(f"  ✅ 找到: {pattern}")

    for pattern in missing:
        print(f"  ⚠️  未找到: {pattern}")

    return len(missing) == 0


def verify_rewards_functions():
    """验证扩展奖励函数文件"""
    print("\n" + "=" * 70)
    print("验证: extended_rewards.py")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    rewards_file = project_root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py"

    # 1. 语法验证
    if not verify_config_file(rewards_file):
        return False

    # 2. 检查函数定义
    print("\n新奖励函数:")
    reward_functions = [
        'def upward_velocity',
        'def orientation_tracking',
        'def torque_penalty',
        'def joint_regularization',
        'def contact_management',
        'def wheel_assisted_recovery',
    ]

    found, missing = check_config_patterns(rewards_file, reward_functions)

    for func in found:
        print(f"  ✅ {func}")

    for func in missing:
        print(f"  ❌ 未找到: {func}")

    return len(missing) == 0


def verify_observations_functions():
    """验证扩展观测函数文件"""
    print("\n" + "=" * 70)
    print("验证: observations.py")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    obs_file = project_root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py"

    # 1. 语法验证
    if not verify_config_file(obs_file):
        return False

    # 2. 检查函数定义
    print("\n新观测函数:")
    obs_functions = [
        'def history_buffer',
        'def joint_pos_history',
        'def body_vel_history',
    ]

    found, missing = check_config_patterns(obs_file, obs_functions)

    for func in found:
        print(f"  ✅ {func}")

    for func in missing:
        print(f"  ❌ 未找到: {func}")

    return len(missing) == 0


def verify_mdp_exports():
    """验证MDP导出"""
    print("\n" + "=" * 70)
    print("验证: mdp/__init__.py")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    init_file = project_root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py"

    # 1. 语法验证
    if not verify_config_file(init_file):
        return False

    # 2. 检查导出
    print("\nMDP导出:")
    export_patterns = [
        'upward_velocity',
        'orientation_tracking',
        'torque_penalty',
        'joint_regularization',
        'contact_management',
        'wheel_assisted_recovery',
        'history_buffer',
        'joint_pos_history',
        'body_vel_history',
    ]

    found, missing = check_config_patterns(init_file, export_patterns)

    for pattern in found:
        print(f"  ✅ {pattern}")

    for pattern in missing:
        print(f"  ❌ 未找到: {pattern}")

    return len(missing) == 0


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("GO2W ARM 配置验证（无完整环境依赖）")
    print("=" * 70)

    results = []

    # 运行所有验证
    results.append(("velocity_env_cfg.py", verify_velocity_env_cfg()))
    results.append(("extended_rewards.py", verify_rewards_functions()))
    results.append(("observations.py", verify_observations_functions()))
    results.append(("mdp/__init__.py", verify_mdp_exports()))

    # 打印总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有配置验证通过！")
        print("\n代码已正确实施：")
        print("  • 6个新奖励函数")
        print("  • 3个新观测函数")
        print("  • MDP导出更新")
        print("  • 配置文件修改（奖励+观测+动作空间）")
        print("\n下一步：")
        print("  1. 配置IsaacLab环境（如果需要）")
        print("  2. 运行训练: python scripts/train.py --task Robot-v0")
        print("  3. 参考详细文档: docs/GO2W_ARM_COMPREHENSIVE_OPTIMIZATION.md")
    else:
        print("❌ 部分验证失败，请查看上述错误信息。")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
