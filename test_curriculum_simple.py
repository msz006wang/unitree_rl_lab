#!/usr/bin/env python3
"""
简单测试多级姿态恢复课程实现
直接验证修改的文件内容，不需要Isaac Sim环境
"""

import ast
import re
import sys
import os

def test_curriculums_file():
    """测试curriculums.py文件修改"""
    print("=" * 60)
    print("测试1: curriculums.py文件验证")
    print("=" * 60)

    file_path = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 检查POSTURE_CURRICULUM_LEVELS定义
        if "POSTURE_CURRICULUM_LEVELS" not in content:
            print("❌ 未找到POSTURE_CURRICULUM_LEVELS定义")
            return False
        print("✅ 找到POSTURE_CURRICULUM_LEVELS定义")

        # 检查4个级别
        level_count = content.count('"name":')
        if level_count < 4:
            print(f"❌ 级别数量不足，预期4个，实际找到{level_count}个")
            return False
        print(f"✅ 找到{level_count}个级别配置")

        # 检查posture_curriculum_levels函数
        if "def posture_curriculum_levels" not in content:
            print("❌ 未找到posture_curriculum_levels函数")
            return False
        print("✅ 找到posture_curriculum_levels函数")

        # 检查关键参数
        required_params = ["check_interval", "enable_backward", "hysteresis"]
        for param in required_params:
            if param not in content:
                print(f"❌ 未找到参数{param}")
                return False
        print(f"✅ 找到所有必需参数: {required_params}")

        # 检查级别范围
        level_ranges = {
            0: "roll\": (-0.1, 0.1)",
            1: "roll\": (-0.5, 0.5)",
            2: "roll\": (-1.0, 1.0)",
            3: "roll\": (-3.14, 3.14)"
        }

        for level, range_str in level_ranges.items():
            if range_str not in content:
                print(f"❌ Level {level}的roll范围不正确")
                return False
        print("✅ 所有级别的roll范围配置正确")

        print("\n✅ curriculums.py文件测试通过")
        return True

    except Exception as e:
        print(f"❌ curriculums.py文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_init_file():
    """测试__init__.py文件修改"""
    print("\n" + "=" * 60)
    print("测试2: __init__.py文件验证")
    print("=" * 60)

    file_path = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 检查导入
        if "posture_curriculum_levels" not in content:
            print("❌ 未找到posture_curriculum_levels导入")
            return False
        print("✅ 找到posture_curriculum_levels导入")

        if "POSTURE_CURRICULUM_LEVELS" not in content:
            print("❌ 未找到POSTURE_CURRICULUM_LEVELS导入")
            return False
        print("✅ 找到POSTURE_CURRICULUM_LEVELS导入")

        # 检查导入位置（应该在from .curriculums import中）
        if "from .curriculums import" not in content:
            print("❌ 未找到正确的导入语句")
            return False
        print("✅ 找到正确的导入语句")

        print("\n✅ __init__.py文件测试通过")
        return True

    except Exception as e:
        print(f"❌ __init__.py文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_file():
    """测试two_stage_recovery_env_cfg.py文件修改"""
    print("\n" + "=" * 60)
    print("测试3: two_stage_recovery_env_cfg.py文件验证")
    print("=" * 60)

    file_path = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 检查课程启用
        if "posture_curriculum = CurrTerm" not in content:
            print("❌ 未找到posture_curriculum启用")
            return False
        print("✅ 找到posture_curriculum启用")

        # 检查禁用其他课程
        if "terrain_levels = None" not in content:
            print("❌ 未禁用terrain_levels课程")
            return False
        print("✅ terrain_levels课程已禁用")

        if "command_levels = None" not in content:
            print("❌ 未禁用command_levels课程")
            return False
        print("✅ command_levels课程已禁用")

        # 检查使用POSTURE_CURRICULUM_LEVELS
        if "POSTURE_CURRICULUM_LEVELS[0]" not in content:
            print("❌ 未使用POSTURE_CURRICULUM_LEVELS[0]设置初始参数")
            return False
        print("✅ 使用POSTURE_CURRICULUM_LEVELS[0]设置初始参数")

        # 检查奖励权重调整
        reward_checks = {
            "self.rewards.upward.weight = 8.0": "upward权重调整为8.0",
            "self.rewards.base_height_l2.weight = -6.0": "base_height_l2权重调整为-6.0",
            "self.rewards.lin_vel_z_l2.weight = -3.0": "lin_vel_z_l2权重调整为-3.0",
            "wheel_angular_momentum = RewTerm(\n        func=mdp.wheel_angular_momentum_reward,\n        weight=3.0": "wheel_angular_momentum权重调整为3.0"
        }

        for check_str, desc in reward_checks.items():
            if check_str not in content:
                print(f"❌ 未找到{desc}")
                return False
            print(f"✅ {desc}")

        # 检查joint_acc_l2权重
        if "joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-8" not in content:
            print("❌ joint_acc_l2权重未调整为-5e-8")
            return False
        print("✅ joint_acc_l2权重调整为-5e-8")

        # 检查成功判定条件
        success_checks = {
            'self.terminations.success_stable.params["min_upright"] = 0.80': "min_upright调整为0.80",
            'self.terminations.success_stable.params["min_height"] = 0.60': "min_height调整为0.60",
            'self.terminations.success_stable.params["max_tilt"] = 0.40': "max_tilt调整为0.40",
            'self.terminations.success_stable.params["duration"] = 1.0': "duration调整为1.0"
        }

        for check_str, desc in success_checks.items():
            if check_str not in content:
                print(f"❌ 未找到{desc}")
                return False
            print(f"✅ {desc}")

        print("\n✅ two_stage_recovery_env_cfg.py文件测试通过")
        return True

    except Exception as e:
        print(f"❌ two_stage_recovery_env_cfg.py文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_terminations_file():
    """测试terminations.py文件修改"""
    print("\n" + "=" * 60)
    print("测试4: terminations.py文件验证")
    print("=" * 60)

    file_path = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/terminations.py"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 检查函数定义修改
        if "def is_success_stable(" not in content:
            print("❌ 未找到is_success_stable函数")
            return False

        # 提取函数定义部分
        func_start = content.find("def is_success_stable(")
        func_end = content.find("):", func_start) + 2
        func_def = content[func_start:func_end]

        # 检查参数修改
        param_checks = {
            "min_upright: float = 0.80": "min_upright默认值修改为0.80",
            "min_height: float = 0.60": "min_height默认值修改为0.60",
            "max_tilt: float = 0.40": "max_tilt默认值修改为0.40",
            "duration: float = 1.0": "duration默认值修改为1.0"
        }

        for check_str, desc in param_checks.items():
            if check_str not in func_def:
                print(f"❌ {desc}")
                return False
            print(f"✅ {desc}")

        # 检查文档字符串更新
        # 直接在整个文件中查找这些数值，因为文档字符串已经在函数附近
        doc_checks = [
            "0.80",  # min_upright
            "0.60",  # min_height
            "0.40",  # max_tilt
            "1.0"    # duration
        ]

        # 检查函数附近的文档是否包含这些数值
        func_context = content[func_start:func_start + 2000]  # 获取函数定义后的内容

        for check_str in doc_checks:
            if check_str not in func_context:
                print(f"❌ 文档中未找到数值: {check_str}")
                return False
        print("✅ 文档中所有参数值已更新")

        print("\n✅ terminations.py文件测试通过")
        return True

    except Exception as e:
        print(f"❌ terminations.py文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum_levels_logic():
    """测试课程级别逻辑"""
    print("\n" + "=" * 60)
    print("测试5: 课程级别逻辑验证")
    print("=" * 60)

    file_path = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 检查级别描述
        level_descriptions = [
            "Static Anti-Disturbance",
            "Kneeling/Squat Push-Up",
            "Half-Lying Tilt",
            "Extreme Side-Lying"
        ]

        for desc in level_descriptions:
            if desc not in content:
                print(f"❌ 未找到级别描述: {desc}")
                return False
            print(f"✅ 找到级别描述: {desc}")

        # 检查成功阈值
        thresholds = ["0.90", "0.80", "0.70", "0.50"]
        for threshold in thresholds:
            if f'"success_threshold": {threshold}' not in content:
                print(f"❌ 未找到阈值: {threshold}")
                return False
            print(f"✅ 找到阈值: {threshold}")

        # 检查最少episodes
        min_episodes = ["100", "150", "200", "300"]
        for min_ep in min_episodes:
            if f'"min_episodes": {min_ep}' not in content:
                print(f"❌ 未找到min_episodes: {min_ep}")
                return False
            print(f"✅ 找到min_episodes: {min_ep}")

        # 检查级别递增逻辑
        if "if current_level < 3:" not in content:
            print("❌ 未找到级别递增逻辑")
            return False
        print("✅ 找到级别递增逻辑")

        if "if current_level > 0:" not in content:
            print("❌ 未找到向后恢复逻辑")
            return False
        print("✅ 找到向后恢复逻辑")

        # 检查状态管理
        state_vars = [
            "_posture_curriculum_level",
            "_posture_curriculum_episode_count",
            "_posture_curriculum_success_count",
            "_posture_curriculum_timeout_count"
        ]

        for var in state_vars:
            if var not in content:
                print(f"❌ 未找到状态变量: {var}")
                return False
            print(f"✅ 找到状态变量: {var}")

        print("\n✅ 课程级别逻辑测试通过")
        return True

    except Exception as e:
        print(f"❌ 课程级别逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GO2W-ARM 多级姿态恢复课程实现验证")
    print("=" * 60)

    tests = [
        test_curriculums_file,
        test_init_file,
        test_config_file,
        test_terminations_file,
        test_curriculum_levels_logic
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！多级姿态恢复课程已成功实现。")
        print("\n📋 实施要点总结:")
        print("✅ 4级渐进式课程配置完成")
        print("✅ 自动级别递进逻辑实现")
        print("✅ 初始Level 0参数设置")
        print("✅ 奖励权重优化完成")
        print("✅ 成功判定条件优化")
        print("✅ 向后恢复机制实现")
        print("\n🚀 可以开始使用新配置进行训练！")
        return 0
    else:
        print(f"⚠️  {total - passed} 个测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())