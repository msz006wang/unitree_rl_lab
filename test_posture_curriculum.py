#!/usr/bin/env python3
"""
测试多级姿态恢复课程学习功能
验证课程配置、导入和基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

def test_imports():
    """测试导入是否正常"""
    print("=" * 60)
    print("测试1: 导入验证")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp

        # 测试导入课程配置
        assert hasattr(mdp, 'POSTURE_CURRICULUM_LEVELS'), "POSTURE_CURRICULUM_LEVELS 未导出"
        assert hasattr(mdp, 'posture_curriculum_levels'), "posture_curriculum_levels 未导出"

        print("✅ 所有导入测试通过")
        print(f"   - POSTURE_CURRICULUM_LEVELS: {type(mdp.POSTURE_CURRICULUM_LEVELS)}")
        print(f"   - posture_curriculum_levels: {type(mdp.posture_curriculum_levels)}")

        return True
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum_config():
    """测试课程配置是否正确"""
    print("\n" + "=" * 60)
    print("测试2: 课程配置验证")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp

        config = mdp.POSTURE_CURRICULUM_LEVELS

        # 验证配置结构
        assert isinstance(config, dict), "配置应该是字典类型"
        assert len(config) == 4, f"应该有4个级别，实际有{len(config)}个"

        # 验证每个级别的配置
        for level in range(4):
            assert level in config, f"缺少Level {level}"
            level_config = config[level]

            required_keys = ["name", "description", "pose_range", "velocity_range",
                           "success_threshold", "min_episodes", "focus"]
            for key in required_keys:
                assert key in level_config, f"Level {level} 缺少 {key}"

            # 验证pose_range
            pose_range = level_config["pose_range"]
            required_pose_keys = ["roll", "pitch", "yaw", "x", "y", "z"]
            for key in required_pose_keys:
                assert key in pose_range, f"Level {level} pose_range 缺少 {key}"

            # 验证阈值范围
            threshold = level_config["success_threshold"]
            assert 0.0 <= threshold <= 1.0, f"Level {level} 阈值超出范围: {threshold}"

            print(f"✅ Level {level}: {level_config['name']}")
            print(f"   - 描述: {level_config['description']}")
            print(f"   - Roll范围: {pose_range['roll']}")
            print(f"   - 高度范围: {pose_range['z']}")
            print(f"   - 成功阈值: {threshold*100:.0f}%")
            print(f"   - 最少episodes: {level_config['min_episodes']}")
            print(f"   - 重点: {level_config['focus']}")

        print("\n✅ 所有配置测试通过")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_level_progression():
    """测试级别递进逻辑"""
    print("\n" + "=" * 60)
    print("测试3: 级别递进验证")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp

        config = mdp.POSTURE_CURRICULUM_LEVELS

        # 验证难度递增
        prev_roll_range = None
        for level in range(4):
            roll_range = config[level]["pose_range"]["roll"]
            roll_width = roll_range[1] - roll_range[0]

            if prev_roll_range is not None:
                prev_width = prev_roll_range[1] - prev_roll_range[0]
                assert roll_width >= prev_width, f"Level {level} 的roll范围应该大于等于Level {level-1}"

            prev_roll_range = roll_range
            print(f"✅ Level {level} roll范围: {roll_range} (宽度: {roll_width:.2f})")

        # 验证阈值递减（难度增加）
        prev_threshold = None
        for level in range(4):
            threshold = config[level]["success_threshold"]

            if prev_threshold is not None:
                assert threshold <= prev_threshold, f"Level {level} 的阈值应该小于等于Level {level-1}"

            prev_threshold = threshold
            print(f"✅ Level {level} 成功阈值: {threshold*100:.0f}%")

        # 验证最少episodes递增
        prev_min_episodes = None
        for level in range(4):
            min_episodes = config[level]["min_episodes"]

            if prev_min_episodes is not None:
                assert min_episodes >= prev_min_episodes, f"Level {level} 的最少episodes应该大于等于Level {level-1}"

            prev_min_episodes = min_episodes
            print(f"✅ Level {level} 最少episodes: {min_episodes}")

        print("\n✅ 级别递进测试通过")
        return True
    except Exception as e:
        print(f"❌ 级别递进测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_config():
    """测试环境配置是否正确启用课程"""
    print("\n" + "=" * 60)
    print("测试4: 环境配置验证")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.two_stage_recovery_env_cfg import (
            TwoStageRecoveryFlatEnvCfg
        )

        cfg = TwoStageRecoveryFlatEnvCfg()

        # 验证课程已启用
        assert cfg.curriculum.posture_curriculum is not None, "姿态课程未启用"
        print("✅ 姿态课程已启用")

        # 验证初始参数是Level 0
        initial_pose_range = cfg.events.randomize_reset_base.params["pose_range"]
        level_0_pose = mdp.POSTURE_CURRICULUM_LEVELS[0]["pose_range"]

        for key in ["roll", "pitch", "z"]:
            assert abs(initial_pose_range[key][0] - level_0_pose[key][0]) < 0.01, \
                f"初始pose范围{key}不匹配Level 0"
            assert abs(initial_pose_range[key][1] - level_0_pose[key][1]) < 0.01, \
                f"初始pose范围{key}不匹配Level 0"

        print("✅ 初始参数设置为Level 0")
        print(f"   - Roll: {initial_pose_range['roll']}")
        print(f"   - Pitch: {initial_pose_range['pitch']}")
        print(f"   - 高度: {initial_pose_range['z']}")

        # 验证其他课程已禁用
        assert cfg.curriculum.terrain_levels is None, "地形课程应该禁用"
        assert cfg.curriculum.command_levels is None, "命令课程应该禁用"
        print("✅ 竞争课程已正确禁用")

        # 验证奖励权重调整
        assert cfg.rewards.upward.weight == 8.0, f"upward权重应该是8.0，实际是{cfg.rewards.upward.weight}"
        assert cfg.rewards.base_height_l2.weight == -6.0, f"base_height_l2权重应该是-6.0，实际是{cfg.rewards.base_height_l2.weight}"
        assert cfg.rewards.lin_vel_z_l2.weight == -3.0, f"lin_vel_z_l2权重应该是-3.0，实际是{cfg.rewards.lin_vel_z_l2.weight}"
        assert cfg.rewards.wheel_angular_momentum.weight == 3.0, f"wheel_angular_momentum权重应该是3.0，实际是{cfg.rewards.wheel_angular_momentum.weight}"

        print("✅ 奖励权重已正确调整")

        # 验证成功判定条件
        success_params = cfg.terminations.success_stable.params
        assert success_params["min_upright"] == 0.80, f"min_upright应该是0.80，实际是{success_params['min_upright']}"
        assert success_params["min_height"] == 0.60, f"min_height应该是0.60，实际是{success_params['min_height']}"
        assert success_params["max_tilt"] == 0.40, f"max_tilt应该是0.40，实际是{success_params['max_tilt']}"
        assert success_params["duration"] == 1.0, f"duration应该是1.0，实际是{success_params['duration']}"

        print("✅ 成功判定条件已正确优化")
        print(f"   - 最小直立度: {success_params['min_upright']}")
        print(f"   - 最小高度: {success_params['min_height']}m")
        print(f"   - 最大倾斜: {success_params['max_tilt']}rad")
        print(f"   - 持续时间: {success_params['duration']}s")

        print("\n✅ 环境配置测试通过")
        return True
    except Exception as e:
        print(f"❌ 环境配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GO2W-ARM 多级姿态恢复课程测试")
    print("=" * 60)

    tests = [
        test_imports,
        test_curriculum_config,
        test_level_progression,
        test_environment_config
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
        return 0
    else:
        print(f"⚠️  {total - passed} 个测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())