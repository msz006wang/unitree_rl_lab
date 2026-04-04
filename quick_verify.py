#!/usr/bin/env python3
"""快速验证GO2W_ARM迁移功能"""

import sys
sys.path.insert(0, 'source')

def test_imports():
    """测试关键导入"""
    print("测试关键导入...")

    tests = [
        ("观测函数", "unitree_rl_lab.tasks.locomotion.mdp.observations"),
        ("课程学习", "unitree_rl_lab.tasks.locomotion.mdp.curriculums"),
        ("奖励函数", "unitree_rl_lab.tasks.locomotion.mdp.extended_rewards"),
        ("配置文件", "unitree_rl_lab.tasks.locomotion.robots.go2w_arm.two_stage_recovery_env_cfg"),
    ]

    passed = 0
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\n通过: {passed}/{len(tests)}")
    return passed == len(tests)

def test_features():
    """测试迁移的功能"""
    print("\n测试robot_lab_locomanip迁移功能...")

    try:
        # 测试观测函数
        from unitree_rl_lab.tasks.locomotion.mdp.observations import (
            body_state_obs, contact_state_obs, phase_obs,
            body_lin_acc_l2, action_rate_l2
        )
        print("  ✅ 观测函数")

        # 测试课程学习
        from unitree_rl_lab.tasks.locomotion.mdp.curriculums import (
            difficulty_levels_two_stage, command_levels_vel
        )
        print("  ✅ 课程学习")

        # 测试奖励函数
        from unitree_rl_lab.tasks.locomotion.mdp.extended_rewards import (
            two_stage_standing_reward, phase_detection
        )
        print("  ✅ 奖励函数")

        return True
    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("GO2W_ARM 训练环境快速验证\n")

    import_ok = test_imports()
    feature_ok = test_features()

    if import_ok and feature_ok:
        print("\n🎉 所有验证通过！")
        print("可以开始训练:")
        print("  ./scripts/train_go2w_arm_two_stage.sh")
        sys.exit(0)
    else:
        print("\n⚠️ 验证失败")
        sys.exit(1)