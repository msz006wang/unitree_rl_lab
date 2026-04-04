#!/usr/bin/env python3
"""
验证脚本：检查robot_lab_locomanip功能是否正确迁移到GO2W_ARM项目
"""

import sys
import os
sys.path.append('source')

def check_observations():
    """检查观测函数"""
    print("=" * 50)
    print("检查观测函数...")

    try:
        from unitree_rl_lab.unitree_rl_lab.tasks.locomotion.mdp.observations import (
            body_state_obs,
            contact_state_obs,
            phase_obs,
            two_stage_state_obs,
            body_lin_acc_l2,
            action_rate_l2,
            joint_pos_limits,
            joint_vel_limits
        )
        print("✅ 所有观测函数导入成功")
        return True
    except Exception as e:
        print(f"❌ 观测函数导入失败: {e}")
        return False

def check_curriculums():
    """检查课程学习函数"""
    print("=" * 50)
    print("检查课程学习函数...")

    try:
        from unitree_rl_lab.unitree_rl_lab.tasks.locomotion.mdp.curriculums import (
            difficulty_levels_two_stage,
            command_levels_vel
        )
        print("✅ 课程学习函数导入成功")
        return True
    except Exception as e:
        print(f"❌ 课程学习函数导入失败: {e}")
        return False

def check_rewards():
    """检查奖励函数"""
    print("=" * 50)
    print("检查奖励函数...")

    try:
        from unitree_rl_lab.unitree_rl_lab.tasks.locomotion.mdp.extended_rewards import (
            two_stage_standing_reward,
            phase_detection,
            tuck_and_roll_reward,
            asymmetric_kick_reward,
            explode_to_stand_reward
        )
        print("✅ 奖励函数导入成功")
        return True
    except Exception as e:
        print(f"❌ 奖励函数导入失败: {e}")
        return False

def check_config_file():
    """检查配置文件"""
    print("=" * 50)
    print("检查配置文件...")

    config_file = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py"

    try:
        with open(config_file, 'r') as f:
            content = f.read()

        # 检查关键的配置项
        checks = [
            ("body_state = ObsTerm", "body_state观测"),
            ("contact_state = ObsTerm", "contact_state观测"),
            ("phase_state = ObsTerm", "phase_state观测"),
            ("two_stage_standing = RewTerm", "two_stage_standing奖励"),
            ("difficulty_levels = CurrTerm", "difficulty_levels课程"),
            ("success_stand = DoneTerm", "success_stand终止"),
        ]

        all_found = True
        for check, desc in checks:
            if check in content:
                print(f"✅ {desc} 已启用")
            else:
                print(f"❌ {desc} 未找到")
                all_found = False

        return all_found

    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        return False

def main():
    """主验证函数"""
    print("开始验证robot_lab_locomanip功能迁移...")

    results = []
    results.append(check_observations())
    results.append(check_curriculums())
    results.append(check_rewards())
    results.append(check_config_file())

    print("=" * 50)
    print("验证结果总结:")

    if all(results):
        print("🎉 所有验证通过！robot_lab_locomanip功能已成功迁移")
        print("\n迁移的功能包括:")
        print("1. ✅ 两段式状态观测 (body_state_obs, contact_state_obs, phase_obs)")
        print("2. ✅ 高级控制观测 (body_lin_acc_l2, action_rate_l2, limits)")
        print("3. ✅ 课程学习函数 (difficulty_levels_two_stage)")
        print("4. ✅ 两段式奖励函数 (two_stage_standing_reward)")
        print("5. ✅ 成功终止条件 (success_stand)")
        print("\n可以开始使用新的训练配置了！")
    else:
        print("⚠️  部分验证未通过，请检查上述错误")

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)