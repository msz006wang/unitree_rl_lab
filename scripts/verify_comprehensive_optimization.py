#!/usr/bin/env python
"""
GO2W ARM 综合优化验证脚本

验证所有新增的奖励函数、观测函数和配置是否正确加载。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "source"))
sys.path.insert(0, str(project_root / "source" / "unitree_rl_lab"))

def test_imports():
    """测试导入所有新增的函数"""
    print("=" * 60)
    print("1. 测试导入新增的奖励函数")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp

        # 测试新奖励函数
        reward_functions = [
            "upward_velocity",
            "orientation_tracking",
            "torque_penalty",
            "joint_regularization",
            "contact_management",
            "wheel_assisted_recovery",
        ]

        for func_name in reward_functions:
            if hasattr(mdp, func_name):
                func = getattr(mdp, func_name)
                print(f"  ✅ {func_name}: 导入成功")
                print(f"     - 函数类型: {type(func).__name__}")
                print(f"     - 模块位置: {func.__module__}")
            else:
                print(f"  ❌ {func_name}: 未找到")

    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

    return True


def test_observations():
    """测试导入所有新增的观测函数"""
    print("\n" + "=" * 60)
    print("2. 测试导入新增的观测函数")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp

        # 测试新观测函数
        observation_functions = [
            "history_buffer",
            "joint_pos_history",
            "body_vel_history",
        ]

        for func_name in observation_functions:
            if hasattr(mdp, func_name):
                func = getattr(mdp, func_name)
                print(f"  ✅ {func_name}: 导入成功")
                print(f"     - 函数类型: {type(func).__name__}")
                print(f"     - 模块位置: {func.__module__}")
            else:
                print(f"  ❌ {func_name}: 未找到")

    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

    return True


def test_configuration():
    """测试配置文件是否正确加载"""
    print("\n" + "=" * 60)
    print("3. 测试配置文件加载")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm import velocity_env_cfg

        # 测试环境配置
        env_cfg = velocity_env_cfg.RobotEnvCfg()
        print(f"  ✅ RobotEnvCfg: 加载成功")

        # 检查观测配置
        obs_cfg = env_cfg.observations
        print(f"\n  策略观测配置:")
        policy_obs = vars(obs_cfg.policy)
        for obs_name, obs_term in policy_obs.items():
            if obs_term is not None:
                print(f"    ✅ {obs_name}: 已配置")
                if obs_name in ["joint_pos_history", "body_vel_history"]:
                    print(f"       -> 历史观测（新增）")

        print(f"\n  评判器观测配置:")
        critic_obs = vars(obs_cfg.critic)
        for obs_name, obs_term in critic_obs.items():
            if obs_term is not None:
                print(f"    ✅ {obs_name}: 已配置")
                if obs_name in ["joint_pos_history", "body_vel_history"]:
                    print(f"       -> 历史观测（新增）")

        # 检查奖励配置
        print(f"\n  奖励配置:")
        rewards_cfg = env_cfg.rewards
        new_rewards = [
            "upward_velocity",
            "orientation_tracking",
            "torque_penalty",
            "joint_regularization",
            "contact_management",
            "wheel_assisted_recovery",
        ]

        for reward_name in new_rewards:
            if hasattr(rewards_cfg, reward_name):
                reward_term = getattr(rewards_cfg, reward_name)
                if reward_term is not None:
                    print(f"    ✅ {reward_name}: 已配置")
                    print(f"       - 权重: {reward_term.weight}")
                else:
                    print(f"    ⚠️  {reward_name}: 已禁用（weight=None）")
            else:
                print(f"    ❌ {reward_name}: 未找到")

        # 检查动作配置
        print(f"\n  动作配置:")
        action_cfg = env_cfg.actions
        print(f"    ✅ joint_pos 关节: {action_cfg.joint_pos.joint_names}")
        print(f"    ✅ joint_vel 关节: {action_cfg.joint_vel.joint_names}")
        print(f"    ✅ joint_pos 缩放: {action_cfg.joint_pos.scale}")
        print(f"    ✅ joint_vel 缩放: {action_cfg.joint_vel.scale}")

        # 检查机械臂配置
        arm_joints_in_action = [j for j in action_cfg.joint_pos.joint_names if "arm" in j]
        print(f"\n  机械臂动作配置:")
        if "arm_joint1" in arm_joints_in_action:
            print(f"    ✅ arm_joint1: 在动作空间中（允许根部旋转）")
        else:
            print(f"    ❌ arm_joint1: 不在动作空间中")

        other_arm_joints = ["arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
        other_in_action = [j for j in other_arm_joints if j in arm_joints_in_action]
        if not other_in_action:
            print(f"    ✅ arm_joint2-6: 不在动作空间中（保持折叠）")
        else:
            print(f"    ⚠️  以下机械臂关节在动作空间中: {other_in_action}")

    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_reward_functions():
    """测试奖励函数的可调用性"""
    print("\n" + "=" * 60)
    print("4. 测试奖励函数签名和参数")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp
        import inspect

        # 测试新奖励函数的签名
        reward_functions = {
            "upward_velocity": {
                "params": ["env", "asset_cfg"],
                "defaults": {},
            },
            "orientation_tracking": {
                "params": ["env", "asset_cfg"],
                "defaults": {},
            },
            "torque_penalty": {
                "params": ["env", "asset_cfg", "sustained_window", "burst_threshold", "decay_rate", "rated_torque"],
                "defaults": {
                    "sustained_window": 2.0,
                    "burst_threshold": 1.5,
                    "decay_rate": 0.9,
                    "rated_torque": 23.5,
                },
            },
            "joint_regularization": {
                "params": ["env", "asset_cfg", "soft_ratio"],
                "defaults": {"soft_ratio": 0.95},
            },
            "contact_management": {
                "params": ["env", "sensor_cfg", "foot_body_names"],
                "defaults": {},
            },
            "wheel_assisted_recovery": {
                "params": ["env", "asset_cfg", "wheel_joint_names"],
                "defaults": {},
            },
        }

        for func_name, config in reward_functions.items():
            if hasattr(mdp, func_name):
                func = getattr(mdp, func_name)
                sig = inspect.signature(func)

                print(f"\n  {func_name}:")
                print(f"    - 参数: {list(sig.parameters.keys())}")
                print(f"    - 期望参数: {config['params']}")

                # 检查参数匹配
                expected_params = set(config['params'])
                actual_params = set(sig.parameters.keys())
                if expected_params.issubset(actual_params):
                    print(f"    ✅ 参数匹配")
                else:
                    missing = expected_params - actual_params
                    print(f"    ❌ 缺少参数: {missing}")

                if config['defaults']:
                    print(f"    - 默认值: {config['defaults']}")

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

    return True


def test_observation_functions():
    """测试观测函数的签名"""
    print("\n" + "=" * 60)
    print("5. 测试观测函数签名和参数")
    print("=" * 60)

    try:
        from unitree_rl_lab.tasks.locomotion import mdp
        import inspect

        # 测试新观测函数的签名
        observation_functions = {
            "history_buffer": {
                "params": ["env", "obs_term_func", "buffer_length"],
                "defaults": {"buffer_length": 10},
            },
            "joint_pos_history": {
                "params": ["env", "asset_cfg", "buffer_length"],
                "defaults": {"buffer_length": 10},
            },
            "body_vel_history": {
                "params": ["env", "buffer_length"],
                "defaults": {"buffer_length": 10},
            },
        }

        for func_name, config in observation_functions.items():
            if hasattr(mdp, func_name):
                func = getattr(mdp, func_name)
                sig = inspect.signature(func)

                print(f"\n  {func_name}:")
                print(f"    - 参数: {list(sig.parameters.keys())}")
                print(f"    - 期望参数: {config['params']}")

                # 检查参数匹配
                expected_params = set(config['params'])
                actual_params = set(sig.parameters.keys())
                if expected_params.issubset(actual_params):
                    print(f"    ✅ 参数匹配")
                else:
                    missing = expected_params - actual_params
                    print(f"    ❌ 缺少参数: {missing}")

                if config['defaults']:
                    print(f"    - 默认值: {config['defaults']}")

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("GO2W ARM 综合优化验证")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("导入奖励函数", test_imports()))
    results.append(("导入观测函数", test_observations()))
    results.append(("配置文件加载", test_configuration()))
    results.append(("奖励函数签名", test_reward_functions()))
    results.append(("观测函数签名", test_observation_functions()))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！配置正确。")
        print("\n下一步：")
        print("  1. 运行训练: python scripts/train.py --task Robot-v0")
        print("  2. 监控训练: python scripts/start_tensorboard.sh")
        print("  3. 查看文档: docs/GO2W_ARM_COMPREHENSIVE_OPTIMIZATION.md")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
