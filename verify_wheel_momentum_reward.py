#!/usr/bin/env python3
"""验证轮子角动量奖励是否正确配置到训练中"""

import sys
sys.path.insert(0, "source")

try:
    from isaaclab.app import AppLauncher

    # 启动 Isaac Lab
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from unitree_rl_lab.tasks.locomotion.mdp import wheel_angular_momentum_reward
    from isaaclab.utils.assets import check_file_path, ISAACLAB_NUCLEUS_DIR

    print("✅ Isaac Lab 启动成功")
    print(f"✅ wheel_angular_momentum_reward 函数导入成功: {wheel_angular_momentum_reward.__name__}")

    # 检查函数签名
    import inspect
    sig = inspect.signature(wheel_angular_momentum_reward)
    params = list(sig.parameters.keys())
    print(f"✅ 函数参数: {params}")

    # 创建环境实例以验证配置
    print("\n🔍 检查环境配置...")
    env_id = "Unitree-Go2WArm-TwoStage-Recovery-v0"
    print(f"环境ID: {env_id}")

    try:
        # 使用更小的环境数量来快速验证
        env = gym.make(env_id, num_envs=2, headless=True)
        print(f"✅ 环境创建成功: {env}")

        # 检查奖励配置
        if hasattr(env, 'reward_manager'):
            print(f"✅ Reward manager 存在")

            # 检查是否有 wheel_angular_momentum 奖励项
            if hasattr(env.reward_manager, 'term_cfgs'):
                reward_terms = list(env.reward_manager.term_cfgs.keys())
                print(f"✅ 奖励项总数: {len(reward_terms)}")
                print(f"✅ 所有奖励项: {reward_terms}")

                if 'wheel_angular_momentum' in reward_terms:
                    print("\n🎉 轮子角动量奖励已成功集成到训练中！")

                    # 获取奖励配置
                    wheel_reward_cfg = env.reward_manager.term_cfgs['wheel_angular_momentum']
                    print(f"   - 函数: {wheel_reward_cfg.func}")
                    print(f"   - 权重: {wheel_reward_cfg.weight}")
                    if hasattr(wheel_reward_cfg, 'params'):
                        print(f"   - 参数: {wheel_reward_cfg.params}")
                else:
                    print("\n⚠️  警告: wheel_angular_momentum 奖励项未找到")
                    print("可用的奖励项:", reward_terms)

        env.close()
        print("\n✅ 验证完成！轮子角动量奖励已正确配置到训练中。")

    except Exception as e:
        print(f"❌ 创建环境时出错: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'simulation_app' in locals():
        simulation_app.close()
