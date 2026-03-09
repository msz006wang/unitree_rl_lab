#!/usr/bin/env python3
"""快速测试导入"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_lab"))

try:
    # 测试mdp导入
    from unitree_rl_lab.tasks.locomotion import mdp

    # 检查关键函数
    functions_to_check = [
        "randomize_rigid_body_inertia",
        "randomize_rigid_body_com",  # 使用别名
        "action_mirror",
        "action_sync",
    ]

    print("检查关键函数...")
    for func_name in functions_to_check:
        if hasattr(mdp, func_name):
            print(f"✅ {func_name}")
        else:
            print(f"❌ {func_name} - 缺失!")
            sys.exit(1)

    # 尝试导入配置
    from unitree_rl_lab.tasks.locomotion.robots.go2w import velocity_env_cfg
    print("\n✅ 配置导入成功!")

    # 尝试实例化
    cfg = velocity_env_cfg.RobotEnvCfg()
    print(f"✅ 配置实例化成功!")
    print(f"   - 环境数: {cfg.scene.num_envs}")
    print(f"   - 惯量随机化: {hasattr(cfg.events, 'randomize_rigid_body_inertia')}")
    print(f"   - 动作镜像: {hasattr(cfg.rewards, 'action_mirror')}")
    print(f"   - 动作同步: {hasattr(cfg.rewards, 'action_sync')}")

    print("\n✅ 所有测试通过!")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
