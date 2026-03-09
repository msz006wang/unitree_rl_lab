#!/usr/bin/env python3
"""验证GO2W配置导入"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_lab"))

try:
    print("测试导入...")
    from unitree_rl_lab.tasks.locomotion.robots.go2w import velocity_env_cfg
    print("✅ 成功导入 velocity_env_cfg")

    # 检查新功能
    cfg = velocity_env_cfg.RobotEnvCfg()
    print("✅ 成功创建配置实例")

    # 验证事件
    print("\n检查事件配置:")
    print(f"  - randomize_rigid_body_inertia: {hasattr(cfg.events, 'randomize_rigid_body_inertia')}")
    print(f"  - randomize_com_positions: {hasattr(cfg.events, 'randomize_com_positions')}")

    # 验证奖励
    print("\n检查奖励配置:")
    print(f"  - action_mirror: {hasattr(cfg.rewards, 'action_mirror')}")
    print(f"  - action_sync: {hasattr(cfg.rewards, 'action_sync')}")

    # 检查 randomize_rigid_body_com 函数签名
    from unitree_rl_lab.tasks.locomotion import mdp
    import inspect

    func = mdp.randomize_rigid_body_com
    sig = inspect.signature(func)
    print(f"\nrandomize_rigid_body_com 参数: {list(sig.parameters.keys())}")

    print("\n✅ 所有检查通过!")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
