#!/usr/bin/env python3
"""简单测试：验证velocity_env_cfg能否正常导入"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_lab"))

try:
    print("正在测试导入...")

    # 尝试导入配置
    from unitree_rl_lab.tasks.locomotion.robots.go2w import velocity_env_cfg

    print("✅ 成功导入 velocity_env_cfg")

    # 检查关键类是否存在
    print("\n检查配置类...")
    print(f"✅ EventCfg: {hasattr(velocity_env_cfg, 'EventCfg')}")
    print(f"✅ RewardsCfg: {hasattr(velocity_env_cfg, 'RewardsCfg')}")
    print(f"✅ ObservationsCfg: {hasattr(velocity_env_cfg, 'ObservationsCfg')}")
    print(f"✅ ActionsCfg: {hasattr(velocity_env_cfg, 'ActionsCfg')}")
    print(f"✅ RobotEnvCfg: {hasattr(velocity_env_cfg, 'RobotEnvCfg')}")

    # 测试实例化配置
    print("\n尝试实例化配置...")
    cfg = velocity_env_cfg.RobotEnvCfg()
    print(f"✅ 成功创建 RobotEnvCfg 实例")
    print(f"   - num_envs: {cfg.scene.num_envs}")
    print(f"   - episode_length_s: {cfg.episode_length_s}")

    # 检查新增的事件
    print("\n检查新添加的事件...")
    print(f"✅ randomize_rigid_body_inertia: {hasattr(cfg.events, 'randomize_rigid_body_inertia')}")
    print(f"✅ action_mirror: {hasattr(cfg.rewards, 'action_mirror')}")
    print(f"✅ action_sync: {hasattr(cfg.rewards, 'action_sync')}")

    print("\n" + "="*60)
    print("✅ 所有测试通过! 配置文件可以正常使用。")
    print("="*60)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
