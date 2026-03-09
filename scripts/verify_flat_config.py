#!/usr/bin/env python3
"""验证Flat配置是否正确"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_lab"))

try:
    print("正在验证 RobotFlatEnvCfg 配置...")

    from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import RobotFlatEnvCfg

    # 创建配置实例
    cfg = RobotFlatEnvCfg()

    print("\n✅ 配置创建成功!")
    print("\n检查关键配置:")
    print(f"  - 地形类型: {cfg.scene.terrain.terrain_type}")
    print(f"  - 地形生成器: {cfg.scene.terrain.terrain_generator}")
    print(f"  - 高度扫描器: {cfg.scene.height_scanner}")
    print(f"  - 高度扫描器(基座): {cfg.scene.height_scanner_base}")
    print(f"  - 观测空间(height_scan - policy): {cfg.observations.policy.height_scan}")
    print(f"  - 观测空间(height_scan - critic): {cfg.observations.critic.height_scan}")
    print(f"  - 地形课程: {cfg.curriculum.terrain_levels}")

    # 验证传感器配置
    available_sensors = []
    if hasattr(cfg.scene, 'height_scanner') and cfg.scene.height_scanner is not None:
        available_sensors.append('height_scanner')
    if hasattr(cfg.scene, 'height_scanner_base') and cfg.scene.height_scanner_base is not None:
        available_sensors.append('height_scanner_base')
    if hasattr(cfg.scene, 'contact_forces') and cfg.scene.contact_forces is not None:
        available_sensors.append('contact_forces')

    print(f"\n可用传感器: {available_sensors}")

    # 验证观测项
    policy_obs = []
    if hasattr(cfg.observations, 'policy') and cfg.observations.policy is not None:
        for attr in dir(cfg.observations.policy):
            if not attr.startswith('_') and not callable(getattr(cfg.observations.policy, attr)):
                obs_term = getattr(cfg.observations.policy, attr)
                if obs_term is not None and hasattr(obs_term, 'func'):
                    policy_obs.append(attr)

    print(f"\n策略观测项: {policy_obs}")

    # 检查是否没有height_scan
    if 'height_scan' not in policy_obs:
        print("\n✅ height_scan 已成功从观测空间中移除!")
    else:
        print("\n❌ height_scan 仍在观测空间中!")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ RobotFlatEnvCfg 配置验证通过!")
    print("="*60)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
