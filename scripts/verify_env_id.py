#!/usr/bin/env python3
"""验证环境注册配置"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source"))

# 只读取注册文件，不运行 Isaac Sim
with open("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__init__.py") as f:
    content = f.read()

print("=" * 60)
print("GO2W ARM 环境注册信息")
print("=" * 60)

import re
registrations = re.findall(r'id="([^"]+)"', content)

print("\n✅ 已注册的环境ID:")
for i, env_id in enumerate(registrations, 1):
    marker = "🎯" if "TwoStage" in env_id else "  "
    print(f"  {i}. {marker} {env_id}")

print("\n" + "=" * 60)
print("推荐使用的环境ID:")
print("=" * 60)
print(f"  ✅ Unitree-Go2WArm-TwoStage-Recovery-v0 (两段式恢复训练)")
print(f"  ✅ Unitree-Go2WArm-Velocity-Flat-v0 (平地速度控制)")
print(f"  ✅ Unitree-Go2WArm-Velocity-Rough-v0 (复杂地形速度控制)")

print("\n" + "=" * 60)
print("训练命令示例:")
print("=" * 60)
print("  ./scripts/train_go2w_arm_two_stage.sh")
print("  python3 scripts/train_fixed.py --task Unitree-Go2WArm-TwoStage-Recovery-v0 --headless --num_envs 4096")
print()
