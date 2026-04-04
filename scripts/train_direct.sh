#!/bin/bash
# 直接训练脚本 - 绕过所有已知问题

set -e

echo "🚀 GO2W ARM 直接训练脚本"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 激活环境
source /home/jay/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

echo "✅ Conda环境已激活: env_isaaclab"
echo "📁 项目根目录: ${PROJECT_ROOT}"

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH}"
export ISAACSIM_PATH="/home/jay/IsaacLab/apps/isaacsim_4_5"
export ISAACLAB_PATH="/home/jay/IsaacLab"

# 关键修复
export _GLIBCXX_ASSERTIONS=0
export MALLOC_CHECK_=0
export CARB_APP_DISABLE_FILE_WATCHING=1

# 进入项目目录
cd "${PROJECT_ROOT}"

echo ""
echo "📝 训练配置:"
echo "   任务: Unitree-Go2WArm-TwoStage-Recovery-v0"
echo "   环境: 4 (测试用)"
echo "   最大迭代: 10"
echo ""

echo "🔧 系统检查:"
echo "   可用内存: $(free -h | grep "Mem:" | awk '{print $7}')"
echo "   可用磁盘: $(df -h /home/jay/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/ | tail -1 | awk '{print $4}')"

echo ""
echo "🚀 启动训练..."
echo "========================================="

# 直接运行Python训练代码，不通过bash包装
python3 -u << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import os
import sys

# Setup paths
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, f"{PROJECT_ROOT}/source")

print("🎯 GO2W ARM 两段式恢复训练")
print(f"📁 项目: {PROJECT_ROOT}")
print("🔧 环境初始化...")

# Import only what we absolutely need
import torch
import gymnasium as gym

# Configure torch with safest settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.beterministic = False
torch.backends.cudnn.benchmark = False

print("📚 PyTorch配置完成")

# Import AppLauncher LAST
from isaaclab.app import AppLauncher
import argparse

# Create absolutely minimal parser
parser = argparse.ArgumentParser(description="GO2W ARM Training")
parser.add_argument("--task", type=str, default="Unitree-Go2WArm-TwoStage-Recovery-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
import sys
original_args = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Clear original args

args = parser.parse_args()

print("✅ 参数解析完成")

# Create minimal headless argument
if not hasattr(args, 'headless'):
    args.headless = True

print(f"📊 配置:")
print(f"   任务: {args.task}")
print(f"   环境: {args.num_envs}")
print(f"   迭代: {args.max_iterations}")
print(f"   种子: {args.seed}")
print(f"   无GUI: {args.headless}")

# Launch Isaac Sim
print("🎮 启动Isaac Sim...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("✅ Isaac Sim启动成功!")

# Now import tasks
import unitree_rl_lab.tasks
print("✅ 任务模块导入成功!")

# Test with minimal setup
print("🧪 开始最小化测试...")

try:
    import time
    print("🏗 创建环境...")
    env = gym.make(args.task)
    print(f"✅ 环境创建成功: {args.task}")

    print("🔄 运行测试循环...")
    for step in range(min(args.max_iterations, 3)):  # Limit to 3 steps for testing
        obs, _, _, _, info = env.reset()
        print(f"   步骤 {step+1}/{min(args.max_iterations, 3)}: 观测形状 {obs.shape}")
        time.sleep(0.1)  # Small delay between steps

    print("✅ 测试完成!")

    env.close()

    print("🎉 成功！环境测试通过，Isaac Sim工作正常！")
    print("💡 现在可以使用此方法进行完整训练")
    print("   只需增加 --max_iterations 和 --num_envs 参数")

    sys.exit(0)

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    print("📋 调试信息:")
    print(f"   项目路径: {PROJECT_ROOT}")
    print(f"   Python路径: {sys.path[:3]}")
    sys.exit(1)

finally:
    if 'simulation_app' in locals():
        print("🔄 关闭Isaac Sim...")
        simulation_app.close()
        print("✅ Isaac Sim已关闭")

PYTHON_SCRIPT
'

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅✅✅ 训练成功完成！"
    echo ""
    echo "🎯 下一步操作:"
    echo "   1. 现在可以进行完整训练"
    echo "   2. 使用更多环境: --num_envs 64"
    echo "   3. 增加训练迭代: --max_iterations 10000"
else
    echo "❌ 训练失败，退出代码: $EXIT_CODE"
    echo ""
    echo "🔍 故障排除建议:"
    echo "   1. 检查完整错误日志"
    echo "   2. 确认CUDA驱动正常工作"
    echo "   3. 尝试使用GUI模式测试: --headless False"
fi
echo "========================================="
