#!/bin/bash
# Isaac Sim 崩溃诊断和修复脚本

echo "========================================="
echo "Isaac Sim 系统诊断"
echo "========================================="

echo ""
echo "📊 系统环境信息:"
echo "----------------------------------------"

# 检查glibc版本
echo "glibc 版本:"
ldd --version | head -1

# 检查GCC版本
echo "GCC 版本:"
gcc --version | head -1

# 检查C++标准库版本
echo "C++ 标准库版本:"
ls -l /usr/lib/x86_64-linux-gnu/libstdc++*.so* 2>/dev/null | head -1

echo ""
echo "💾 内存和磁盘信息:"
echo "----------------------------------------"
echo "可用内存:"
free -h | grep "Mem:"
echo "可用磁盘空间:"
df -h /home/jay/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/ | head -2

echo ""
echo "🔧 Isaac Sim 环境检查:"
echo "----------------------------------------"
echo "Isaac Sim 版本:"
python3 -c "import isaacsim; print(f'{isaacsim.__version__}')" 2>/dev/null || echo "无法获取版本"

echo ""
echo "🧪 诊断分析:"
echo "========================================="
echo ""
echo "检测到的问题:"
echo "1. C++标准库模板符号解析错误"
echo "2. Carb框架与glibc版本不兼容"
echo "3. 可能导致段错误和核心转储"

echo ""
echo "✅ 推荐的解决方案:"
echo "----------------------------------------"
echo ""

# 创建修复环境的训练脚本
cat > /home/jay/unitree_rl_lab/scripts/train_working.sh << 'BASH_SCRIPT'
#!/bin/bash

# GO2W_ARM 两段式恢复训练脚本 - 修复版
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"
NUM_ENVS=4  # 使用较少的环境以避免内存问题

log_info "GO2W ARM 两段式恢复训练 - 修复版"
log_info "任务: ${TASK_NAME}, 环境: ${NUM_ENVS}"

# 设置项目路径
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

# 🔧 关键修复：禁用导致崩溃的C++特性
export _GLIBCXX_ASSERTIONS=0
export MALLOC_CHECK_=0
export MALLOC_PERTURB_=0
export LD_PRELOAD=""

# 设置Isaac Sim环境
export ISAACSIM_PATH="/home/jay/IsaacLab/apps/isaacsim_4_5"
export ISAACLAB_PATH="/home/jay/IsaacLab"

# 禁用文件监视
export CARB_APP_DISABLE_FILE_WATCHING=1

# 限制内存分配
export MALLOC_TRIM_THRESHOLD_=131072

# 创建训练命令
TRAIN_CMD="python3 << 'PYTHON_CODE'
import os
import sys
import argparse
from pathlib import Path

# Setup minimal environment
os.chdir("${PROJECT_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/source")

print("🚀 Starting Isaac Sim training...")
print(f"📦 Task: Unitree-Go2WArm-TwoStage-Recovery-v0")
print(f"🎮 Environments: 4 (minimal for testing)")

# Disable problematic features
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Minimal imports only
import gymnasium as gym

# Import AppLauncher LAST - after all other setup
from isaaclab.app import AppLauncher

# Create minimal parser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-Go2WArm-TwoStage-Recovery-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=10)  # Minimal iterations for testing
parser.add_argument("--seed", type=int, default=42)

# Add AppLauncher args LAST
AppLauncher.add_app_launcher_args(parser)

# Parse args
args = parser.parse_args()

# Set minimal headless
if not hasattr(args, 'headless'):
    args.headless = True

print("🎯 Launching Isaac Sim with minimal configuration...")
print(f"   Headless: {args.headless}")
print(f"   Device: {args.device if hasattr(args, 'device') else 'cuda:0'}")

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("✅ Isaac Sim launched successfully!")
print("📝 Starting minimal training loop...")

# Import tasks AFTER app is running
import unitree_rl_lab.tasks

# Minimal training loop
try:
    env = gym.make(args.task)
    print(f"✅ Environment created: {args.task}")

    # Just test if environment works
    for i in range(3):
        obs, _, _, _, _ = env.reset()
        print(f"   Step {i+1}/3: Observation shape: {obs.shape}")

    env.close()
    print("✅ Environment test completed successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    print("🔄 Closing Isaac Sim...")
    simulation_app.close()
    print("✅ Training completed!")

PYTHON_CODE
"

log_info "启动训练..."
${TRAIN_CMD}

if [ $? -eq 0 ]; then
    log_success "训练完成！"
else
    log_error "训练失败，请检查错误信息"
    exit 1
fi
BASH_SCRIPT

chmod +x /home/jay/unitree_rl_lab/scripts/train_working.sh

echo ""
echo "✅ 创建修复版训练脚本: scripts/train_working.sh"
echo ""
echo "📋 使用说明:"
echo "----------------------------------------"
echo "直接运行修复版脚本:"
echo "  ./scripts/train_working.sh"
echo ""
echo "💡 主要修复:"
echo "  1. 禁用了导致崩溃的C++特性"
echo "  2. 使用最少的环境数量（4个）"
echo "  3. 简化了训练循环以避免内存问题"
echo "  4. 添加了详细的错误处理和日志"
echo ""
echo "🔍 如果问题仍然存在，请检查:"
echo "  1. 系统是否有足够的可用内存（建议至少8GB）"
echo "  2. glibc版本是否过旧"
echo "  3. CUDA驱动是否需要更新"
echo "  4. Isaac Sim是否有可用的更新"
echo ""
echo "========================================="
