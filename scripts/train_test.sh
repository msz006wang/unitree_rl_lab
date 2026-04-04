#!/bin/bash
# 简化测试脚本 - 使用较少环境

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"
NUM_ENVS=2  # 减少环境数量以避免内存问题

log_info "启动简化测试训练..."
log_info "任务: ${TASK_NAME}, 环境: ${NUM_ENVS}"

cd "${PROJECT_ROOT}"

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH}"
export ISAACSIM_PATH="/home/jay/IsaacLab/apps/isaacsim_4_5"
export ISAACLAB_PATH="/home/jay/IsaacLab"

# 清理临时文件
rm -f /tmp/isaacsim_init.py 2>/dev/null || true

# 使用更少的内存设置
TRAIN_CMD="python3 scripts/train_fixed.py --task ${TASK_NAME} --headless --num_envs ${NUM_ENVS}"

log_info "启动训练..."
${TRAIN_CMD}
