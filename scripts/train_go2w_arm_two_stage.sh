#!/bin/bash

# GO2W_ARM 两段式恢复训练脚本
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"
NUM_ENVS=4096
OUTPUT_DIR="${PROJECT_ROOT}/output/go2w_arm_two_stage_recovery"

log_info "创建输出目录..."
mkdir -p "${OUTPUT_DIR}"

log_info "设置Python路径..."
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

# 设置Isaac Sim环境变量
export ISAACSIM_PATH="/home/jay/IsaacLab/apps/isaacsim_4_5"
export ISAACLAB_PATH="/home/jay/IsaacLab"

# 禁用文件监视以避免磁盘空间问题
export CARB_APP_DISABLE_FILE_WATCHING=1

# 使用修复后的训练脚本，GUI模式显示
TRAIN_CMD="python3 scripts/train_fixed.py --task ${TASK_NAME} --num_envs ${NUM_ENVS}"

log_info "启动训练..."
${TRAIN_CMD}

