#!/bin/bash

# GO2W_ARM 两段式恢复训练脚本 - 支持多级姿态恢复课程
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_curriculum() { echo -e "${CYAN}[CURRICULUM]${NC} $1"; }

# ============================================================================
# 配置参数
# ============================================================================

# 任务选择：使用两段式恢复环境配置
TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"  # 修复：使用正确的环境名称
# 注意：该环境已内置 Flat 地形配置（TwoStageRecoveryFlatEnvCfg）

# 训练参数
NUM_ENVS=4096
HEADLESS=false  # 是否使用无头模式（无GUI）
MAX_ITERATIONS=1000000  # 最大迭代次数（修复：使用 max_iterations 而非 num_steps）

# 输出配置
OUTPUT_DIR="${PROJECT_ROOT}/logs/rsl_rl/unitree_go2warm_twostage_recovery_curriculum"
TENSORBOARD_PORT=6006

# 多级姿态恢复课程配置
CURRICULUM_ENABLED=true
CURRICULUM_CHECK_INTERVAL=100
CURRICULUM_ENABLE_BACKWARD=true

# ============================================================================
# 显示配置信息
# ============================================================================

echo -e "${CYAN}=============================================================================${NC}"
echo -e "${CYAN}GO2W-ARM 多级姿态恢复课程训练${NC}"
echo -e "${CYAN}=============================================================================${NC}"
echo ""
log_info "任务配置："
echo "  - 任务名称: ${TASK_NAME}"
echo "  - 环境数量: ${NUM_ENVS}"
echo "  - 最大迭代次数: ${MAX_ITERATIONS}"
echo "  - 无头模式: ${HEADLESS}"
echo ""
log_curriculum "多级姿态恢复课程配置："
echo "  - 课程启用: ${CURRICULUM_ENABLED}"
echo "  - 检查间隔: ${CURRICULUM_CHECK_INTERVAL} episodes"
echo "  - 向后恢复: ${CURRICULUM_ENABLE_BACKWARD}"
echo ""
log_curriculum "课程级别详情："
echo "  - Level 0: 静态抗扰平衡 (±5°, 存活率>90%)"
echo "  - Level 1: 跪伏/深蹲推举 (±30°, 站起率>80%)"
echo "  - Level 2: 半侧卧倾斜 (±60°, 恢复率>70%)"
echo "  - Level 3: 极端侧卧孤岛 (±180°, 掌握率>50%)"
echo ""
log_info "输出配置："
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - TensorBoard端口: ${TENSORBOARD_PORT}"
echo ""
echo -e "${CYAN}=============================================================================${NC}"
echo ""

# ============================================================================
# 环境设置
# ============================================================================

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

# ============================================================================
# 启动TensorBoard（后台运行）
# ============================================================================

log_info "启动TensorBoard监控服务..."

# 尝试启动 TensorBoard，如果失败则记录日志但继续训练
tensorboard --logdir="${OUTPUT_DIR}" --port=${TENSORBOARD_PORT} --host=0.0.0.0 > /tmp/tensorboard_${TENSORBOARD_PORT}.log 2>&1 &
TENSORBOARD_PID=$!

# 等待 2 秒让 TensorBoard 尝试启动
sleep 2

# 检查 TensorBoard 是否成功启动
if ps -p ${TENSORBOARD_PID} > /dev/null; then
    log_success "TensorBoard已启动，PID: ${TENSORBOARD_PID}"
    log_info "访问地址: http://localhost:${TENSORBOARD_PORT}"
    log_info "注意: TensorBoard 可能在训练目录创建后才能正常显示数据"
else
    log_warning "TensorBoard启动失败（目录可能不存在），将在训练开始后自动可用"
    log_info "访问地址: http://localhost:${TENSORBOARD_PORT}"
    log_info "或运行: tensorboard --logdir=${OUTPUT_DIR} --port=${TENSORBOARD_PORT}"
    TENSORBOARD_PID=""
fi

# ============================================================================
# 构建训练命令
# ============================================================================

# 基础训练命令
TRAIN_CMD="python3 scripts/train_fixed.py --task ${TASK_NAME} --num_envs ${NUM_ENVS}"

# 添加最大迭代次数
TRAIN_CMD="${TRAIN_CMD} --max_iterations ${MAX_ITERATIONS}"

# 添加无头模式参数
if [ "${HEADLESS}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --headless"
fi

# ============================================================================
# 显示训练命令和开始训练
# ============================================================================

log_info "训练命令："
echo "${TRAIN_CMD}"
echo ""
log_info "开始训练..."

# 训练完成后清理TensorBoard进程
function cleanup() {
    log_info "训练结束，清理TensorBoard进程..."
    if [ -n "${TENSORBOARD_PID}" ] && ps -p ${TENSORBOARD_PID} > /dev/null; then
        kill ${TENSORBOARD_PID}
        log_success "TensorBoard进程已终止"
    fi
}

# 设置退出时执行清理
trap cleanup EXIT

# 执行训练
${TRAIN_CMD}

# ============================================================================
# 训练完成总结
# ============================================================================

echo ""
echo -e "${CYAN}=============================================================================${NC}"
log_success "训练完成！"
echo -e "${CYAN}=============================================================================${NC}"
echo ""
log_info "训练结果位置："
echo "  - 模型文件: ${OUTPUT_DIR}/*/model_*.pt"
echo "  - 训练日志: ${OUTPUT_DIR}/*/events.out.tfevents.*"
echo "  - 配置文件: ${OUTPUT_DIR}/*/params/"
echo ""
log_info "查看训练曲线："
echo "  - TensorBoard: http://localhost:${TENSORBOARD_PORT}"
echo "  - 或运行: tensorboard --logdir=${OUTPUT_DIR} --port=${TENSORBOARD_PORT}"
echo ""
log_curriculum "课程学习关键指标："
echo "  - Episode_Termination/success_stable: 成功率（预期从0%提升到50-75%）"
echo "  - Episode_Reward/success_stable_reward: 成功奖励（预期从0提升到500-700）"
echo "  - Episode_Reward/base_height_l2: 高度控制（应该逐渐减小）"
echo "  - Episode_Reward/joint_acc_l2: 关节冲击（应该逐渐减小）"
echo ""
log_info "下一步操作："
echo "  1. 使用训练好的模型进行测试: python scripts/play.py --task ${TASK_NAME} --checkpoint ${OUTPUT_DIR}/<path>/model_*.pt"
echo "  2. 分析训练结果: python analyze_training_metrics.py"
echo "  3. 导出部署模型: python export_deploy.py --checkpoint ${OUTPUT_DIR}/<path>/model_*.pt"
echo ""
echo -e "${CYAN}=============================================================================${NC}"


