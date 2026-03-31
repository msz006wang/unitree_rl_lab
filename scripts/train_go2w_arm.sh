#!/bin/bash
# GO2W-Arm 综合优化训练脚本
# 基于新的奖励函数、历史观测、动作空间优化
# 日期: 2026-03-31

# 设置环境变量
export PYTHONPATH=/home/jay/unitree_rl_lab/source:$PYTHONPATH

echo "========================================"
echo "GO2W-Arm 综合优化训练 (v2.0-final)"
echo "========================================"
echo ""
echo "新优化特性:"
echo "  ✅ 6个新奖励函数（upward_velocity, orientation_tracking, torque_penalty等）"
echo "  ✅ 10帧历史观测（joint_pos_history, body_vel_history）"
echo "  ✅ 机械臂策略优化（夹紧+根部旋转）"
echo "  ✅ 轮足协同奖励（wheel_assisted_recovery）"
echo ""

# 默认参数
TASK_NAME="Robot-v0"
TRAIN_HEADLESS=false
ENABLE_TENSORBOARD=true
SKIP_VERIFY=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --arx5-flat)
            TASK_NAME="Unitree-Go2WArm-Velocity-Flat-v0"
            ;;
        --arx5-rough)
            TASK_NAME="Unitree-Go2WArm-Velocity-Rough-v0"
            ;;
        --no-verify)
            SKIP_VERIFY=true
            ;;
        --headless)
            TRAIN_HEADLESS=true
            ;;
        --no-tensorboard)
            ENABLE_TENSORBOARD=false
            ;;
        --tensorboard-dir=*)
            TENSORBOARD_DIR="$2"
            shift
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH="$2"
            shift
            ;;
        --epochs=*)
            NUM_EPOCHS="$2"
            shift
            ;;
        --num-envs=*)
            NUM_ENVS="$2"
            shift
            ;;
        --help|-h)
            echo "GO2W-Arm 综合优化训练脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "训练类型（默认: ARX5平地）:"
            echo "  --arx5-flat     ARX5机械臂 - 平地环境 | ✅ 默认"
            echo "  --arx5-rough     ARX5机械臂 - 粗糙地形 | -"
            echo ""
            echo "选项:"
            echo "  --no-verify       跳过配置验证"
            echo "  --headless        无头模式（适合服务器）"
            echo "  --no-tensorboard  不启动TensorBoard"
            echo "  --tensorboard-dir=DIR 指定TensorBoard日志目录"
            echo "  --checkpoint=PATH   从检查点恢复训练"
            echo "  --epochs=N       训练轮数"
            echo "  --num-envs=N     环境数量"
            echo ""
            echo "新优化特性:"
            echo "  • 向上速度奖励（upward_velocity, weight=2.0）"
            echo "  • 姿态跟踪奖励（orientation_tracking, weight=1.5）"
            echo "  • 扭矩惩罚（torque_penalty, weight=-0.01）"
            echo "  • 关节正则化（joint_regularization, weight=-0.5）"
            echo "  • 接触管理（contact_management, weight=-0.3）"
            echo "  • 轮足协同（wheel_assisted_recovery, weight=0.5）"
            echo "  • 历史观测（10帧关节位置+身体速度）"
            echo "  • 机械臂策略优化（arm_joint1可旋转，arm_joint2-6折叠）"
            echo ""
            echo "TensorBoard监控重点:"
            echo "  rewards/upward_velocity: 向上速度（目标>0）"
            echo "  rewards/orientation_tracking: 姿态（目标>0.8）"
            echo "  rewards/torque_penalty: 扭矩使用（目标<0.005）"
            echo "  rewards/contact_management: 非足端接触（目标接近0）"
            echo ""
            echo "示例命令:"
            echo "  $0 --arx5-flat"
            echo "  $0 --arx5-flat --num-envs 4096 --epochs 10"
            echo "  $0 --no-verify --headless"
            echo ""
            exit 0
            ;;
        *)
            echo "⚠️ 未知选项: $1"
            echo "使用 --help 查看所有选项"
            exit 1
            ;;
    esac
    shift
done

# 检查训练类型（如果未指定，默认ARX5平地）
if [ -z "$TASK_NAME" ]; then
    TASK_NAME="Unitree-Go2WArm-Velocity-Flat-v0"
fi

# ========================================
# 步骤1: 配置验证（可选）
# ========================================
if [ "$SKIP_VERIFY" != true ]; then
    echo ""
    echo "========================================"
    echo "步骤 1/5: 配置验证"
    echo "========================================"
    echo ""

    CONFIG_FILE="/home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py"
    OBS_FILE="/home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py"
    MDP_FILE="/home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py"

    if [ -f "$CONFIG_FILE" ] && [ -f "$OBS_FILE" ] && [ -f "$MDP_FILE" ]; then
        echo "✓ 配置文件存在"
    else
        echo "⚠️ 部分配置文件缺失，但继续训练"
        echo "   建议手动运行: python scripts/validate_config.py"
    fi

    echo ""
fi

# ========================================
# 步骤2: 环境检查
# ========================================
echo ""
echo "========================================"
echo "步骤 2/5: 检查训练环境"
echo "========================================"
echo ""

# 检查Python环境
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ 错误: Python环境未配置或torch未安装"
    echo ""
    echo "请确保已安装必要的依赖："
    echo "  - Python >= 3.10"
    echo "  - PyTorch"
    echo "  - IsaacLab (可选，训练脚本会自动加载）"
    echo ""
    echo "设置环境: export PYTHONPATH=/home/jay/unitree_rl_lab/source:\$PYTHONPATH"
    exit 1
fi

# 检查IsaacLab（如果需要）
# 注意：由于pxr模块问题，验证脚本不依赖完整IsaacLab环境
echo "✓ Python和PyTorch已安装"

# ========================================
# 步骤3: TensorBoard监控（可选）
# ========================================
if [ "$ENABLE_TENSORBOARD" = true ]; then
    echo ""
    echo "========================================"
    echo "步骤 3/5: TensorBoard监控"
    echo "========================================"
    echo ""

    TENSORBOARD_DIR=${TENSORBOARD_DIR:-"/home/jay/unitree_rl_lab/logs/tensorboard"}

    # 检查TensorBoard是否已运行
    if pgrep -x "tensorboard" > /dev/null 2>&1; then
        echo "⚠️  TensorBoard已在运行"
        echo "   进程ID: $(pgrep -x tensorboard | head -n 1 | awk '{print $2}')"
        echo ""
        read -p "是否终止现有TensorBoard? (y/N): " -n 1 -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "tensorboard"
            echo "✓ 已终止现有TensorBoard"
            sleep 2
        fi
    fi

    # 启动TensorBoard
    echo "启动TensorBoard..."
    echo "  端口: $TENSORBOARD_PORT"
    echo "  日志目录: $TENSORBOARD_DIR"
    echo ""

    # 后台启动TensorBoard
    mkdir -p "$TENSORBOARD_DIR"
    nohup python -m tensorboard.main --logdir="$TENSORBOARD_DIR" --port=$TENSORBOARD_PORT --host=0.0.0.0 > "$TENSORBOARD_DIR/tensorboard.log" 2>&1 &

    # 等待TensorBoard启动
    sleep 3

    # 检查TensorBoard是否成功启动
    if pgrep -x "tensorboard" > /dev/null 2>&1; then
        echo "✓ TensorBoard已启动"
        echo "  进程ID: $(pgrep -x tensorboard | head -n 1 | awk '{print $2}')"
    else
        echo "❌ TensorBoard启动失败"
        echo "   请检查日志: cat $TENSORBOARD_DIR/tensorboard.log"
    fi

    echo ""
fi

# ========================================
# 步骤4: 训练准备
# ========================================
echo ""
echo "========================================"
echo "步骤 4/5: 训练准备"
echo "========================================"
echo ""

# 训练参数
LOG_DIR="/home/jay/unitree_rl_lab/logs"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}

echo "训练配置:"
echo "  任务名称: $TASK_NAME"
echo "  日志目录: $LOG_DIR"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  检查点路径: 默认（自动保存）"
else
    echo "  检查点路径: $CHECKPOINT_PATH"
fi

# 环境参数
if [ "$TRAIN_HEADLESS" = true ]; then
    echo " 显示模式: headless（无GUI，适合服务器）"
else
    echo "  显示模式: 有GUI（默认，需要X11）"
fi

# ========================================
# 步骤5: 启动训练
# ========================================
echo ""
echo "========================================"
echo "步骤 5/5: 启动训练"
echo "========================================"
echo ""

# 创建日志目录
mkdir -p "$LOG_DIR"

# 构建训练命令
TRAIN_CMD="python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py --task $TASK_NAME"

# 添加其他参数
if [ -n "$NUM_ENVS" ]; then
    TRAIN_CMD="$TRAIN_CMD --num-envs $NUM_ENVS"
fi

# 检查点恢复
if [ -n "$CHECKPOINT_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --load $CHECKPOINT_PATH"
fi

# 显示训练命令
echo "执行命令:"
echo "  $TRAIN_CMD"
echo ""

# 训练前信息
echo "新优化特性:"
echo "  • 6个新奖励函数（upward_velocity, orientation_tracking, torque_penalty等）"
echo "  • 10帧历史观测（joint_pos_history, body_vel_history）"
echo "  • 机械臂策略优化（arm_joint1可旋转，arm_joint2-6折叠）"
echo ""

# 直接启动训练（不要求确认）
echo "🚀 开始训练..."
echo "  任务: $TASK_NAME"
echo "  进程ID: $$"
echo ""

# 运行训练
eval $TRAIN_CMD &
TRAIN_PID=$!

# 等待几秒确保训练启动
sleep 5

# 检查训练是否正常启动
if ! ps -p $TRAIN_PID > /dev/null; then
    echo "❌ 训练进程已退出"
    echo "   请检查错误日志"
    sleep 2
    cat "$LOG_DIR/stdout.txt" 2>/dev/null || echo "   日志文件不存在"
    exit 1
fi

echo "✓ 训练已启动"
echo ""
echo "监控命令:"
echo "  查看实时日志: tail -f $LOG_DIR/stdout.txt"
echo "  查看训练进度: tail -f $LOG_DIR/stdout.txt 2>/dev/null || echo "   (日志文件将创建)"
echo "  查看TensorBoard: http://localhost:$TENSORBOARD_PORT"
echo "  查看GPU使用: watch -n 1 nvidia-smi"
echo ""
echo "停止训练: 按 Ctrl+C"
echo ""

# 捕获Ctrl+C信号以优雅停止
trap 'echo ""; echo ""; echo "========================================"; echo "训练已停止"; echo "========================================"; kill $TRAIN_PID 2>/dev/null; exit 0' SIGINT SIGTERM

# 等待训练进程
wait $TRAIN_PID

TRAIN_EXIT_CODE=$?
TRAIN_END_TIME=$(date +%s)
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))

echo ""
echo "========================================"
echo "训练完成"
echo "========================================"
echo "退出码: $TRAIN_EXIT_CODE"
echo "训练时长: $TRAIN_DURATION 秒"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练正常完成"
else
    echo "⚠️  训练异常退出（码: $TRAIN_EXIT_CODE）"
fi

echo ""
echo "训练日志位置:"
echo "  • 主日志: $LOG_DIR/train.log"
echo "  • 输出日志: $LOG_DIR/stdout.txt"
echo "  • TensorBoard: $TENSORBOARD_DIR"
echo "  • 检查点: (保存在logs/目录）"
echo ""
echo "重新训练: $0 --arx5-flat --checkpoint <checkpoint_path>"
