#!/bin/bash
# 查看最新训练的 TensorBoard

echo "========================================"
echo "最新训练 TensorBoard 查看脚本"
echo "========================================"
echo ""

cd /home/jay/unitree_rl_lab

# 获取最新训练目录
EXPERIMENT="unitree_go2warm_velocity_flat_v0"

if [ ! -d "logs/rsl_rl/$EXPERIMENT" ]; then
    echo "❌ 找不到实验目录: $EXPERIMENT"
    echo ""
    echo "可用的实验:"
    ls -d logs/rsl_rl/
    exit 1
fi

# 获取最新时间戳目录
LATEST_DIR=$(ls -t logs/rsl_rl/$EXPERIMENT | head -1)

echo "📂 实验类型: $EXPERIMENT"
echo "🕐 最新训练: $LATEST_DIR"
echo ""

# 检查端口是否被占用
PORT=6006
while netstat -tuln 2>/dev/null | grep -q ":$PORT "; do
    echo "端口 $PORT 被占用，尝试 $((PORT + 1))"
    PORT=$((PORT + 1))
done

echo "✅ 在端口 $PORT 上启动 TensorBoard..."
echo "📊 日志目录: logs/rsl_rl/$EXPERIMENT/$LATEST_DIR"
echo "🌐 访问地址: http://localhost:$PORT"
echo ""
echo "使用 Ctrl+C 停止 TensorBoard"
echo ""

tensorboard --logdir="logs/rsl_rl/$EXPERIMENT/$LATEST_DIR" --port=$PORT