#!/bin/bash
# TensorBoard 启动脚本

echo "========================================"
echo "TensorBoard 启动脚本"
echo "========================================"
echo ""

cd /home/jay/unitree_rl_lab

# 检查端口是否被占用
PORT=6006
while netstat -tuln 2>/dev/null | grep -q ":$PORT "; do
    echo "端口 $PORT 被占用，尝试 $((PORT + 1))"
    PORT=$((PORT + 1))
done

echo "✅ 在端口 $PORT 上启动 TensorBoard..."
echo "📊 日志目录: logs/rsl_rl"
echo "🌐 访问地址: http://localhost:$PORT"
echo ""
echo "使用 Ctrl+C 停止 TensorBoard"
echo ""

tensorboard --logdir=logs/rsl_rl --port=$PORT