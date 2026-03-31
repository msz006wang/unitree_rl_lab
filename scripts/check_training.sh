#!/bin/bash
# 训练状态检查脚本

echo "========================================"
echo "GO2W ARM 训练状态检查"
echo "========================================"
echo ""

# 检查训练进程
TRAINING_PROCESS=$(ps aux | grep -v grep | grep "train.py" | grep "rsl_rl")

if [ -n "$TRAINING_PROCESS" ]; then
    echo "✅ 训练进程运行中:"
    echo "$TRAINING_PROCESS"
else
    echo "❌ 没有找到训练进程"
fi

echo ""

# 检查GPU使用
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU 状态:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "🖥️  无法获取 GPU 状态 (nvidia-smi 不可用)"
fi

echo ""

# 检查日志目录
EXPERIMENT="unitree_go2warm_velocity_flat_v0"

if [ -d "logs/rsl_rl/$EXPERIMENT" ]; then
    echo "📂 实验目录: logs/rsl_rl/$EXPERIMENT"

    # 列出所有训练运行
    echo ""
    echo "📊 训练运行记录:"
    ls -lt logs/rsl_rl/$EXPERIMENT/ | tail -5

    # 检查最新训练
    LATEST_DIR=$(ls -t logs/rsl_rl/$EXPERIMENT 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo ""
        echo "🕐 最新训练: $LATEST_DIR"

        # 检查事件文件
        EVENT_COUNT=$(find logs/rsl_rl/$EXPERIMENT/$LATEST_DIR -name "*.tfevents*" 2>/dev/null | wc -l)
        echo "📈 事件文件数量: $EVENT_COUNT"
    fi
else
    echo "❌ 找不到实验目录: $EXPERIMENT"
fi

echo ""
echo "========================================"