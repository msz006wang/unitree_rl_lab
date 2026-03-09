#!/bin/bash
# 快速测试训练配置（只运行很短时间来验证配置）

set -e

echo "========================================="
echo "快速测试改进的G1训练配置"
echo "========================================="
echo ""

# 检查环境
if [ -z "$ISAAC_SIM_PATH" ] && [ -z "$ISAAC_PATH" ]; then
    echo "⚠️  警告: 未检测到 Isaac Sim 环境变量"
    echo ""
    echo "请确保已设置以下环境变量之一："
    echo "  - ISAAC_SIM_PATH"
    echo "  - ISAAC_PATH"
    echo ""
    echo "或运行以下命令激活环境："
    echo "  conda activate env_isaaclab"
    echo ""
fi

# 使用较小的环境数量和很短的时间来测试
echo "配置："
echo "  任务: Unitree-G1-29dof-Velocity-Improved"
echo "  环境数量: 64 (测试用)"
echo "  测试时间: 约30秒"
echo ""

read -p "是否开始测试? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "测试已取消"
    exit 0
fi

echo ""
echo "开始测试..."
echo ""

# 启动训练（后台运行，30秒后自动终止）
timeout 30s python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Velocity-Improved \
    --num_envs 64 \
    --headless \
    2>&1 | tee /tmp/training_test.log &

TRAIN_PID=$!

# 监控输出
echo "训练进程 PID: $TRAIN_PID"
echo "等待30秒以测试配置..."
echo ""

# 等待30秒或进程结束
for i in {1..30}; do
    sleep 1
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo ""
        echo "训练进程已结束"
        break
    fi
    echo -n "."
done

echo ""
echo ""

# 检查进程状态
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "✅ 训练进程仍在运行 - 配置加载成功！"
    echo "正在终止测试进程..."
    kill $TRAIN_PID 2>/dev/null || true
    wait $TRAIN_PID 2>/dev/null || true
else
    # 检查退出状态
    wait $TRAIN_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 124 ]; then
        echo "✅ 测试超时（正常）- 配置加载成功！"
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 训练正常退出 - 配置加载成功！"
    else
        echo "❌ 训练失败，退出码: $EXIT_CODE"
        echo ""
        echo "查看日志:"
        tail -20 /tmp/training_test.log
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "🎉 测试完成！"
echo "========================================="
echo ""
echo "配置验证成功！现在可以开始完整训练："
echo "  ./scripts/quick_start.sh train-improved-small  # 512 envs"
echo "  ./scripts/quick_start.sh train-improved        # 4096 envs"
echo ""
