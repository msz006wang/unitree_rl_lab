#!/bin/bash
# 监控G1训练进程
# Monitor G1 Training Process

echo "=========================================="
echo "G1机器人训练监控 / G1 Robot Training Monitor"
echo "=========================================="
echo ""

# 检查训练进程
echo "1. 训练进程状态 / Training Process Status:"
echo "-------------------------------------------"
TRAINING_PID=$(ps aux | grep "python.*train.py.*Unitree-G1-29dof-Velocity" | grep -v grep | awk '{print $2}')
if [ -n "$TRAINING_PID" ]; then
    echo "✅ 训练进程运行中 / Training process is running"
    echo "   PID: $TRAINING_PID"
    ps -p $TRAINING_PID -o pid,ppid,cmd,%mem,%cpu,etime --no-headers 2>/dev/null || echo "   进程详情不可用"
else
    echo "❌ 未找到训练进程 / No training process found"
fi

echo ""
echo "2. GPU使用情况 / GPU Usage:"
echo "-------------------------------------------"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "   GPU信息不可用"

echo ""
echo "3. 系统资源 / System Resources:"
echo "-------------------------------------------"
echo "   CPU使用率 / CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   内存使用 / Memory Usage: $(free -h | awk '/Mem:/ {printf "%s/%s (%.1f%%)", $3, $2, ($3/$2)*100}')"

echo ""
echo "4. 训练日志（最后20行）/ Training Logs (Last 20 Lines):"
echo "-------------------------------------------"
tail -20 /tmp/training_output.log | grep -v "^$" || tail -20 /tmp/training_output.log

echo ""
echo "5. 检查最新模型保存 / Check Latest Model Saves:"
echo "-------------------------------------------"
LATEST_LOG=$(ls -t logs/rsl_rl/Unitree-G1-29dof-Velocity/ 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   最新日志目录 / Latest log dir: $LATEST_LOG"
    ls -lh logs/rsl_rl/Unitree-G1-29dof-Velocity/$LATEST_LOG/model_*.pt 2>/dev/null | tail -5 || echo "   暂无保存的模型"
else
    echo "   尚未创建训练日志目录"
fi

echo ""
echo "=========================================="
echo "监控完成 / Monitor Completed"
echo "=========================================="
