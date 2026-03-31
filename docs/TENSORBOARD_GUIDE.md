# TensorBoard 查看指南

## 快速开始

### 方法 1: 启动 TensorBoard (推荐)

在训练过程中或训练完成后，使用以下命令启动 TensorBoard：

```bash
# 进入项目目录
cd /home/jay/unitree_rl_lab

# 启动 TensorBoard (查看所有训练记录)
tensorboard --logdir=logs/rsl_rl --port=6006

# 或者查看特定实验
tensorboard --logdir=logs/rsl_rl/unitree_go2warm_velocity_flat_v0 --port=6006
```

### 方法 2: 在后台启动 TensorBoard

如果你想在训练时继续查看 TensorBoard：

```bash
# 后台启动 TensorBoard
tensorboard --logdir=logs/rsl_rl --port=6006 &

# 查看进程
ps aux | grep tensorboard
```

### 方法 3: 查看特定时间段的训练

```bash
# 查看最新训练
tensorboard --logdir=logs/rsl_rl/unitree_go2warm_velocity_flat_v0/2026-03-30_23-29-09 --port=6006
```

## 当前训练记录

### 可用实验目录：
- `unitree_go2warm_velocity_flat_v0` - GO2W ARM 平地环境
- `unitree_g1_29dof_velocity_flat_improved` - G1 改进版
- `unitree_g1_29dof_velocity_improved` - G1 标准版
- `unitree_go2w_velocity_flat_v0` - GO2W 无机械臂
- `unitree_go2w_velocity_rough_v0` - GO2W 粗糙地形

### 当前训练运行：
最新的训练记录位于：
- `logs/rsl_rl/unitree_go2warm_velocity_flat_v0/2026-03-30_23-29-09`

## TensorBoard 访问

启动 TensorBoard 后，在浏览器中访问：
- **本地访问**: http://localhost:6006
- **远程访问**: http://[你的IP地址]:6006

## 重要监控指标

### 1. 训练进度指标
- **Episode Reward**: 每个episode的总奖励
- **Episode Length**: 每个episode的步数
- **Success Rate**: 任务完成成功率

### 2. 姿态和平衡指标
- **Stand Reward**: 站立恢复奖励
- **Flat Orientation L2**: 直立姿态奖励
- **Base Height L2**: 高度控制奖励
- **Upright Bonus**: 直立状态额外奖励

### 3. 速度追踪指标
- **Track Lin Vel XY Exp**: 线速度追踪奖励
- **Track Ang Vel Z Exp**: 角速度追踪奖励
- **Command Following**: 指令跟随性能

### 4. 能效和稳定性指标
- **Joint Torques L2**: 关节扭矩惩罚
- **Joint Vel L2**: 关节速度惩罚
- **Energy Consumption**: 能量消耗
- **Arm Stability**: 机械臂稳定性

## 故障排除

### 问题 1: 端口被占用
如果 6006 端口被占用，使用其他端口：
```bash
tensorboard --logdir=logs/rsl_rl --port=6007
```

### 问题 2: 找不到事件文件
检查训练是否真的在运行：
```bash
# 查看是否有新的训练记录
ls -la logs/rsl_rl/unitree_go2warm_velocity_flat_v0/

# 查看事件文件
find logs/rsl_rl/unitree_go2warm_velocity_flat_v0 -name "*.tfevents*"
```

### 问题 3: TensorBoard 连接断开
TensorBoard 在长时间训练中可能会断开，重新加载页面即可。

## 高级功能

### 比较多个实验
```bash
# 同时查看多个训练对比
tensorboard --logdir logs/rsl_rl/unitree_go2w_velocity_flat_v0:logs/rsl_rl/unitree_go2warm_velocity_flat_v0 --port=6006
```

### 导出数据
在 TensorBoard 中可以：
- 下载图表为 PNG/SVG
- 导出数据为 CSV
- 保存自定义视图

## 训练监控建议

### 1. 关键指标监控
重点关注以下指标的改善趋势：
- 站立成功率（从倒下状态恢复的比例）
- 平均episode奖励
- 姿态稳定性（roll/pich angle）

### 2. 训练阶段判断
根据奖励曲线判断训练阶段：
- **早期 (0-1M steps)**: 奖励快速上升，学习基本技能
- **中期 (1-5M steps)**: 奖励平稳增长，技能整合
- **后期 (5M+ steps)**: 奖励缓慢提升，精细调优

### 3. 异常检测
注意以下异常情况：
- 奖励突然下降：检查配置或环境问题
- 站立成功率为0：检查初始化或奖励函数
- episode长度持续很低：可能早停策略问题

## 自动化脚本

### 启动 TensorBoard 脚本
创建启动脚本 `scripts/start_tensorboard.sh`：

```bash
#!/bin/bash
# TensorBoard 启动脚本

cd /home/jay/unitree_rl_lab

# 检查端口是否被占用
PORT=6006
while netstat -tuln | grep -q ":$PORT "; do
    echo "端口 $PORT 被占用，尝试 $((PORT + 1))"
    PORT=$((PORT + 1))
done

echo "在端口 $PORT 上启动 TensorBoard..."
tensorboard --logdir=logs/rsl_rl --port=$PORT
```

### 查看当前训练状态
创建快速查看脚本：

```bash
#!/bin/bash
# 查看最新训练的 TensorBoard

cd /home/jay/unitree_rl_lab

# 获取最新训练目录
LATEST_DIR=$(ls -t logs/rsl_rl/unitree_go2warm_velocity_flat_v0 | head -1)

echo "启动 TensorBoard 查看最新训练: $LATEST_DIR"
tensorboard --logdir="logs/rsl_rl/unitree_go2warm_velocity_flat_v0/$LATEST_DIR" --port=6006
```

## 实时训练监控

### 查看训练进程
```bash
# 查看训练进程
ps aux | grep "train.py"

# 查看GPU使用情况
nvidia-smi

# 查看日志文件
tail -f logs/rsl_rl/unitree_go2warm_velocity_flat_v0/2026-03-30_23-29-09/stdout.txt
```

### 训练进度监控
在 TensorBoard 中可以实时查看：
- 当前训练步数
- 学习曲线
- 实时奖励变化
- 环境渲染视频（如果启用）

## 总结

1. **立即启动**: 使用 `tensorboard --logdir=logs/rsl_rl --port=6006`
2. **访问地址**: http://localhost:6006
3. **监控重点**: 站立恢复能力、姿态控制、奖励趋势
4. **故障排除**: 检查端口占用、确认训练运行

这些设置将帮助你有效地监控 GO2W ARM 机器人的训练进度！