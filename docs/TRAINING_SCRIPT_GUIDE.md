# GO2W ARM 综合优化 - 训练脚本使用指南

## 快速开始

### 基本训练（ARX5平地）
```bash
./scripts/train_go2w_arm.sh --arx5-flat
```

### 带TensorBoard监控
```bash
./scripts/train_go2w_arm.sh --arx5-flat --tensorboard-dir /home/jay/unitree_rl_lab/logs/tensorboard
```

### 从检查点恢复训练
```bash
./scripts/train_go2w_arm.sh --checkpoint /home/jay/unitree_rl_lab/logs/checkpoints/epoch_100.ckpt
```

### 无头模式（服务器训练）
```bash
./scripts/train_go2w_arm.sh --arx5-flat --headless
```

### 指定环境数量
```bash
./scripts/train_go2w_arm.sh --arx5-flat --num-envs 8192
```

### 跳过配置验证
```bash
./scripts/train_go2w_arm.sh --arx5-flat --no-verify
```

---

## 完整选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| --arx5-flat | ARX5机械臂 - 平地环境 | 默认 |
| --arx5-rough | ARX5机械臂 - 粗糙地形 | - |
| --no-verify | 跳过配置验证 | false |
| --headless | 无头模式（适合服务器） | false |
| --no-tensorboard | 不启动TensorBoard | false |
| --tensorboard-dir=DIR | 指定TensorBoard日志目录 | logs/tensorboard |
| --checkpoint=PATH | 从检查点恢复训练 | 自动 |
| --epochs=N | 训练轮数 | 默认（由配置决定） |
| --num-envs=N | 环境数量 | 默认（由配置决定） |
| --help, -h | 显示帮助信息 | - |

---

## 训练流程

### 步骤1: 配置验证
- 运行`verify_config_only.py`
- 检查所有6个新奖励函数
- 检查历史观测配置
- 验证动作空间优化
- 确认机械臂策略

### 步骤2: 环境检查
- 检查Python环境
- 验证PyTorch安装
- 显示环境变量设置

### 步骤3: TensorBoard监控
- 自动检测现有TensorBoard进程
- 询问是否终止（如已运行）
- 后台启动TensorBoard（端口6006）
- 访问地址: http://localhost:6006

### 步骤4: 训练准备
- 创建日志目录
- 配置训练参数
- 显示训练前确认信息

### 步骤5: 启动训练
- 前台或后台启动训练
- 捕获训练进程ID
- 等待5秒确认启动

### 步骤6: 训练监控
- 实时监控命令提示
- 显示TensorBoard访问地址
- 优雅处理Ctrl+C信号
- 训练完成后显示统计信息

---

## TensorBoard监控重点

### 新增奖励函数监控

**1. upward_velocity（向上速度奖励）**
- 目标值: >0（有向上的速度）
- 预期: 初期较低（0.1-0.3），中期上升（0.4-0.6）
- 意义: 机器人学会了向上蹬地

**2. orientation_tracking（姿态跟踪奖励）**
- 目标值: >0.8（接近直立）
- 预期: 初期0.5-0.7，中期0.75-0.9
- 意义: 机器人保持直立姿态

**3. torque_penalty（扭矩惩罚）**
- 目标值: <0.005（接近零）
- 预期: 初期0.01-0.05，中期<0.01
- 意义: 无持续过载，电机健康

**4. joint_regularization（关节正则化）**
- 目标值: <0.1（接近零）
- 预期: 逐渐降低（0.3→0.2→0.1）
- 意义: 关节不卡在限位

**5. contact_management（接触管理）**
- 目标值: <0.1（接近零）
- 预期: 逐渐减少非期望接触
- 意义: 简化接触模式

**6. wheel_assisted_recovery（轮足协同）**
- 目标值: >0.1（有协同行为）
- 预期: 初期0，中期逐渐学习
- 意义: 学会利用轮子辅助恢复

### 性能指标

**episode_length（回合长度）**
- 初期: 8-12秒（学习基本站立）
- 中期: 15-18秒（稳定站立）
- 后期: >20秒（优化恢复策略）

**success_rate（成功率）**
- 初期: 40-50%（基本恢复能力）
- 中期: 60-75%（稳定恢复）
- 后期: >80%（高效恢复）

**total_reward（总奖励）**
- 初期: 100-200
- 中期: 200-400
- 后期: >400

---

## 训练阶段建议

### 阶段1: 基础站立（0-500K steps）
**目标**: 学会基本站立和保持

**重点监控**:
- orientation_tracking是否接近1.0
- episode_length是否增加
- 总奖励是否上升

**调优建议**:
- 降低upward_velocity权重（1.0-1.5）
- 增加orientation_tracking权重（1.5-2.0）
- 确保机械臂保持折叠

### 阶段2: 优化恢复（500K-1M steps）
**目标**: 提高恢复速度和成功率

**重点监控**:
- upward_velocity是否稳定激活
- torque_penalty是否保持低值
- wheel_assisted_recovery是否开始激活

**调优建议**:
- 恢复upward_velocity权重（2.0）
- 增加wheel_assisted_recovery权重（0.5-0.8）
- 微调torque_penalty参数

### 阶段3: 鲁棒性测试（1M-3M steps）
**目标**: 提高适应性和鲁棒性

**重点监控**:
- 不同地形下的稳定性
- 外部干扰后的恢复能力
- 总奖励的一致性

**调优建议**:
- 启用粗糙地形训练
- 添加随机推力
- 优化所有奖励权重

---

## 常见问题排查

### 问题: 训练无法启动
**解决方案**:
1. 检查Python环境: `python --version`
2. 检查PyTorch: `python -c "import torch; print(torch.__version__)"`
3. 查看错误日志: `cat logs/train.log`

### 问题: 配置验证失败
**解决方案**:
1. 检查velocity_env_cfg.py语法
2. 运行`python scripts/verify_config_only.py`
3. 检查所有6个奖励函数是否已配置

### 问题: TensorBoard无法访问
**解决方案**:
1. 检查TensorBoard进程: `pgrep -x tensorboard`
2. 查看TensorBoard日志: `cat logs/tensorboard/tensorboard.log`
3. 重新启动TensorBoard: 使用`--no-tensorboard`选项

### 问题: 机器人无法站立
**可能原因**:
- upward_velocity权重过低
- orientation_tracking权重不足
- 机械臂未保持折叠

**解决方案**:
- 增加upward_velocity权重（1.5-3.0）
- 增加orientation_tracking权重（2.0-2.5）
- 确认arm_joint2-6不在动作空间

### 问题: 扭矩过高
**可能原因**:
- torque_penalty权重过高
- sustained_window参数过小

**解决方案**:
- 降低torque_penalty权重（-0.005）
- 增加sustained_window（3.0-5.0）
- 增加burst_threshold（2.0-3.0）

### 问题: 机械臂摆动
**可能原因**:
- arm_joint1在动作空间且scale过大
- 未设置arm_joint2-6为固定

**解决方案**:
- 减小arm_joint1的scale（0.05）
- 确认arm_joint2-6不在动作空间
- 增加arm_stability权重

---

## 训练完成检查清单

### 代码质量
- [ ] 配置验证通过
- [ ] 所有新奖励函数已配置
- [ ] 历史观测已添加
- [ ] 动作空间已优化
- [ ] TensorBoard正常运行

### 训练质量
- [ ] episode_length稳定增加
- [ ] orientation_tracking >0.8
- [ ] torque_penalty <0.01
- [ ] contact_management <0.1
- [ ] wheel_assisted_recovery >0.1

### 部署准备
- [ ] 检查点保存正常
- [ ] 评估策略在测试集表现
- [ ] TensorBoard日志保存完整
- [ ] 训练时间文档化

---

## 进阶技巧

### 并行训练
```bash
# 使用不同的随机种子启动多个训练
./scripts/train_go2w_arm.sh --arx5-flat --seed 1 --num-envs 4096 &
./scripts/train_go2w_arm.sh --arx5-flat --seed 2 --num-envs 4096 &
./scripts/train_go2w_arm.sh --arx5-flat --seed 3 --num-envs 4096 &
./scripts/train_go2w_arm.sh --arx5-flat --seed 4 --num-envs 4096 &
```

### 课程学习策略
1. **奖励权重课程**: 根据训练进度逐步调整
2. **地形难度课程**: 从平地→小起伏→中等起伏→大起伏
3. **随机化强度课程**: 从弱→中等→强

### 超参数优化
- 增加--num-envs参数（8192, 16384）
- 调整学习率和批次大小
- 使用梯度累积和混合精度训练

---

## 相关文档

- [GO2W_ARM_COMPREHENSIVE_OPTIMIZATION.md](GO2W_ARM_COMPREHENSIVE_OPTIMIZATION.md) - 完整优化方案
- [GO2W_ARM_CHANGES_SUMMARY.md](GO2W_ARM_CHANGES_SUMMARY.md) - 实施总结
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - 完成报告
- [verify_config_only.py](../scripts/verify_config_only.py) - 配置验证
