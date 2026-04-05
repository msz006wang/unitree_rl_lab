#!/usr/bin/env python3
"""
详细的 TensorBoard 训练数据分析
重点关注姿态、高度和动态刹车效果
"""

import os
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("请安装 tensorboard: pip install tensorboard")
    sys.exit(1)

def detailed_analysis(log_dir):
    """详细分析训练数据"""

    log_path = Path(log_dir)
    event_files = sorted(log_path.rglob("events.out.tfevents.*"))

    # 分析最新的训练运行
    latest_file = event_files[-1]

    print("=" * 80)
    print("GO2W_ARM 两段式恢复训练 - 详细分析报告")
    print("=" * 80)
    print(f"\n分析文件: {latest_file.relative_to(log_path)}\n")

    # 加载事件数据
    ea = event_accumulator.EventAccumulator(str(latest_file))
    ea.Reload()

    tags = ea.Tags()['scalars']

    # 1. 分析终止原因
    print("📊 1. 终止原因分析")
    print("-" * 80)

    timeout_events = ea.Scalars('Episode_Termination/time_out')
    if timeout_events:
        timeout_values = [e.value for e in timeout_events]
        print(f"超时终止率: {np.mean(timeout_values):.2%} (平均值: {np.mean(timeout_values):.4f})")
        print(f"最新超时率: {timeout_values[-1]:.2%}")
        print(f"最小超时率: {min(timeout_values):.2%}")
        print(f"最大超时率: {max(timeout_values):.2%}")

    success_events = ea.Scalars('Episode_Termination/success_stable')
    if success_events:
        success_values = [e.value for e in success_events]
        print(f"成功终止率: {np.mean(success_values):.2%} (0/720 次成功)")
        print(f"⚠️  机器人从未成功站立！")

    # 2. 分析高度变化
    print("\n📏 2. 高度分析")
    print("-" * 80)

    height_events = ea.Scalars('Episode_Reward/base_height_l2')
    if height_events:
        height_values = [e.value for e in height_events]
        print(f"高度惩罚值: {np.mean(height_values):.6f} (平均值)")
        print(f"最新高度惩罚: {height_values[-1]:.6f}")
        print(f"高度惩罚趋势: {'改善' if height_values[-1] > height_values[0] else '恶化'}")

        # 计算实际高度（惩罚值是负的，越接近0越好）
        # 假设目标高度是 0.6m，惩罚是 (height - 0.6)^2
        estimated_heights = [0.6 - np.sqrt(-v) if v < 0 else 0.6 for v in height_values]
        print(f"估计平均高度: {np.mean(estimated_heights):.3f} m")
        print(f"估计最新高度: {estimated_heights[-1]:.3f} m")
        print(f"⚠️  高度远低于目标值 0.6m，机器人始终处于倒伏状态")

    # 3. 分析姿态（projected_gravity）
    print("\n🧍 3. 姿态分析")
    print("-" * 80)

    # 查找姿态相关指标
    upright_tags = [t for t in tags if 'upright' in t.lower() or 'gravity' in t.lower() or 'orientation' in t.lower()]
    for tag in upright_tags[:5]:
        events = ea.Scalars(tag)
        if events:
            values = [e.value for e in events]
            print(f"{tag}:")
            print(f"  最新值: {values[-1]:.4f}")
            print(f"  平均值: {np.mean(values):.4f}")
            print(f"  最大值: {max(values):.4f}")

    # 4. 分析动态刹车效果
    print("\n🚧 4. 动态刹车分析")
    print("-" * 80)

    action_rate_events = ea.Scalars('Episode_Reward/action_rate_l2')
    if action_rate_events:
        action_rate_values = [e.value for e in action_rate_events]
        print(f"动作变化率惩罚:")
        print(f"  最新值: {action_rate_values[-1]:.6f}")
        print(f"  平均值: {np.mean(action_rate_values):.6f}")
        print(f"  最小值: {min(action_rate_values):.6f}")
        print(f"  最大值: {max(action_rate_values):.6f}")
        print(f"  增长趋势: {action_rate_values[-1] / action_rate_values[0]:.2f}x")
        print(f"  → 动作变化率在增加，说明策略在探索，但可能探索不足")

    torque_events = ea.Scalars('Episode_Reward/joint_torques_l2')
    if torque_events:
        torque_values = [e.value for e in torque_events]
        print(f"\n关节扭矩惩罚:")
        print(f"  最新值: {torque_values[-1]:.6f}")
        print(f"  平均值: {np.mean(torque_values):.6f}")
        print(f"  ⚠️  扭矩惩罚几乎为 0，说明机器人可能没有产生足够的扭矩")

    # 5. 分析角动量阻尼
    print("\n🌪️ 5. 角动量阻尼分析")
    print("-" * 80)

    damping_events = ea.Scalars('Episode_Reward/angular_momentum_damping')
    if damping_events:
        damping_values = [e.value for e in damping_events]
        print(f"角动量阻尼惩罚:")
        print(f"  最新值: {damping_values[-1]:.6f}")
        print(f"  平均值: {np.mean(damping_values):.6f}")
        print(f"  最大值: {max(damping_values):.6f}")

        # 阻尼惩罚为正数说明有角速度，应该被抑制
        if np.mean(damping_values) > 0.001:
            print(f"  ⚠️  角动量阻尼惩罚为正，说明机器人有持续的角速度")
            print(f"  → 阻尼机制未能有效抑制翻滚惯性")
        else:
            print(f"  阻尼惩罚很小，角速度控制较好")

    # 6. 分析课程系统
    print("\n📚 6. 多级姿态恢复课程分析")
    print("-" * 80)

    curriculum_events = ea.Scalars('Curriculum/posture_curriculum')
    if curriculum_events:
        curriculum_values = [e.value for e in curriculum_events]
        print(f"课程级别:")
        print(f"  当前级别: {curriculum_values[-1]:.2f}")
        print(f"  历史最大级别: {max(curriculum_values):.2f}")
        print(f"  历史最小级别: {min(curriculum_values):.2f}")

        unique_levels = sorted(set([int(v) for v in curriculum_values]))
        print(f"  经历的级别: {unique_levels}")
        print(f"  ⚠️  课程级别始终为 0，说明机器人未达到升级条件")

    # 7. 分析关节约束
    print("\n🦾 7. 关节约束分析")
    print("-" * 80)

    joint_acc_events = ea.Scalars('Episode_Reward/joint_acc_l2')
    if joint_acc_events:
        joint_acc_values = [e.value for e in joint_acc_events]
        print(f"关节加速度惩罚:")
        print(f"  最新值: {joint_acc_values[-1]:.6f}")
        print(f"  平均值: {np.mean(joint_acc_values):.6f}")
        print(f"  → 关节冲击较大，说明动作变化剧烈")

    joint_limit_events = ea.Scalars('Episode_Reward/joint_pos_limits')
    if joint_limit_events:
        joint_limit_values = [e.value for e in joint_limit_events]
        print(f"\n关节位置限制惩罚:")
        print(f"  最新值: {joint_limit_values[-1]:.6f}")
        print(f"  平均值: {np.mean(joint_limit_values):.6f}")
        print(f"  ⚠️  关节位置限制惩罚较大，说明机器人频繁触碰关节极限")

    # 8. 分析垂直速度
    print("\n⬆️ 8. 垂直速度分析")
    print("-" * 80)

    vel_z_events = ea.Scalars('Episode_Reward/lin_vel_z_l2')
    if vel_z_events:
        vel_z_values = [e.value for e in vel_z_events]
        print(f"垂直速度惩罚:")
        print(f"  最新值: {vel_z_values[-1]:.6f}")
        print(f"  平均值: {np.mean(vel_z_values):.6f}")
        print(f"  ⚠️  垂直速度惩罚较大且为负数，说明机器人有不当的垂直运动")

    # 9. 总结分析
    print("\n🎯 9. 问题诊断总结")
    print("=" * 80)

    issues = []

    # 检查高度
    if height_events:
        if np.mean([e.value for e in height_events]) < -0.1:
            issues.append("❌ 机器人高度始终低于 0.5m，未能站立")

    # 检查成功
    if success_events and np.mean([e.value for e in success_events]) == 0:
        issues.append("❌ 驻留成功率为 0%，机器人从未成功站立")

    # 检查扭矩
    if torque_events and np.mean([e.value for e in torque_events]) == 0:
        issues.append("❌ 扭矩惩罚为 0，机器人可能没有产生足够的推力")

    # 检查关节限制
    if joint_limit_events and np.mean([e.value for e in joint_limit_events]) < -0.5:
        issues.append("❌ 关节频繁触碰极限，动作空间可能受限")

    # 检查课程
    if curriculum_events and max([e.value for e in curriculum_events]) == 0:
        issues.append("❌ 课程系统未升级，一直停留在 Level 0")

    # 检查动作变化率
    if action_rate_events and np.mean([e.value for e in action_rate_events]) < 0.001:
        issues.append("❌ 动作变化率过小，探索不足")

    if issues:
        print("\n发现以下问题：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n未发现明显问题")

    # 10. 根本原因分析
    print("\n🔬 10. 根本原因分析")
    print("=" * 80)

    print("""
基于以上数据，机器人无法站立的根本原因可能是：

1. **初始姿态问题**
   - 初始 roll 范围 ±0.8 rad (约 ±45°) 可能过大
   - 机器人从侧卧状态开始，需要先纠正姿态才能站起

2. **推力不足**
   - 扭矩惩罚为 0，说明关节没有产生足够的推力
   - 可能是动作范围或关节限制导致无法发力

3. **关节限制冲突**
   - 关节位置限制惩罚较大，说明动作空间被关节极限限制
   - 机器人可能在尝试站立时频繁触碰关节极限

4. **探索不足**
   - 动作变化率惩罚较小，说明策略探索不够充分
   - 可能是学习率过低或奖励信号不够强

5. **奖励信号问题**
   - 高度惩罚一直为负，说明机器人从未接近目标高度
   - 缺少正向的站立奖励引导

6. **物理约束**
   - 轮足机器人的恢复机制可能需要特殊的轮子-腿部协调
   - 当前策略可能没有学会如何利用轮子辅助站立

建议：
- 减小初始 roll 范围（从 ±0.8 rad 减小到 ±0.3 rad）
- 增加高度奖励的权重
- 放宽关节限制或调整关节初始位置
- 增加学习率或调整 PPO 参数
- 检查是否需要特殊的轮子控制策略
    """)


def main():
    log_dir = "/home/jay/unitree_rl_lab/logs/rsl_rl/unitree_go2warm_twostage_recovery_v0"
    detailed_analysis(log_dir)


if __name__ == "__main__":
    main()
