#!/usr/bin/env python3
"""
TensorBoard 训练趋势可视化分析
"""

import os
import sys
from pathlib import Path
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("请安装 tensorboard: pip install tensorboard")
    sys.exit(1)

def get_metric_data(ea, tag):
    """获取指标数据"""
    try:
        events = ea.Scalars(tag)
        if not events:
            return None, None
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except:
        return None, None

def analyze_trends(log_dir):
    """分析训练趋势"""

    log_path = Path(log_dir)
    event_files = sorted(log_path.rglob("events.out.tfevents.*"))

    # 分析最新的训练运行
    latest_file = event_files[-1]

    print("=" * 80)
    print("GO2W_ARM 训练趋势分析")
    print("=" * 80)
    print(f"\n分析文件: {latest_file.relative_to(log_path)}\n")

    # 加载事件数据
    ea = event_accumulator.EventAccumulator(str(latest_file))
    ea.Reload()

    # 1. 分析动态刹车趋势
    print("🚧 动态刹车趋势分析")
    print("-" * 80)

    steps, action_rate = get_metric_data(ea, 'Episode_Reward/action_rate_l2')
    if steps and action_rate:
        print(f"动作变化率惩罚 (action_rate_l2):")
        print(f"  初始值: {action_rate[0]:.6f}")
        print(f"  最新值: {action_rate[-1]:.6f}")
        print(f"  平均值: {np.mean(action_rate):.6f}")
        print(f"  增长倍数: {action_rate[-1] / action_rate[0]:.2f}x")

        # 分析增长趋势
        if len(action_rate) > 10:
            early_avg = np.mean(action_rate[:len(action_rate)//4])
            late_avg = np.mean(action_rate[-len(action_rate)//4:])
            growth = (late_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
            print(f"  趋势分析: 早期平均 {early_avg:.6f} → 晚期平均 {late_avg:.6f}")
            print(f"  增长率: {growth*100:.1f}%")

            if growth > 1.0:
                print(f"  ⚠️  动作变化率快速增长，可能说明策略在过度探索")
            elif growth > 0.1:
                print(f"  ✓ 动作变化率适度增长，探索合理")
            else:
                print(f"  ⚠️  动作变化率增长缓慢，探索可能不足")

    # 2. 分析角动量阻尼趋势
    print(f"\n🌪️ 角动量阻尼趋势分析")
    print("-" * 80)

    steps, damping = get_metric_data(ea, 'Episode_Reward/angular_momentum_damping')
    if steps and damping:
        print(f"角动量阻尼惩罚:")
        print(f"  初始值: {damping[0]:.6f}")
        print(f"  最新值: {damping[-1]:.6f}")
        print(f"  平均值: {np.mean(damping):.6f}")
        print(f"  最大值: {max(damping):.6f}")

        # 分析阻尼激活情况
        activation_count = sum(1 for v in damping if v > 0.001)
        print(f"  强阻尼激活次数 (>0.001): {activation_count}/{len(damping)} ({100*activation_count/len(damping):.1f}%)")

        if activation_count > len(damping) * 0.5:
            print(f"  ⚠️  阻尼频繁激活，说明机器人有持续的角速度")
            print(f"  → 阻尼机制可能强度不够，或激活阈值需要调整")
        elif activation_count > 0:
            print(f"  ✓ 阻尼偶尔激活，角速度控制基本正常")
        else:
            print(f"  ✓ 阻尼几乎不激活，角速度控制良好")

    # 3. 分析高度趋势
    print(f"\n📏 高度变化趋势")
    print("-" * 80)

    steps, height = get_metric_data(ea, 'Episode_Reward/base_height_l2')
    if steps and height:
        print(f"高度惩罚:")
        print(f"  初始值: {height[0]:.6f}")
        print(f"  最新值: {height[-1]:.6f}")
        print(f"  平均值: {np.mean(height):.6f}")
        print(f"  估计初始高度: {0.6 - np.sqrt(-height[0]) if height[0] < 0 else 0.6:.3f} m")
        print(f"  估计最新高度: {0.6 - np.sqrt(-height[-1]) if height[-1] < 0 else 0.6:.3f} m")

        # 分析高度改善趋势
        if len(height) > 10:
            early_avg = np.mean(height[:len(height)//4])
            late_avg = np.mean(height[-len(height)//4:])
            improvement = (late_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
            print(f"  趋势分析: 早期平均 {early_avg:.6f} → 晚期平均 {late_avg:.6f}")
            print(f"  改善率: {improvement*100:.1f}%")

            if improvement > 0.1:
                print(f"  ✓ 高度在改善，机器人正在学习站立")
            elif improvement > -0.1:
                print(f"  ⚠️  高度无明显改善，学习停滞")
            else:
                print(f"  ❌ 高度在恶化，机器人正在'学会'倒下")

    # 4. 分析关节限制趋势
    print(f"\n🦾 关节限制趋势")
    print("-" * 80)

    steps, joint_limit = get_metric_data(ea, 'Episode_Reward/joint_pos_limits')
    if steps and joint_limit:
        print(f"关节位置限制惩罚:")
        print(f"  初始值: {joint_limit[0]:.6f}")
        print(f"  最新值: {joint_limit[-1]:.6f}")
        print(f"  平均值: {np.mean(joint_limit):.6f}")

        # 分析严重触碰极限的次数
        severe_limit_count = sum(1 for v in joint_limit if v < -1.0)
        print(f"  严重触碰极限次数 (<-1.0): {severe_limit_count}/{len(joint_limit)} ({100*severe_limit_count/len(joint_limit):.1f}%)")

        if severe_limit_count > len(joint_limit) * 0.5:
            print(f"  ❌ 频繁严重触碰关节极限，动作空间严重受限")
            print(f"  → 可能需要调整关节初始位置或放宽关节限制")
        elif severe_limit_count > 0:
            print(f"  ⚠️  偶尔触碰关节极限，需要注意")
        else:
            print(f"  ✓ 关节限制控制良好")

    # 5. 分析课程进展
    print(f"\n📚 课程进展趋势")
    print("-" * 80)

    steps, curriculum = get_metric_data(ea, 'Curriculum/posture_curriculum')
    if steps and curriculum:
        print(f"课程级别:")
        print(f"  初始级别: {curriculum[0]:.2f}")
        print(f"  最新级别: {curriculum[-1]:.2f}")
        print(f"  历史最大级别: {max(curriculum):.2f}")

        if max(curriculum) > 0:
            print(f"  ✓ 课程有进展，经历了 {len(set([int(v) for v in curriculum]))} 个不同级别")
        else:
            print(f"  ⚠️  课程从未升级，一直停留在 Level 0")
            print(f"  → 机器人未达到升级条件（存活率、站立率等）")

    # 6. 分析训练稳定性
    print(f"\n📊 训练稳定性分析")
    print("-" * 80)

    # 分析超时率稳定性
    steps, timeout = get_metric_data(ea, 'Episode_Termination/time_out')
    if steps and timeout:
        timeout_std = np.std(timeout)
        timeout_mean = np.mean(timeout)
        print(f"超时终止率:")
        print(f"  平均值: {timeout_mean:.4f}")
        print(f"  标准差: {timeout_std:.4f}")
        print(f"  变异系数: {timeout_std/timeout_mean*100:.1f}%")

        if timeout_std / timeout_mean < 0.1:
            print(f"  ✓ 训练稳定，超时率一致")
        elif timeout_std / timeout_mean < 0.3:
            print(f"  ⚠️  训练基本稳定，有一定波动")
        else:
            print(f"  ❌ 训练不稳定，超时率波动很大")

    # 7. 三段式动态刹车效果分析
    print(f"\n🚦 三段式动态刹车效果分析")
    print("-" * 80)

    if steps and action_rate:
        # 假设 Z < 0.5 时动作变化率应该很小
        # Z 在 0.5-0.85 之间时应该逐渐增加
        # Z >= 0.85 时应该最大

        print(f"理论预期:")
        print(f"  Z < 0.5 (倒地): action_rate ≈ 0 (极小惩罚，允许探索)")
        print(f"  0.5 ≤ Z < 0.85 (过渡): action_rate 逐渐增加 (平滑过渡)")
        print(f"  Z ≥ 0.85 (站立): action_rate 最大 (全额惩罚，收敛动作)")

        print(f"\n实际观察:")
        print(f"  action_rate 从 {action_rate[0]:.6f} 增长到 {action_rate[-1]:.6f}")
        print(f"  增长了 {action_rate[-1] / action_rate[0]:.2f} 倍")

        if action_rate[-1] > 0.005:
            print(f"  ✓ 动作变化率已经达到较高水平")
            print(f"  → 说明策略在尝试探索，但可能还没有学会如何有效利用三段式机制")
        else:
            print(f"  ⚠️  动作变化率仍然较低")
            print(f"  → 可能是机器人始终处于倒地状态（Z < 0.5），刹车机制未完全激活")

    # 8. 综合分析
    print(f"\n🎯 综合分析")
    print("=" * 80)

    print("""
基于趋势分析，可以得出以下结论：

1. **动态刹车机制**:
   - 动作变化率在增长，说明策略在探索
   - 但增长可能是由于倒地状态的随机探索，而非有效的站立尝试
   - 三段式机制可能没有正确触发，因为机器人从未达到 Z > 0.5

2. **角动量阻尼**:
   - 阻尼惩罚存在且偶尔激活
   - 说明机器人有角速度，但阻尼未能完全抑制翻滚
   - 可能是因为机器人始终处于倒地状态，阻尼机制（Z > 0.8 才激活）未触发

3. **高度控制**:
   - 高度惩罚始终为负且数值较大
   - 估计高度约 0.1m，远低于目标 0.6m
   - 高度没有明显改善趋势，说明机器人没有学会如何增加高度

4. **关节限制**:
   - 关节位置限制惩罚较大，说明频繁触碰极限
   - 这可能是机器人无法站立的关键原因
   - 当机器人尝试站立时，关节很快就达到极限，无法继续发力

5. **课程系统**:
   - 课程级别始终为 0，从未升级
   - 说明机器人未达到任何升级条件
   - 课程系统本身工作正常，但机器人没有达到性能要求

6. **训练稳定性**:
   - 超时率接近 100%，说明几乎所有的 episode 都是因为超时而终止
   - 训练基本稳定，但结果是一致的失败

**根本原因推测**:

1. **初始姿态问题**: ±0.8 rad 的 roll 范围过大，机器人从侧卧状态开始
2. **关节限制冲突**: 当机器人尝试站立时，关节很快就达到极限
3. **轮子未利用**: 轮足机器人可能需要利用轮子产生推力，但策略可能没有学会
4. **动作空间不足**: 可能需要更大的动作范围或不同的动作表示
5. **奖励信号引导不足**: 高度惩罚是负的，没有正向的"站立尝试"奖励

**动态刹车、角动量阻尼、驻留成功的影响**:

- 这些机制本身设计合理，但由于机器人从未达到触发条件（Z > 0.5 或 Z > 0.8），所以几乎没有发挥实际作用
- 它们的问题是"门槛太高"，要求机器人先达到一定姿态才能激活
- 但机器人正是因为无法达到这些姿态才需要这些机制的帮助
- 这形成了一个"鸡生蛋、蛋生鸡"的死循环

**多级姿态恢复课程的影响**:

- 课程系统工作正常，但由于机器人性能太差，始终停留在 Level 0
- 这实际上是一个"保护机制"，防止机器人过早进入更难的姿态
- 但也意味着训练效率低下，机器人没有机会学习更难的情况
    """)


def main():
    log_dir = "/home/jay/unitree_rl_lab/logs/rsl_rl/unitree_go2warm_twostage_recovery_v0"
    analyze_trends(log_dir)


if __name__ == "__main__":
    main()
