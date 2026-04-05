#!/usr/bin/env python3
"""
TensorBoard 训练数据分析脚本
分析 GO2W_ARM 两段式恢复训练的各项指标
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("请安装 tensorboard: pip install tensorboard")
    sys.exit(1)

def analyze_tensorboard_logs(log_dir):
    """分析 TensorBoard 日志"""

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        return

    # 查找所有 events 文件
    event_files = sorted(log_path.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"错误: 未找到 TensorBoard events 文件在 {log_dir}")
        return

    print(f"找到 {len(event_files)} 个训练运行\n")

    for event_file in event_files:
        print("=" * 80)
        print(f"分析: {event_file.relative_to(log_path)}")
        print("=" * 80)

        # 加载事件数据
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()

        # 获取所有标量标签
        tags = ea.Tags()['scalars']

        # 按类别分组
        reward_tags = [t for t in tags if 'Episode_Reward' in t or 'reward' in t.lower()]
        termination_tags = [t for t in tags if 'Episode_Termination' in t or 'termination' in t.lower()]
        curriculum_tags = [t for t in tags if 'curriculum' in t.lower() or 'level' in t.lower()]
        brake_tags = [t for t in tags if 'brake' in t.lower() or 'action_rate' in t.lower() or 'torque' in t.lower()]
        damping_tags = [t for t in tags if 'angular' in t.lower() or 'momentum' in t.lower() or 'damping' in t.lower()]
        success_tags = [t for t in tags if 'success' in t.lower()]

        print(f"\n📊 总计 {len(tags)} 个指标标签")

        # 分析每个类别
        if success_tags:
            print(f"\n✅ 驻留成功指标 ({len(success_tags)}):")
            for tag in sorted(success_tags):
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.4f}")
                    print(f"    最大值: {max(values):.4f}")
                    print(f"    平均值: {np.mean(values):.4f}")
                    print(f"    数据点数: {len(events)}")

        if brake_tags:
            print(f"\n🚧 动态刹车指标 ({len(brake_tags)}):")
            for tag in sorted(brake_tags):
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.6f}")
                    print(f"    最小值: {min(values):.6f}")
                    print(f"    最大值: {max(values):.6f}")
                    print(f"    平均值: {np.mean(values):.6f}")
                    print(f"    数据点数: {len(events)}")

        if damping_tags:
            print(f"\n🌪️ 角动量阻尼指标 ({len(damping_tags)}):")
            for tag in sorted(damping_tags):
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.6f}")
                    print(f"    最小值: {min(values):.6f}")
                    print(f"    最大值: {max(values):.6f}")
                    print(f"    平均值: {np.mean(values):.6f}")
                    print(f"    数据点数: {len(events)}")

        if curriculum_tags:
            print(f"\n📚 多级姿态恢复课程指标 ({len(curriculum_tags)}):")
            for tag in sorted(curriculum_tags):
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.4f}")
                    print(f"    最小值: {min(values):.4f}")
                    print(f"    最大值: {max(values):.4f}")
                    print(f"    平均值: {np.mean(values):.4f}")
                    print(f"    数据点数: {len(events)}")

        if termination_tags:
            print(f"\n⏹️ 终止指标 ({len(termination_tags)}):")
            for tag in sorted(termination_tags):
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.4f}")
                    print(f"    最大值: {max(values):.4f}")
                    print(f"    数据点数: {len(events)}")

        if reward_tags:
            print(f"\n🎁 奖励指标 ({len(reward_tags)}):")
            for tag in sorted(reward_tags)[:10]:  # 只显示前10个
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"  {tag}:")
                    print(f"    最新值: {values[-1]:.4f}")
                    print(f"    平均值: {np.mean(values):.4f}")
                    print(f"    数据点数: {len(events)}")

        # 分析特定关键指标的趋势
        print(f"\n🔍 关键指标详细分析:")

        # 1. 检查是否成功站立
        success_stable_tags = [t for t in success_tags if 'stable' in t.lower()]
        if success_stable_tags:
            print(f"\n  驻留成功:")
            for tag in success_stable_tags:
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    success_count = sum(1 for v in values if v > 0)
                    print(f"    {tag}:")
                    print(f"      成功次数: {success_count}/{len(events)} ({100*success_count/len(events):.2f}%)")
                    if success_count > 0:
                        success_indices = [i for i, v in enumerate(values) if v > 0]
                        print(f"      首次成功在第 {success_indices[0]+1} 步")
                        print(f"      最后成功在第 {success_indices[-1]+1} 步")
                    else:
                        print(f"      ⚠️  未检测到任何成功！")

        # 2. 分析姿态指标
        upright_tags = [t for t in tags if 'upright' in t.lower() or 'projected_gravity' in t.lower()]
        if upright_tags:
            print(f"\n  姿态指标:")
            for tag in upright_tags:
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"    {tag}:")
                    print(f"      最新值: {values[-1]:.4f}")
                    print(f"      平均值: {np.mean(values):.4f}")
                    upright_count = sum(1 for v in values if v > 0.85)
                    print(f"      直立次数 (Z>0.85): {upright_count}/{len(events)} ({100*upright_count/len(events):.2f}%)")

        # 3. 分析高度指标
        height_tags = [t for t in tags if 'height' in t.lower()]
        if height_tags:
            print(f"\n  高度指标:")
            for tag in height_tags:
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"    {tag}:")
                    print(f"      最新值: {values[-1]:.4f}")
                    print(f"      平均值: {np.mean(values):.4f}")
                    high_count = sum(1 for v in values if v > 0.6)
                    print(f"      高度>0.6m次数: {high_count}/{len(events)} ({100*high_count/len(events):.2f}%)")

        # 4. 分析动作和扭矩
        action_tags = [t for t in tags if 'action' in t.lower()]
        if action_tags:
            print(f"\n  动作指标:")
            for tag in action_tags[:5]:
                events = ea.Scalars(tag)
                if events:
                    values = [e.value for e in events]
                    print(f"    {tag}:")
                    print(f"      最新值: {values[-1]:.6f}")
                    print(f"      平均值: {np.mean(values):.6f}")

        print(f"\n{'=' * 80}\n")


def main():
    log_dir = "/home/jay/unitree_rl_lab/logs/rsl_rl/unitree_go2warm_twostage_recovery_v0"
    analyze_tensorboard_logs(log_dir)


if __name__ == "__main__":
    main()
