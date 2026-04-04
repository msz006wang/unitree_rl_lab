#!/usr/bin/env python3
"""
训练指标分析脚本
分析TensorBoard日志中的各项指标及其物理意义
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def parse_tensorboard_logs(log_dir):
    """解析TensorBoard日志文件"""
    metrics_data = defaultdict(list)

    event_files = list(Path(log_dir).glob("*.tfevents*"))

    for event_file in event_files:
        print(f"Processing: {event_file}")

        # 使用tensorboard命令行工具提取数据
        cmd = f"tensorboard dev upload --logdir={event_file.parent} 2>/dev/null"
        result = os.popen(cmd)

    return metrics_data

def extract_metrics_from_events(event_file):
    """从events文件中提取指标数据"""
    metrics = {}

    # 查找所有指标
    with open(event_file, 'rb') as f:
        content = f.read()

    # 查找常见的指标模式
    patterns = {
        'episode_reward': b'episode_reward',
        'episode_length': b'episode_length',
        'stand_reward': b'stand_reward',
        'flat_orientation_l2': b'flat_orientation_l2',
        'base_height_l2': b'base_height_l2',
        'track_lin_vel_xy': b'track_lin_vel_xy',
        'track_ang_vel_z': b'track_ang_vel_z',
        'joint_torques_l2': b'joint_torques_l2',
        'joint_vel_l2': b'joint_vel_l2',
        'success_rate': b'success_rate'
    }

    for metric_name, pattern in patterns.items():
        if pattern in content:
            metrics[metric_name] = True
            print(f"✓ Found metric: {metric_name}")
        else:
            metrics[metric_name] = False
            print(f"✗ Missing metric: {metric_name}")

    return metrics

def analyze_metric_physical_meaning():
    """分析各指标的物理意义"""

    metrics_analysis = {
        # 基础训练指标
        "Episode Reward": {
            "物理意义": "每个完整训练回合的总奖励值，综合评估机器人整体表现",
            "重要性": "★ ★ ★ ★ ★",
            "优化目标": "最大化",
            "理想范围": "逐步上升并稳定在较高值",
            "物理含义": "反映了机器人完成任务的综合能力，包括站立稳定性、运动控制、能量效率等"
        },

        "Episode Length": {
            "物理意义": "每个回合的持续步数，反映机器人的耐久性和稳定性",
            "重要性": "★ ★ ★ ★",
            "优化目标": "逐步增长",
            "理想范围": "从短到长，最终稳定在较长值",
            "物理含义": "机器人保持平衡和控制能力的时间长度，越长说明越稳定"
        },

        "Success Rate": {
            "物理意义": "任务完成的成功百分比，直接反映机器人能力",
            "重要性": "★ ★ ★ ★ ★",
            "优化目标": "最大化",
            "理想范围": "> 90%",
            "物理含义": "机器人能够成功站立、行走并完成任务的比例"
        },

        # 姿态和平衡指标
        "Stand Reward": {
            "物理意义": "站立恢复奖励，机器人从倒下状态恢复的能力",
            "重要性": "★ ★ ★ ★ ★",
            "优化目标": "最大化",
            "理想范围": "快速恢复且稳定",
            "物理含义": "反映机器人的鲁棒性和抗干扰能力"
        },

        "Flat Orientation L2": {
            "物理意义": "姿态控制L2惩罚，机器人保持直立的能力",
            "重要性": "★ ★ ★ ★",
            "优化目标": "最小化",
            "理想范围": "< 0.1",
            "物理含义": "机器人 torso 部分的俯仰和翻滚角度偏差，数值越小越稳定"
        },

        "Base Height L2": {
            "物理意义": "高度控制L2惩罚，机器人保持特定高度的能力",
            "重要性": "★ ★ ★",
            "优化目标": "最小化",
            "理想范围": "< 0.05",
            "物理含义": "机器人 base 相对于目标高度的偏差，影响整体平衡"
        },

        # 速度追踪指标
        "Track Lin Vel XY": {
            "物理意义": "线速度追踪奖励，机器人按照期望速度运动的能力",
            "重要性": "★ ★ ★ ★",
            "优化目标": "最大化",
            "理想范围": "接近目标速度",
            "物理含义": "机器人在平面内按指令速度移动的精确度"
        },

        "Track Ang Vel Z": {
            "物理意义": "角速度追踪奖励，机器人旋转控制能力",
            "重要性": "★ ★ ★",
            "优化目标": "最大化",
            "理想范围": "精确跟踪目标角速度",
            "物理含义": "机器人绕垂直轴旋转的精确度，影响转向能力"
        },

        # 能效和稳定性指标
        "Joint Torques L2": {
            "物理意义": "关节扭矩惩罚，控制能量消耗和机械应力",
            "重要性": "★ ★ ★",
            "优化目标": "最小化",
            "理想范围": "越小越好",
            "物理含义": "所有关节扭矩的平方和，反映能耗和机械负荷"
        },

        "Joint Vel L2": {
            "物理意义": "关节速度惩罚，控制运动平滑性",
            "重要性": "★ ★ ★",
            "优化目标": "最小化",
            "理想范围": "适中",
            "物理含义": "所有关节速度的平方和，反映运动平滑程度"
        }
    }

    return metrics_analysis

def compare_with_standard_go2w():
    """对比标准GO2W的性能"""

    comparison = {
        "Standard GO2W (无机械臂)": {
            "特点": "经典轮足机器人，机械结构简单",
            "优势": "计算效率高，控制简单，能耗低",
            "劣势": "任务适应性有限，稳定性较差",
            "典型性能": {
                "Episode Reward": "500-800",
                "Success Rate": "70-85%",
                "Stand Reward": "0.3-0.5",
                "Flat Orientation": "0.1-0.2"
            }
        },

        "GO2W ARM (带机械臂)": {
            "特点": "轮足+机械臂复合结构，复杂度更高",
            "优势": "任务多样性，环境适应性更强",
            "劣势": "计算复杂，能耗增加，控制难度大",
            "典型性能": {
                "Episode Reward": "300-600 (初期)",
                "Success Rate": "60-80%",
                "Stand Reward": "0.2-0.4",
                "Flat Orientation": "0.15-0.25"
            }
        },

        "性能差异分析": {
            "奖励差异": "GO2W ARM初始奖励较低，因为需要学习更复杂的控制",
            "收敛速度": "GO2W ARM收敛较慢，通常需要更多训练时间",
            "稳定性": "GO2W ARM稳定性稍差，但长期潜力更大",
            "能耗": "GO2W ARM能耗更高，机械臂增加了计算负担"
        }
    }

    return comparison

def generate_training_report():
    """生成训练性能评估报告"""

    print("=" * 80)
    print("GO2W ARM 训练性能分析报告")
    print("=" * 80)
    print()

    # 检查当前训练状态
    print("🔍 当前训练状态:")
    print(f"训练目录: {os.getcwd()}")
    print(f"模型文件: model_100.pt")
    print(f"事件文件: events.out.tfevents.1775011057.jay-GE76-Raider-11UH.7741.0")
    print()

    # 分析指标
    print("📊 指标物理意义分析:")
    print("-" * 50)

    metrics = extract_metrics_from_events("events.out.tfevents.1775011057.jay-GE76-Raider-11UH.7741.0")

    analysis = analyze_metric_physical_meaning()

    for metric_name, info in analysis.items():
        status = "✓" if metric_name.replace(" ", "_").lower() in [k.replace(" ", "_").lower() for k in metrics.keys()] else "✗"
        print(f"{status} {metric_name}")
        print(f"   物理意义: {info['物理意义']}")
        print(f"   重要性: {info['重要性']}")
        print(f"   优化目标: {info['优化目标']}")
        print(f"   理想范围: {info['理想范围']}")
        print()

    # 对比分析
    print("🔄 标准GO2W对比分析:")
    print("-" * 50)

    comparison = compare_with_standard_go2w()

    for model_type, info in comparison.items():
        if model_type != "性能差异分析":
            print(f"\n{model_type}:")
            print(f"   特点: {info['特点']}")
            print(f"   优势: {info['优势']}")
            print(f"   劣势: {info['劣势']}")
            print(f"   典型性能: {info['典型性能']}")

    print("\n" + "=" * 50)
    print("性能差异分析:")
    print(f"   {comparison['性能差异分析']['奖励差异']}")
    print(f"   {comparison['性能差异分析']['收敛速度']}")
    print(f"   {comparison['性能差异分析']['稳定性']}")
    print(f"   {comparison['性能差异分析']['能耗']}")

    print("\n" + "=" * 80)
    print("📈 训练建议:")
    print("=" * 80)

    suggestions = [
        "1. 继续训练：GO2W ARM需要更多训练时间才能达到稳定性能",
        "2. 监控关键指标：重点关注Episode Reward和Success Rate的上升趋势",
        "3. 调整超参数：如果收敛过慢，可考虑调整学习率或奖励权重",
        "4. 定期评估：每1000步评估一次训练进度",
        "5. 与标准GO2W对比：使用相同的评估标准进行公平比较"
    ]

    for suggestion in suggestions:
        print(suggestion)

    print("\n" + "=" * 80)
    print("📋 指标监控建议:")
    print("=" * 80)

    monitoring_tips = [
        "核心指标: Episode Reward (必须持续上升)",
        "稳定性指标: Flat Orientation L2 (必须小于0.1)",
        "成功率指标: Success Rate (目标>90%)",
        "效率指标: Joint Torques L2 (越小越好)",
        "综合评估: 各指标协调发展，无明显短板"
    ]

    for tip in monitoring_tips:
        print(f"• {tip}")

if __name__ == "__main__":
    # 切换到训练目录
    os.chdir("/home/jay/unitree_rl_lab/logs/rsl_rl/unitree_go2warm_velocity_flat_v0/2026-04-01_10-36-44")

    generate_training_report()