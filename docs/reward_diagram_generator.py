#!/usr/bin/env python3
"""
生成奖励函数关系示意图
这个脚本创建一个简化的奖励函数依赖关系图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'SimSun', 'AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色方案
colors = {
    'primary': '#2E86AB',      # 蓝色 - 主要任务
    'safety': '#E74C3C',      # 红色 - 安全约束
    'efficiency': '#F59E0B',    # 橙色 - 能量效率
    'stability': '#28A745',     # 绿色 - 稳定性
    'quality': '#6C63B5',      # 紫色 - 运动质量
    'control': '#95A5A6',      # 粉色 - 控制质量
}

# 绘制中心目标节点
ax.text(7, 8, '目标函数\n最大化总奖励',
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))

# 绘制主要奖励类别的箭头和节点
# 速度跟踪
ax.add_patch(FancyArrowPatch((7, 7.5), (4, 5.5), width=2, head_width=15, head_length=20,
                          fc=colors['primary'], ec=colors['primary'], alpha=0.7))
ax.text(4, 5.5, '速度跟踪\n(权重 4.5)', ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['primary'], alpha=0.3))

# 安全约束
ax.add_patch(FancyArrowPatch((7, 7.5), (10, 5.5), width=2, head_width=15, head_length=20,
                          fc=colors['safety'], ec=colors['safety'], alpha=0.7))
ax.text(10, 5.5, '安全约束\n(权重 -6.0)', ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['safety'], alpha=0.3))

# 姿态控制
ax.add_patch(FancyArrowPatch((7, 6.5), (6, 4.5), width=1.5, head_width=12, head_length=18,
                          fc=colors['stability'], ec=colors['stability'], alpha=0.6))
ax.text(6, 4.5, '姿态控制\n(权重 -2.05)', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['stability'], alpha=0.3))

# 能量效率
ax.add_patch(FancyArrowPatch((7, 6.5), (8, 4.5), width=1.5, head_width=12, head_length=18,
                          fc=colors['efficiency'], ec=colors['efficiency'], alpha=0.6))
ax.text(8, 4.5, '能量效率\n(权重 -5e-5)', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['efficiency'], alpha=0.3))

# 运动质量
ax.add_patch(FancyArrowPatch((7, 6.5), (9, 5.5), width=1.2, head_width=10, head_length=16,
                          fc=colors['quality'], ec=colors['quality'], alpha=0.5))
ax.text(9, 5.5, '步态质量\n(权重 0.1)', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['quality'], alpha=0.3))

# 控制质量
ax.add_patch(FancyArrowPatch((7, 6.5), (11, 4.5), width=1.2, head_width=10, head_length=16,
                          fc=colors['control'], ec=colors['control'], alpha=0.5))
ax.text(11, 4.5, '控制质量\n(权重 -0.01)', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['control'], alpha=0.3))

# 添加相互作用说明
ax.text(7, 3, '相互关系与权衡', ha='center', va='center', fontsize=12, fontweight='bold')

# 绘制相互作用图示
# 速度 vs 能量 (冲突)
ax.plot([4, 3.5], [2.5, 1.5], 'o-', color='gray', alpha=0.5, linewidth=2)
ax.text(2.5, 2.5, '冲突', ha='center', fontsize=9, color='red', rotation=90)

# 稳定性 vs 机动性 (权衡)
ax.plot([6, 3.5], [2.5, 1.5], 'o-', color='gray', alpha=0.5, linewidth=2)
ax.text(6, 2.5, '权衡', ha='center', fontsize=9, color='orange', rotation=90)

# 协同关系 (协调)
ax.plot([8, 3.5], [2.5, 1.5], 'o-', color='gray', alpha=0.5, linewidth=2)
ax.text(8, 2.5, '协同', ha='center', fontsize=9, color='green', rotation=90)

# 底部添加具体奖励函数分解
ax.text(7, 1.2, '详细权重分解', ha='center', fontsize=11, fontweight='bold')

# 速度跟踪详细
ax.text(4, 1, '• track_lin_vel_xy: 3.0\n• track_ang_vel_z: 1.5', ha='center', fontsize=8, color=colors['primary'])

# 安全约束详细
ax.text(10, 1, '• joint_pos_limits: -5.0\n• undesired_contacts: -1.0', ha='center', fontsize=8, color=colors['safety'])

# 姿态控制详细
ax.text(6, 1.2, '• base_height: 0.0\n• flat_orientation: 0.0\n• lin_vel_z: -2.0', ha='center', fontsize=8, color=colors['stability'])

# 能量效率详细
ax.text(8, 1.2, '• joint_torques: -2.5e-5\n• joint_power: -2e-5\n• joint_acc: -2.5e-7', ha='center', fontsize=8, color=colors['efficiency'])

# 添加物理含义说明
ax.text(7, 0.5, '物理意义: 速度精度 × 姿态稳定 + 能耗效率 × 运动质量 × 控制质量 - 安全约束',
         ha='center', fontsize=10, style='italic', color='darkblue')

plt.tight_layout()
plt.savefig('/home/jay/unitree_rl_lab/docs/reward_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("奖励函数关系图已生成: docs/reward_diagram.png")
