"""
配置对比工具
用于比较原始配置和改进配置的差异
"""

import sys
sys.path.insert(0, 'source/unitree_rl_lab')

from unitree_rl_lab.tasks.locomotion.robots.g1.dof.velocity_env_cfg import RobotEnvCfg as OriginalCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.dof.velocity_env_cfg_improved import RobotEnvCfg as ImprovedCfg


def compare_configs():
    """对比原始配置和改进配置"""

    print("=" * 80)
    print("G1机器人训练配置对比")
    print("=" * 80)
    print()

    # 创建配置实例
    original = OriginalCfg()
    improved = ImprovedCfg()

    # 对比reward权重
    print("📊 Reward权重对比:")
    print("-" * 80)
    print(f"{'Reward项':<30} {'原始值':>15} {'改进值':>15} {'变化':>15}")
    print("-" * 80)

    reward_names = [
        'track_lin_vel_xy',
        'track_ang_vel_z',
        'alive',
        'base_linear_velocity',
        'base_angular_velocity',
        'joint_vel',
        'joint_acc',
        'action_rate',
        'dof_pos_limits',
        'energy',
        'flat_orientation_l2',
        'base_height',
    ]

    for name in reward_names:
        if hasattr(original.rewards, name):
            orig_weight = getattr(original.rewards, name).weight
            if hasattr(improved.rewards, name):
                imp_weight = getattr(improved.rewards, name).weight
                change = imp_weight - orig_weight
                change_str = f"({change:+.2f})"
                print(f"{name:<30} {orig_weight:>15.2f} {imp_weight:>15.2f} {change_str:>15}")

    # 新增的reward
    print("\n🆕 新增Reward项:")
    print("-" * 80)
    new_rewards = [
        'survival',
        'distance_traveled',
        'energy_efficiency',
        'consistent_velocity',
        'fall_recovery',
        'stand_up_progress',
        'upright_orientation',
    ]

    for name in new_rewards:
        if hasattr(improved.rewards, name):
            weight = getattr(improved.rewards, name).weight
            print(f"{name:<30} {'-':>15} {weight:>15.2f} {'(NEW)':>15}")

    # Action配置对比
    print("\n⚙️  Action配置对比:")
    print("-" * 80)
    orig_scale = original.actions.JointPositionAction.scale
    imp_scale = improved.actions.JointPositionAction.scale
    print(f"{'Action scale':<30} {orig_scale:>15.2f} {imp_scale:>15.2f} ({(imp_scale-orig_scale):+.2f})")

    # Episode长度对比
    print("\n⏱️  Episode配置对比:")
    print("-" * 80)
    orig_ep_len = original.episode_length_s
    imp_ep_len = improved.episode_length_s
    print(f"{'Episode长度 (秒)':<30} {orig_ep_len:>15.1f} {imp_ep_len:>15.1f} ({(imp_ep_len-orig_ep_len):+.1f})")

    # 终止条件对比
    print("\n🚪 终止条件对比:")
    print("-" * 80)

    orig_min_height = original.terminations.base_height.params['minimum_height']
    imp_min_height = improved.terminations.base_height.params['minimum_height']
    print(f"{'最小高度 (m)':<30} {orig_min_height:>15.2f} {imp_min_height:>15.2f} ({(imp_min_height-orig_min_height):+.2f})")

    orig_angle = original.terminations.bad_orientation.params['limit_angle']
    imp_angle = improved.terminations.bad_orientation.params['limit_angle']
    print(f"{'最大倾斜角度 (rad)':<30} {orig_angle:>15.2f} {imp_angle:>15.2f} ({(imp_angle-orig_angle):+.2f})")

    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print("✅ 主要改进:")
    print("  1. 添加了生存奖励 (0.5) - 鼓励长时间行走")
    print("  2. 添加了摔倒恢复奖励 (5.0) - 支持摔倒后重新站立")
    print("  3. 增加了Action scale (0.3→0.5) - 支持更大幅度运动")
    print("  4. 延长了Episode长度 (20→25秒) - 有更多时间恢复")
    print("  5. 放宽了终止条件 - 允许机器人摔倒后尝试恢复")
    print("  6. 添加了距离、能量效率、速度一致性等辅助奖励")
    print()


if __name__ == "__main__":
    try:
        compare_configs()
    except ImportError as e:
        print(f"错误: 无法导入配置文件")
        print(f"请确保在项目根目录运行此脚本")
        print(f"详细错误: {e}")
        sys.exit(1)
