#!/usr/bin/env python3
"""测试mdp模块中的函数是否都可用"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_lab"))

try:
    from unitree_rl_lab.tasks.locomotion import mdp

    # velocity_env_cfg.py 中使用的所有函数
    required_functions = [
        "action_mirror",
        "action_rate_l2",
        "action_sync",
        "ang_vel_xy_l2",
        "apply_external_force_torque",
        "base_ang_vel",
        "base_height_l2",
        "base_lin_vel",
        "body_lin_acc_l2",
        "command_levels_vel",
        "contact_forces",
        "feet_air_time",
        "feet_air_time_variance_penalty",
        "feet_contact",
        "feet_contact_without_cmd",
        "feet_height",
        "feet_height_body",
        "feet_slide",
        "feet_stumble",
        "flat_orientation_l2",
        "generated_commands",
        "height_scan",
        "illegal_contact",
        "is_terminated",
        "joint_acc_l2",
        "joint_mirror",
        "joint_pos_limits",
        "joint_pos_penalty",
        "joint_pos_rel",
        "joint_pos_rel_without_wheel",
        "joint_power",
        "joint_torques_l2",
        "joint_vel_l2",
        "joint_vel_limits",
        "joint_vel_rel",
        "last_action",
        "lin_vel_z_l2",
        "projected_gravity",
        "push_by_setting_velocity",
        "randomize_actuator_gains",
        "randomize_rigid_body_com",
        "randomize_rigid_body_inertia",  # 新添加的
        "randomize_rigid_body_mass",
        "randomize_rigid_body_material",
        "reset_joints_by_scale",
        "reset_root_state_uniform",
        "stand_still",
        "terrain_levels_vel",
        "time_out",
        "track_ang_vel_z_exp",
        "track_lin_vel_xy_exp",
        "undesired_contacts",
        "upward",
        "wheel_vel_penalty",
    ]

    print("检查mdp模块中的函数可用性...")
    print("=" * 60)

    missing_functions = []
    available_functions = []

    for func_name in required_functions:
        if hasattr(mdp, func_name):
            available_functions.append(func_name)
            print(f"✅ {func_name}")
        else:
            missing_functions.append(func_name)
            print(f"❌ {func_name} - 缺失!")

    print("=" * 60)
    print(f"\n总计: {len(available_functions)}/{len(required_functions)} 个函数可用")

    if missing_functions:
        print(f"\n⚠️  缺失的函数 ({len(missing_functions)}):")
        for func_name in missing_functions:
            print(f"  - {func_name}")
        sys.exit(1)
    else:
        print("\n✅ 所有必需的函数都可用!")
        sys.exit(0)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
