#!/usr/bin/env python3
"""GO2W ARM初始状态验证脚本

这个脚本用于验证新的初始状态随机化配置是否按预期工作。
"""

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.app import AppLauncher

# 创建仿真应用
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

# 导入配置
import sys
sys.path.insert(0, "/home/jay/unitree_rl_lab/source")
from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg


def check_initial_state():
    """检查初始状态随机化配置"""

    print("=" * 60)
    print("GO2W ARM 初始状态验证")
    print("=" * 60)
    print()

    # 创建环境
    env_cfg = RobotFlatEnvCfg()
    env_cfg.scene.num_envs = 100  # 创建100个环境进行统计
    env_cfg.scene.terrain.terrain_type = "plane"  # 使用平面地形

    # 创建仿真环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("📊 初始状态随机化配置:")
    print("-" * 60)

    # 打印配置参数
    reset_params = env_cfg.events.randomize_reset_base.params

    print("📍 位置范围 (pose_range):")
    print(f"   X: {reset_params['pose_range']['x']} m")
    print(f"   Y: {reset_params['pose_range']['y']} m")
    print(f"   Z: {reset_params['pose_range']['z']} m")
    print()

    print("🔄 姿态角度 (roll, pitch, yaw):")
    print(f"   Roll:  {reset_params['pose_range']['roll']} rad")
    print(f"   Pitch: {reset_params['pose_range']['pitch']} rad")
    print(f"   Yaw:   {reset_params['pose_range']['yaw']} rad")
    print()

    print("🚀 初始速度范围 (velocity_range):")
    print(f"   X: {reset_params['velocity_range']['x']} m/s")
    print(f"   Y: {reset_params['velocity_range']['y']} m/s")
    print(f"   Z: {reset_params['velocity_range']['z']} m/s")
    print(f"   Roll:  {reset_params['velocity_range']['roll']} rad/s")
    print(f"   Pitch: {reset_params['velocity_range']['pitch']} rad/s")
    print(f"   Yaw:   {reset_params['velocity_range']['yaw']} rad/s")
    print()

    print("=" * 60)
    print("🧪 测试100个随机初始状态:")
    print("=" * 60)
    print()

    # 重置环境多次，收集统计数据
    num_resets = 100
    positions = []
    orientations = []
    velocities = []

    for i in range(num_resets):
        env.reset()
        robot = env.scene["robot"]

        # 获取基座位置
        base_pos = robot.data.root_state_w[:, :3][0].cpu().numpy()
        positions.append(base_pos)

        # 获取基座姿态（四元数转换为欧拉角）
        base_quat = robot.data.root_state_w[:, 3:7][0].cpu().numpy()
        orientation = sim_utils.quat_to_euler_xyz(base_quat)
        orientations.append(orientation)

        # 获取基座速度
        base_vel = robot.data.root_state_w[:, 7:13][0].cpu().numpy()
        velocities.append(base_vel)

    # 转换为numpy数组
    positions = np.array(positions)
    orientations = np.array(orientations)
    velocities = np.array(velocities)

    # 计算统计数据
    print("📍 位置统计 (m):")
    print(f"   X: mean={positions[:, 0].mean():.3f}, std={positions[:, 0].std():.3f}, min={positions[:, 0].min():.3f}, max={positions[:, 0].max():.3f}")
    print(f"   Y: mean={positions[:, 1].mean():.3f}, std={positions[:, 1].std():.3f}, min={positions[:, 1].min():.3f}, max={positions[:, 1].max():.3f}")
    print(f"   Z: mean={positions[:, 2].mean():.3f}, std={positions[:, 2].std():.3f}, min={positions[:, 2].min():.3f}, max={positions[:, 2].max():.3f}")
    print()

    print("🔄 姿态角度统计 (rad / deg):")
    print(f"   Roll:  mean={orientations[:, 0].mean():.3f} ({np.degrees(orientations[:, 0].mean()):.1f}°), "
          f"std={orientations[:, 0].std():.3f} ({np.degrees(orientations[:, 0].std()):.1f}°)")
    print(f"   Pitch: mean={orientations[:, 1].mean():.3f} ({np.degrees(orientations[:, 1].mean()):.1f}°), "
          f"std={orientations[:, 1].std():.3f} ({np.degrees(orientations[:, 1].std()):.1f}°)")
    print(f"   Yaw:   mean={orientations[:, 2].mean():.3f} ({np.degrees(orientations[:, 2].mean()):.1f}°), "
          f"std={orientations[:, 2].std():.3f} ({np.degrees(orientations[:, 2].std()):.1f}°)")
    print()

    print("🚀 初始速度统计:")
    print(f"   线速度X: mean={velocities[:, 0].mean():.3f} m/s, std={velocities[:, 0].std():.3f} m/s")
    print(f"   线速度Y: mean={velocities[:, 1].mean():.3f} m/s, std={velocities[:, 1].std():.3f} m/s")
    print(f"   线速度Z: mean={velocities[:, 2].mean():.3f} m/s, std={velocities[:, 2].std():.3f} m/s")
    print(f"   角速度Roll:  mean={velocities[:, 3].mean():.3f} rad/s, std={velocities[:, 3].std():.3f} rad/s")
    print(f"   角速度Pitch: mean={velocities[:, 4].mean():.3f} rad/s, std={velocities[:, 4].std():.3f} rad/s")
    print(f"   角速度Yaw:   mean={velocities[:, 5].mean():.3f} rad/s, std={velocities[:, 5].std():.3f} rad/s")
    print()

    # 计算倾斜角度（与垂直方向的夹角）
    tilt_angles = np.degrees(np.arccos(np.abs(orientations[:, 0]) + np.abs(orientations[:, 1])))

    print("=" * 60)
    print("✅ 关键指标验证:")
    print("=" * 60)
    print()

    print(f"🎯 倾斜角度统计:")
    print(f"   平均倾斜角度: {tilt_angles.mean():.1f}°")
    print(f"   最大倾斜角度: {tilt_angles.max():.1f}°")
    print(f"   倾斜角度 < 30°: {np.sum(tilt_angles < 30)} / {len(tilt_angles)} ({100*np.mean(tilt_angles < 30):.1f}%)")
    print(f"   倾斜角度 < 20°: {np.sum(tilt_angles < 20)} / {len(tilt_angles)} ({100*np.mean(tilt_angles < 20):.1f}%)")
    print(f"   倾斜角度 < 10°: {np.sum(tilt_angles < 10)} / {len(tilt_angles)} ({100*np.mean(tilt_angles < 10):.1f}%)")
    print()

    print(f"📏 高度统计:")
    print(f"   平均高度: {positions[:, 2].mean():.3f} m")
    print(f"   最小高度: {positions[:, 2].min():.3f} m")
    print(f"   最大高度: {positions[:, 2].max():.3f} m")
    print(f"   高度 > 0.35m: {np.sum(positions[:, 2] > 0.35)} / {len(positions)} ({100*np.mean(positions[:, 2] > 0.35):.1f}%)")
    print(f"   高度 > 0.40m: {np.sum(positions[:, 2] > 0.40)} / {len(positions)} ({100*np.mean(positions[:, 2] > 0.40):.1f}%)")
    print()

    # 验证是否符合预期
    print("=" * 60)
    print("🔍 配置验证结果:")
    print("=" * 60)
    print()

    checks = []

    # 检查位置范围
    x_range_ok = (positions[:, 0].min() >= -0.2) and (positions[:, 0].max() <= 0.2)
    y_range_ok = (positions[:, 1].min() >= -0.2) and (positions[:, 1].max() <= 0.2)
    z_range_ok = (positions[:, 2].min() >= 0.35) and (positions[:, 2].max() <= 0.5)
    checks.append(("位置X范围", x_range_ok))
    checks.append(("位置Y范围", y_range_ok))
    checks.append(("位置Z范围", z_range_ok))

    # 检查角度范围
    roll_angle_ok = (orientations[:, 0].min() >= -0.3) and (orientations[:, 0].max() <= 0.3)
    pitch_angle_ok = (orientations[:, 1].min() >= -0.3) and (orientations[:, 1].max() <= 0.3)
    checks.append(("翻滚角范围", roll_angle_ok))
    checks.append(("俯仰角范围", pitch_angle_ok))

    # 检查速度范围
    lin_vel_ok = (np.abs(velocities[:, :3]).max() <= 0.15)  # 允许一点误差
    ang_vel_ok = (np.abs(velocities[:, 3:6]).max() <= 0.15)  # 允许一点误差
    checks.append(("线速度范围", lin_vel_ok))
    checks.append(("角速度范围", ang_vel_ok))

    # 打印检查结果
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("🎉 所有检查通过！初始状态配置符合预期。")
    else:
        print("⚠️  部分检查未通过，请检查配置参数。")

    print()
    print("=" * 60)

    # 关闭环境
    simulation_app.close()


if __name__ == "__main__":
    check_initial_state()
