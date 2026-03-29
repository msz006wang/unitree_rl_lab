"""
改进的G1速度跟踪环境配置 - 平地地形
Improved G1 Velocity Tracking Environment Configuration - Flat Terrain

这是velocity_env_cfg_improved.py的平地版本，使用简化地形进行基础训练。
This is the flat terrain version of velocity_env_cfg_improved.py for basic training.

用途：
Use case:
- 基础步态训练 (Basic gait training)
- 配置验证 (Configuration validation)
- 快速测试 (Quick testing)
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
# 导入扩展的reward函数
from unitree_rl_lab.tasks.locomotion.mdp import extended_rewards

# 平地地形配置（简化版本）
FLAT_TERRAIN_CFG = terrain_gen.MeshPlaneTerrainCfg()


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for flat terrain scene with G1 robot."""

    # ground terrain - 使用平地
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval - 推力事件以提高鲁棒性
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.2),  # 前进速度
            lin_vel_y=(-0.5, 0.5),  # 侧向速度
            ang_vel_z=(-0.3, 0.3)   # 转向速度
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for MDP."""

    # 使用中等scale以平衡稳定性和灵活性
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.35,  # 修复后的scale值
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observations specifications for MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Improved Reward terms for MDP (Flat Terrain Version)."""

    # ========== 任务相关奖励 ==========

    # 线速度跟踪（主要任务）
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,  # 修复后的权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 角速度跟踪
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,  # 修复后的权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ========== 长时间行走奖励 ==========

    # 生存奖励（每个时间步给予正奖励）
    survival = RewTerm(
        func=extended_rewards.survival_reward,
        weight=0.5,
        params={}
    )

    # 行走距离奖励
    distance_traveled = RewTerm(
        func=extended_rewards.distance_traveled_reward,
        weight=0.3,
        params={"command_name": "base_velocity"}
    )

    # 能量效率奖励
    energy_efficiency = RewTerm(
        func=extended_rewards.energy_efficiency_reward,
        weight=0.1,
        params={}
    )

    # 速度一致性奖励（减少速度波动）
    consistent_velocity = RewTerm(
        func=extended_rewards.consistent_velocity_reward,
        weight=0.2,
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # ========== 摔倒恢复奖励 ==========

    # 摔倒恢复奖励（从摔倒状态恢复）
    fall_recovery = RewTerm(
        func=extended_rewards.fall_recovery_reward,
        weight=0.5,  # 修复后的权重
        params={}
    )

    # 站起进度奖励
    stand_up_progress = RewTerm(
        func=extended_rewards.stand_up_progress_reward,
        weight=0.3,  # 修复后的权重
        params={"target_height": 0.78, "std": 0.2}
    )

    # 直立姿态奖励
    upright_orientation = RewTerm(
        func=extended_rewards.upright_orientation_reward,
        weight=0.5,
        params={"std": 0.3}
    )

    # ========== 基础保持奖励 ==========

    # 存活奖励（原始）
    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # ========== 姿态和稳定性惩罚 ==========

    # 线速度Z轴惩罚（防止跳跃）
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # 角速度XY平面惩罚
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 身体姿态惩罚
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)  # 修复后的权重

    # 身体高度惩罚
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-8.0,  # 修复后的权重
        params={"target_height": 0.78}
    )

    # ========== 关节相关惩罚 ==========

    # 关节速度惩罚
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    # 关节加速度惩罚
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # 动作变化率惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # 关节位置限制惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # 能量消耗惩罚
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # 手臂关节保持
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    # 腰部关节保持
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],
            )
        },
    )

    # 腿部关节保持
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # ========== 足部相关奖励 ==========

    # 步态奖励
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # 足部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # 足部高度奖励
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # ========== 其他惩罚 ==========

    # 非期望接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 修复后的终止条件（适度收紧）
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.12}  # 修复后的值
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.1}  # 修复后的值
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for MDP."""

    # 平地地形不需要课程学习
    pass


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Improved configuration for locomotion velocity-tracking environment on flat terrain."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 25.0  # 保持25秒episode长度

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # G1稳定性修复：使用Mimic配置的初始关节角度和高度
        if hasattr(self.scene.robot, 'init_state'):
            # 使用Mimic配置的初始高度0.76m
            self.scene.robot.init_state.pos = (0.0, 0.0, 0.76)
            # 使用更合理的初始关节角度（参考Mimic配置）
            self.scene.robot.init_state.joint_pos = {
                "left_hip_pitch_joint": -0.312,
                "right_hip_pitch_joint": -0.312,
                "left_hip_roll_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.669,
                "right_knee_joint": 0.669,
                "left_ankle_pitch_joint": -0.363,
                "right_ankle_pitch_joint": -0.363,
                "left_ankle_roll_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_pitch_joint": 0.2,
                "left_shoulder_roll_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.6,
                "right_elbow_joint": 0.6,
                "left_wrist_roll_joint": 0.15,
                "right_wrist_roll_joint": -0.15,
                "left_wrist_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            }


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Configuration for play/testing environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
