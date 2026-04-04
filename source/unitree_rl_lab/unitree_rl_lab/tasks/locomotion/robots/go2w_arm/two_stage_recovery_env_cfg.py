# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for GO2W-Arm two-stage recovery environment based on robot_lab_locomanip."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_GO2W_ARM_ARX5_CFG as ROBOT_CFG
)
from unitree_rl_lab.tasks.locomotion import mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with GO2W-Arm robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
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
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
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
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=5.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=1.0
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP - Based on robot_lab_locomanip configuration."""

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.torque_adaptive, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "full_penalty_weight": -2.5e-5, "reduced_penalty_weight": -2.5e-6, "orientation_threshold": 0.5}
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),  # 匹配轮子
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )

    # Action penalties
    action_rate_l2 = RewTerm(func=mdp.action_rate_adaptive, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot"), "full_penalty_weight": -0.01, "reduced_penalty_weight": -0.001, "orientation_threshold": 0.5})

    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )
    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.contact_adaptive,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*foot).*$"]), "full_penalty_weight": -1.0, "reduced_penalty_weight": -0.1, "orientation_threshold": 0.5},
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Note: 3D velocity tracking rewards (track_lin_vel_xyz_exp, track_ang_vel_xyz_exp) are not available in current isaaclab version
    # and are commented out in robot_lab_locomanip reference project as well

    # Others
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "command_name": "base_velocity",
        },
    )
    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.2,
            "command_name": "base_velocity",
        },
    )
    upward = RewTerm(func=mdp.upward_orientation, weight=5.0)  # 基于姿态的向上奖励，权重+5.0

    # Two-stage recovery specific: Wheel angular momentum reward
    # 鼓励悬空轮子急加速，利用角动量守恒产生翻滚扭矩
    wheel_angular_momentum = RewTerm(
        func=mdp.wheel_angular_momentum_reward,
        weight=2.0,  # 较高的权重，鼓励利用角动量进行恢复
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),  # 匹配所有轮子（foot）
            "wheel_joint_names": ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"],
            "weight": 1.0,
            "contact_threshold": 1.0,  # 接触力小于1.0N认为轮子悬空
        },
    )

    # Two-stage recovery specific rewards (disabled - robot_lab_locomanip doesn't use standing rewards)
    # two_stage_standing_reward = RewTerm(
    #     func=mdp.two_stage_standing_reward,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*foot).*$"]), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )


@configclass
class TwoStageRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for GO2W-Arm two-stage recovery environment based on robot_lab_locomanip."""

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

    # Wheel-legged robot specific parameters
    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    arm_joint_names = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names
    # fmt: on

    def __post_init__(self):
        """Post initialization - Based on robot_lab_locomanip configuration."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # ------------------------------Wheel-Legged Specific Settings------------------------------
        # Update sensor prim paths
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # Use joint_pos_rel_without_wheel to exclude wheel positions from observations
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        # Set observation scales
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # Remove base_lin_vel from policy observations
        self.observations.policy.base_lin_vel = None
        # Remove height_scan from policy observations
        self.observations.policy.height_scan = None
        # Update joint observations for specific joint groups
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # Set action scale for different joint groups
        # 针对 Two-Stage Recovery 恢复训练，增大 arm_joint1 的缩放因子至 0.5
        # 使策略能够通过快速甩动大质量机械臂，产生足以打破侧卧静态平衡的惯性力矩
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "arm_joint1": 0.5,  # 增大至 0.5，允许快速甩动产生惯性力矩
            "^(?!.*_hip_joint)(?!arm_joint1).*": 0.25,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}

        # Leg + all arm joints in action space - enable dynamic arm swinging
        self.actions.joint_pos.joint_names = self.leg_joint_names + self.arm_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # Update event parameters for base link
        # 探索期：大幅放开 roll 范围至 ±1.6 rad (±90°) 以训练两段式恢复能力
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.35, 0.5),
                "roll": (-1.6, 1.6),  # ±90°，大幅放开以训练侧倾恢复能力
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # Based on robot_lab_locomanip configuration
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties - 使用自适应惩罚，允许倒地时产生大扭矩
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.rewards.joint_torques_l2.params["full_penalty_weight"] = -2.5e-5
        self.rewards.joint_torques_l2.params["reduced_penalty_weight"] = -2.5e-6  # 倒地时降低10倍
        self.rewards.joint_torques_l2.params["orientation_threshold"] = 0.5  # Z<0.5时降低惩罚
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties - 使用自适应惩罚，允许倒地时疯狂探索
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_rate_l2.params["full_penalty_weight"] = -0.01
        self.rewards.action_rate_l2.params["reduced_penalty_weight"] = -0.001  # 倒地时降低10倍
        self.rewards.action_rate_l2.params["orientation_threshold"] = 0.5  # Z<0.5时降低惩罚

        # Contact sensor - 使用自适应惩罚，允许倒地时借力
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["full_penalty_weight"] = -1.0
        self.rewards.undesired_contacts.params["reduced_penalty_weight"] = -0.1  # 倒地时降低10倍
        self.rewards.undesired_contacts.params["orientation_threshold"] = 0.5  # Z<0.5时降低惩罚
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 5.0  # 基于姿态的向上奖励，权重+5.0，高权重促进恢复

        # ------------------------------Terminations------------------------------
        # Disable illegal contact termination
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # Disable command levels curriculum
        self.curriculum.command_levels = None

        # Disable zero-weight rewards to optimize computation
        self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """Disable rewards with zero weight to optimize computation."""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and hasattr(reward_attr, 'weight') and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


@configclass
class TwoStageRecoveryPlayEnvCfg(TwoStageRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.ranges.__class__(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )


@configclass
class TwoStageRecoveryFlatEnvCfg(TwoStageRecoveryEnvCfg):
    """Flat terrain training configuration for two-stage recovery."""

    def __post_init__(self):
        # Execute parent class initialization first
        super().__post_init__()

        # ==================== Terrain Settings ====================
        # Change terrain type to plane
        self.scene.terrain.terrain_type = "plane"
        # Remove terrain generator
        self.scene.terrain.terrain_generator = None

        # ==================== Sensor Settings ====================
        # Disable height scanners (not needed for flat terrain)
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # ==================== Observations ====================
        # Explicitly disable height scan observations
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ==================== Rewards ====================
        # Flat terrain doesn't need height scan based rewards
        if hasattr(self.rewards, 'base_height_l2') and self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None

        # ==================== Curriculum ====================
        # Disable terrain difficulty curriculum (no terrain on flat ground)
        self.curriculum.terrain_levels = None

        # ==================== Re-disable zero-weight rewards ====================
        self.disable_zero_weight_rewards()

        print("✅ Using GO2W-Arm Two-Stage Recovery flat terrain configuration (TwoStageRecoveryFlatEnvCfg)")
        print(f"   - Terrain type: {self.scene.terrain.terrain_type}")
        print(f"   - Terrain generator: {self.scene.terrain.terrain_generator}")
        print(f"   - Height scanner: {self.scene.height_scanner}")
        print(f"   - Terrain curriculum: {self.curriculum.terrain_levels}")
