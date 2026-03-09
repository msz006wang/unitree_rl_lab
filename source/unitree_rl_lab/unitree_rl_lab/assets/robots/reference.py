import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from msz006_go2w.assets import ISAACLAB_ASSETS_DATA_DIR

UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        # replace_cylinder_with_capsule=False,      # 参数在这里不再需要，因为已经在转换时处理了
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/go2w_description/urdf/go2w_description.urdf",
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.5,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],     # 轮子关节名称
            effort_limit=23.5,                      # 限制执行器（电机）能输出的最大力矩。
            velocity_limit=30.0,                    # 限制执行器（电机）的最大速度。
            stiffness=0.0,                          # 限制执行器（电机）的刚度。
            damping=0.5,                            # 限制执行器（电机）的阻尼。
            friction=0.0,                           # 限制执行器（电机）的摩擦。
        ),
    },
)