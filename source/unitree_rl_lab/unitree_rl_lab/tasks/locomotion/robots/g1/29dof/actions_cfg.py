"""
改进的G1动作空间配置
参考了以下项目：
- FRASA (Fall Recovery And Stand-up Agent)
- HoST (Humanoid Standing-up Control)
- walk-these-ways (通用足式机器人控制)
"""

import isaaclab.mdp as mdp
from isaaclab.managers import ActionTermCfg, ActionTermCfg
from isaaclab.utils import configclass


@configclass
class JointPositionActionCfg(mdp.JointPositionActionCfg):
    """标准关节位置控制"""
    asset_name = "robot"
    joint_names = [".*"]
    scale = 0.5  # 增加动作范围以支持更大幅度运动
    use_default_offset = True
    clip = None  # 不限制动作范围，让网络学习最优控制


@configclass
class JointPositionVelocityActionCfg(ActionTermCfg):
    """混合位置-速度控制（用于更精细的控制）"""
    class_type = "JointPositionVelocityAction"

    asset_name = "robot"
    joint_names = [".*"]

    # 位置控制参数
    pos_scale = 0.3
    pos_use_default_offset = True

    # 速度控制参数
    vel_scale = 0.5

    # 控制混合权重
    position_weight = 0.7  # 70%位置控制
    velocity_weight = 0.3  # 30%速度控制


@configclass
class PDTargetPositionActionCfg(ActionTermCfg):
    """PD目标位置控制（参考HoST和FRASA的设计）"""
    class_type = "PDTargetPositionAction"

    asset_name = "robot"
    joint_names = [".*"]

    # PD目标计算公式：target_pos = current_pos + scale * action
    scale = 0.25

    # PD控制器参数（在IsaacLab中配置）
    # 这些参数会在robot的USD文件中设置
    # stiffness: 关节刚度
    # damping: 关节阻尼

    use_default_offset = True


@configclass
class HybridActionsCfg:
    """
    混合动作配置，根据训练阶段选择不同的动作空间

    阶段1：基础行走 - 使用标准关节位置控制
    阶段2：精细控制 - 使用混合位置-速度控制
    阶段3：高级技能 - 使用PD目标控制
    """

    # 基础关节位置控制（推荐用于初始训练）
    joint_position = JointPositionActionCfg()

    # 混合位置-速度控制（用于精细控制）
    # joint_position_velocity = JointPositionVelocityActionCfg()

    # PD目标控制（用于高级技能）
    # pd_target_position = PDTargetPositionActionCfg()
