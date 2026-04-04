"""
两段式恢复专用动作空间配置
"""

import isaaclab.mdp as mdp
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass


@configclass
class PhaseSelectionActionCfg(ActionTermCfg):
    """阶段选择动作配置

    这个动作项允许策略网络选择当前应该执行哪个阶段的动作：
    - 0: 趴伏阶段的动作（蜷缩+翻滚）
    - 1: 侧卧阶段的动作（不对称蹬腿）
    - 2: 站立阶段的动作（爆发起立）

    物理意义：
    1. 策略路由：让网络自动判断当前阶段并选择合适的动作策略
    2. 任务分解：将复杂的恢复任务分解为简单的阶段选择
    3. 学习效率：降低单次学习任务的复杂度，提高训练效率

    实现方法：
    - 输出一个3维的one-hot向量，表示当前选择的阶段
    - 通过离散选择实现阶段切换
    - 结合连续动作执行具体的关节控制

    Args:
        asset_name: 机器人资产名称
        num_phases: 阶段数量
        scale: 动作缩放因子
    """
    class_type = "PhaseSelectionAction"

    asset_name = "robot"
    num_phases = 3  # 0=趴伏, 1=侧卧, 2=站立
    scale = 1.0


@configclass
class TwoStageActionsCfg:
    """
    两段式恢复混合动作配置

    结合离散的阶段选择和连续的关节控制，实现两段式恢复策略。

    动作组成：
    1. 阶段选择（离散）：决定当前应该执行哪个阶段
    2. 关节位置控制（连续）：根据选定阶段执行具体的关节动作
    3. 轮子速度控制（连续）：控制轮子的锁死和转动
    """

    # 阶段选择动作
    phase_selection = PhaseSelectionActionCfg()

    # 阶段一动作：蜷缩+翻滚
    joint_position_phase1 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "arm_joint1",
        ],
        scale=0.3,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )

    # 阶段二动作：爆发起立
    joint_position_phase2 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "arm_joint1",
        ],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )

    # 轮子速度控制
    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"],
        scale=3.0,
        use_default_offset=True,
    )