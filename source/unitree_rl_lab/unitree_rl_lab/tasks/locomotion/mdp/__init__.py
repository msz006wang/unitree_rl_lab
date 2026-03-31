from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from . import events  # noqa: F401
from . import extended_rewards  # noqa: F401
from . import terminations  # noqa: F401
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

# 导出events模块中的新增函数
from .events import (  # noqa: F401
    _randomize_prop_by_op,  # 内部函数
    randomize_rigid_body_inertia,  # 新增：转动惯量随机化
)

# 导出扩展奖励函数
from .extended_rewards import (  # noqa: F401
    action_mirror,
    action_sync,
    wheel_vel_penalty,
    feet_air_time,
    survival_reward,
    distance_traveled_reward,
    energy_efficiency_reward,
    fall_recovery_reward,
    is_fallen,
    upright_orientation_reward,
    # 新增GO2W ARM专用奖励函数
    upward_velocity,
    orientation_tracking,
    torque_penalty,
    joint_regularization,
    contact_management,
    wheel_assisted_recovery,
)

# 导出扩展观测函数
from .observations import (  # noqa: F401
    joint_pos_rel_without_wheel,
    gait_phase,
    phase,
    # 新增历史观测函数
    history_buffer,
    joint_pos_history,
    body_vel_history,
)
