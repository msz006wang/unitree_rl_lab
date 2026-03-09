# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """
    地形边界检测终止函数

    物理意义：
    1. 边界保护：防止机器人走出训练地形，避免在未定义区域的行为
    2. 训练一致性：确保所有环境实例都在相同的地形范围内训练
    3. 安全性：避免机器人在地形边缘可能出现的物理异常
    4. 数据质量：防止机器人在地形边缘产生异常数据

    工作原理：
    - 根据地形类型（plane 或 generator）采用不同的检测策略
    - plane：无限平面，无边界，永不终止
    - generator：有限地形，计算边界并检测机器人位置
    - 考虑安全缓冲区，在机器人接近边界时提前终止

    边界检测：
    - 计算地形的总尺寸（包括边界）
    - 检查机器人位置是否超出安全区域
    - 超出边界则触发终止

    Args:
        env: 强化学习环境实例
        asset_cfg: 机器人资产配置，默认为"robot"
        distance_buffer: 安全缓冲距离（米），默认3.0米
                       机器人在距离边界小于此距离时就会终止

    Returns:
        终止标志张量，形状为 [num_envs]
        True 表示该环境实例需要终止，False 表示继续
    """

    # ========== 检查地形类型 ==========
    # env.scene.cfg.terrain.terrain_type: 地形类型配置
    # 可能的值：
    #   - "plane": 无限平面地形
    #   - "generator": 生成的有限地形
    if env.scene.cfg.terrain.terrain_type == "plane":
        # ========== 无限平面地形 ==========
        # plane 地形是无限延伸的平面，没有边界
        # 因此机器人永远不会超出边界
        #
        # 返回全 False 的张量，表示所有环境实例都不需要终止
        #
        # torch.zeros(..., dtype=torch.bool): 创建全0的布尔张量
        # - env.num_envs: 并行环境数量
        # - dtype=torch.bool: 数据类型为布尔值
        # - device=env.device: 张量存储在正确的设备上（CPU或GPU）
        #
        # False 表示不终止
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # ========== 生成的有限地形 ==========
        # generator 地形是由多个地形块拼接而成的有限地形
        # 需要计算地形的总尺寸并检测机器人是否超出边界

        # ========== 获取地形生成器配置 ==========
        # env.scene.terrain.cfg.terrain_generator: 地形生成器的配置对象

        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator

        # ========== 获取单个地形块的尺寸 ==========
        # grid_width: 单个地形块的宽度（x方向）
        # grid_length: 单个地形块的长度（y方向）
        # 单位：米
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols

        # ========== 获取边界宽度 ==========
        # border_width: 地形周围的边界宽度
        # 这个边界通常用于平滑过渡或避免边缘效应
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # 地形结构示意图：
        #
        #   y (map_height)
        #   ↑
        #   │  ┌─────────────────────────────┐
        #   │  │ border (border_width)      │
        #   │  ├─────────────────────────────┤
        #   │  │ terrain blocks              │
        #   │  │ ┌───┬───┬───┐             │
        #   │  │ │   │   │   │ n_cols      │
        #   │  │ ├───┼───┼───┤             │
        #   │  │ │   │   │   │             │
        #   │  │ └───┴───┴───┘             │
        #   │  │     n_rows                 │
        #   │  ├─────────────────────────────┤
        #   │  │ border (border_width)      │
        #   │  └─────────────────────────────┘
        #   │
        #   └──────────────────────────────→ x (map_width)

        # ========== 获取机器人实例 ==========
        # 从场景中获取机器人实例
        # 类型提示用于代码可读性和IDE支持
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        # ========== 检测机器人是否超出x方向边界 ==========
        # asset.data.root_pos_w: 机器人根部在世界坐标系中的位置
        # 形状: [num_envs, 3]
        #   - 3: [x, y, z] 三个方向的位置分量

        # [:, 0]: 取x坐标
        # 形状: [num_envs]
        # 表示每个环境实例中机器人的x坐标

        # torch.abs(...): 计算x坐标的绝对值
        # 因为地形是以原点为中心对称的
        # x坐标的正负表示机器人在原点的左侧或右侧
        # 形状: [num_envs]

        # 0.5 * map_width: 地形宽度的一半
        # 表示从中心到边缘的距离

        # 0.5 * map_width - distance_buffer: 安全区域的边界
        # 减去 distance_buffer 创建一个安全缓冲区
        # 机器人在距离实际边界 distance_buffer 米时就会触发终止
        #
        # 示例：
        # - map_width = 20米
        # - distance_buffer = 3米
        # - 0.5 * map_width = 10米（从中心到边缘）
        # - 安全边界 = 10 - 3 = 7米
        # - 当 |x| > 7米时触发终止
        # - 即机器人在 x > 7 或 x < -7 时终止
        # - 实际地形边缘在 x = ±10米
        # - 所以机器人在距离边缘3米时就会终止

        # > 0.5 * map_width - distance_buffer: 判断是否超出安全边界
        # 结果: 布尔张量，True表示超出边界，False表示在安全区域内
        # 形状: [num_envs]
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
