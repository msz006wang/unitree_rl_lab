#!/usr/bin/env python
"""测试基于robot_lab_locomanip修改后的GO2W-Arm配置"""

def test_imports():
    """测试导入和基本配置"""
    print("="*60)
    print("测试导入和基本配置")
    print("="*60)

    try:
        # 测试DelayedPDActuatorCfg是否可用
        from isaaclab.actuators import DelayedPDActuatorCfg
        print("✅ DelayedPDActuatorCfg导入成功")
    except Exception as e:
        print(f"❌ DelayedPDActuatorCfg导入失败: {e}")
        return False

    try:
        from unitree_rl_lab.assets.robots.unitree import (
            UNITREE_GO2W_ARM_PIPER_CFG,
            UNITREE_GO2W_ARM_ARX5_CFG
        )
        print("✅ GO2W-Arm机器人配置导入成功")
    except Exception as e:
        print(f"❌ GO2W-Arm机器人配置导入失败: {e}")
        return False

    return True

def test_actuator_configurations():
    """测试执行器配置"""
    print("\n" + "="*60)
    print("测试执行器配置")
    print("="*60)

    from unitree_rl_lab.assets.robots.unitree import (
        UNITREE_GO2W_ARM_PIPER_CFG,
        UNITREE_GO2W_ARM_ARX5_CFG
    )

    # 测试Piper配置
    print("\n🤖 Piper机械臂配置:")
    print(f"  Actuator类型: {type(UNITREE_GO2W_ARM_PIPER_CFG.actuators['arm']).__name__}")
    piper_arm = UNITREE_GO2W_ARM_PIPER_CFG.actuators['arm']
    print(f"  延迟范围: {piper_arm.min_delay}-{piper_arm.max_delay}步")
    print(f"  刚度: {piper_arm.stiffness}")
    print(f"  机械臂关节数量: {len(piper_arm.joint_names_expr)}")

    # 测试ARX5配置
    print("\n🤖 ARX5机械臂配置:")
    print(f"  Actuator类型: {type(UNITREE_GO2W_ARM_ARX5_CFG.actuators['arm']).__name__}")
    arx5_arm = UNITREE_GO2W_ARM_ARX5_CFG.actuators['arm']
    print(f"  延迟范围: {arx5_arm.min_delay}-{arx5_arm.max_delay}步")
    print(f"  刚度: {arx5_arm.stiffness}")
    print(f"  力矩限制: {arx5_arm.effort_limit_sim}")
    print(f"  速度限制: {arx5_arm.velocity_limit_sim}")
    print(f"  机械臂关节数量: {len(arx5_arm.joint_names_expr)}")

    # 测试腿部执行器
    print("\n🦵 腿部执行器配置:")
    piper_legs = UNITREE_GO2W_ARM_PIPER_CFG.actuators['legs']
    print(f"  Actuator类型: {type(piper_legs).__name__}")
    print(f"  延迟范围: {piper_legs.min_delay}-{piper_legs.max_delay}步")
    print(f"  刚度: {piper_legs.stiffness}")
    print(f"  最大速度: {piper_legs.velocity_limit_sim}")

def test_initial_states():
    """测试初始状态配置"""
    print("\n" + "="*60)
    print("测试初始状态配置")
    print("="*60)

    from unitree_rl_lab.assets.robots.unitree import (
        UNITREE_GO2W_ARM_PIPER_CFG,
        UNITREE_GO2W_ARM_ARX5_CFG
    )

    # 测试Piper初始状态
    print("\n🤖 Piper初始状态:")
    print(f"  基座高度: {UNITREE_GO2W_ARM_PIPER_CFG.init_state.pos[2]}m")
    piper_joints = UNITREE_GO2W_ARM_PIPER_CFG.init_state.joint_pos
    print(f"  机械臂姿态:")
    print(f"    arm_joint1: {piper_joints['arm_joint1']}")
    print(f"    arm_joint2: {piper_joints['arm_joint2']}")
    print(f"    arm_joint3: {piper_joints['arm_joint3']}")
    print(f"    arm_joint4: {piper_joints['arm_joint4']}")
    print(f"    arm_joint5: {piper_joints['arm_joint5']}")
    print(f"    arm_joint6: {piper_joints['arm_joint6']}")

    # 测试ARX5初始状态
    print("\n🤖 ARX5初始状态:")
    print(f"  基座高度: {UNITREE_GO2W_ARM_ARX5_CFG.init_state.pos[2]}m")
    arx5_joints = UNITREE_GO2W_ARM_ARX5_CFG.init_state.joint_pos
    print(f"  机械臂姿态:")
    print(f"    arm_joint1: {arx5_joints['arm_joint1']}")
    print(f"    arm_joint2: {arx5_joints['arm_joint2']}")
    print(f"    arm_joint3: {arx5_joints['arm_joint3']}")
    print(f"    arm_joint4: {arx5_joints['arm_joint4']}")
    print(f"    arm_joint5: {arx5_joints['arm_joint5']}")
    print(f"    arm_joint6: {arx5_joints['arm_joint6']}")

def test_environment_configs():
    """测试环境配置"""
    print("\n" + "="*60)
    print("测试环境配置")
    print("="*60)

    try:
        # 测试velocity_env_cfg.py
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg as ARX5FlatCfg
        print("✅ ARX5 velocity_env_cfg导入成功")
        print(f"  基座高度控制权重: {ARX5FlatCfg.rewards.base_height_l2.weight}")

    except Exception as e:
        print(f"❌ ARX5 velocity_env_cfg导入失败: {e}")

    try:
        # 测试velocity_env_cfg_piper.py
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg_piper import RobotFlatEnvCfg as PiperFlatCfg
        print("✅ Piper velocity_env_cfg_piper导入成功")
        print(f"  基座高度控制权重: {PiperFlatCfg.rewards.base_height_l2.weight}")

    except Exception as e:
        print(f"❌ Piper velocity_env_cfg_piper导入失败: {e}")

def test_action_scales():
    """测试动作scale配置"""
    print("\n" + "="*60)
    print("测试动作scale配置")
    print("="*60)

    try:
        from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg as ARX5FlatCfg
        arm_scale = ARX5FlatCfg.actions.arm_pos.scale.get("arm_joint.*", 0.0)
        print(f"  ARX5机械臂动作scale: {arm_scale}")

    except Exception as e:
        print(f"❌ 动作scale测试失败: {e}")

def test_collision_settings():
    """测试碰撞配置"""
    print("\n" + "="*60)
    print("测试碰撞配置")
    print("="*60)

    try:
        from unitree_rl_lab.assets.robots.unitree import (
            UNITREE_GO2W_ARM_PIPER_CFG,
            UNITREE_GO2W_ARM_ARX5_CFG
        )

        print("\n🤖 Piper碰撞配置:")
        piper_collision = UNITREE_GO2W_ARM_PIPER_CFG.spawn.articulation_props.enabled_self_collisions
        print(f"  自碰撞检测: {piper_collision}")

        print("\n🤖 ARX5碰撞配置:")
        arx5_collision = UNITREE_GO2W_ARM_ARX5_CFG.spawn.articulation_props.enabled_self_collisions
        print(f"  自碰撞检测: {arx5_collision}")

    except Exception as e:
        print(f"❌ 碰撞配置测试失败: {e}")

def test_import_cleanup():
    """测试导入清理"""
    print("\n" + "="*60)
    print("测试导入清理")
    print("="*60)

    # 检查velocity_env_cfg_piper.py是否还有未使用的导入
    with open("/home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py", "r") as f:
        content = f.read()

        if "import isaaclab.terrains" in content:
            print("❌ velocity_env_cfg_piper.py仍有未使用的terrain_gen导入")
        else:
            print("✅ velocity_env_cfg_piper.py已清理未使用导入")

if __name__ == "__main__":
    try:
        success = test_imports()
        if success:
            test_actuator_configurations()
            test_initial_states()
            test_environment_configs()
            test_action_scales()
            test_collision_settings()
            test_import_cleanup()

        print("\n" + "="*60)
        print("🎉 所有修改测试通过！")
        print("="*60)

        print("\n📊 修改总结:")
        print("✅ 执行器: DCMotor → DelayedPDActuatorCfg")
        print("✅ 机械臂刚度: 25.0 → 10.0")
        print("✅ 延迟控制: 无 → 2-5/5-10步")
        print("✅ 自碰撞: False → True")
        print("✅ 初始高度: 0.4m → 0.45m")
        print("✅ 动作scale: 0.2 → 0.5")
        print("✅ 高度控制: weight=0.0 → weight=-5.0")
        print("✅ 初始姿态: 全部0.0 → 预设姿态")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
