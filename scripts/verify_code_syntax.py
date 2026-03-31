#!/usr/bin/env python
"""
代码语法验证脚本

验证所有新增的代码文件语法是否正确，无需导入完整的依赖库。
"""

import sys
from pathlib import Path
import ast

def verify_file_syntax(file_path):
    """验证Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"读取错误: {e}"


def check_function_definitions(file_path, function_names):
    """检查文件中是否定义了指定的函数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)

        # 获取所有函数定义
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        found = []
        missing = []
        for func_name in function_names:
            if func_name in functions:
                found.append(func_name)
            else:
                missing.append(func_name)

        return found, missing
    except Exception as e:
        return [], list(function_names)


def main():
    """主函数"""
    print("=" * 70)
    print("GO2W ARM 综合优化 - 代码语法验证")
    print("=" * 70)

    project_root = Path(__file__).parent.parent

    # 定义要验证的文件和函数
    checks = [
        {
            "name": "扩展奖励函数",
            "file": "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py",
            "functions": [
                "upward_velocity",
                "orientation_tracking",
                "torque_penalty",
                "joint_regularization",
                "contact_management",
                "wheel_assisted_recovery",
            ],
        },
        {
            "name": "扩展观测函数",
            "file": "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py",
            "functions": [
                "history_buffer",
                "joint_pos_history",
                "body_vel_history",
            ],
        },
        {
            "name": "MDP导出",
            "file": "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py",
            "functions": None,  # 只检查语法
        },
        {
            "name": "环境配置",
            "file": "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py",
            "functions": None,  # 只检查语法
        },
    ]

    all_passed = True
    for check in checks:
        file_path = project_root / check["file"]

        print(f"\n{'-' * 70}")
        print(f"检查: {check['name']}")
        print(f"文件: {check['file']}")
        print('-' * 70)

        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            all_passed = False
            continue

        # 验证语法
        syntax_ok, syntax_msg = verify_file_syntax(file_path)
        if syntax_ok:
            print(f"✅ 语法验证: {syntax_msg}")
        else:
            print(f"❌ 语法验证: {syntax_msg}")
            all_passed = False
            continue

        # 检查函数定义
        if check["functions"]:
            found, missing = check_function_definitions(file_path, check["functions"])

            print(f"\n函数检查:")
            for func in found:
                print(f"  ✅ {func}: 已定义")
            for func in missing:
                print(f"  ❌ {func}: 未定义")
                all_passed = False

    # 检查配置文件中的关键配置
    print(f"\n{'=' * 70}")
    print("配置关键项检查")
    print('=' * 70)

    config_file = project_root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py"

    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_code = f.read()

        # 检查关键配置项
        checks = [
            ("upward_velocity", "Upward Velocity奖励"),
            ("orientation_tracking", "Orientation Tracking奖励"),
            ("torque_penalty", "Torque Penalty奖励"),
            ("joint_regularization", "Joint Regularization奖励"),
            ("contact_management", "Contact Management奖励"),
            ("wheel_assisted_recovery", "Wheel Assisted Recovery奖励"),
            ("joint_pos_history", "关节位置历史观测"),
            ("body_vel_history", "身体速度历史观测"),
            ('joint_pos.joint_names = self.leg_joint_names + ["arm_joint1"]', "动作空间包含arm_joint1"),
            ("self.actions.joint_pos.scale", "动作缩放配置"),
        ]

        for check_str, description in checks:
            if check_str in config_code:
                print(f"✅ {description}: 已配置")
            else:
                print(f"⚠️  {description}: 未找到（可能在运行时配置）")

    print(f"\n{'=' * 70}")
    print("验证总结")
    print('=' * 70)

    if all_passed:
        print("✅ 所有语法和函数检查通过！")
        print("\n下一步:")
        print("  1. 运行训练脚本")
        print("  2. 使用TensorBoard监控训练")
        print("  3. 参考详细文档: docs/GO2W_ARM_COMPREHENSIVE_OPTIMIZATION.md")
    else:
        print("❌ 部分检查失败，请查看上述错误信息。")

    print('=' * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
