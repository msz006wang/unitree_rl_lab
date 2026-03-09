#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件语法验证脚本
Configuration File Syntax Validation Script

这个脚本验证地形配置文件的语法正确性，无需运行Isaac Sim。
This script validates the syntax of terrain configuration files without running Isaac Sim.
"""

import ast
import sys
import os
from pathlib import Path


def verify_config_file_syntax(file_path):
    """验证Python配置文件的语法 Verify Python config file syntax"""
    print(f"验证配置文件 / Verifying config file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Parse the file to check syntax
        ast.parse(code)
        print("  ✅ 语法正确 / Syntax is valid")
        return True, code
    except SyntaxError as e:
        print(f"  ❌ 语法错误 / Syntax error: {e}")
        return False, None
    except Exception as e:
        print(f"  ❌ 错误 / Error: {e}")
        return False, None


def check_terrain_config(content):
    """检查地形配置的关键特征 Check key features of terrain config"""
    print("\n检查地形配置特征 / Checking terrain configuration features:")

    checks = {
        'num_rows=16': False,
        'PROGRESSIVE_TERRAINS_CFG': False,
        'curriculum=True': False,
        'sub_terrains': False,
        'rough_terrain': False,
        'gentle_slopes': False,
        'stairs': False,
        'obstacles': False,
    }

    for check in checks.keys():
        if check in content:
            checks[check] = True
            print(f"  ✅ 找到 / Found: {check}")
        else:
            print(f"  ❌ 缺失 / Missing: {check}")

    all_passed = all(checks.values())
    if all_passed:
        print("\n✅ 所有关键特征都存在 / All key features present")
    else:
        print("\n⚠️  部分特征缺失 / Some features missing")

    return all_passed


def check_config_classes(content):
    """检查配置类的关键修改 Check key modifications to config classes"""
    print("\n检查配置类修改 / Checking config class modifications:")

    checks = {
        'EventCfg': 'physics_material',
        'CommandsCfg': 'base_velocity',
        'ActionsCfg': 'JointPositionAction',
        'TerminationsCfg': 'base_height',
        'CurriculumCfg': 'terrain_levels',
    }

    all_passed = True
    for class_name, key_feature in checks.items():
        if class_name in content and key_feature in content:
            print(f"  ✅ {class_name} 包含 {key_feature}")
        else:
            print(f"  ❌ {class_name} 或 {key_feature} 未找到")
            all_passed = False

    return all_passed


def main():
    """主函数 Main function"""
    print("="*80)
    print("地形配置验证工具 / Terrain Configuration Validation Tool")
    print("="*80 + "\n")

    # Path to the config file
    config_path = Path("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py")

    # Check if file exists
    if not config_path.exists():
        print(f"❌ 配置文件不存在 / Config file not found: {config_path}")
        return False

    print(f"配置文件路径 / Config file path: {config_path}")
    print()

    # Verify syntax
    success, content = verify_config_file_syntax(config_path)
    if not success:
        return False

    # Check terrain config
    terrain_ok = check_terrain_config(content)

    # Check config classes
    classes_ok = check_config_classes(content)

    # Summary
    print("\n" + "="*80)
    print("验证摘要 / Validation Summary")
    print("="*80)
    print(f"语法验证 / Syntax Validation: {'✅ 通过 / Passed' if success else '❌ 失败 / Failed'}")
    print(f"地形配置 / Terrain Config: {'✅ 通过 / Passed' if terrain_ok else '❌ 失败 / Failed'}")
    print(f"配置类 / Config Classes: {'✅ 通过 / Passed' if classes_ok else '❌ 失败 / Failed'}")
    print("="*80)

    if success and terrain_ok and classes_ok:
        print("\n🎉 所有验证通过！配置文件已准备好使用。")
        print("🎉 All validations passed! Config file is ready to use.\n")
        return True
    else:
        print("\n⚠️  部分验证失败，请检查配置。")
        print("⚠️  Some validations failed, please check the configuration.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
