#!/usr/bin/env python3
"""验证 wheel_angular_momentum_reward 修复"""

import ast

# 读取文件
with open('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py', 'r') as f:
    content = f.read()

# 解析AST
tree = ast.parse(content)

# 查找 wheel_angular_momentum_reward 函数
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'wheel_angular_momentum_reward':
        print("✅ 找到 wheel_angular_momentum_reward 函数")

        # 检查函数体中是否还有对 contact_sensor.cfg.body_names 的引用
        func_source = ast.unparse(node)
        if 'contact_sensor.cfg.body_names' in func_source:
            print("❌ 错误：函数体中仍然包含 contact_sensor.cfg.body_names")
            print("这会导致 AttributeError")
        else:
            print("✅ 已移除 contact_sensor.cfg.body_names 的引用")

        # 检查是否使用了 sensor_cfg.body_ids
        if 'sensor_cfg.body_ids' in func_source:
            print("✅ 正确使用 sensor_cfg.body_ids 获取 body 索引")
        else:
            print("⚠️  警告：未找到 sensor_cfg.body_ids 的使用")

        print("\n函数参数:")
        for arg in node.args.args:
            print(f"  - {arg.arg}")
        break
else:
    print("❌ 未找到 wheel_angular_momentum_reward 函数")

print("\n✅ 修复验证完成！")
