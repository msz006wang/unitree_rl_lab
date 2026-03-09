#!/usr/bin/env python3
"""Verify UNITREE_GO2W_CFG configuration file (No imports required).

This script checks the configuration file structure without importing any modules.
"""

import re
from pathlib import Path


def verify_go2w_config():
    """Verify the UNITREE_GO2W_CFG configuration by parsing the file."""
    print("=" * 80)
    print("Verifying UNITREE_GO2W_CFG Configuration File")
    print("=" * 80)

    # Path to the unitree.py file
    config_file = Path("source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py")

    if not config_file.exists():
        print(f"✗ Configuration file not found: {config_file}")
        return False

    print(f"✓ Configuration file found: {config_file}\n")

    # Read the file
    content = config_file.read_text()

    # Find all UNITREE_GO2W_CFG definitions (including commented ones)
    go2w_pattern = r'^(#)?\s*UNITREE_GO2W_CFG\s*=\s*(\w+)'
    matches = list(re.finditer(go2w_pattern, content, re.MULTILINE))

    if not matches:
        print("✗ UNITREE_GO2W_CFG definition not found")
        return False

    # Get the last non-commented definition
    active_match = None
    for match in matches:
        if not match.group(1):  # Not commented
            active_match = match

    if not active_match:
        print("✗ No active (non-commented) UNITREE_GO2W_CFG found")
        return False

    cfg_class = active_match.group(2)
    print(f"✓ Found UNITREE_GO2W_CFG = {cfg_class}\n")

    # Extract the UNITREE_GO2W_CFG definition
    # Start from the active match position
    start_idx = active_match.start()
    if start_idx == -1:
        print("✗ Could not find configuration start")
        return False

    # Find matching closing parenthesis
    depth = 0
    in_string = False
    escape_next = False
    start_paren = content.find('(', start_idx)

    if start_paren == -1:
        print("✗ No opening parenthesis found")
        return False

    for i in range(start_paren, len(content)):
        char = content[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char in ('"', "'"):
            if not in_string:
                in_string = char
            elif in_string == char:
                in_string = False
            continue

        if in_string:
            continue

        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    else:
        print("✗ Could not find matching closing parenthesis")
        return False

    config_text = content[start_idx:end_idx]

    # Verify key components
    print("1. Configuration Structure:")
    checks = []

    # Check for spawn configuration
    if 'spawn=' in config_text:
        checks.append(('✓', 'spawn configuration found'))
        if 'UnitreeUrdfFileCfg' in config_text:
            checks.append(('✓', 'using UnitreeUrdfFileCfg'))
            # Try to extract asset path
            asset_match = re.search(r'asset_path\s*=\s*f?"([^"]*)"', config_text)
            if asset_match:
                asset_path = asset_match.group(1).replace('{ISAACLAB_ASSETS_DATA_DIR}', '')
                checks.append(('  ', f'  asset path: {asset_path}'))
        elif 'UnitreeUsdFileCfg' in config_text:
            checks.append(('✓', 'using UnitreeUsdFileCfg'))
    else:
        checks.append(('✗', 'spawn configuration missing'))

    # Check for init_state
    if 'init_state=' in config_text:
        checks.append(('✓', 'init_state configuration found'))
        # Try to extract position
        pos_match = re.search(r'pos\s*=\s*\(([^)]+)\)', config_text)
        if pos_match:
            checks.append(('  ', f'  initial position: ({pos_match.group(1)})'))
    else:
        checks.append(('✗', 'init_state configuration missing'))

    # Check for actuators
    if 'actuators=' in config_text:
        checks.append(('✓', 'actuators configuration found'))
        # Count actuator groups
        actuator_matches = re.findall(r'"(\w+)":\s*\w+', config_text)
        if actuator_matches:
            checks.append(('  ', f'  actuator groups: {", ".join(actuator_matches)}'))
    else:
        checks.append(('✗', 'actuators configuration missing'))

    # Check for joint_sdk_names
    if 'joint_sdk_names=' in config_text:
        checks.append(('✓', 'joint_sdk_names found'))
        # Try to count joints
        names_match = re.search(r'joint_sdk_names\s*=\s*\[(.*?)\]', config_text, re.DOTALL)
        if names_match:
            joint_names = [name.strip().strip('"\'') for name in names_match.group(1).split(',') if name.strip()]
            if joint_names:
                checks.append(('  ', f'  total joints: {len(joint_names)}'))
    else:
        checks.append(('✗', 'joint_sdk_names missing'))

    # Check for soft_joint_pos_limit_factor
    if 'soft_joint_pos_limit_factor' in config_text:
        checks.append(('✓', 'soft_joint_pos_limit_factor specified'))
    else:
        checks.append(('  ', 'soft_joint_pos_limit_factor not specified (using default)'))

    for status, message in checks:
        print(f"   {status} {message}")
    print()

    # Check for commented out original configuration
    # Look before the active configuration
    context_start = max(0, start_idx - 2000)
    context = content[context_start:start_idx]
    if '# Original configuration' in context or '# UNITREE_GO2W_CFG' in context:
        print("2. Backup Configuration:")
        print("   ✓ Original configuration preserved as comment\n")
    else:
        print("2. Backup Configuration:")
        print("   (no backup comment found)\n")

    # Check configuration style matches reference.py
    print("3. Configuration Style:")
    style_checks = []

    if 'UnitreeUrdfFileCfg' in config_text:
        style_checks.append(('✓', 'Uses UnitreeUrdfFileCfg (URDF format)'))

    if 'ImplicitActuatorCfg' in config_text:
        style_checks.append(('✓', 'Uses ImplicitActuatorCfg'))

    if 'effort_limit_sim' in config_text or 'velocity_limit_sim' in config_text:
        style_checks.append(('✓', 'Uses simulation limit parameters (*_sim)'))

    if '"legs"' in config_text and '"wheels"' in config_text:
        style_checks.append(('✓', 'Separates legs and wheels actuators'))

    for status, message in style_checks:
        print(f"   {status} {message}")
    print()

    print("=" * 80)
    print("✓ Configuration file verification completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Ensure asset files exist at the specified paths")
    print("2. Run full simulation test with: ./scripts/run_go2w_test.sh --headless")

    return True


def main():
    """Main function."""
    try:
        success = verify_go2w_config()
        if success:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
