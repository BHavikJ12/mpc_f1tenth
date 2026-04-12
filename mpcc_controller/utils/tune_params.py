#!/usr/bin/env python3
"""
Interactive parameter tuning GUI for MPCC controller

Usage:
    ros2 run mpcc_controller tune_params
"""

import sys
import yaml
from pathlib import Path


def main():
    print("="*60)
    print("MPCC Parameter Tuning")
    print("="*60)
    print()
    print("This utility helps you tune MPCC parameters interactively.")
    print()
    
    # Find params file
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_dir = get_package_share_directory('mpcc_controller')
        params_file = Path(pkg_dir) / 'config' / 'mpcc_params.yaml'
    except:
        params_file = Path('config/mpcc_params.yaml')
    
    if not params_file.exists():
        print(f"Error: Cannot find {params_file}")
        print("Please run from the mpcc_controller package directory")
        sys.exit(1)
    
    # Load current parameters
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    
    mpcc_params = params.get('mpcc_controller', {}).get('ros__parameters', {})
    
    print(f"Current parameters loaded from: {params_file}")
    print()
    
    # Display and allow editing
    print("Current Cost Weights:")
    print(f"  q_contour (contouring error penalty): {mpcc_params.get('q_contour', 10.0)}")
    print(f"  q_lag (lag error penalty): {mpcc_params.get('q_lag', 100.0)}")
    print(f"  gamma (progress reward): {mpcc_params.get('gamma', 1.0)}")
    print(f"  q_slack (track boundary penalty): {mpcc_params.get('q_slack', 1000.0)}")
    print()
    
    print("Tuning Guidelines:")
    print("="*60)
    print("q_contour (default: 10.0)")
    print("  - Higher = stay closer to centerline")
    print("  - Lower = allow more deviation for speed")
    print()
    print("q_lag (default: 100.0)")
    print("  - Higher = follow reference progress more closely")
    print("  - Lower = allow more lag for better racing line")
    print()
    print("gamma (default: 1.0)")
    print("  - Higher = encourage faster speeds")
    print("  - Lower = prioritize accuracy over speed")
    print()
    print("q_slack (default: 1000.0)")
    print("  - Very high to avoid track violations")
    print("  - Don't change unless you understand the consequences")
    print()
    print("="*60)
    print()
    
    # Interactive editing
    print("Would you like to modify parameters? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
        print("\nEnter new values (press Enter to keep current value):")
        print()
        
        # q_contour
        current = mpcc_params.get('q_contour', 10.0)
        new_val = input(f"q_contour [{current}]: ").strip()
        if new_val:
            try:
                mpcc_params['q_contour'] = float(new_val)
            except ValueError:
                print("Invalid input, keeping current value")
        
        # q_lag
        current = mpcc_params.get('q_lag', 100.0)
        new_val = input(f"q_lag [{current}]: ").strip()
        if new_val:
            try:
                mpcc_params['q_lag'] = float(new_val)
            except ValueError:
                print("Invalid input, keeping current value")
        
        # gamma
        current = mpcc_params.get('gamma', 1.0)
        new_val = input(f"gamma [{current}]: ").strip()
        if new_val:
            try:
                mpcc_params['gamma'] = float(new_val)
            except ValueError:
                print("Invalid input, keeping current value")
        
        # Update params dict
        params['mpcc_controller']['ros__parameters'] = mpcc_params
        
        # Save
        print("\nSave changes? (y/n): ", end='')
        if input().strip().lower() == 'y':
            with open(params_file, 'w') as f:
                yaml.dump(params, f, default_flow_style=False)
            print(f"\nParameters saved to: {params_file}")
            print("\nRebuild the package for changes to take effect:")
            print("  cd ~/ros2_ws")
            print("  colcon build --packages-select mpcc_controller")
            print("  source install/setup.bash")
        else:
            print("\nChanges discarded")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
