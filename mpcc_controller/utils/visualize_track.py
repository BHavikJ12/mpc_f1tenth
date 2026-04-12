#!/usr/bin/env python3
"""
Visualize track waypoints

Usage:
    ros2 run mpcc_controller visualize_track <waypoints.csv>
"""

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def load_waypoints(filepath):
    """Load waypoints from CSV"""
    waypoints = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x_m'])
            y = float(row['y_m'])
            v = float(row.get('vx_mps', 0.0))
            waypoints.append([x, y, v])
    
    return np.array(waypoints)


def visualize(waypoints):
    """Create visualization of track"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Track layout
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'b-o', markersize=3, linewidth=2)
    ax1.plot(waypoints[0, 0], waypoints[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(waypoints[-1, 0], waypoints[-1, 1], 'ro', markersize=12, label='End')
    
    # Annotate every 10th point
    for i in range(0, len(waypoints), 10):
        ax1.annotate(str(i), (waypoints[i, 0], waypoints[i, 1]),
                    fontsize=8, color='red')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Track Centerline ({len(waypoints)} waypoints)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # Right: Velocity profile
    distances = np.sqrt(np.diff(waypoints[:, 0])**2 + np.diff(waypoints[:, 1])**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    
    ax2.plot(cumulative_dist, waypoints[:, 2], 'b-', linewidth=2)
    ax2.set_xlabel('Distance along track (m)')
    ax2.set_ylabel('Target velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    total_length = cumulative_dist[-1]
    avg_spacing = np.mean(distances)
    
    stats_text = f'Track length: {total_length:.2f}m\n'
    stats_text += f'Waypoints: {len(waypoints)}\n'
    stats_text += f'Avg spacing: {avg_spacing:.3f}m\n'
    stats_text += f'Min velocity: {waypoints[:, 2].min():.2f}m/s\n'
    stats_text += f'Max velocity: {waypoints[:, 2].max():.2f}m/s'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print("TRACK STATISTICS")
    print(f"{'='*60}")
    print(f"Total length: {total_length:.2f} meters")
    print(f"Number of waypoints: {len(waypoints)}")
    print(f"Average spacing: {avg_spacing:.3f} meters")
    print(f"Velocity range: [{waypoints[:, 2].min():.2f}, {waypoints[:, 2].max():.2f}] m/s")
    print(f"Bounding box:")
    print(f"  X: [{waypoints[:, 0].min():.2f}, {waypoints[:, 0].max():.2f}]")
    print(f"  Y: [{waypoints[:, 1].min():.2f}, {waypoints[:, 1].max():.2f}]")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run mpcc_controller visualize_track <waypoints.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"Loading waypoints from: {filepath}")
    waypoints = load_waypoints(filepath)
    print(f"Loaded {len(waypoints)} waypoints")
    
    visualize(waypoints)


if __name__ == '__main__':
    main()
