#!/usr/bin/env python3
"""
Visualize map, waypoints, and car position to debug coordinate alignment

Usage:
    python3 visualize_map_waypoints.py Spielberg.png Spielberg.yaml waypoints.csv

Optional: Add car position
    python3 visualize_map_waypoints.py Spielberg.png Spielberg.yaml waypoints.csv 1.27 0.70
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import csv
import sys
from PIL import Image


def load_map_yaml(yaml_file):
    """Load map metadata from YAML file"""
    with open(yaml_file, 'r') as f:
        map_data = yaml.safe_load(f)
    
    resolution = map_data['resolution']  # meters/pixel
    origin = map_data['origin']  # [x, y, theta] in meters
    
    print(f"Map metadata:")
    print(f"  Resolution: {resolution} m/pixel")
    print(f"  Origin: {origin} (x, y, theta)")
    
    return resolution, origin


def load_waypoints(csv_file):
    """Load waypoints from CSV file"""
    waypoints = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x_m'])
            y = float(row['y_m'])
            waypoints.append([x, y])
    
    waypoints = np.array(waypoints)
    print(f"\nWaypoints:")
    print(f"  Count: {len(waypoints)}")
    print(f"  X range: [{waypoints[:, 0].min():.2f}, {waypoints[:, 0].max():.2f}] m")
    print(f"  Y range: [{waypoints[:, 1].min():.2f}, {waypoints[:, 1].max():.2f}] m")
    
    return waypoints


def world_to_pixel(x, y, origin, resolution, image_height):
    """
    Convert world coordinates to pixel coordinates
    
    Args:
        x, y: World coordinates (meters)
        origin: Map origin [x_origin, y_origin, theta]
        resolution: meters per pixel
        image_height: Height of image in pixels (for Y-axis flip)
    
    Returns:
        px, py: Pixel coordinates
    """
    # Offset by origin
    x_rel = x - origin[0]
    y_rel = y - origin[1]
    
    # Convert to pixels
    px = x_rel / resolution
    py = y_rel / resolution
    
    # Flip Y (image coordinates have Y down, world has Y up)
    py = image_height - py
    
    return px, py


def pixel_to_world(px, py, origin, resolution, image_height):
    """Convert pixel coordinates to world coordinates"""
    # Flip Y back
    py_world = image_height - py
    
    # Convert to meters
    x = px * resolution + origin[0]
    y = py_world * resolution + origin[1]
    
    return x, y


def visualize_map_waypoints(map_file, yaml_file, waypoints_file, car_x=None, car_y=None):
    """
    Create comprehensive visualization of map and waypoints
    
    Args:
        map_file: Path to map PNG
        yaml_file: Path to map YAML
        waypoints_file: Path to waypoints CSV
        car_x, car_y: Optional car position in world coordinates
    """
    # Load map image
    print(f"Loading map: {map_file}")
    img = Image.open(map_file)
    img_array = np.array(img)
    image_height, image_width = img_array.shape[:2]
    print(f"  Image size: {image_width} x {image_height} pixels")
    
    # Load metadata
    resolution, origin = load_map_yaml(yaml_file)
    
    # Load waypoints
    waypoints = load_waypoints(waypoints_file)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ============================================
    # LEFT PLOT: Pixel Coordinates (Image View)
    # ============================================
    ax1.imshow(img_array, cmap='gray', origin='upper')
    ax1.set_title('Map in Pixel Coordinates\n(Raw Image View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    
    # Convert waypoints to pixel coordinates
    waypoints_px = np.array([
        world_to_pixel(wp[0], wp[1], origin, resolution, image_height)
        for wp in waypoints
    ])
    
    # Plot waypoints
    ax1.plot(waypoints_px[:, 0], waypoints_px[:, 1], 'b-', linewidth=2, label='Centerline', alpha=0.7)
    ax1.scatter(waypoints_px[:, 0], waypoints_px[:, 1], c='blue', s=10, zorder=5, alpha=0.5)
    
    # Mark first waypoint
    ax1.scatter(waypoints_px[0, 0], waypoints_px[0, 1], c='green', s=100, 
                marker='*', zorder=10, label='First waypoint', edgecolors='black', linewidths=1)
    
    # Mark origin (world 0,0)
    origin_px = world_to_pixel(0, 0, origin, resolution, image_height)
    ax1.scatter(origin_px[0], origin_px[1], c='red', s=200, marker='X', 
                zorder=10, label='World origin (0, 0)', edgecolors='black', linewidths=2)
    
    # Mark map origin (from YAML)
    map_origin_px = world_to_pixel(origin[0], origin[1], origin, resolution, image_height)
    ax1.scatter(map_origin_px[0], map_origin_px[1], c='orange', s=150, marker='s',
                zorder=10, label=f'Map origin {origin[:2]}', edgecolors='black', linewidths=1)
    
    # Mark car position if provided
    if car_x is not None and car_y is not None:
        car_px = world_to_pixel(car_x, car_y, origin, resolution, image_height)
        ax1.scatter(car_px[0], car_px[1], c='magenta', s=200, marker='D',
                    zorder=10, label=f'Car ({car_x:.2f}, {car_y:.2f})', 
                    edgecolors='black', linewidths=2)
        
        # Draw line to nearest waypoint
        distances = np.sqrt((waypoints[:, 0] - car_x)**2 + (waypoints[:, 1] - car_y)**2)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        nearest_px = waypoints_px[nearest_idx]
        ax1.plot([car_px[0], nearest_px[0]], [car_px[1], nearest_px[1]], 
                 'r--', linewidth=2, label=f'Distance: {nearest_dist:.2f}m')
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ============================================
    # RIGHT PLOT: World Coordinates (ROS View)
    # ============================================
    
    # Calculate world coordinate bounds
    world_width = image_width * resolution
    world_height = image_height * resolution
    
    world_x_min = origin[0]
    world_x_max = origin[0] + world_width
    world_y_min = origin[1]
    world_y_max = origin[1] + world_height
    
    # Show map in world coordinates
    extent = [world_x_min, world_x_max, world_y_min, world_y_max]
    ax2.imshow(img_array, cmap='gray', origin='lower', extent=extent, alpha=0.5)
    
    ax2.set_title('Map in World Coordinates\n(ROS/AMCL View)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('World X (meters)')
    ax2.set_ylabel('World Y (meters)')
    
    # Plot waypoints in world coordinates
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=2, label='Centerline', alpha=0.7)
    ax2.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', s=10, zorder=5, alpha=0.5)
    
    # Mark first waypoint
    ax2.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=100, 
                marker='*', zorder=10, label=f'First: ({waypoints[0, 0]:.2f}, {waypoints[0, 1]:.2f})', 
                edgecolors='black', linewidths=1)
    
    # Mark world origin
    ax2.scatter(0, 0, c='red', s=200, marker='X', zorder=10, 
                label='World origin (0, 0)', edgecolors='black', linewidths=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Mark map origin
    ax2.scatter(origin[0], origin[1], c='orange', s=150, marker='s', zorder=10,
                label=f'Map origin ({origin[0]:.2f}, {origin[1]:.2f})', 
                edgecolors='black', linewidths=1)
    
    # Mark car position
    if car_x is not None and car_y is not None:
        ax2.scatter(car_x, car_y, c='magenta', s=200, marker='D', zorder=10,
                    label=f'Car ({car_x:.2f}, {car_y:.2f})', edgecolors='black', linewidths=2)
        
        # Distance to nearest waypoint
        ax2.plot([car_x, waypoints[nearest_idx, 0]], 
                 [car_y, waypoints[nearest_idx, 1]], 
                 'r--', linewidth=2, label=f'Distance: {nearest_dist:.2f}m')
        
        # Print diagnostic info
        print(f"\n🚗 Car Position Analysis:")
        print(f"  Car: ({car_x:.2f}, {car_y:.2f})")
        print(f"  Nearest waypoint: ({waypoints[nearest_idx, 0]:.2f}, {waypoints[nearest_idx, 1]:.2f})")
        print(f"  Distance to track: {nearest_dist:.2f} m")
        
        if nearest_dist < 1.0:
            print(f"  ✅ GOOD: Car is on the track!")
        elif nearest_dist < 5.0:
            print(f"  ⚠️  WARNING: Car is {nearest_dist:.1f}m from track")
        else:
            print(f"  ❌ ERROR: Car is {nearest_dist:.1f}m off track!")
    
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # ============================================
    # Add info text
    # ============================================
    info_text = f"""Map Info:
Resolution: {resolution} m/pixel
Origin: ({origin[0]:.2f}, {origin[1]:.2f}) m
Image: {image_width} × {image_height} px
World size: {world_width:.2f} × {world_height:.2f} m

Waypoints: {len(waypoints)} points
X: [{waypoints[:, 0].min():.2f}, {waypoints[:, 0].max():.2f}] m
Y: [{waypoints[:, 1].min():.2f}, {waypoints[:, 1].max():.2f}] m"""
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'map_waypoints_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    
    plt.show()


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 visualize_map_waypoints.py <map.png> <map.yaml> <waypoints.csv> [car_x] [car_y]")
        print("\nExample:")
        print("  python3 visualize_map_waypoints.py Spielberg.png Spielberg.yaml waypoints.csv")
        print("  python3 visualize_map_waypoints.py Spielberg.png Spielberg.yaml waypoints.csv 1.27 0.70")
        sys.exit(1)
    
    map_file = sys.argv[1]
    yaml_file = sys.argv[2]
    waypoints_file = sys.argv[3]
    
    car_x = None
    car_y = None
    if len(sys.argv) >= 6:
        car_x = float(sys.argv[4])
        car_y = float(sys.argv[5])
    
    print("=" * 70)
    print("MAP AND WAYPOINT VISUALIZATION")
    print("=" * 70)
    
    visualize_map_waypoints(map_file, yaml_file, waypoints_file, car_x, car_y)


if __name__ == "__main__":
    main()