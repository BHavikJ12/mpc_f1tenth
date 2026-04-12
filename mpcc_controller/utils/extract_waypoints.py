#!/usr/bin/env python3
"""
Extract centerline waypoints from occupancy grid map
Clean version - only fixes coordinate alignment
"""
import numpy as np
import cv2
import yaml
import csv
import sys
import os
from skimage.morphology import skeletonize
from scipy import ndimage
import matplotlib.pyplot as plt

def find_map_files():
    """Try to find map files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [script_dir, os.path.join(script_dir, '..'), os.getcwd()]
    
    for base_path in search_paths:
        png_path = os.path.join(base_path, 'levine.png')
        yaml_path = os.path.join(base_path, 'levine.yaml')
        if os.path.exists(png_path) and os.path.exists(yaml_path):
            return png_path, yaml_path
    return None, None

def main():
    # Find files
    if len(sys.argv) >= 3:
        image_file, yaml_file = sys.argv[1], sys.argv[2]
    else:
        image_file, yaml_file = find_map_files()
        if image_file is None:
            print("ERROR: Could not find levine.png and levine.yaml")
            sys.exit(1)
    
    print(f"Loading: {image_file}")
    
    # Load map
    map_img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        print(f"ERROR: Cannot read {image_file}")
        sys.exit(1)
    
    with open(yaml_file) as f:
        metadata = yaml.safe_load(f)
    
    resolution = metadata['resolution']
    origin = metadata['origin'][:2]
    
    print(f"Image: {map_img.shape}, Resolution: {resolution}m/px, Origin: {origin}")
    
    # Get free space (white = free)
    free_space = (map_img > 200).astype(np.uint8)
    print(f"Free space pixels: {np.sum(free_space)}")
    
    # Distance transform: each pixel = distance to nearest obstacle
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    
    # Keep only pixels far from walls
    min_dist_pixels = 5  # ~0.25m safety margin
    safe_corridor = (dist_transform > min_dist_pixels).astype(np.uint8)
    print(f"Safe corridor pixels (>{min_dist_pixels}px from walls): {np.sum(safe_corridor)}")
    
    if np.sum(safe_corridor) < 100:
        print(f"WARNING: Very few safe pixels! Reducing threshold...")
        min_dist_pixels = 3
        safe_corridor = (dist_transform > min_dist_pixels).astype(np.uint8)
        print(f"Retrying with min_dist={min_dist_pixels}: {np.sum(safe_corridor)} pixels")
    
    # Skeleton of safe corridor = centerline
    skeleton = skeletonize(safe_corridor > 0)
    
    # Close small gaps
    kernel_close = np.ones((3, 3), dtype=bool)
    skeleton = ndimage.binary_closing(skeleton, structure=kernel_close, iterations=1)
    
    skeleton_pixels = np.argwhere(skeleton)
    print(f"✓ Skeleton pixels: {len(skeleton_pixels)}")
    
    if len(skeleton_pixels) == 0:
        print("ERROR: No skeleton found!")
        sys.exit(1)
    
    # ========================================================================
    # CRITICAL FIX: Correct coordinate transformation
    # ========================================================================
    waypoints = []
    for pixel in skeleton_pixels:
        row, col = pixel
        
        # Convert pixel coordinates to world coordinates
        # Image: row=0 is top, col=0 is left
        # World: origin is bottom-left, Y increases upward
        x = origin[0] + col * resolution
        y = origin[1] + (map_img.shape[0] - 1 - row) * resolution
        
        waypoints.append([x, y])
    
    waypoints = np.array(waypoints)
    print(f"Converted to world coordinates: {len(waypoints)} points")
    
    # Order waypoints (nearest neighbor)
    print("Ordering waypoints...")
    distances_from_origin = np.linalg.norm(waypoints, axis=1)
    start_idx = np.argmin(distances_from_origin)
    
    ordered = [waypoints[start_idx]]
    remaining = set(range(len(waypoints))) - {start_idx}
    
    max_gap = 1.5  # meters
    
    while remaining:
        last = ordered[-1]
        remaining_pts = waypoints[list(remaining)]
        dists = np.linalg.norm(remaining_pts - last, axis=1)
        nearest_dist = np.min(dists)
        
        if nearest_dist > max_gap:
            print(f"Warning: Gap detected ({nearest_dist:.2f}m)")
            if len(ordered) > len(waypoints) * 0.3:
                break
        
        nearest_idx = list(remaining)[np.argmin(dists)]
        ordered.append(waypoints[nearest_idx])
        remaining.remove(nearest_idx)
    
    waypoints_ordered = np.array(ordered)
    print(f"✓ Ordered: {len(waypoints_ordered)} points")
    
    # Downsample
    target_spacing = 0.5
    sampled = [waypoints_ordered[0]]
    for i in range(1, len(waypoints_ordered)):
        if np.linalg.norm(waypoints_ordered[i] - sampled[-1]) >= target_spacing:
            sampled.append(waypoints_ordered[i])
    
    waypoints_final = np.array(sampled)
    print(f"✓ Downsampled: {len(waypoints_final)} waypoints")
    
    # Smooth
    if len(waypoints_final) > 10:
        window = 5
        smoothed = np.copy(waypoints_final)
        for i in range(window, len(waypoints_final) - window):
            smoothed[i] = np.mean(waypoints_final[i-window:i+window+1], axis=0)
        waypoints_final = smoothed
        print(f"✓ Smoothed with window={window}")
    
    # Save CSV
    output_file = 'waypoints.csv'
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x_m', 'y_m', 'vx_mps'])
        for pt in waypoints_final:
            writer.writerow([f"{pt[0]:.6f}", f"{pt[1]:.6f}", "2.0"])
    
    print(f"\n✓ Saved {len(waypoints_final)} waypoints to {output_file}")
    
    # Statistics
    if len(waypoints_final) > 1:
        dists = np.linalg.norm(np.diff(waypoints_final, axis=0), axis=1)
        print(f"\nStatistics:")
        print(f"  Track length: {np.sum(dists):.2f}m")
        print(f"  Spacing: {np.mean(dists):.3f}m avg")
        print(f"  Bounds: X=[{waypoints_final[:,0].min():.1f}, {waypoints_final[:,0].max():.1f}]")
        print(f"          Y=[{waypoints_final[:,1].min():.1f}, {waypoints_final[:,1].max():.1f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Top-left: Original map
    axes[0,0].imshow(map_img, cmap='gray', origin='lower')
    axes[0,0].set_title('Original Map')
    
    # Top-right: Distance transform
    axes[0,1].imshow(dist_transform, cmap='hot', origin='lower')
    axes[0,1].contour(safe_corridor, colors='cyan', linewidths=2)
    axes[0,1].set_title(f'Distance Transform (threshold={min_dist_pixels}px)')
    
    # Bottom-left: Skeleton
    axes[1,0].imshow(map_img, cmap='gray', origin='lower')
    skel_y, skel_x = np.where(skeleton)
    axes[1,0].plot(skel_x, skel_y, 'r.', markersize=1)
    axes[1,0].set_title(f'Skeleton ({len(skeleton_pixels)} pixels)')
    
    # Bottom-right: Final waypoints in world coords
    # CRITICAL: Must match coordinate system used in conversion
    x_min = origin[0]
    x_max = origin[0] + map_img.shape[1] * resolution
    y_min = origin[1]
    y_max = origin[1] + map_img.shape[0] * resolution
    extent = [x_min, x_max, y_min, y_max]
    
    # FIXED: Use origin='upper' because we already flipped Y in conversion
    # If we use origin='lower', we flip TWICE (once in conversion, once in display)
    axes[1,1].imshow(map_img, cmap='gray', origin='upper', extent=extent)
    axes[1,1].plot(waypoints_final[:,0], waypoints_final[:,1], 
                   'r-', linewidth=2, label='Centerline')
    axes[1,1].plot(waypoints_final[0,0], waypoints_final[0,1], 
                   'go', markersize=10, label='Start')
    axes[1,1].set_xlabel('X (m)')
    axes[1,1].set_ylabel('Y (m)')
    axes[1,1].set_title(f'Final Waypoints ({len(waypoints_final)} points)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('waypoints_final.png', dpi=150)
    print(f"✓ Saved waypoints_final.png")
    plt.show()
    
    print("\n✓ Done!")

if __name__ == '__main__':
    main()


def main(args=None):
    """Entry point for ROS2 run command"""
    import sys
    if args is None:
        args = sys.argv[1:]
    
    # Run the original script logic
    if len(args) >= 2:
        image_file, yaml_file = args[0], args[1]
    else:
        image_file, yaml_file = find_map_files()
        if image_file is None:
            print("ERROR: Could not find levine.png and levine.yaml")
            sys.exit(1)
    
    # Rest of main() continues as before...
    print(f"Loading: {image_file}")
    
    # Load map
    map_img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    # ... (rest of the existing code)
