#!/usr/bin/env python3
"""
Improved centerline extraction using distance transform
Works better for narrow tracks like Levine
"""
import numpy as np
import cv2
import yaml
import csv
import sys
import os
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def find_map_files():
    """Try to find map files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [script_dir, os.path.join(script_dir, '..'), os.getcwd()]
    
    for base_path in search_paths:
        png_path = os.path.join(base_path, 'Spielberg.png')
        yaml_path = os.path.join(base_path, 'Spielberg.yaml')
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
    
    # ====================================================================
    # IMPROVED: Use distance transform instead of simple erosion
    # ====================================================================
    
    # Threshold to get free space (white = free)
    free_space = (map_img > 200).astype(np.uint8)
    print(f"Free space pixels: {np.sum(free_space)}")
    
    # Distance transform: each pixel = distance to nearest obstacle
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    
    # Visualize distance transform (for debugging)
    plt.figure(figsize=(10, 10))
    plt.imshow(dist_transform, cmap='hot')
    plt.colorbar(label='Distance to wall (pixels)')
    plt.title('Distance Transform')
    plt.savefig('distance_transform.png', dpi=150)
    print("✓ Saved distance_transform.png (for debugging)")
    plt.close()
    
    # Keep only pixels far from walls
    # Adjust this threshold: higher = stays further from walls
    # For Levine at 0.05m/pixel: 
    #   5 pixels = 0.25m, 10 pixels = 0.5m, 15 pixels = 0.75m
    min_dist_pixels = 5  # ~0.25m safety margin (REDUCED to avoid gaps)
    
    safe_corridor = (dist_transform > min_dist_pixels).astype(np.uint8)
    print(f"Safe corridor pixels (>{min_dist_pixels}px from walls): {np.sum(safe_corridor)}")
    
    # Check if corridor is empty
    if np.sum(safe_corridor) < 100:
        print(f"WARNING: Very few safe pixels! Try lower min_dist_pixels")
        print(f"Current: {min_dist_pixels}, try: {min_dist_pixels//2}")
        min_dist_pixels = max(3, min_dist_pixels // 2)
        safe_corridor = (dist_transform > min_dist_pixels).astype(np.uint8)
        print(f"Retrying with min_dist={min_dist_pixels}: {np.sum(safe_corridor)} pixels")
    
    # Skeleton of safe corridor = centerline
    skeleton = skeletonize(safe_corridor > 0)
    
    # ADDED: Close small gaps in skeleton to prevent disconnected segments
    from scipy import ndimage
    kernel_close = np.ones((3, 3), dtype=bool)
    skeleton = ndimage.binary_closing(skeleton, structure=kernel_close, iterations=3)
    print("✓ Applied morphological closing to connect gaps")
    
    skeleton_pixels = np.argwhere(skeleton)
    
    print(f"✓ Skeleton pixels: {len(skeleton_pixels)}")
    
    if len(skeleton_pixels) == 0:
        print("ERROR: No skeleton found! Track might be too narrow.")
        print("Try reducing min_dist_pixels in the script.")
        sys.exit(1)
    
    # Convert to world coordinates
    waypoints = []
    for pixel in skeleton_pixels:
        row, col = pixel
        x = origin[0] + col * resolution
        y = origin[1] + (map_img.shape[0] - row) * resolution
        waypoints.append([x, y])
    
    waypoints = np.array(waypoints)
    
    # Order waypoints (nearest neighbor with gap detection)
    print("Ordering waypoints...")
    
    # Start from point closest to origin (usually bottom-left)
    distances_from_origin = np.linalg.norm(waypoints, axis=1)
    start_idx = np.argmin(distances_from_origin)
    
    ordered = [waypoints[start_idx]]
    remaining = set(range(len(waypoints))) - {start_idx}
    
    # Maximum allowed gap between consecutive waypoints
    max_gap = 1.5  # meters - prevents jumping across track
    
    while remaining:
        last = ordered[-1]
        remaining_pts = waypoints[list(remaining)]
        dists = np.linalg.norm(remaining_pts - last, axis=1)
        nearest_dist = np.min(dists)
        
        # Check for large gaps (disconnected segments)
        if nearest_dist > max_gap:
            print(f"Warning: Gap detected ({nearest_dist:.2f}m > {max_gap}m)")
            # Don't jump - just stop this segment
            if len(ordered) > len(waypoints) * 0.3:  # Got at least 30%
                print(f"Stopping ordering - got {len(ordered)}/{len(waypoints)} waypoints")
                break
            else:
                # Try starting new segment
                print("Attempting to start new segment...")
        
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
        print(f"  Spacing: {np.mean(dists):.3f}m avg, {np.min(dists):.3f}m min, {np.max(dists):.3f}m max")
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
    extent = [origin[0], origin[0] + map_img.shape[1]*resolution,
              origin[1], origin[1] + map_img.shape[0]*resolution]
    axes[1,1].imshow(map_img, cmap='gray', origin='upper', extent=extent)
    axes[1,1].plot(waypoints_final[:,0], waypoints_final[:,1], 
                   'r-', linewidth=2, label='Centerline')
    axes[1,1].plot(waypoints_final[0,0], waypoints_final[0,1], 
                   'go', markersize=10, label='Start')
    axes[1,1].set_xlabel('X (m)')
    axes[1,1].set_ylabel('Y (m)')
    axes[1,1].set_title(f'Final Waypoints ({len(waypoints_final)} points)')
    axes[1,1].legend()
    axes[1,1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('waypoints_extraction_process.png', dpi=150)
    print(f"✓ Saved waypoints_extraction_process.png")
    plt.show()
    
    print("\n✓ Done!")

if __name__ == '__main__':
    main()