"""
useOpen3DVisualizationPLYpoint cloud file
Support loading、show、rotate、Interactive operations such as zooming
"""

import open3d as o3d
import numpy as np
import argparse
from pathlib import Path


def visualize_pointcloud(
    ply_path,
    point_size=2.0,
    background_color=(0.1, 0.1, 0.1),
    window_name="Point Cloud Viewer",
    width=1280,
    height=720,
):
    """
    VisualizationPLYpoint cloud file
    
    Args:
        ply_path: PLYfile path
        point_size: dot size
        background_color: background color (R, G, B)
        window_name: window name
        width: window width
        height: window height
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLYFile does not exist: {ply_path}")
    
    print(f"Load point cloud file: {ply_path}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if len(pcd.points) == 0:
        raise ValueError("Point cloud file is empty")
    
    print(f"Point cloud information:")
    print(f"  Points: {len(pcd.points):,}")
    print(f"  color: {'yes' if pcd.has_colors() else 'no'}")
    print(f"  Normal: {'yes' if pcd.has_normals() else 'no'}")
    
    # Compute point cloud boundaries
    points = np.asarray(pcd.points)
    print(f"  scope:")
    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Computing Point Cloud Center
    center = points.mean(axis=0)
    print(f"  center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # if there is no color，Color based on height
    if not pcd.has_colors():
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_normalized = (z_values - z_min) / (z_max - z_min + 1e-8)
        
        # usejet colormap
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_normalized
        colors[:, 1] = 1.0 - np.abs(z_normalized - 0.5) * 2
        colors[:, 2] = 1.0 - z_normalized
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("  Automatically colored based on height")
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = point_size
    
    # Set up view controls
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # Run the visualization
    print("\nVisual control:")
    print("  - left mouse button drag: rotate")
    print("  - Right mouse button drag: Pan")
    print("  - mouse wheel: Zoom")
    print("  - according to 'H' key: show help")
    print("  - according to 'Q' key or close the window: quit")
    
    vis.run()
    vis.destroy_window()
    
    print("End of visualization")


def visualize_multiple_pointclouds(
    ply_paths,
    point_size=2.0,
    background_color=(0.1, 0.1, 0.1),
    window_name="Multiple Point Clouds Viewer",
    width=1280,
    height=720,
):
    """
    Visualize multiplePLYpoint cloud file
    
    Args:
        ply_paths: PLYfile path list
        point_size: dot size
        background_color: background color (R, G, B)
        window_name: window name
        width: window width
        height: window height
    """
    if not ply_paths:
        raise ValueError("Point cloud path list is empty")
    
    print(f"load {len(ply_paths)} point cloud files")
    
    # Predefined color list（red and green，Used to compare forecasts andGT）
    colors_list = [
        [1.0, 0.0, 0.0],  # red（predict）
        [0.0, 1.0, 0.0],  # green（GT）
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # blue
    ]
    
    all_points = []
    all_pcds = []
    
    for idx, ply_path in enumerate(ply_paths):
        ply_path = Path(ply_path)
        
        if not ply_path.exists():
            print(f"warn: File does not exist，jump over: {ply_path}")
            continue
        
        print(f"\n[{idx + 1}/{len(ply_paths)}] load: {ply_path.name}")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        if len(pcd.points) == 0:
            print(f"  warn: Point cloud is empty，jump over")
            continue
        
        print(f"  Points: {len(pcd.points):,}")
        
        # if there is no color，Use predefined colors
        if not pcd.has_colors():
            color = colors_list[idx % len(colors_list)]
            pcd.paint_uniform_color(color)
            print(f"  Color applied: {color}")
        
        all_points.append(np.asarray(pcd.points))
        all_pcds.append(pcd)
    
    if not all_pcds:
        raise ValueError("No point clouds successfully loaded")
    
    # Calculate global boundaries
    all_points_array = np.vstack(all_points)
    print(f"\nTotal points: {len(all_points_array):,}")
    print(f"overall scope:")
    print(f"  X: [{all_points_array[:, 0].min():.3f}, {all_points_array[:, 0].max():.3f}]")
    print(f"  Y: [{all_points_array[:, 1].min():.3f}, {all_points_array[:, 1].max():.3f}]")
    print(f"  Z: [{all_points_array[:, 2].min():.3f}, {all_points_array[:, 2].max():.3f}]")
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    
    # Add all point clouds
    for pcd in all_pcds:
        vis.add_geometry(pcd)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = point_size
    
    # Set up view controls
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # Run the visualization
    print("\nVisual control:")
    print("  - left mouse button drag: rotate")
    print("  - Right mouse button drag: Pan")
    print("  - mouse wheel: Zoom")
    print("  - according to 'H' key: show help")
    print("  - according to 'Q' key or close the window: quit")
    
    vis.run()
    vis.destroy_window()
    
    print("End of visualization")


def main():
    parser = argparse.ArgumentParser(
        description='useOpen3DVisualizationPLYpoint cloud file'
    )
    
    parser.add_argument('ply_files', nargs='+', type=str,
                       help='PLYfile path（Support multiple files）')
    parser.add_argument('--point_size', type=float, default=2.0,
                       help='dot size（default：2.0）')
    parser.add_argument('--background', type=float, nargs=3, default=[0.1, 0.1, 0.1],
                       help='background color R G B（default：0.1 0.1 0.1）')
    parser.add_argument('--width', type=int, default=1280,
                       help='window width（default：1280）')
    parser.add_argument('--height', type=int, default=720,
                       help='window height（default：720）')
    
    args = parser.parse_args()
    
    if len(args.ply_files) == 1:
        visualize_pointcloud(
            ply_path=args.ply_files[0],
            point_size=args.point_size,
            background_color=tuple(args.background),
            width=args.width,
            height=args.height,
        )
    else:
        visualize_multiple_pointclouds(
            ply_paths=args.ply_files,
            point_size=args.point_size,
            background_color=tuple(args.background),
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
