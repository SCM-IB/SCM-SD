# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("open3d not found. Point cloud saving will use PLY format manually.")

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("pynvml not found. GPU monitoring will use torch only.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def get_gpu_info():
    """
    Get GPU usage information.
    
    Returns:
        dict: Dictionary containing GPU information including:
            - gpu_name: GPU device name
            - memory_used: Used memory in MB
            - memory_total: Total memory in MB
            - memory_percent: Memory usage percentage
            - utilization: GPU utilization percentage
            - temperature: GPU temperature in Celsius
    """
    gpu_info = {
        "gpu_name": "N/A",
        "memory_used": 0,
        "memory_total": 0,
        "memory_percent": 0,
        "utilization": 0,
        "temperature": 0
    }
    
    if HAS_NVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_info["gpu_name"] = gpu_name.decode('utf-8')
                else:
                    gpu_info["gpu_name"] = gpu_name
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info["memory_used"] = mem_info.used / 1024 / 1024
                gpu_info["memory_total"] = mem_info.total / 1024 / 1024
                gpu_info["memory_percent"] = (mem_info.used / mem_info.total) * 100
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info["utilization"] = util.gpu
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_info["temperature"] = temp
                except:
                    pass
        except Exception as e:
            print(f"Error getting GPU info from NVML: {e}")
    
    elif torch.cuda.is_available():
        try:
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["memory_total"] = props.total_memory / 1024 / 1024
            gpu_info["memory_used"] = torch.cuda.memory_allocated(0) / 1024 / 1024
            gpu_info["memory_percent"] = (gpu_info["memory_used"] / gpu_info["memory_total"]) * 100
        except Exception as e:
            print(f"Error getting GPU info from torch: {e}")
    
    return gpu_info


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Add GPU monitoring GUI elements
    with server.gui.add_folder("GPU Monitor"):
        gui_gpu_name = server.gui.add_text("GPU Name", initial_value="N/A", disabled=True)
        gui_gpu_memory = server.gui.add_text("Memory Usage", initial_value="0 / 0 MB (0%)", disabled=True)
        gui_gpu_util = server.gui.add_text("GPU Utilization", initial_value="0%", disabled=True)
        gui_gpu_temp = server.gui.add_text("Temperature", initial_value="0°C", disabled=True)

    # Add point cloud export GUI elements
    with server.gui.add_folder("Point Cloud Export"):
        gui_save_colored = server.gui.add_button("Save Colored Point Cloud", icon=None)
        gui_save_uncolored = server.gui.add_button("Save Uncolored Point Cloud", icon=None)
        gui_save_combined = server.gui.add_button("Save Combined Scene (Points + Cameras)", icon=None)

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    def update_gpu_info():
        """Update GPU information display."""
        gpu_info = get_gpu_info()
        gui_gpu_name.value = gpu_info["gpu_name"]
        gui_gpu_memory.value = f"{gpu_info['memory_used']:.1f} / {gpu_info['memory_total']:.1f} MB ({gpu_info['memory_percent']:.1f}%)"
        gui_gpu_util.value = f"{gpu_info['utilization']}%"
        gui_gpu_temp.value = f"{gpu_info['temperature']}°C"

    def select_save_path():
        """Open a file dialog to select the save path."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            title="Save Point Cloud"
        )
        root.destroy()
        return file_path

    def select_save_folder():
        """Open a folder dialog to select the save folder."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title="Select Folder to Save NVM Files")
        root.destroy()
        return folder_path

    def save_point_cloud_colored():
        """Save the current colored point cloud to a PLY file."""
        try:
            file_path = select_save_path()
            if not file_path:
                print("Save cancelled by user")
                return

            current_percentage = gui_points_conf.value
            threshold_val = np.percentile(conf_flat, current_percentage)
            conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

            if gui_frame_selector.value == "All":
                frame_mask = np.ones_like(conf_mask, dtype=bool)
            else:
                selected_idx = int(gui_frame_selector.value)
                frame_mask = frame_indices == selected_idx

            combined_mask = conf_mask & frame_mask
            points_to_save = points_centered[combined_mask]
            colors_to_save = colors_flat[combined_mask]

            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_to_save)
                pcd.colors = o3d.utility.Vector3dVector(colors_to_save.astype(np.float32) / 255.0)
                o3d.io.write_point_cloud(file_path, pcd)
            else:
                save_ply_manual(file_path, points_to_save, colors_to_save, has_color=True)

            print(f"Colored point cloud saved to {file_path}")
            print(f"Saved {len(points_to_save)} points")
        except Exception as e:
            print(f"Error saving colored point cloud: {e}")

    def save_point_cloud_uncolored():
        """Save the current uncolored point cloud to a PLY file."""
        try:
            file_path = select_save_path()
            if not file_path:
                print("Save cancelled by user")
                return

            current_percentage = gui_points_conf.value
            threshold_val = np.percentile(conf_flat, current_percentage)
            conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

            if gui_frame_selector.value == "All":
                frame_mask = np.ones_like(conf_mask, dtype=bool)
            else:
                selected_idx = int(gui_frame_selector.value)
                frame_mask = frame_indices == selected_idx

            combined_mask = conf_mask & frame_mask
            points_to_save = points_centered[combined_mask]

            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_to_save)
                o3d.io.write_point_cloud(file_path, pcd)
            else:
                save_ply_manual(file_path, points_to_save, None, has_color=False)

            print(f"Uncolored point cloud saved to {file_path}")
            print(f"Saved {len(points_to_save)} points")
        except Exception as e:
            print(f"Error saving uncolored point cloud: {e}")

    def save_ply_manual(file_path: str, points: np.ndarray, colors: np.ndarray = None, has_color: bool = True):
        """Manually save point cloud to PLY format without open3d."""
        num_points = len(points)
        with open(file_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color and colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(num_points):
                if has_color and colors is not None:
                    f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
                else:
                    f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")

    def save_combined_scene():
        """Save the current scene to two PLY files: point cloud and camera frustums."""
        try:
            folder_path = select_save_folder()
            if not folder_path:
                print("Save cancelled by user")
                return

            current_percentage = gui_points_conf.value
            threshold_val = np.percentile(conf_flat, current_percentage)
            conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

            if gui_frame_selector.value == "All":
                frame_mask = np.ones_like(conf_mask, dtype=bool)
            else:
                selected_idx = int(gui_frame_selector.value)
                frame_mask = frame_indices == selected_idx

            combined_mask = conf_mask & frame_mask
            points_to_save = points[combined_mask]
            colors_to_save = colors_flat[combined_mask]

            # Save point cloud to PLY file
            points_path = os.path.join(folder_path, "points.ply")
            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_to_save)
                pcd.colors = o3d.utility.Vector3dVector(colors_to_save.astype(np.float32) / 255.0)
                o3d.io.write_point_cloud(points_path, pcd)
            else:
                save_ply_manual(points_path, points_to_save, colors_to_save, has_color=True)

            # Save camera frustums to PLY file (as line segments)
            cameras_path = os.path.join(folder_path, "cameras.ply")
            save_camera_frustums_ply(cameras_path)

            print(f"Combined scene saved to {folder_path}")
            print(f"  - Point cloud: {points_path}")
            print(f"  - Camera frustums: {cameras_path}")
            print(f"  - Saved {len(points_to_save)} points and {S} cameras")
        except Exception as e:
            print(f"Error saving combined scene: {e}")

    def save_camera_frustums_ply(file_path: str):
        """Save camera frustums as line segments in PLY format."""
        vertices = []
        edges = []
        colors = []

        for i in range(S):
            cam2world = cam_to_world[i]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world)
            
            # Get camera center
            cam_center = T_world_camera.translation()
            
            # Get image dimensions
            img = images[i]
            h, w = img.shape[1], img.shape[2]
            
            # Get intrinsics
            fx = intrinsics_cam[i, 0, 0]
            fy = intrinsics_cam[i, 1, 1]
            cx = intrinsics_cam[i, 0, 2]
            cy = intrinsics_cam[i, 1, 2]
            
            # Compute frustum corners in camera space
            depth = 0.0005
            corners_cam = np.array([
                [-cx, -cy, fx],
                [w - cx, -cy, fx],
                [w - cx, h - cy, fy],
                [-cx, h - cy, fy]
            ]) * depth / np.array([fx, fy, 1])
            
            # Transform corners to world space
            corners_world = []
            for corner in corners_cam:
                point_cam = np.array([corner[0], corner[1], corner[2], 1.0])
                point_world = (T_world_camera.as_matrix() @ point_cam)[:3]
                corners_world.append(point_world)
            
            corners_world = np.array(corners_world)
            
            # Add vertices
            base_idx = len(vertices)
            vertices.append(cam_center)
            vertices.extend(corners_world)
            
            # Add edges (camera center to each corner)
            for j in range(4):
                edges.append([base_idx, base_idx + 1 + j])
            
            # Add edges (corners to form rectangle)
            edges.append([base_idx + 1, base_idx + 2])
            edges.append([base_idx + 2, base_idx + 3])
            edges.append([base_idx + 3, base_idx + 4])
            edges.append([base_idx + 4, base_idx + 1])
            
            # Add colors (yellow for cameras)
            for _ in range(5):
                colors.append([255, 255, 0])
        
        vertices = np.array(vertices)
        edges = np.array(edges)
        colors = np.array(colors)
        
        # Write PLY file with edges
        with open(file_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            for i, (v, c) in enumerate(zip(vertices, colors)):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
            
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")

    @gui_save_colored.on_click
    def _(_) -> None:
        save_point_cloud_colored()

    @gui_save_uncolored.on_click
    def _(_) -> None:
        save_point_cloud_uncolored()

    @gui_save_combined.on_click
    def _(_) -> None:
        save_combined_scene()

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            last_gpu_update = time.time()
            while True:
                current_time = time.time()
                if current_time - last_gpu_update >= 1.0:
                    update_gpu_info()
                    last_gpu_update = current_time
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        last_gpu_update = time.time()
        while True:
            current_time = time.time()
            if current_time - last_gpu_update >= 1.0:
                update_gpu_info()
                last_gpu_update = current_time
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    ckpt_path = "./model_tracker_fixed_e20.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    # model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
