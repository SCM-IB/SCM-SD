"""
Depth map backprojected to world coordinate system（Scaled alignment and caching）
Back-project the multi-view depth map into the world coordinate system centered on the first frame
Support scale alignment for monocular depth estimation，And save the aligned depth map and scale parameters

Notice：This code assumes that the input depth map is an inverse depth/disparity map（inverse depth），
This isDepthAnythingCommon output formats for monocular depth estimation models。

The output depth map format is the same asScanNet/7-ScenesBe consistent：
- Format: 16single channelPNG
- unit: mm(mm)
- Invalid value: 0
"""

import numpy as np
import cv2
import os
from pathlib import Path
import struct
import collections
import argparse
from tqdm import tqdm
import open3d as o3d
import json
import time as time
from joblib import delayed, Parallel

# COLMAPData structure definition
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read data from binary file"""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """readCOLMAPofcameras.bindocument"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODELS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """readCOLMAPofimages.bindocument"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24* num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
        return images


def read_points3D_binary(path_to_model_file):
    """readCOLMAPofpoints3D.bindocument"""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def qvec2rotmat(qvec):
    """Quaternion to rotation matrix"""
    return np.array(
        [
            [
                1- 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_camera_intrinsics(camera):
    """Get camera intrinsic parameter matrix"""
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
    elif camera.model == "PINHOLE":
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
    elif camera.model in ["SIMPLE_RADIAL", "RADIAL", "OPENCV"]:
        fx = camera.params[0]
        fy = camera.params[1] if len(camera.params) > 1 else camera.params[0]
        cx = camera.params[2] if camera.model == "OPENCV" else camera.params[1]
        cy = camera.params[3] if camera.model == "OPENCV" else camera.params[2]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def get_scales_for_image(image_id, cameras, images, points3d_ordered, depth_dir):
    """
    Calculate depth alignment parameters for a single image（scaleandoffset）
    
    Key points：
    1. DepthAnythingThe output is the inverse depth（inverse depth）
    2. COLMAPThe depth needs to be converted to inverse depth for alignment
    3. Use the median andMADPerform robust alignment
    
    Args:
        image_id: imageID
        cameras: COLMAPcamera dictionary
        images: COLMAPimage dictionary
        points3d_ordered: sorted by index3DPoint array
        depth_dir: Depth map directory
    
    Returns:
        dict: Includeimage_name, scale, offsetdictionary，orNone（if failed）
    """
    image_meta = images[image_id]
    cam_intrinsic = cameras[image_meta.camera_id]
    
    # Get the observed image3Dpoint index
    pts_idx = image_meta.point3D_ids
    
    # Create effective pointsmask
    mask = pts_idx >= 0
    mask = mask & (pts_idx < len(points3d_ordered))
    
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]
    if len(pts_idx) == 0:
        return None
    
    # get3DPoint coordinates
    pts = points3d_ordered[pts_idx]
    
    # Will3DPoint conversion to camera coordinate system: P_cam = R @ P_world + t
    R = qvec2rotmat(image_meta.qvec)
    pts_cam = np.dot(pts, R.T) + image_meta.tvec
    
    # calculateCOLMAPThe inverse depth of
    inv_depth_colmap = 1.0 / pts_cam[..., 2]
    
    # Build depth map path
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    image_name_no_ext = image_meta.name[:-n_remove]
    depth_path = os.path.join(depth_dir, f"{image_name_no_ext}.png")
    
    # Read monocular inverse depth map
    inv_depth_mono_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if inv_depth_mono_map is None:
        return None
    
    # Make sure it is single channel
    if inv_depth_mono_map.ndim != 2:
        inv_depth_mono_map = inv_depth_mono_map[..., 0]
    
    # Convert tofloat32and normalize（Assume stored as16Bit）
    inv_depth_mono_map = inv_depth_mono_map.astype(np.float32) / (2**16)
    
    # Calculate the scaling ratio of the depth map and camera intrinsic parameters
    scale_factor = inv_depth_mono_map.shape[0] / cam_intrinsic.height
    
    # Will2DConvert point coordinates to depth map coordinate system
    maps = (valid_xys * scale_factor).astype(np.float32)
    
    # Create effective pointsmask
    valid = (
        (maps[..., 0] >= 0) & 
        (maps[..., 1] >= 0) & 
        (maps[..., 0] < cam_intrinsic.width * scale_factor) & 
        (maps[..., 1] < cam_intrinsic.height * scale_factor) & 
        (inv_depth_colmap > 0)
    )
    
    # Check if there are enough valid points，and whether the depth range is sufficient
    if valid.sum() > 10 and (inv_depth_colmap.max() - inv_depth_colmap.min()) > 1e-3:
        maps = maps[valid, :]
        inv_depth_colmap = inv_depth_colmap[valid]
        
        # usecv2.remapPerform sub-pixel sampling
        inv_depth_mono = cv2.remap(
            inv_depth_mono_map,
            maps[..., 0], 
            maps[..., 1], 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # deal withremapoutput dimensions
        if inv_depth_mono.ndim > 1:
            inv_depth_mono = inv_depth_mono[..., 0]
        
        # Use the median andMAD（Mean Absolute Deviation）Perform robust alignment
        # alignment formula：inv_depth_colmap = scale * inv_depth_mono + offset
        # calculateCOLMAPMedian sum of inverse depthMAD
        t_colmap = np.median(inv_depth_colmap)
        s_colmap = np.mean(np.abs(inv_depth_colmap - t_colmap))
        
        # Calculate the median sum of monocular inverse depthsMAD
        t_mono = np.median(inv_depth_mono)
        s_mono = np.mean(np.abs(inv_depth_mono - t_mono))
        
        # calculatescaleandoffset
        if s_mono > 1e-8:  # avoid division by zero
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale
        else:
            scale = 0
            offset = 0
    else:
        scale = 0
        offset = 0
    
    return {
        "image_name": image_name_no_ext,
        "scale": float(scale), 
        "offset": float(offset),
        "num_valid_points": int(valid.sum()) if 'valid' in dir() else 0
    }


def save_depth_scannet_format(depth_meters, save_path, max_depth_meters=65.535):
    """
    Save the depth map asScanNet/7-ScenesCompatible formats
    
    Format specifications：
    - 16single channelPNG
    - unit：mm（mm）
    - Invalid depth：0
    - maximum depth：65.535rice（65535mm，16bit maximum value）
    
    Args:
        depth_meters: Depth map，Unit is meter [H, W]
        save_path: save path
        max_depth_meters: Maximum effective depth（rice），default65.535m
    """
    # Convert to millimeters
    depth_mm = depth_meters * 1000.0
    
    # Handle invalid values（Negative or out-of-range values ​​are set to0）
    invalid_mask = (depth_meters <= 0) | (depth_meters > max_depth_meters) | np.isnan(depth_meters) | np.isinf(depth_meters)
    depth_mm[invalid_mask] = 0
    
    # crop to16bit range [0, 65535]
    depth_mm = np.clip(depth_mm, 0, 65535)
    
    # Convert touint16
    depth_uint16 = depth_mm.astype(np.uint16)
    
    # save asPNG
    cv2.imwrite(str(save_path), depth_uint16)


def load_depth_scannet_format(depth_path):
    """
    loadScanNet/7-Scenesformat depth map
    
    Args:
        depth_path: Depth map path
    Returns:
        depth_meters: Depth map，Unit is meter [H, W]
        invalid_mask: Invalid pixelsmask [H, W]
    """
    depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    if depth_uint16 is None:
        raise ValueError(f"Failed to load depth map: {depth_path}")
    
    # Make sure it is single channel
    if depth_uint16.ndim == 3:
        depth_uint16 = depth_uint16[:, :, 0]
    
    # Convert to meters
    depth_meters = depth_uint16.astype(np.float32) / 1000.0
    
    # invalidmask（0Indicates invalid）
    invalid_mask = depth_uint16 == 0
    
    return depth_meters, invalid_mask


def compute_aligned_depth(inv_depth_mono_map, scale, offset, max_depth_meters=65.535):
    """
    according toscaleandoffsetCalculate aligned depth map
    
    alignment formula：aligned_inv_depth = scale * inv_depth_mono + offsetaligned_depth = 1.0 / aligned_inv_depth
    
    Args:
        inv_depth_mono_map: Monocular inverse depth map [H, W]，normalized to[0, 1]
        scale: scaling factor
        offset: offset
        max_depth_meters: maximum depth value（rice）
    
    Returns:
        aligned_depth: Aligned depth map，Unit is meter [H, W]
    """
    # Apply alignment transformation
    aligned_inv_depth = scale * inv_depth_mono_map + offset
    
    # Invalid creationmask（Inverse depth is too small or negative）
    invalid_mask = aligned_inv_depth <= 1e-8
    
    # Calculate depth safely
    aligned_inv_depth = np.maximum(aligned_inv_depth, 1e-8)
    aligned_depth = 1.0 / aligned_inv_depth
    
    # Crop to reasonable range
    aligned_depth = np.clip(aligned_depth, 0, max_depth_meters)
    
    # Set invalid area to0
    aligned_depth[invalid_mask] = 0
    
    return aligned_depth


def depth_to_points(depth, K, color_image=None, valid_mask=None):
    """
    Convert depth map to3Dpoint cloud（camera coordinate system）
    
    Args:
        depth: Depth map [H, W]，Unit is meter
        K: Camera internal parameter matrix [3, 3]
        color_image: Optional color image [H, W, 3]
        valid_mask: Optional valid pixelsmask [H, W]
    
    Returns:
        points_cam: in camera coordinate system3Dpoint [N, 3]
        colors: corresponding color [N, 3] or None
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    height, width = depth.shape
    
    # Create a pixel coordinate grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # flatten
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    
    # The build worksmask
    if valid_mask is not None:
        valid = valid_mask.flatten() & (depth_flat > 0) & (depth_flat < 65.535)
    else:
        valid = (depth_flat > 0) & (depth_flat < 65.535)
    
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    
    if len(depth_flat) == 0:
        return np.zeros((0, 3)), None
    
    # Back-project to camera coordinate system
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth_flat / fx
    y = (v - cy) * depth_flat / fy
    z = depth_flat
    
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Extract color
    colors = None
    if color_image is not None:
        colors = color_image[v.astype(int), u.astype(int)] / 255.0
        if colors.shape[1] == 3:  # BGR to RGB
            colors = colors[:, [2, 1, 0]]
    
    return points_cam, colors


def transform_points_to_world(points_cam, R, t, R_ref=None, t_ref=None):
    """Convert points from camera coordinate system to world coordinate system"""
    # camera coordinate system toCOLMAPworld coordinate system: P_world = R^T @ (P_cam - t)
    points_world = (R.T @ (points_cam.T - t.reshape(3, 1))).T
    
    # If a reference frame is provided，Convert to reference frame coordinate system
    if R_ref is not None and t_ref is not None:
        points_world = (R_ref @ points_world.T + t_ref.reshape(3, 1)).T
    
    return points_world


def process_dataset(data_root, output_path, downsample=1, save_aligned_depth=True, 
                   max_depth=65.535, depth_dir_name='depth_rgb', n_jobs=-1,
                   target_height=None, target_width=None):
    """
    Process the entire data set，Back-project all depth maps to the world coordinate system centered on the first frame
    
    The output depth map format is the same asScanNet/7-ScenesBe consistent：
    - 16single channelPNG
    - unit：mm（mm）
    - Invalid depth：0
    
    Args:
        data_root: Dataset root directory
        output_path: Output point cloud path
        downsample: Downsampling factor
        save_aligned_depth: Whether to save the aligned depth map
        max_depth: maximum depth value（rice），default65.535m（16bit maximum value）
        depth_dir_name: Depth map directory name
        n_jobs: Number of parallel tasks（-1means using allCPUcore）
        target_height: target depth map height（NoneRepresents the height using camera intrinsic parameters）
        target_width: Target depth map width（NoneRepresents the width using camera intrinsic parameters）
    """
    data_root = Path(data_root)
    rgb_dir = data_root / "rgb"
    colmap_dir = data_root / "colmap" / "sparse" / "0"
    depth_dir = data_root / depth_dir_name
    
    # Create output directory
    if save_aligned_depth:
        aligned_depth_dir = data_root / "depth_aligned"
        aligned_depth_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Depth map back-projection processing（Align with scale）")
    print("=" * 70)
    print(f"data root directory: {data_root}")
    print(f"Depth map directory: {depth_dir}")
    print(f"Save aligned depth map: {'yes' if save_aligned_depth else 'no'}")
    print(f"Output format: ScanNet/7-Scenescompatible（16BitPNG，unit：mm）")
    print(f"maximum depth: {max_depth:.3f}rice ({max_depth*1000:.0f}mm)")
    if save_aligned_depth:
        print(f"Align depth map directory: {aligned_depth_dir}")
    print("=" * 70)
    
    # readCOLMAPdata
    print("\n[1/5] readCOLMAPdata...")
    cameras = read_cameras_binary(str(colmap_dir / "cameras.bin"))
    images = read_images_binary(str(colmap_dir / "images.bin"))
    
    points3D_path = colmap_dir / "points3D.bin"
    if not points3D_path.exists():
        raise FileNotFoundError(f"not foundpoints3D.bin: {points3D_path}")
    
    points3D = read_points3D_binary(str(points3D_path))
    print(f"  Number of cameras: {len(cameras)}")
    print(f"  number of images: {len(images)}")
    print(f"sparse3DPoints: {len(points3D)}")
    
    # Build an index-sorted3DPoint array
    pts_indices = np.array([points3D[key].id for key in points3D])
    pts_xyzs = np.array([points3D[key].xyz for key in points3D])
    points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs
    
    # Sort by image name
    sorted_image_ids = sorted(images.keys())
    
    # Get the first frame as reference frame
    first_image_id = sorted_image_ids[0]
    first_image = images[first_image_id]
    R_ref = qvec2rotmat(first_image.qvec)
    t_ref = first_image.tvec
    
    print(f"\n[2/5] Reference frame settings")
    print(f"  reference frame: {first_image.name}")
    print(f"  reference frameID: {first_image_id}")
    
    # Calculate depth alignment parameters（Use parallel processing）
    print(f"[3/5] Calculate depth scale parameters...")
    depth_param_list = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(get_scales_for_image)(
            image_id, cameras, images, points3d_ordered, str(depth_dir)
        ) for image_id in tqdm(sorted_image_ids, desc="Scale calculation")
    )
    
    # Build parameter dictionary
    depth_params = {}
    for depth_param in depth_param_list:
        if depth_param is not None:
            depth_params[depth_param["image_name"]] = {
                "scale": depth_param["scale"],
                "offset": depth_param["offset"],
                "num_valid_points": depth_param.get("num_valid_points", 0)
            }
    
    # Save the scale parameters asJSON
    scale_params_file = data_root / "depth_params.json"
    with open(scale_params_file, 'w') as f:
        json.dump(depth_params, f, indent=2)
    print(f"  Scale parameters have been saved to: {scale_params_file}")
    
    # Statistical scale information
    valid_params = [p for p in depth_params.values() if p['scale'] != 0]
    if valid_params:
        scales = [p['scale'] for p in valid_params]
        offsets = [p['offset'] for p in valid_params]
        print(f"  Effective number of aligned images: {len(valid_params)}/{len(depth_params)}")
        print(f"  scale range: [{np.min(scales):.4f}, {np.max(scales):.4f}]")
        print(f"  scale mean: {np.mean(scales):.4f}")
        print(f"  offset range: [{np.min(offsets):.4f}, {np.max(offsets):.4f}]")
    
    # Save the aligned depth map and generate a point cloud
    print(f"[4/5] Save aligned depth map and generate point cloud...")
    all_points = []
    all_colors = []
    
    # Statistical depth range
    depth_stats = {
        'min_depth': float('inf'),
        'max_depth': 0,
        'mean_depths': []
    }
    
    for image_id in tqdm(sorted_image_ids, desc="Point cloud generation"):
        image = images[image_id]
        camera = cameras[image.camera_id]
        
        # Build file path
        n_remove = len(image.name.split('.')[-1]) + 1
        image_name_no_ext = image.name[:-n_remove]
        
        # Check if there are valid alignment parameters
        if image_name_no_ext not in depth_params:
            continue
        params = depth_params[image_name_no_ext]
        if params['scale'] == 0:
            continue
        
        # Load the original inverse depth map
        depth_path = depth_dir / f"{image_name_no_ext}.png"
        if not depth_path.exists():
            continue
        
        inv_depth_mono_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if inv_depth_mono_map is None:
            continue
        
        if inv_depth_mono_map.ndim != 2:
            inv_depth_mono_map = inv_depth_mono_map[..., 0]
        
        inv_depth_mono_map = inv_depth_mono_map.astype(np.float32) / (2**16)
        
        # Calculate aligned depth（unit：rice）
        aligned_depth = compute_aligned_depth(
            inv_depth_mono_map, 
            params['scale'], 
            params['offset'],
            max_depth_meters=max_depth
        )
        
        # Resize depth map to match target resolution or camera intrinsics
        target_h = target_height if target_height is not None else camera.height
        target_w = target_width if target_width is not None else camera.width
        
        if aligned_depth.shape[0] != target_h or aligned_depth.shape[1] != target_w:
            # For depth map，Keep edges sharp using nearest neighbor interpolation
            aligned_depth = cv2.resize(
                aligned_depth, 
                (target_w, target_h), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Save the aligned depth map（ScanNet/7-ScenesFormat）
        if save_aligned_depth:
            aligned_depth_path = aligned_depth_dir / f"{image_name_no_ext}.png"
            save_depth_scannet_format(aligned_depth, aligned_depth_path, max_depth)
        
        # Update depth statistics
        valid_depth = aligned_depth[aligned_depth > 0]
        if len(valid_depth) > 0:
            depth_stats['min_depth'] = min(depth_stats['min_depth'], valid_depth.min())
            depth_stats['max_depth'] = max(depth_stats['max_depth'], valid_depth.max())
            depth_stats['mean_depths'].append(valid_depth.mean())
        
        # loadRGB
        rgb_path = rgb_dir / image.name
        if not rgb_path.exists():
            color_image = None
        else:
            color_image = cv2.imread(str(rgb_path))
        if color_image is not None:
            # AdjustmentRGBSize to match depth map
            if color_image.shape[0] != target_h or color_image.shape[1] != target_w:
                color_image = cv2.resize(color_image, (target_w, target_h))
        
        # Get camera internal parameters（Need to scale according to resolution）
        K = get_camera_intrinsics(camera)
        scale_x = target_w / camera.width
        scale_y = target_h / camera.height
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        
        # Downsampling（for point cloud generation）
        if downsample > 1:
            aligned_depth_ds = aligned_depth[::downsample, ::downsample]
            if color_image is not None:
                color_image_ds = color_image[::downsample, ::downsample]
            else:
                color_image_ds = None
            K_down = K_scaled.copy()
            K_down[0, 0] /= downsample
            K_down[1, 1] /= downsample
            K_down[0, 2] /= downsample
            K_down[1, 2] /= downsample
        else:
            aligned_depth_ds = aligned_depth
            color_image_ds = color_image
            K_down = K_scaled
        # Depth map to point cloud
        points_cam, colors = depth_to_points(aligned_depth_ds, K_down, color_image_ds)
        
        if len(points_cam) == 0:
            continue
        
        # Convert to world coordinate system
        points_world = transform_points_to_world(points_cam, R, t, R_ref, t_ref)
        
        all_points.append(points_world)
        if colors is not None:
            all_colors.append(colors)
    
    # Merge all point clouds
    print("\n[5/5] Merge and save point clouds...")
    
    if not all_points:
        print("warn: No point cloud data is generated")
        return
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors) if all_colors else None
    
    print(f"  Total points: {len(all_points):,}")
    
    # Save point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_points)
    # if all_colors is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # output_path = Path(output_path)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # o3d.io.write_point_cloud(str(output_path), pcd)
    # print(f"  Point cloud saved to: {output_path}")
    
    # Statistics
    print("" + "=" * 70)
    print("Processing completed！")
    print("=" * 70)
    print(f"Total points: {len(all_points):,}")
    print(f"point cloud range:")
    print(f"  X: [{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}] m")
    print(f"  Y: [{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}] m")
    print(f"  Z: [{all_points[:, 2].min():.3f}, {all_points[:, 2].max():.3f}] m")
    
    if depth_stats['mean_depths']:
        print(f"\nDepth Statistics:")
        print(f"  minimum depth: {depth_stats['min_depth']:.3f} m ({depth_stats['min_depth']*1000:.0f} mm)")
        print(f"  maximum depth: {depth_stats['max_depth']:.3f} m ({depth_stats['max_depth']*1000:.0f} mm)")
        print(f"  average depth: {np.mean(depth_stats['mean_depths']):.3f} m ({np.mean(depth_stats['mean_depths'])*1000:.0f} mm)")
    
    print(f"Align depth map format information:")
    print(f"  Format: 16single channelPNG")
    print(f"  unit: mm (mm)")
    print(f"  Invalid value: 0")
    print(f"  Maximum representable depth: 65.535 m (65535 mm)")


def verify_depth_format(depth_path):
    """
    Verify that the depth map format is consistent withScanNet/7-Scenescompatible
    
    Args:
        depth_path: Depth map path
    Returns:
        dict: format information
    """
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    if depth_raw is None:
        return {"error": "Unable to read file"}
    
    info = {
        "path": str(depth_path),
        "shape": depth_raw.shape,
        "dtype": str(depth_raw.dtype),
        "channels": 1 if depth_raw.ndim == 2 else depth_raw.shape[2],
        "min_value": int(depth_raw.min()),
        "max_value": int(depth_raw.max()),
        "is_scannet_compatible": False
    }
    
    # Check for compatibility
    if depth_raw.dtype == np.uint16 and (depth_raw.ndim == 2 or depth_raw.shape[2] == 1):
        info["is_scannet_compatible"] = True
        info["min_depth_mm"] = int(depth_raw[depth_raw > 0].min()) if (depth_raw > 0).any() else 0
        info["max_depth_mm"] = int(depth_raw.max())
        info["min_depth_m"] = info["min_depth_mm"] / 1000.0
        info["max_depth_m"] = info["max_depth_mm"] / 1000.0
        info["invalid_pixel_count"] = int((depth_raw == 0).sum())
        info["valid_pixel_count"] = int((depth_raw > 0).sum())
    
    return info


def process_scenes(dataset_root, downsample=1, save_aligned_depth=True, 
                   max_depth=65.535, depth_dir_name='depth_rgb', n_jobs=-1,
                   target_height=1024, target_width=1280):
    """
    deal withdatasetAll scenes in the folder
    
    Args:
        dataset_root: datasetroot directory（Contains multiple scene folders）
        downsample: Downsampling factor
        save_aligned_depth: Whether to save the aligned depth map
        max_depth: maximum depth value（rice），default65.535m（16bit maximum value）
        depth_dir_name: Depth map directory name
        n_jobs: Number of parallel tasks（-1means using allCPUcore）
        target_height: target depth map height（NoneRepresents the height using camera intrinsic parameters）
        target_width: Target depth map width（NoneRepresents the width using camera intrinsic parameters）
    """
    dataset_root = Path(dataset_root)
    
    print("=" * 70)
    print("Batch processingdatasetscenes in")
    print("=" * 70)
    print(f"Dataset root directory: {dataset_root}")
    print("=" * 70)
    
    # Get all scene folders
    scene_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print("mistake: No scene folders found")
        return
    
    print(f"\nturn up {len(scene_dirs)} scenes:")
    for scene_dir in scene_dirs:
        print(f"  - {scene_dir.name}")
    print()
    
    # Work on every scenario
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        print(f"\n{'=' * 70}")
        print(f"Handle the scene: {scene_name}")
        print(f"{'=' * 70}")
        
        # Set scene output path（andrgbFolders juxtaposed）
        scene_output_path = scene_dir / f"{scene_name}.ply"
        
        try:
            process_dataset(
                data_root=str(scene_dir),
                output_path=str(scene_output_path),
                downsample=downsample,
                save_aligned_depth=save_aligned_depth,
                max_depth=max_depth,
                depth_dir_name=depth_dir_name,
                n_jobs=n_jobs,
                target_height=target_height,
                target_width=target_width
            )
            print(f"scene {scene_name} Processing completed！")
        except Exception as e:
            print(f"mistake: scene {scene_name} Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print("All scene processing completed！")
    print(f"{'=' * 70}")


def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description='Depth map back-projection processing（Align with scale）- outputScanNet/7-ScenesCompatible formats'
    )
    
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='datasetroot directory（Contains multiple scene folders）')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Point cloud downsampling factor（default：1，No sampling）')
    parser.add_argument('--save_aligned_depth', action='store_true', default=True,
                       help='Save the aligned depth map（default：enable）')
    parser.add_argument('--no_save_aligned_depth', dest='save_aligned_depth', action='store_false',
                       help='Do not save aligned depth map')
    parser.add_argument('--max_depth', type=float, default=65.535,
                       help='maximum depth value（rice），default65.535m（16bit maximum value）')
    parser.add_argument('--depth_dir', type=str, default='depth_rgb',
                       help='Enter the depth map directory name（default：depth_rgb）')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel tasks（-1means using allCPUcore）')
    parser.add_argument('--target_height', type=int, default=None,
                       help='target depth map height（default：Use camera intrinsic height）')
    parser.add_argument('--target_width', type=int, default=None,
                       help='Target depth map width（default：Use camera intrinsic width）')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verify the format of the specified depth map（No processing is performed）')
    
    args = parser.parse_args()
    
    # If authentication mode is specified
    if args.verify:
        print("Verify depth map format...")
        info = verify_depth_format(args.verify)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    
    start_time = time.time()
    
    process_scenes(
        dataset_root=args.dataset_root,
        downsample=args.downsample,
        save_aligned_depth=args.save_aligned_depth,
        max_depth=args.max_depth,
        depth_dir_name=args.depth_dir,
        n_jobs=args.n_jobs,
        target_height=args.target_height,
        target_width=args.target_width
    )
    
    elapsed = time.time() - start_time
    print(f"\ntotal processing time: {elapsed:.2f} Second")


if __name__ == '__main__':
    main()
