# COLMAP-style Dataset Loader

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct
import open3d as o3d
import cv2
from sympy import false, true
from torch.nn.modules.linear import F


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_file: str) -> Dict[int, Dict]:
    cameras = {}
    with open(path_to_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                         format_char_sequence="d" * num_params)
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path_to_file: str) -> Dict[int, Dict]:

    images = {}
    with open(path_to_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                        format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                        format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                    tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = {
                "camera_id": camera_id,
                "name": image_name,
                "qvec": qvec,
                "tvec": tvec,
                "xys": np.array(x_y_id_s),
            }
    return images


class CameraModel:

    def __init__(self, model_id, model_name, num_params):
        self.model_id = model_id
        self.model_name = model_name
        self.num_params = num_params


CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def qvec2rotmat(qvec):

    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    q = np.array([x, y, z, w])
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    return R


def rotmat2qvec(R):

    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def get_camera_intrinsics(camera: Dict) -> np.ndarray:

    model = camera["model"]
    params = camera["params"]
    width = camera["width"]
    height = camera["height"]

    if model == "PINHOLE":
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    elif model == "SIMPLE_PINHOLE":
        fx = params[0]
        fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model == "OPENCV":
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def get_camera_extrinsics(image: Dict) -> np.ndarray:
    qvec = image["qvec"]
    tvec = image["tvec"]
    R = qvec2rotmat(qvec)
    
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = tvec
    return extrinsics


def get_camera_to_world_matrix(image: Dict) -> np.ndarray:

    world_to_camera = get_camera_extrinsics(image)
    camera_to_world = np.linalg.inv(world_to_camera)
    return camera_to_world


def create_camera_frustum(length: float = 0.2, width: float = 0.1) -> o3d.geometry.TriangleMesh:

    vertices = np.array([
        [0, 0, 0],           # Camera center (apex)
        [-width/2, -width/2, length],  # Bottom-left
        [width/2, -width/2, length],   # Bottom-right
        [width/2, width/2, length],    # Top-right
        [-width/2, width/2, length],   # Top-left
    ])
    
    triangles = np.array([
        [0, 1, 2],  # Side face 1
        [0, 2, 3],  # Side face 2
        [0, 3, 4],  # Side face 3
        [0, 4, 1],  # Side face 4
        [1, 2, 3],  # Near plane 1
        [1, 3, 4],  # Near plane 2
    ])
    
    frustum = o3d.geometry.TriangleMesh()
    frustum.vertices = o3d.utility.Vector3dVector(vertices)
    frustum.triangles = o3d.utility.Vector3iVector(triangles)
    frustum.compute_vertex_normals()
    
    return frustum


class COLMAPDataset:

    def __init__(self, data_root: str, rgb_dir: str = "thermal", depth_dir: Optional[str] = None):

        self.data_root = Path(data_root)
        self.rgb_dir = self.data_root / rgb_dir
        self.depth_dir = self.data_root / depth_dir if depth_dir else None

        sparse_dir = self.data_root / "colmap" / "sparse" / "0"
        self.cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        self.images = read_images_binary(str(sparse_dir / "images.bin"))

        images_unsorted = self.images.copy()

        self.image_list = sorted(images_unsorted.values(), key = lambda x : x['name'])

        valid_image_list = []
        missing_images = []
        for image_info in self.image_list:
            image_path = self.rgb_dir / image_info["name"]
            if image_path.exists():
                valid_image_list.append(image_info)
            else:
                missing_images.append(image_info["name"])
        
        self.valid_image_list = sorted(valid_image_list, key = lambda x : x['name'])
        self.image_list = valid_image_list
        self.num_images = len(self.image_list)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found in {self.rgb_dir}:")
            for name in missing_images[:5]:
                print(f"  - {name}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
        
        print(f"Loaded {len(self.image_list)} valid images from COLMAP metadata")
        
    def __len__(self) -> int:
        return len(self.image_list)

    
    def __getitem__(self, idx: int) -> Dict:

        image_info = self.image_list[idx]
        camera_id = image_info["camera_id"]
        camera_info = self.cameras[camera_id]

        image_path = self.rgb_dir / image_info["name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        depth_map = None
        if self.depth_dir:
            base_name = Path(image_info["name"]).stem
            for ext in ['.png', '.tiff', '.tif']:
                depth_path = self.depth_dir / (base_name + ext)
                if depth_path.exists():
                    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    
                    if depth_raw is not None:
                        if depth_raw.dtype == np.uint16:
                            depth_map = depth_raw.astype(np.float32) / 1000.0
                        else:
                            depth_map = depth_raw.astype(np.float32)
                    break

        intrinsics = get_camera_intrinsics(camera_info)
        extrinsics = get_camera_extrinsics(image_info)
        camera_to_world = get_camera_to_world_matrix(image_info)
        
        return {
            "image_id": idx,
            "image_name": image_info["name"],
            "image_path": str(image_path),
            "camera_id": camera_id,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "camera_to_world": camera_to_world,
            "depth_map": depth_map,
        }
    
    def get_all_camera_poses(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        c2ws = []
        available_frame_ids = []
        
        for i, image_info in enumerate(self.image_list):
            camera_to_world = get_camera_to_world_matrix(image_info)
            c2ws.append(camera_to_world)
            available_frame_ids.append(i)
        
        c2ws = np.stack(c2ws)
        first_gt_pose = c2ws[0].copy()

        c2ws = np.linalg.inv(c2ws[0]) @ c2ws
        available_frame_ids = np.array(available_frame_ids)

        return c2ws, first_gt_pose, available_frame_ids
    
    def get_image_paths(self) -> List[Path]:

        image_paths = []
        for image_info in self.image_list:
            image_path = self.rgb_dir / image_info["name"]
            image_paths.append(image_path)
        return image_paths
    
    def sample_indices(self, num_samples: int, mode: str = "uniform", range_size: int = 25) -> np.ndarray:
        num_images = self.num_images
        
        if num_samples <= 0:
            return np.array([], dtype=int)
        
        if num_samples >= num_images:
            return np.arange(num_images, dtype=int)
        
        if mode == "uniform":
            indices = np.linspace(0, num_images - 1, num_samples, dtype=int)
        elif mode == "random":
            indices = np.random.choice(num_images, num_samples, replace=False)
            indices = np.sort(indices)
        elif mode == "range":
            center_idx = np.random.randint(0, num_images)

            half_range = range_size // 2

            candidate_indices = []
            for i in range(-half_range, half_range + 1):
                idx = (center_idx + i) % num_images
                candidate_indices.append(idx)
            
            candidate_indices = np.array(candidate_indices)

            if len(candidate_indices) >= num_samples:
                selected = np.random.choice(candidate_indices, num_samples, replace=False)
                indices = selected
            else:
                indices = candidate_indices
            
            indices = np.sort(indices)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        
        return indices


    def visualize_multi_view_pointclouds(
        self,
        num_views: int = 3,
        sample_size_per_view: int = 10000,
        point_size: float = 2.0,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        depth_cutoff: float = 10.0,
        use_icp: bool = True,
        icp_distance_threshold: float = 0.1,
        icp_max_iterations: int = 50,
        use_colored_icp: bool = False,
    ):

        if self.depth_dir is None:
            print("Error: Depth directory is not set, cannot generate point clouds.")
            return
        
        if self.num_images < num_views:
            num_views = self.num_images
        
        if num_views == 0:
            print("Error: No available images.")
            return

        random_indices = np.random.choice(self.num_images, num_views, replace=False)
        random_indices = np.sort(random_indices)

        colors_list = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        ]
        
        point_clouds = []

        
        for view_idx, image_idx in enumerate(random_indices):
            sample = self[image_idx]
            depth_map = sample["depth_map"]
            
            if depth_map is None: continue
            
            intrinsics = sample["intrinsics"]
            camera_to_world = sample["camera_to_world"]

            rgb_img = cv2.imread(sample["image_path"])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (depth_map.shape[1], depth_map.shape[0]))
            
            height, width = depth_map.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            u = u.flatten()
            v = v.flatten()
            depth_flat = depth_map.flatten()
            rgb_flat = rgb_img.reshape(-1, 3) / 255.0

            valid_mask = (depth_flat > 0.001) & (depth_flat < depth_cutoff)
            u, v, depth_flat = u[valid_mask], v[valid_mask], depth_flat[valid_mask]
            rgb_flat = rgb_flat[valid_mask]
            
            if len(depth_flat) == 0: continue

            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            x_cam = (u - cx) * depth_flat / fx
            y_cam = (v - cy) * depth_flat / fy
            z_cam = depth_flat
            
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

            points_homogeneous = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
            points_world = (camera_to_world @ points_homogeneous.T).T[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)

            pcd.colors = o3d.utility.Vector3dVector(rgb_flat)

            if len(pcd.points) > sample_size_per_view:
                 pcd = pcd.random_down_sample(sample_size_per_view / len(pcd.points))

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            vis_color = colors_list[view_idx % len(colors_list)]
            
            point_clouds.append({
                "pcd": pcd,
                "name": sample['image_name'],
                "vis_color": vis_color,
                "original_colors": np.asarray(pcd.colors)  # Backup original colors
            })
            
        if not point_clouds:
            print("Error: No point cloud was generated.")
            return

        if use_icp and len(point_clouds) > 1:
            print("\n=== Starting ICP refinement ===")

            reference_pcd = point_clouds[0]["pcd"]
            
            for i in range(1, len(point_clouds)):
                source_dict = point_clouds[i]
                source_pcd = source_dict["pcd"]
                name = source_dict["name"]

                try:
                    if use_colored_icp:
                        result_icp = o3d.pipelines.registration.registration_colored_icp(
                            source_pcd, reference_pcd,
                            max_correspondence_distance=icp_distance_threshold,
                            init=np.eye(4),
                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=icp_max_iterations)
                        )
                        method_name = "Colored ICP"

                    else:
                        result_icp = o3d.pipelines.registration.registration_icp(
                            source_pcd, reference_pcd,
                            max_correspondence_distance=icp_distance_threshold,
                            init=np.eye(4),
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations)
                        )
                        method_name = "Point-to-Plane"

                    source_pcd.transform(result_icp.transformation)
                    print(f"  [{i}/{len(point_clouds)-1}] {name} -> Reference: {method_name} Fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}")
                    
                except Exception as e:
                    print(f"  ICP failed for {name}: {e}. Fallback to original pose.")

        vis_geometries = []
        for pc_dict in point_clouds:
            pcd = pc_dict["pcd"]
            pcd.paint_uniform_color(pc_dict["vis_color"]) 
            vis_geometries.append(pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Improved Multi-View Alignment", width=1280, height=720)
        
        for geom in vis_geometries:
            vis.add_geometry(geom)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.point_size = point_size

        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    """
    Usage example.
    """
    print("=== COLMAP Dataset Loader ===")

    dataset = COLMAPDataset(
        r"E:/Paper2/1/vggt-our/dataset_example/Transmission_Tower",
        rgb_dir="rgb",
        depth_dir="depth_aligned"
    )
    
    print(f"Dataset contains {len(dataset)} images.")
    print(f"Number of cameras: {len(dataset.cameras)}")
    
    # Inspect samples.
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nImage {i} info:")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Image name: {sample['image_name']}")
        print(f"  Camera ID: {sample['camera_id']}")
        print(f"  Intrinsics shape: {sample['intrinsics'].shape}")
        print(f"  Extrinsics shape: {sample['extrinsics'].shape}")

    c2ws, first_gt_pose, frame_ids = dataset.get_all_camera_poses()
    print("\nAll camera poses:")
    print(f"  Number of poses: {len(c2ws)}")
    print(f"  Pose array shape: {c2ws.shape}")
    print(f"  First pose (in first-frame coordinates):\n{c2ws[0]}")
    
    # # Visualize multi-view point clouds.
    # print("\n=== Multi-view point cloud visualization ===")
    # dataset.visualize_multi_view_pointclouds(
    #     num_views=3,
    #     sample_size_per_view=100000,
    #     point_size=2.0,
    #     background_color=(0.1, 0.1, 0.1),
    #     depth_cutoff=50.0,
    #     use_icp=True,
    #     icp_distance_threshold=0.1,
    #     icp_max_iterations=30,
    # )
    
