import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import yaml

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    get_vgg_input_imgs,
    load_images_rgb,
    eval_trajectory,
    compute_average_metrics_and_save,
    calculate_auc,
    compute_relative_pose_errors,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt_pointcloud_evaluator import (
    evaluate_pointcloud_reconstruction,
    evaluate_pointcloud_reconstruction_da,
)
from colmap_dataset_loader import COLMAPDataset


POSE_SUMMARY_KEYS = [
    "ate",
    "rpe_rot",
    "rpe_trans",
    "rra_3",
    "rra_5",
    "rra_15",
    "rra_30",
    "rta_3",
    "rta_5",
    "rta_15",
    "rta_30",
    "auc_3",
    "auc_5",
    "auc_15",
    "auc_30",
]

POINTCLOUD_SUMMARY_KEYS = [
    "acc",
    "acc_med",
    "comp",
    "comp_med",
    "nc1",
    "nc1_med",
    "nc2",
    "nc2_med",
    "nc",
    "nc_med",
    "acc_da",
    "comp_da",
    "overall",
    "precision",
    "recall",
    "fscore_da",
]

DEPTH_SUMMARY_KEYS = [
    "abs_rel_scale",
    "delta_1.25_scale",
    "abs_rel_scale_shift",
    "delta_1.25_scale_shift",
    "abs_rel_mono",
    "delta_1.25_mono",
]

POINTCLOUD_BACKPROJECTION_MODE = "pred_pose_pred_depth"


def visualize_point_clouds_open3d(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    save_path: Path,
    title: str = "Point Cloud Comparison",
    sample_size: int = 10000,
):

    if len(pred_points) > sample_size:
        pred_indices = np.random.choice(len(pred_points), sample_size, replace=False)
        pred_points_sampled = pred_points[pred_indices]
    else:
        pred_points_sampled = pred_points
    
    if len(gt_points) > sample_size:
        gt_indices = np.random.choice(len(gt_points), sample_size, replace=False)
        gt_points_sampled = gt_points[gt_indices]
    else:
        gt_points_sampled = gt_points
    
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_points_sampled)
    pcd_pred.paint_uniform_color([1, 0, 0])
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_points_sampled)
    pcd_gt.paint_uniform_color([0, 1, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd_pred)
    vis.add_geometry(pcd_gt)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis.run()
    vis.capture_screen_image(str(save_path))
    vis.destroy_window()
    
    print(f"Point cloud visualization saved to: {save_path}")


def generate_gt_pointcloud_from_colmap(
    selected_samples: List[dict],
    max_points: Optional[int] = None,
    target_width: int = 518,
    target_height: int = 518,
) -> Tuple[np.ndarray, np.ndarray]:

    import cv2
    
    all_gt_points = []
    all_gt_masks = []
    depth_list = []
    
    for sample in selected_samples:
        depth_map = sample["depth_map"]
        
        if depth_map is None:
            print(f"Warning: No depth map for image {sample['name']}")
            continue

        extrinsics = sample["extrinsics"]
        intrinsics = sample["intrinsics"].copy()

        img = cv2.imread(sample["image_path"])
        if img is None:
            print(f"Warning: Could not read image {sample['image_path']}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if depth_map.shape[:2] != (target_height, target_width):
            depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            scale_x = target_width / img.shape[1]
            scale_y = target_height / img.shape[0]
            intrinsics[0, 0] = fx * scale_x
            intrinsics[1, 1] = fy * scale_y
            intrinsics[0, 2] = cx * scale_x
            intrinsics[1, 2] = cy * scale_y
        
        _depth = depth_map[np.newaxis, np.newaxis, ...]
        _depth = _depth.reshape(-1, _depth.shape[2], _depth.shape[3], _depth.shape[1])

        world_points = unproject_depth_map_to_point_map(
            _depth,
            extrinsics[np.newaxis, :3, :],
            intrinsics[np.newaxis, ...]
        )
        world_points = world_points[0]

        valid_mask = np.isfinite(world_points).all(axis=2) & (depth_map > 0)
        
        all_gt_points.append(world_points)
        all_gt_masks.append(valid_mask)
        depth_list.append(_depth)
    if len(all_gt_points) == 0:
        return None, None

    depth_squeezed = [arr.squeeze(0) for arr in depth_list]
    depth = np.stack(depth_squeezed, axis=0)  # (S, H, W, 1)
    gt_points = np.stack(all_gt_points, axis=0)  # (S, H, W, 3)
    gt_valid_mask = np.stack(all_gt_masks, axis=0)  # (S, H, W)
    
    return gt_points, gt_valid_mask, depth


def generate_pred_pointcloud_from_vggt(
    predictions: dict,
    vgg_input: torch.Tensor,
    depth_conf_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:

    depth_tensor = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    
    depth_np = depth_tensor[0].detach().float().cpu().numpy()  # (S, H, W, 1)
    depth_conf_np = depth_conf[0].detach().float().cpu().numpy()  # (S, H, W)

    # Only the predicted-pose + predicted-depth backprojection mode is supported.
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
    )
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()  # (S, 3, 4)
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()  # (S, 3, 3)

    if depth_np.shape[0] != extrinsic_np.shape[0]:
        raise ValueError(
            f"Depth frame count ({depth_np.shape[0]}) does not match pose frame count "
            f"({extrinsic_np.shape[0]}) in {POINTCLOUD_BACKPROJECTION_MODE}"
        )

    pred_points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
    return pred_points, depth_conf_np


def evaluate_scene_and_save(
    scene: str,
    c2ws: np.ndarray,
    first_gt_pose: np.ndarray,
    frame_ids: List[int],
    all_cam_to_world_mat: List[np.ndarray],
    all_world_points: List[np.ndarray],
    output_scene_dir: Path,
    gt_points: np.ndarray,
    gt_valid_mask: np.ndarray,
    inference_time_ms: float,
    plot_flag: bool,
    max_points: Optional[int] = None,
    pc_metrics: dict = None,
    depth_metrics: dict = None,
) -> Optional[dict]:
    """
    Evaluate scene and save results.
    
    Args:
        scene: Scene name
        c2ws: Ground truth camera poses
        first_gt_pose: First frame ground truth pose
        frame_ids: Frame IDs
        all_cam_to_world_mat: Predicted camera poses
        all_world_points: Predicted point clouds
        output_scene_dir: Output directory
        gt_points: GT point cloud from depth backprojection (S, H, W, 3)
        gt_valid_mask: GT valid mask (S, H, W)
        chamfer_max_dist: Maximum Chamfer distance
        inference_time_ms: Inference time
        plot_flag: Whether to plot
        max_points: Maximum points limit
        pc_metrics: Point cloud reconstruction metrics
        depth_metrics: Depth estimation metrics (Abs Rel, δ1.25)
    
    Returns:
        Dictionary of metrics
    """
    if not all_cam_to_world_mat:
        print(f"Skipping {scene}: failed to obtain valid camera poses")
        return None

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    poses_gt = c2ws
    pred_w2cs = np.array(all_cam_to_world_mat)
    traj_est_poses = np.linalg.inv(pred_w2cs)
    n = min(len(traj_est_poses), len(poses_gt))
    timestamps = frame_ids[:n]
    stats_aligned, traj_plot, _ = eval_trajectory(
        traj_est_poses[:n], poses_gt[:n], timestamps, align=True
    )

    if n >= 2:
        poses_gt_w2c = np.linalg.inv(poses_gt[:n])
        poses_est_w2c = pred_w2cs[:n]
        r_error, t_error = compute_relative_pose_errors(poses_est_w2c, poses_gt_w2c, n)

        for threshold in [3, 5, 15, 30]:
            stats_aligned[f"rra_{threshold}"] = float(np.mean(r_error < threshold) * 100.0)
            stats_aligned[f"rta_{threshold}"] = float(np.mean(t_error < threshold) * 100.0)
            auc_value, _ = calculate_auc(r_error, t_error, max_threshold=threshold)
            stats_aligned[f"auc_{threshold}"] = float(auc_value * 100.0)
    else:
        print(f"Warning: Not enough poses ({n}) for AUC calculation, need at least 2")
        for threshold in [3, 5, 15, 30]:
            stats_aligned[f"rra_{threshold}"] = None
            stats_aligned[f"rta_{threshold}"] = None
            stats_aligned[f"auc_{threshold}"] = None

    metrics = {
        "ate": stats_aligned["ate"],
        "rpe_rot": stats_aligned["rpe_rot"],
        "rpe_trans": stats_aligned["rpe_trans"],
    }
    
    for pose_key in [
        "rra_3",
        "rra_5",
        "rra_15",
        "rra_30",
        "rta_3",
        "rta_5",
        "rta_15",
        "rta_30",
        "auc_3",
        "auc_5",
        "auc_15",
        "auc_30",
    ]:
        if pose_key in stats_aligned and stats_aligned[pose_key] is not None:
            metrics[pose_key] = stats_aligned[pose_key]

    if pc_metrics is not None:
        metrics.update(pc_metrics)

    if depth_metrics is not None:
        metrics.update(depth_metrics)
    
    if plot_flag:
        traj_plot.save(output_scene_dir / "trajectory_plot.png")

    import json
    with open(output_scene_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def evaluate_pointcloud_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_conf: np.ndarray,
    gt_valid_mask: np.ndarray,
    use_proj: bool = True,
    use_icp: bool = True,
    max_points: Optional[int] = None,
    da3_threshold: float = 0.05,
    da3_down_sample: Optional[float] = None,
    debug: bool = False,
) -> dict:

    metrics = evaluate_pointcloud_reconstruction(
        pred_points=pred_points,
        gt_points=gt_points,
        pred_conf=pred_conf,
        gt_valid_mask=gt_valid_mask,
        use_proj=use_proj,
        use_icp=use_icp,
        max_points=max_points,
        debug=debug,
    )
    
    # Add Depth-Anything-3 metrics
    metrics_da = evaluate_pointcloud_reconstruction_da(
        pred_points=pred_points,
        gt_points=gt_points,
        pred_conf=pred_conf,
        gt_valid_mask=gt_valid_mask,
        use_proj=use_proj,
        use_icp=use_icp,
        max_points=max_points,
        debug=debug,
    )
    
    # Merge the metrics
    metrics.update(metrics_da)
    
    return metrics


def _prepare_depth_array(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 2:
        depth = depth[None, ...]
    return depth


def _solve_scale_only(pred_valid: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    scale = torch.nanmean(gt_valid) / torch.clamp(torch.nanmean(pred_valid), min=1e-8)
    for _ in range(10):
        residuals = scale * pred_valid - gt_valid
        weights = 1.0 / (residuals.abs() + 1e-8)
        numer = torch.sum(weights * pred_valid * gt_valid)
        denom = torch.sum(weights * pred_valid.square()).clamp(min=1e-8)
        scale = numer / denom
    return scale.clamp(min=1e-3).detach()


def _solve_median_scale(pred_valid: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    return (torch.median(gt_valid) / torch.clamp(torch.median(pred_valid), min=1e-8)).detach()


def _solve_scale_shift(pred_valid: torch.Tensor, gt_valid: torch.Tensor) -> Tuple[float, float]:
    scale = torch.tensor(
        [_solve_median_scale(pred_valid, gt_valid).item()],
        dtype=pred_valid.dtype,
        device=pred_valid.device,
        requires_grad=True,
    )
    shift = torch.tensor(
        [0.0],
        dtype=pred_valid.dtype,
        device=pred_valid.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([scale, shift], lr=1e-4)
    prev_loss = None
    for _ in range(1000):
        optimizer.zero_grad()
        aligned = scale * pred_valid + shift
        loss = torch.abs(aligned - gt_valid).sum()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        if prev_loss is not None and abs(prev_loss - loss_value) < 1e-6:
            break
        prev_loss = loss_value
    return float(scale.detach().item()), float(shift.detach().item())


def _evaluate_depth_frame(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_depth: Optional[float] = None,
    alignment_mode: str = "scale",
) -> Dict[str, float]:
    pred_tensor = torch.from_numpy(np.asarray(pred_depth, dtype=np.float32))
    gt_tensor = torch.from_numpy(np.asarray(gt_depth, dtype=np.float32))

    mask = torch.isfinite(pred_tensor) & torch.isfinite(gt_tensor) & (gt_tensor > 0)
    if valid_mask is not None:
        mask &= torch.from_numpy(np.asarray(valid_mask).astype(bool))
    if max_depth is not None:
        mask &= gt_tensor < max_depth

    pred_valid = pred_tensor[mask]
    gt_valid = gt_tensor[mask]
    if pred_valid.numel() == 0 or gt_valid.numel() == 0:
        return {"abs_rel": float("inf"), "delta_1.25": 0.0, "valid_pixels": 0}

    if alignment_mode == "scale":
        scale = _solve_scale_only(pred_valid, gt_valid)
        pred_valid = pred_valid * scale
    elif alignment_mode == "median_scale":
        scale = _solve_median_scale(pred_valid, gt_valid)
        pred_valid = pred_valid * scale
    elif alignment_mode == "scale_shift":
        scale, shift = _solve_scale_shift(pred_valid, gt_valid)
        pred_valid = pred_valid * scale + shift
    else:
        raise ValueError(f"Unsupported depth alignment mode: {alignment_mode}")

    pred_valid = torch.clamp(pred_valid, min=1e-5)
    gt_valid = torch.clamp(gt_valid, min=1e-5)
    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()
    max_ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_125 = torch.mean((max_ratio < 1.25).float()).item()
    return {
        "abs_rel": float(abs_rel),
        "delta_1.25": float(delta_125),
        "valid_pixels": int(pred_valid.numel()),
    }


def evaluate_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_valid_mask: np.ndarray = None,
    max_depth: float = None,
    alignment_mode: str = "scale",
    per_frame: bool = False,
) -> dict:
    """
    Pi3-style depth metrics.

    `per_frame=False` matches the video/multi-view setting with a single sequence-level
    alignment. `per_frame=True` matches the monocular setting where each frame is aligned
    independently.
    """
    pred_depth = _prepare_depth_array(pred_depth)
    gt_depth = _prepare_depth_array(gt_depth)
    if gt_valid_mask is not None:
        gt_valid_mask = _prepare_depth_array(gt_valid_mask).astype(bool)

    num_frames = min(pred_depth.shape[0], gt_depth.shape[0])
    pred_depth = pred_depth[:num_frames]
    gt_depth = gt_depth[:num_frames]
    if gt_valid_mask is not None:
        gt_valid_mask = gt_valid_mask[:num_frames]

    if not per_frame:
        mask = gt_valid_mask if gt_valid_mask is not None else None
        return _evaluate_depth_frame(
            pred_depth.reshape(-1),
            gt_depth.reshape(-1),
            None if mask is None else mask.reshape(-1),
            max_depth=max_depth,
            alignment_mode=alignment_mode,
        )

    frame_metrics = []
    for frame_idx in range(num_frames):
        frame_metrics.append(
            _evaluate_depth_frame(
                pred_depth[frame_idx],
                gt_depth[frame_idx],
                None if gt_valid_mask is None else gt_valid_mask[frame_idx],
                max_depth=max_depth,
                alignment_mode=alignment_mode,
            )
        )

    total_valid = sum(metric["valid_pixels"] for metric in frame_metrics)
    if total_valid == 0:
        return {"abs_rel": float("inf"), "delta_1.25": 0.0, "valid_pixels": 0}

    return {
        "abs_rel": float(
            np.average(
                [metric["abs_rel"] for metric in frame_metrics],
                weights=[metric["valid_pixels"] for metric in frame_metrics],
            )
        ),
        "delta_1.25": float(
            np.average(
                [metric["delta_1.25"] for metric in frame_metrics],
                weights=[metric["valid_pixels"] for metric in frame_metrics],
            )
        ),
        "valid_pixels": int(total_valid),
    }


def get_all_scenes(data_dir: Path, num_scenes: int) -> List[str]:
    """Get all scenes from data directory."""
    all_scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if len(all_scenes) > num_scenes:
        sample_interval = max(1, len(all_scenes) // num_scenes)
        return all_scenes[::sample_interval][:num_scenes]
    return all_scenes


def find_max_checkpoint(ckpt_dir: Path) -> Optional[Path]:

    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None
    
    checkpoint_files = list(ckpt_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        return None
    
    max_num = -1
    max_ckpt = None
    
    for ckpt_file in checkpoint_files:
        try:
            num = int(ckpt_file.stem.split("_")[1])
            if num > max_num:
                max_num = num
                max_ckpt = ckpt_file
        except (IndexError, ValueError):
            continue
    
    return max_ckpt


def get_output_dir_with_sequence(
    base_output_dir: Path,
    scene: str,
    input_frame: int,
    max_sequence: int = 500,
) -> Tuple[Path, int]:

    base_name = f"input_frame_{input_frame}"
    
    for seq_num in range(1, max_sequence + 1):
        if seq_num == 1:
            output_scene_dir = base_output_dir / base_name / scene
        else:
            output_scene_dir = base_output_dir / f"{base_name}_{seq_num}" / scene
        
        if not (output_scene_dir / "metrics.json").exists():
            return output_scene_dir, seq_num

    return output_scene_dir, max_sequence


def evaluate_single_checkpoint(
    model: VGGT,
    ckpt_path: Path,
    data_dir: Path,
    output_path: Path,
    input_frame: int,
    sample_mode: str,
    sample_range_size: int,
    num_trials: int,
    num_scenes: int,
    depth_conf_thresh: float,
    use_proj: bool,
    use_icp: bool,
    max_points: Optional[int],
    da3_threshold: float,
    da3_down_sample: Optional[float],
    max_sequence: int,
    plot: bool,
    debug: bool,
):

    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"{'='*70}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    
    # Save hyperparameter settings
    import json
    hyperparams = {
        "data_dir": str(data_dir),
        "output_path": str(output_path),
        "plot": plot,
        "depth_conf_thresh": depth_conf_thresh,
        "pointcloud_backprojection_mode": POINTCLOUD_BACKPROJECTION_MODE,
        "input_frame": input_frame,
        "sample_mode": sample_mode,
        "sample_range_size": sample_range_size,
        "num_trials": num_trials,
        "num_scenes": num_scenes,
        "ckpt_path": str(ckpt_path),
        "use_proj": use_proj,
        "use_icp": use_icp,
        "max_points": max_points,
        "da3_threshold": da3_threshold,
        "da3_down_sample": da3_down_sample,
        "max_sequence": max_sequence,
        "random_seed": int(time.time()),
    }

    hyperparams_dir = output_path / f"input_frame_{input_frame}"
    hyperparams_dir.mkdir(parents=True, exist_ok=True)
    
    hyperparams_path = hyperparams_dir / "hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)


    all_scenes_metrics = {"scenes": {}, "average": {}}
    dtype = torch.bfloat16
    
    for scene in get_all_scenes(data_dir, int(999)):
        scene_dir = data_dir / f"{scene}"

        dataset = COLMAPDataset(str(scene_dir), rgb_dir="thermal", depth_dir="depth_aligned")
        dataset_size = len(dataset)
        print(f"Dataset loaded: {dataset_size} images")

        if input_frame > dataset_size:
            print(f"Skipping {scene}: dataset size ({dataset_size}) < input_frame ({input_frame})")
            continue

        adjusted_sample_range = min(sample_range_size, dataset_size - 1)
        if adjusted_sample_range != sample_range_size:
            print(f"Adjusting sample_range_size from {sample_range_size} to {adjusted_sample_range} (dataset size: {dataset_size})")
        else:
            adjusted_sample_range = sample_range_size

        scene_trial_metrics = []

        for trial_idx in range(num_trials):
            print(f"\n{'='*70}")
            print(f"Scene: {scene} | Trial: {trial_idx + 1}/{num_trials}")
            print(f"{'='*70}")
            np.random.seed(int(time.time()))

            selected_indices = dataset.sample_indices(input_frame, mode=sample_mode, range_size=adjusted_sample_range)
            
            print(f"Sampled indices: {selected_indices}")
            
            if len(selected_indices) == 0:
                print(f"No valid images found in {scene_dir}")
                continue

            scene_base_dir = output_path / f"input_frame_{input_frame}" / scene
            scene_base_dir.mkdir(parents=True, exist_ok=True)

            if num_trials > 1:
                trial_dir = scene_base_dir / f"trial_{trial_idx + 1}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                output_scene_dir = trial_dir
            else:
                output_scene_dir = scene_base_dir
            
            print(f"🚩Processing {scene}, sampled No.{selected_indices[0]} images from {len(dataset)} total images")
            print(f"📁Output directory: {output_scene_dir} (trial: {trial_idx + 1})")

            selected_samples = [dataset[idx] for idx in selected_indices]
            selected_image_paths = [sample["image_path"] for sample in selected_samples]
            selected_c2ws_raw = np.stack([sample["camera_to_world"] for sample in selected_samples])
            selected_frame_ids = list(range(len(selected_indices)))

            if len(selected_c2ws_raw) > 0:
                first_sample_pose = selected_c2ws_raw[0].copy()
                selected_c2ws = np.linalg.inv(selected_c2ws_raw[0]) @ selected_c2ws_raw
            else:
                raise ValueError("No valid camera poses sampled in the dataset.")
            all_cam_to_world_mat = []
            all_world_points = []

            try:
                images = load_images_rgb(selected_image_paths)

                if not images or len(images) < 2:
                    print(f"Skipping {scene} (trial {trial_idx + 1}): insufficient valid images")
                    continue

                images_array = np.stack(images)
                vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
                print(f"Patch dimensions: {patch_width}x{patch_height}")

                model.update_patch_dimensions(patch_width, patch_height)

                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        vgg_input_cuda = vgg_input.cuda()
                        predictions = model(vgg_input_cuda)
                torch.cuda.synchronize()
                end = time.time()
                inference_time_ms = (end - start) * 1000.0
                print(f"Inference time: {inference_time_ms:.2f}ms")

                pred_points, pred_conf = generate_pred_pointcloud_from_vggt(
                    predictions,
                    vgg_input,
                    depth_conf_thresh,
                )

                target_width = vgg_input.shape[3]
                target_height = vgg_input.shape[2]
                #print(f"Target point cloud dimensions: {target_width}x{target_height}")
                gt_points, gt_valid_mask, _depth = generate_gt_pointcloud_from_colmap(
                    selected_samples, max_points, target_width, target_height
                )

                if gt_points is None:
                    print(f"Skipping {scene} (trial {trial_idx + 1}): failed to generate GT point cloud")
                    continue

                print("Evaluating depth estimation metrics...")
                depth_scale = evaluate_depth_metrics(
                    pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                    gt_depth=_depth,
                    gt_valid_mask=None,
                    max_depth=80.0,
                    alignment_mode="scale",
                    per_frame=False,
                )
                depth_scale_shift = evaluate_depth_metrics(
                    pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                    gt_depth=_depth,
                    gt_valid_mask=None,
                    max_depth=80.0,
                    alignment_mode="scale_shift",
                    per_frame=False,
                )
                depth_mono = evaluate_depth_metrics(
                    pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                    gt_depth=_depth,
                    gt_valid_mask=None,
                    max_depth=80.0,
                    alignment_mode="median_scale",
                    per_frame=True,
                )
                depth_metrics = {
                    "abs_rel_scale": depth_scale["abs_rel"],
                    "delta_1.25_scale": depth_scale["delta_1.25"],
                    "abs_rel_scale_shift": depth_scale_shift["abs_rel"],
                    "delta_1.25_scale_shift": depth_scale_shift["delta_1.25"],
                    "abs_rel_mono": depth_mono["abs_rel"],
                    "delta_1.25_mono": depth_mono["delta_1.25"],
                }
                # print(f"  Abs Rel: {depth_metrics['abs_rel']:.6f}")
                # print(f"  δ < 1.25: {depth_metrics['delta_1.25']:.6f}")

                print("Evaluating point cloud reconstruction metrics...")
                pc_metrics = evaluate_pointcloud_metrics(
                    pred_points=pred_points,
                    gt_points=gt_points,
                    pred_conf=pred_conf,
                    gt_valid_mask=gt_valid_mask,
                    use_proj=use_proj,
                    use_icp=use_icp,
                    max_points=max_points,
                    da3_threshold=da3_threshold,
                    da3_down_sample=da3_down_sample,
                    debug=debug,
                )
                
                # print(f"Point Cloud Metrics:")
                # print(f"  ACC: {pc_metrics['acc']:.6f}")
                # print(f"  ACC_MED: {pc_metrics['acc_med']:.6f}")
                # print(f"  COMP: {pc_metrics['comp']:.6f}")
                # print(f"  COMP_MED: {pc_metrics['comp_med']:.6f}")
                # print(f"  NC1: {pc_metrics['nc1']:.6f}")
                # print(f"  NC1_MED: {pc_metrics['nc1_med']:.6f}")
                # print(f"  NC2: {pc_metrics['nc2']:.6f}")
                # print(f"  NC2_MED: {pc_metrics['nc2_med']:.6f}")
                # print(f"  NC: {pc_metrics['nc']:.6f}")
                # print(f"  NC_MED: {pc_metrics['nc_med']:.6f}")

                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
                )
                extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
                camera_poses_w2c = np.eye(4)[None, :, :].repeat(extrinsic_np.shape[0], axis=0)
                camera_poses_w2c[:, :3, :4] = extrinsic_np
                all_cam_to_world_mat = list(camera_poses_w2c)

                if not all_cam_to_world_mat:
                    print(
                        f"Skipping {scene} (trial {trial_idx + 1}): failed to obtain valid camera poses"
                    )
                    continue

                metrics = evaluate_scene_and_save(
                    scene,
                    selected_c2ws,
                    first_sample_pose,
                    selected_frame_ids,
                    all_cam_to_world_mat,
                    all_world_points,
                    output_scene_dir,
                    gt_points,
                    gt_valid_mask,
                    inference_time_ms,
                    plot,
                    max_points,
                    pc_metrics,
                    depth_metrics,
                )
                if metrics is not None:
                    metrics['sampled_indices'] = selected_indices.tolist()
                    scene_trial_metrics.append(metrics)

            except Exception as e:
                print(f"Error processing scene {scene} (trial {trial_idx + 1}): {e}")
                import traceback
                traceback.print_exc()
        

        if scene_trial_metrics:
            print(f"\n{'='*70}")
            print(f"Scene: {scene} - Summary of {len(scene_trial_metrics)} trials")
            print(f"{'='*70}")

            scene_avg_metrics = {}
            scene_std_metrics = {}
        
            for key in scene_trial_metrics[0].keys():
                values = [m[key] for m in scene_trial_metrics]
                scene_avg_metrics[key] = float(np.mean(values))
                scene_std_metrics[f"{key}_std"] = float(np.std(values))

                for i, trial_metrics in enumerate(scene_trial_metrics):
                    print(f"\nTrial {i + 1}:")
                    for key in POSE_SUMMARY_KEYS + POINTCLOUD_SUMMARY_KEYS + DEPTH_SUMMARY_KEYS:
                        if key in trial_metrics:
                            print(f"  {key}: {trial_metrics[key]:.6f}")

                print(f"\nAverage across {len(scene_trial_metrics)} trials:")
                for key in POSE_SUMMARY_KEYS + POINTCLOUD_SUMMARY_KEYS + DEPTH_SUMMARY_KEYS:
                    if key in scene_avg_metrics:
                        print(f"  {key}: {scene_avg_metrics[key]:.6f} ± {scene_std_metrics[f'{key}_std']:.6f}")

            all_scenes_metrics["scenes"][scene] = scene_avg_metrics
            all_scenes_metrics["scenes"][scene].update(scene_std_metrics)

            trial_results_path = scene_base_dir / "trial_results.json"
            import json
            with open(trial_results_path, 'w') as f:
                json.dump({
                "scene": scene,
                "num_trials": len(scene_trial_metrics),
                "trials": scene_trial_metrics,
                "average": scene_avg_metrics,
                "std": scene_std_metrics
            }, f, indent=4)
            print(f"\nTrial results saved to: {trial_results_path}")

    input_frame_dir = output_path / f"input_frame_{input_frame}"
    compute_average_metrics_and_save(
        all_scenes_metrics,
        output_path,
        input_frame,
        input_frame_dir
    )
    
    return all_scenes_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="eval_vggt/eval_config2.yaml",
        help="Path to evaluation config file (YAML format)",
    )
    parser.add_argument(
        "--data_dir", type=Path, default="./data/colmap"
    )
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=1.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=3,
        help="Maximum number of frames selected for processing per scene (deprecated, use --input_frames)",
    )
    parser.add_argument(
        "--input_frames",
        type=int,
        nargs="+",
        default=[3, 4, 6, 8, 10, 12],
        help="List of input frame counts to evaluate (default: [3, 4, 6, 8, 10, 12])",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="range",
        choices=["uniform", "random", "range"],
        help="Sampling mode for image indices: uniform, random, or range",
    )
    parser.add_argument(
        "--sample_range_size",
        type=int,
        default=8,
        help="Range size for range sampling mode (default: 25, i.e., +-12)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of independent sampling trials per scene (default: 1)",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to evaluate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./ckpt/model_tracker_fixed_e20.pt",
        help="Path to model checkpoint file (deprecated, use --ckpt_dirs)",
    )
    parser.add_argument(
        "--ckpt_dirs",
        type=str,
        nargs="+",
        default=None,
        help="List of checkpoint directories to evaluate (e.g., F:/Paper2/results/SCMIB_01/ckpts F:/Paper2/results/train_kd_05/ckpts)",
    )
    parser.add_argument(
        "--use_proj",
        action="store_true",
        help="Deprecated for Pi3-consistent evaluation; Umeyama Sim(3) alignment is always enabled",
    )
    parser.add_argument(
        "--use_icp",
        action="store_true",
        help="Deprecated for Pi3-consistent evaluation; ICP refinement is always enabled",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Maximum number of points for point cloud evaluation; None matches Pi3 behavior",
    )
    parser.add_argument(
        "--da3_threshold",
        type=float,
        default=0.05,
        help="Depth-Anything-3 F-score distance threshold",
    )
    parser.add_argument(
        "--da3_down_sample",
        type=float,
        default=None,
        help="Depth-Anything-3 evaluation voxel downsample size",
    )
    parser.add_argument(
        "--max_sequence",
        type=int,
        default=500,
        help="Maximum sequence number for output directories (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for point cloud metrics",
    )
    args = parser.parse_args()

    if args.config is not None:
        print(f"\n{'='*70}")
        print(f"Loading config file: {args.config}")
        print(f"{'='*70}")
        
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'data_dir' in config:
            args.data_dir = Path(config['data_dir'])
        if 'output_path' in config:
            args.output_path = Path(config['output_path'])
        if 'plot' in config:
            args.plot = config['plot']
        if 'depth_conf_thresh' in config:
            args.depth_conf_thresh = config['depth_conf_thresh']
        if 'input_frames' in config:
            args.input_frames = config['input_frames']
        if 'sample_mode' in config:
            args.sample_mode = config['sample_mode']
        if 'sample_range_size' in config:
            args.sample_range_size = config['sample_range_size']
        if 'num_trials' in config:
            args.num_trials = config['num_trials']
        if 'num_scenes' in config:
            args.num_scenes = config['num_scenes']
        if 'ckpt_dirs' in config:
            args.ckpt_dirs = config['ckpt_dirs']
        if 'use_proj' in config:
            if config['use_proj']:
                args.use_proj = True
        if 'use_icp' in config:
            if config['use_icp']:
                args.use_icp = True
        if 'max_points' in config:
            args.max_points = config['max_points']
        if 'da3_threshold' in config:
            args.da3_threshold = config['da3_threshold']
        if 'da3_down_sample' in config:
            args.da3_down_sample = config['da3_down_sample']
        if 'max_sequence' in config:
            args.max_sequence = config['max_sequence']
        if 'debug' in config:
            if config['debug']:
                args.debug = True
        
        print("Config loaded successfully!")
        print(f"{'='*70}\n")

    # Pi3-consistent point cloud evaluation always uses Umeyama Sim(3) + ICP.
    args.use_proj = True
    args.use_icp = True

    if args.ckpt_dirs is not None:
        print(f"\n{'='*70}")
        print("EVALUATION MODES: Batch")
        print(f"{'='*70}")

        ckpt_configs = []
        for ckpt_dir in args.ckpt_dirs:
            ckpt_dir_path = Path(ckpt_dir)
            ckpt_path = find_max_checkpoint(ckpt_dir_path)

            if ckpt_path is None:
                print(f"Warning: No checkpoint found in {ckpt_dir}")
                continue

            model_name = ckpt_dir_path.parent.name

            output_base_path = ckpt_dir_path.parent / "eval_results"
            
            ckpt_configs.append({
                "model_name": model_name,
                "ckpt_path": ckpt_path,
                "output_path": output_base_path,
            })
            
            print(f"  Model: {model_name}")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"  Output: {output_base_path}")
        
        if not ckpt_configs:
            print("Error: No valid checkpoints found")
            exit(1)

        print(
            f"\nWill evaluate {len(ckpt_configs)} models, with each model evaluated under {len(args.input_frames)} input-frame-count configurations")
        print(f"Input frame counts: {args.input_frames}")
        print(f"{'=' * 70}\n")

        model = VGGT()
        print(f"\n{'='*70}")
        print(f"Loading model!")

        for ckpt_config in ckpt_configs:
            model_name = ckpt_config["model_name"]
            ckpt_path = ckpt_config["ckpt_path"]
            output_path = ckpt_config["output_path"]

            print(f"\n{'#' * 70}")
            print(f"# Starting evaluation for model: {model_name}")
            print(f"# Checkpoint: {ckpt_path}")
            print(f"# Output directory: {output_path}")
            print(f"{'#' * 70}\n")

            for input_frame in args.input_frames:
                # Set the sampling range based on the number of input frames
                if input_frame in [3, 4, 6]:
                    current_sample_range = 12
                elif input_frame in [8, 10, 12]:
                    current_sample_range = 24
                else:
                    current_sample_range = args.sample_range_size

                
                try:
                    evaluate_single_checkpoint(
                        model=model,
                        ckpt_path=ckpt_path,
                        data_dir=args.data_dir,
                        output_path=output_path,
                        input_frame=input_frame,
                        sample_mode=args.sample_mode,
                        sample_range_size=current_sample_range,
                        num_trials=args.num_trials,
                        num_scenes=args.num_scenes,
                        depth_conf_thresh=args.depth_conf_thresh,
                        use_proj=args.use_proj,
                        use_icp=args.use_icp,
                        max_points=args.max_points,
                        da3_threshold=args.da3_threshold,
                        da3_down_sample=args.da3_down_sample,
                        max_sequence=args.max_sequence,
                        plot=args.plot,
                        debug=args.debug,
                    )
                except Exception as e:
                    print(f"Error evaluating {model_name} with input_frame={input_frame}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n{'#'*70}")
        print(f"# Done!!!")
        print(f"{'#'*70}\n")
    
    else:

        import json
        hyperparams = {
            "data_dir": str(args.data_dir),
            "output_path": str(args.output_path),
            "plot": args.plot,
            "depth_conf_thresh": args.depth_conf_thresh,
            "pointcloud_backprojection_mode": POINTCLOUD_BACKPROJECTION_MODE,
            "input_frame": args.input_frame,
            "sample_mode": args.sample_mode,
            "sample_range_size": args.sample_range_size,
            "num_trials": args.num_trials,
            "num_scenes": args.num_scenes,
            "ckpt_path": args.ckpt_path,
            "use_proj": args.use_proj,
            "use_icp": args.use_icp,
            "max_points": args.max_points,
            "da3_threshold": args.da3_threshold,
            "da3_down_sample": args.da3_down_sample,
            "max_sequence": args.max_sequence,
            "random_seed": int(time.time()),
        }

        hyperparams_dir = args.output_path / f"_{args.input_frame}_{args.sample_range_size}"
        hyperparams_dir.mkdir(parents=True, exist_ok=True)
        
        hyperparams_path = hyperparams_dir / "hyperparams.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)

        print(f"{'=' * 70}")
        print("Hyperparameter settings have been saved")
        print(f"{'=' * 70}")
        print(f"Save path: {hyperparams_path}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_path}")
        print(f"Input frame count: {args.input_frame}")
        print(f"Sampling mode: {args.sample_mode}")
        print(f"Sampling range size: {args.sample_range_size}")
        print(f"Number of trials: {args.num_trials}")
        print(f"Depth confidence threshold: {args.depth_conf_thresh}")
        print(f"Use projection alignment: {args.use_proj}")
        print(f"Use ICP refinement: {args.use_icp}")
        print(f"Maximum number of points: {args.max_points}")
        print(f"DA3 F-score threshold: {args.da3_threshold}")
        print(f"DA3 evaluation downsampling: {args.da3_down_sample}")
        print(f"Random seed: {hyperparams['random_seed']}")
        print(f"{'=' * 70}\n")

        all_scenes_metrics = {"scenes": {}, "average": {}}
        dtype = torch.bfloat16
        
        model = VGGT()
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model = model.cuda().eval()

        for scene in get_all_scenes(args.data_dir, int(999)):
            scene_dir = args.data_dir / f"{scene}"

            dataset = COLMAPDataset(str(scene_dir), rgb_dir="thermal", depth_dir="depth_aligned")
            print(f"Dataset loaded: {len(dataset)} images")

            scene_trial_metrics = []

            for trial_idx in range(args.num_trials):
                print(f"\n{'='*70}")
                print(f"Scene: {scene} | Trial: {trial_idx + 1}/{args.num_trials}")
                print(f"{'='*70}")

                selected_indices = dataset.sample_indices(args.input_frame, mode=args.sample_mode, range_size=args.sample_range_size)
                
                print(f"Sampled indices: {selected_indices}")
                
                if len(selected_indices) == 0:
                    print(f"No valid images found in {scene_dir}")
                    continue

                scene_base_dir = args.output_path / f"_{args.input_frame}_{args.sample_range_size}" / scene
                scene_base_dir.mkdir(parents=True, exist_ok=True)

                if args.num_trials > 1:
                    trial_dir = scene_base_dir / f"trial_{trial_idx + 1}"
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    output_scene_dir = trial_dir
                else:
                    output_scene_dir = scene_base_dir
                
                print(f"🚩Processing {scene}, sampled No.{selected_indices[0]} images from {len(dataset)} total images")
                print(f"📁Output directory: {output_scene_dir} (trial: {trial_idx + 1})")

                selected_samples = [dataset[idx] for idx in selected_indices]
                selected_image_paths = [sample["image_path"] for sample in selected_samples]
                selected_c2ws_raw = np.stack([sample["camera_to_world"] for sample in selected_samples])
                selected_frame_ids = list(range(len(selected_indices)))

                if len(selected_c2ws_raw) > 0:
                    first_sample_pose = selected_c2ws_raw[0].copy()
                    selected_c2ws = np.linalg.inv(selected_c2ws_raw[0]) @ selected_c2ws_raw
                else:
                    raise ValueError("No valid camera poses sampled in the dataset.")
                all_cam_to_world_mat = []
                all_world_points = []

                try:
                    images = load_images_rgb(selected_image_paths)

                    if not images or len(images) < 2:
                        print(f"Skipping {scene} (trial {trial_idx + 1}): insufficient valid images")
                        continue

                    images_array = np.stack(images)
                    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
                    print(f"Patch dimensions: {patch_width}x{patch_height}")

                    model.update_patch_dimensions(patch_width, patch_height)

                    torch.cuda.synchronize()
                    start = time.time()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype):
                            vgg_input_cuda = vgg_input.cuda()
                            predictions = model(vgg_input_cuda)
                    torch.cuda.synchronize()
                    end = time.time()
                    inference_time_ms = (end - start) * 1000.0
                    print(f"Inference time: {inference_time_ms:.2f}ms")

                    pred_points, pred_conf = generate_pred_pointcloud_from_vggt(
                        predictions,
                        vgg_input,
                        args.depth_conf_thresh,
                    )

                    target_width = vgg_input.shape[3]
                    target_height = vgg_input.shape[2]
                    #print(f"Target point cloud dimensions: {target_width}x{target_height}")
                    gt_points, gt_valid_mask, _depth = generate_gt_pointcloud_from_colmap(
                        selected_samples, args.max_points, target_width, target_height
                    )

                    if gt_points is None:
                        print(f"Skipping {scene} (trial {trial_idx + 1}): failed to generate GT point cloud")
                        continue

                    print("Evaluating depth estimation metrics...")
                    depth_scale = evaluate_depth_metrics(
                        pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                        gt_depth=_depth,
                        gt_valid_mask=None,
                        max_depth=80.0,
                        alignment_mode="scale",
                        per_frame=False,
                    )
                    depth_scale_shift = evaluate_depth_metrics(
                        pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                        gt_depth=_depth,
                        gt_valid_mask=None,
                        max_depth=80.0,
                        alignment_mode="scale_shift",
                        per_frame=False,
                    )
                    depth_mono = evaluate_depth_metrics(
                        pred_depth=predictions["depth"][0].detach().float().cpu().numpy(),
                        gt_depth=_depth,
                        gt_valid_mask=None,
                        max_depth=80.0,
                        alignment_mode="median_scale",
                        per_frame=True,
                    )
                    depth_metrics = {
                        "abs_rel_scale": depth_scale["abs_rel"],
                        "delta_1.25_scale": depth_scale["delta_1.25"],
                        "abs_rel_scale_shift": depth_scale_shift["abs_rel"],
                        "delta_1.25_scale_shift": depth_scale_shift["delta_1.25"],
                        "abs_rel_mono": depth_mono["abs_rel"],
                        "delta_1.25_mono": depth_mono["delta_1.25"],
                    }
                    # print(f"  Abs Rel: {depth_metrics['abs_rel']:.6f}")
                    # print(f"  δ < 1.25: {depth_metrics['delta_1.25']:.6f}")

                    # Evaluate point cloud reconstruction metrics（ACC, COMP, NC，acc_da, comp_da,overall, fscore_da）
                    print("Evaluating point cloud reconstruction metrics...")
                    pc_metrics = evaluate_pointcloud_metrics(
                        pred_points=pred_points,
                        gt_points=gt_points,
                        pred_conf=pred_conf,
                        gt_valid_mask=gt_valid_mask,
                        use_proj=args.use_proj,
                        use_icp=args.use_icp,
                        max_points=args.max_points,
                        da3_threshold=args.da3_threshold,
                        da3_down_sample=args.da3_down_sample,
                        debug=args.debug,
                    )
                    
                    # print(f"Point Cloud Metrics:")
                    # print(f"  ACC: {pc_metrics['acc']:.6f}")
                    # print(f"  ACC_MED: {pc_metrics['acc_med']:.6f}")
                    # print(f"  COMP: {pc_metrics['comp']:.6f}")
                    # print(f"  COMP_MED: {pc_metrics['comp_med']:.6f}")
                    # print(f"  NC1: {pc_metrics['nc1']:.6f}")
                    # print(f"  NC1_MED: {pc_metrics['nc1_med']:.6f}")
                    # print(f"  NC2: {pc_metrics['nc2']:.6f}")
                    # print(f"  NC2_MED: {pc_metrics['nc2_med']:.6f}")
                    # print(f"  NC: {pc_metrics['nc']:.6f}")
                    # print(f"  NC_MED: {pc_metrics['nc_med']:.6f}")

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
                    )
                    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
                    camera_poses_w2c = np.eye(4)[None, :, :].repeat(extrinsic_np.shape[0], axis=0)
                    camera_poses_w2c[:, :3, :4] = extrinsic_np
                    all_cam_to_world_mat = list(camera_poses_w2c)

                    all_world_points = []
                    for frame_idx in range(pred_points.shape[0]):
                        points = pred_points[frame_idx].reshape(-1, 3)
                        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
                        valid_points = points[valid_mask]
                        if len(valid_points) > 0:
                            all_world_points.append(valid_points)

                    if not all_cam_to_world_mat or not all_world_points:
                        print(
                            f"Skipping {scene} (trial {trial_idx + 1}): failed to obtain valid camera poses or point clouds"
                        )
                        continue

                    metrics = evaluate_scene_and_save(
                        scene,
                        selected_c2ws,
                        first_sample_pose,
                        selected_frame_ids,
                        all_cam_to_world_mat,
                        all_world_points,
                        output_scene_dir,
                        gt_points,
                        gt_valid_mask,
                        inference_time_ms,
                        args.plot,
                        args.max_points,
                        pc_metrics,
                        depth_metrics,
                    )
                    if metrics is not None:
                        scene_trial_metrics.append(metrics)

                except Exception as e:
                    print(f"Error processing scene {scene} (trial {trial_idx + 1}): {e}")
                    import traceback
                    traceback.print_exc()
            

            if scene_trial_metrics:
                print(f"\n{'='*70}")
                print(f"Scene: {scene} - Summary of {len(scene_trial_metrics)} trials")
                print(f"{'='*70}")

                scene_avg_metrics = {}
                scene_std_metrics = {}
            
                for key in scene_trial_metrics[0].keys():
                    values = [m[key] for m in scene_trial_metrics]
                    scene_avg_metrics[key] = float(np.mean(values))
                    scene_std_metrics[f"{key}_std"] = float(np.std(values))

                for i, trial_metrics in enumerate(scene_trial_metrics):
                    print(f"\nTrial {i + 1}:")
                    for key in POSE_SUMMARY_KEYS + POINTCLOUD_SUMMARY_KEYS + DEPTH_SUMMARY_KEYS:
                        if key in trial_metrics:
                            print(f"  {key}: {trial_metrics[key]:.6f}")

                print(f"\nAverage across {len(scene_trial_metrics)} trials:")
                for key in POSE_SUMMARY_KEYS + POINTCLOUD_SUMMARY_KEYS + DEPTH_SUMMARY_KEYS:
                    if key in scene_avg_metrics:
                        print(f"  {key}: {scene_avg_metrics[key]:.6f} ± {scene_std_metrics[f'{key}_std']:.6f}")
            

                all_scenes_metrics["scenes"][scene] = scene_avg_metrics
                all_scenes_metrics["scenes"][scene].update(scene_std_metrics)

                trial_results_path = scene_base_dir / "trial_results.json"
                import json
                with open(trial_results_path, 'w') as f:
                    json.dump({
                    "scene": scene,
                    "num_trials": len(scene_trial_metrics),
                    "trials": scene_trial_metrics,
                    "average": scene_avg_metrics,
                    "std": scene_std_metrics
                }, f, indent=4)
                print(f"\nTrial results saved to: {trial_results_path}")

        input_frame_dir = args.output_path / f"_{args.input_frame}_{args.sample_range_size}"
        compute_average_metrics_and_save(
            all_scenes_metrics,
            args.output_path,
            args.input_frame,
            input_frame_dir
        )


# python ./eval/eval_script.py --data_dir dataset --use_proj --use_icp
