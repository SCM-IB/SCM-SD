# VGGT Point Cloud Reconstruction Evaluator
# Transferable point cloud reconstruction index calculation tool，based on7andNDataset evaluation process

import torch
import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial import KDTree as DAKDTree
from scipy.spatial import cKDTree as KDTree


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    UmeyamaAlgorithm to achieve point cloud alignment
    
    Args:
        src: Source point cloud (N, 3)
        dst: target point cloud (N, 3)
        with_scale: Whether to estimate scale factors
    
    Returns:
        (scale, rotation, translation) scale factor、rotation matrix、translation vector
    """
    assert src.shape == dst.shape
    N, dim = src.shape
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    
    Sigma = dst_c.T @ src_c / N  # (3,3)
    
    U, D, Vt = np.linalg.svd(Sigma)
    
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    
    R = U @ S @ Vt
    
    if with_scale:
        var_src = (src_c**2).sum() / N
        s = (D * S.diagonal()).sum() / var_src
    else:
        s = 1.0
    
    t = mu_dst - s * R @ mu_src
    
    return s, R, t


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Apply rigid body transformation to point cloud
    
    Args:
        points: point cloud (N, 3)
        scale: scale factor
        rotation: rotation matrix (3, 3)
        translation: translation vector (3,)
    
    Returns:
        Transformed point cloud (N, 3)
    """
    return (scale * (rotation @ points.T)).T + translation


def get_valid_points(points: np.ndarray, valid_mask: Optional[np.ndarray] = None, max_points: Optional[int] = None) -> np.ndarray:
    """
    Obtain valid point cloud data and sample it
    
    Args:
        points: Raw point cloud data
        valid_mask: valid point mask
        max_points: Maximum points limit
    
    Returns:
        Processed point cloud (N, 3)
    """
    # Flatten point cloud
    points_flat = points.reshape(-1, 3)
    
    # Apply effective mask
    if valid_mask is not None:
        mask_flat = valid_mask.reshape(-1)
        points_flat = points_flat[mask_flat]
    
    # Remove invalid points（NaNorInf）
    valid_points_mask = np.isfinite(points_flat).all(axis=1)
    points_flat = points_flat[valid_points_mask]
    
    # Limit points
    if max_points is not None and len(points_flat) > max_points:
        sample_indices = np.random.choice(len(points_flat), max_points, replace=False)
        points_flat = points_flat[sample_indices]
    
    return points_flat


def get_corresponding_points(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_valid_mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect pixel-aligned point correspondences for Sim(3) alignment.
    """
    pred_flat = pred_points.reshape(-1, 3)
    gt_flat = gt_points.reshape(-1, 3)

    common_mask = np.isfinite(pred_flat).all(axis=1) & np.isfinite(gt_flat).all(axis=1)
    if pred_mask is not None:
        common_mask &= pred_mask.reshape(-1).astype(bool)
    if gt_valid_mask is not None:
        common_mask &= gt_valid_mask.reshape(-1).astype(bool)

    pred_corr = pred_flat[common_mask]
    gt_corr = gt_flat[common_mask]

    if max_points is not None and len(pred_corr) > max_points:
        sample_indices = np.random.choice(len(pred_corr), max_points, replace=False)
        pred_corr = pred_corr[sample_indices]
        gt_corr = gt_corr[sample_indices]

    return pred_corr, gt_corr


def estimate_normals(points: np.ndarray, radius: float = None, max_nn: int = None) -> np.ndarray:
    """
    Estimate point cloud normals
    
    Args:
        points: point cloud (N, 3)
        radius: search radius（if forNone，Use default parameters）
        max_nn: Maximum number of neighbors（if forNone，Use default parameters）
    
    Returns:
        normal vector (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Use default parameters，and eval_7andN.py Be consistent
    pcd.estimate_normals()
    return np.asarray(pcd.normals)


def accuracy(gt_points: np.ndarray, pred_points: np.ndarray, gt_normals: np.ndarray, pred_normals: np.ndarray, threshold: float = 0.05, debug: bool = False) -> Tuple[float, float, float, float]:
    """
    Calculate point cloud accuracy metrics
    
    Args:
        gt_points: Ground truth point cloud
        pred_points: Predicted point cloud
        gt_normals: ground truth normal
        pred_normals: Predict normal
        threshold: distance threshold
        debug: Whether to print debugging information
    
    Returns:
        (acc, acc_med, nc1, nc1_med)
        - acc: Accuracy average
        - acc_med: Median precision
        - nc1: Normal consistency average
        - nc1_med: Median normal consistency
    """
    if debug:
        print(f"\n[ACC DEBUG] GTPoints: {len(gt_points)}, Forecast points: {len(pred_points)}")
    
    gt_points_kd_tree = KDTree(np.asarray(gt_points))
    distances, idx = gt_points_kd_tree.query(np.asarray(pred_points), workers=-1)
    normal_consistencies = np.abs(np.sum(np.asarray(gt_normals)[idx] * np.asarray(pred_normals), axis=-1))
    
    if debug:
        print(f"[ACC DEBUG] distance statistics: min={np.min(distances):.6f}, max={np.max(distances):.6f}, mean={np.mean(distances):.6f}, median={np.median(distances):.6f}")
        print(f"[ACC DEBUG] distance distribution: <0.01: {np.sum(distances<0.01)}, <0.05: {np.sum(distances<0.05)}, <0.1: {np.sum(distances<0.1)}, <0.5: {np.sum(distances<0.5)}, >0.5: {np.sum(distances>0.5)}")
    
    acc = np.mean(distances)
    acc_med = np.median(distances)
    
    nc1 = np.mean(normal_consistencies) if len(normal_consistencies) > 0 else 0.0
    nc1_med = np.median(normal_consistencies) if len(normal_consistencies) > 0 else 0.0
    
    return acc, acc_med, nc1, nc1_med


def completion(gt_points: np.ndarray, pred_points: np.ndarray, gt_normals: np.ndarray, pred_normals: np.ndarray, threshold: float = 0.05, debug: bool = False) -> Tuple[float, float, float, float]:
    """
    Calculate point cloud integrity metrics
    
    Args:
        gt_points: Ground truth point cloud
        pred_points: Predicted point cloud
        gt_normals: ground truth normal
        pred_normals: Predict normal
        threshold: distance threshold
        debug: Whether to print debugging information
    
    Returns:
        (comp, comp_med, nc2, nc2_med)
        - comp: Completeness average
        - comp_med: median completeness
        - nc2: Normal consistency average
        - nc2_med: Median normal consistency
    """
    if debug:
        print(f"\n[COMP DEBUG] GTPoints: {len(gt_points)}, Forecast points: {len(pred_points)}")
    
    pred_points_kd_tree = KDTree(np.asarray(pred_points))
    distances, idx = pred_points_kd_tree.query(np.asarray(gt_points), workers=-1)
    normal_consistencies = np.abs(np.sum(np.asarray(gt_normals) * np.asarray(pred_normals)[idx], axis=-1))
    
    if debug:
        print(f"[COMP DEBUG] distance statistics: min={np.min(distances):.6f}, max={np.max(distances):.6f}, mean={np.mean(distances):.6f}, median={np.median(distances):.6f}")
        print(f"[COMP DEBUG] distance distribution: <0.01: {np.sum(distances<0.01)}, <0.05: {np.sum(distances<0.05)}, <0.1: {np.sum(distances<0.1)}, <0.5: {np.sum(distances<0.5)}, >0.5: {np.sum(distances>0.5)}")
    
    comp = np.mean(distances)
    comp_med = np.median(distances)
    
    nc2 = np.mean(normal_consistencies) if len(normal_consistencies) > 0 else 0.0
    nc2_med = np.median(normal_consistencies) if len(normal_consistencies) > 0 else 0.0
    
    return comp, comp_med, nc2, nc2_med


def icp_refinement(src_points: np.ndarray, dst_points: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    useICPAlgorithm fine-tuning point cloud alignment
    
    Args:
        src_points: Source point cloud
        dst_points: target point cloud
        threshold: ICPthreshold
    
    Returns:
        transformation matrix (4, 4)
    """
    # createOpen3Dpoint cloud object
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_points)
    
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst_points)
    
    # implementICP
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_dst,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    
    return reg_p2p.transformation


def evaluate_pointcloud_reconstruction(
    pred_points: np.ndarray,      # Predicted point cloud (H, W, 3) or (B, H, W, 3)
    gt_points: np.ndarray,         # Real point cloud (H, W, 3) or (B, H, W, 3)
    pred_conf: Optional[np.ndarray] = None,  # prediction confidence (H, W) or (B, H, W)
    gt_valid_mask: Optional[np.ndarray] = None,  # Effective masks for real point clouds (H, W) or (B, H, W)
    use_proj: bool = True,         # Whether to use projection transformation
    use_icp: bool = True,          # Whether to useICPfine-tuning
    max_points: Optional[int] = None,       # Maximum points limit
    icp_threshold: float = 0.1,     # ICPthreshold
    normal_radius: float = 0.05,    # normal estimated radius
    debug: bool = False            # Whether to print debugging information
) -> Dict[str, float]:
    """
    Evaluate point cloud reconstruction quality
    
    Args:
        pred_points: Predicted point cloud data
        gt_points: Real point cloud data
        pred_conf: prediction confidence，Used to filter low confidence points
        gt_valid_mask: Effective masks for real point clouds
        use_proj: Whether to useUmeyamaAlgorithm for point cloud alignment
        use_icp: Whether to useICPMake fine adjustments
        max_points: Maximum points limit
        icp_threshold: ICPthreshold
        normal_radius: normal estimated radius
    
    Returns:
        Evaluation Metrics Dictionary，Include：
        - acc: Accuracy average
        - acc_med: Median precision
        - comp: Completeness average
        - comp_med: median completeness
        - nc1: Precision Normal Consistency Average
        - nc1_med: Precision Normal Consistency Median
        - nc2: Integrity Normal Consistency Average
        - nc2_med: Complete normal consistency median
        - nc: Normal consistency average
        - nc_med: Median normal consistency
    """
    # Handle input dimensions
    if pred_points.ndim == 4:
        # Batch processing，Take the first sample
        pred_points = pred_points[0]
    if gt_points.ndim == 4:
        gt_points = gt_points[0]
    if pred_conf is not None and pred_conf.ndim == 3:
        pred_conf = pred_conf[0]
    if gt_valid_mask is not None and gt_valid_mask.ndim == 3:
        gt_valid_mask = gt_valid_mask[0]
    
    # Generate masks for predicted point clouds
    pred_mask = None
    if gt_valid_mask is None and pred_conf is not None:
        pred_mask = pred_conf > 0.0
    
    # Get valid points
    pred_eval_mask = gt_valid_mask if gt_valid_mask is not None else pred_mask
    pred_points_valid = get_valid_points(pred_points, pred_eval_mask, max_points)
    gt_points_valid = get_valid_points(gt_points, gt_valid_mask, max_points)
    
    if debug:
        print(f"\n[DEBUG] Predict effective points in point cloud: {len(pred_points_valid)}")
        print(f"[DEBUG] GTPoint cloud valid points: {len(gt_points_valid)}")
        print(f"[DEBUG] point ratio (pred/gt): {len(pred_points_valid)/len(gt_points_valid) if len(gt_points_valid) > 0 else 0:.3f}")
    
    # Make sure the point cloud is not empty
    if len(pred_points_valid) == 0 or len(gt_points_valid) == 0:
        return {
            'acc': float('inf'), 'acc_med': float('inf'),
            'comp': float('inf'), 'comp_med': float('inf'),
            'nc1': 0.0, 'nc1_med': 0.0,
            'nc2': 0.0, 'nc2_med': 0.0,
            'nc': 0.0, 'nc_med': 0.0
        }
    
    # point cloud alignment
    aligned_pred_points = pred_points_valid.copy()
    
    if use_proj:
        pred_corr, gt_corr = get_corresponding_points(
            pred_points,
            gt_points,
            pred_mask=pred_eval_mask,
            gt_valid_mask=gt_valid_mask,
            max_points=max_points,
        )
        if len(pred_corr) >= 3 and len(gt_corr) >= 3:
            s, R, t = umeyama_alignment(pred_corr, gt_corr, with_scale=True)
            aligned_pred_points = transform_points(pred_points_valid, s, R, t)
    
    if use_icp:
        # useICPMake fine adjustments
        transform = icp_refinement(aligned_pred_points, gt_points_valid, icp_threshold)
        # applicationICPtransform
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(aligned_pred_points)
        pcd_pred.transform(transform)
        aligned_pred_points = np.asarray(pcd_pred.points)
    
    # Estimate normal
    pred_normals = estimate_normals(aligned_pred_points, normal_radius)
    gt_normals = estimate_normals(gt_points_valid, normal_radius)
    
    # Calculate indicators
    acc, acc_med, nc1, nc1_med = accuracy(
        gt_points_valid, aligned_pred_points, gt_normals, pred_normals, debug=debug
    )
    
    comp, comp_med, nc2, nc2_med = completion(
        gt_points_valid, aligned_pred_points, gt_normals, pred_normals, debug=debug
    )
        
    # Calculate normal consistency average
    nc = (nc1 + nc2) / 2
    nc_med = (nc1_med + nc2_med) / 2
    if debug:
        print(f"\n[DEBUG] final indicator:")
        print(f"[DEBUG]   ACC: {acc:.6f}, ACC_MED: {acc_med:.6f}")
        print(f"[DEBUG]   COMP: {comp:.6f}, COMP_MED: {comp_med:.6f}")
        print(f"[DEBUG]   NC1: {nc1:.6f}, NC1_MED: {nc1_med:.6f}")
        print(f"[DEBUG]   NC2: {nc2:.6f}, NC2_MED: {nc2_med:.6f}")
        print(f"[DEBUG]   NC: {nc:.6f}, NC_MED: {nc_med:.6f}")

    return {
        'acc': acc,
        'acc_med': acc_med,
        'comp': comp,
        'comp_med': comp_med,
        'nc1': nc1,
        'nc1_med': nc1_med,
        'nc2': nc2,
        'nc2_med': nc2_med,
        'nc': nc,
        'nc_med': nc_med
    }


def evaluate_vggt_output(
    vggt_predictions: Dict[str, torch.Tensor],
    gt_data: Dict[str, torch.Tensor],
    **kwargs
) -> Dict[str, float]:
    """
    EvaluateVGGTPoint cloud reconstruction output of the model
    
    Args:
        vggt_predictions: VGGTModel’s predicted output，Include：
            - world_points: Predictive3Dpoint cloud (B, S, H, W, 3)
            - world_points_conf: prediction confidence (B, S, H, W)
        gt_data: real data，Include：
            - pts3d: real3Dpoint cloud (B, S, H, W, 3)
            - valid_mask: Effective masks for real point clouds (B, S, H, W)
        **kwargs: passed toevaluate_pointcloud_reconstructionOther parameters of
    
    Returns:
        Evaluation Metrics Dictionary
    """
    # Extract data
    pred_points = vggt_predictions['world_points'].cpu().numpy()
    pred_conf = vggt_predictions['world_points_conf'].cpu().numpy()
    gt_points = gt_data['pts3d'].cpu().numpy()
    gt_valid_mask = gt_data['valid_mask'].cpu().numpy()
    
    # Evaluate point cloud reconstruction
    metrics = evaluate_pointcloud_reconstruction(
        pred_points, gt_points, pred_conf, gt_valid_mask, **kwargs
    )
    
    return metrics


def nn_correspondance_da(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:
    """
    Compute nearest neighbor distances from verts2 to verts1 using KDTree.
    (From Depth-Anything-3)
    
    Args:
        verts1: Reference point cloud [N, 3]
        verts2: Query point cloud [M, 3]
    
    Returns:
        Distance array [M,] - distance from each point in verts2 to nearest in verts1
    """
    if len(verts1) == 0 or len(verts2) == 0:
        return np.array([])
    
    kdtree = DAKDTree(verts1)
    distances, _ = kdtree.query(verts2)
    return distances.reshape(-1)


def evaluate_3d_reconstruction_da(
    pcd_pred: Union[o3d.geometry.PointCloud, np.ndarray],
    pcd_trgt: Union[o3d.geometry.PointCloud, np.ndarray],
    threshold: float = 0.05,
    down_sample: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate 3D reconstruction quality using Depth-Anything-3 metrics.
    
    This function computes:
    - Accuracy (acc_da): Mean distance from predicted points to GT surface
    - Completeness (comp_da): Mean distance from GT points to predicted surface
    - Overall: Average of accuracy and completeness
    - Precision: Fraction of predicted points within threshold of GT
    - Recall: Fraction of GT points within threshold of prediction
    - F-score (fscore_da): Harmonic mean of precision and recall
    
    Args:
        pcd_pred: Predicted point cloud (Open3D or numpy array)
        pcd_trgt: Ground truth point cloud (Open3D or numpy array)
        threshold: Distance threshold for precision/recall (meters)
        down_sample: Voxel size for downsampling (None to skip)
    
    Returns:
        Dict with metrics: acc_da, comp_da, overall, precision, recall, fscore_da
    """
    # Convert to Open3D if needed
    if isinstance(pcd_pred, np.ndarray):
        pcd_pred_o3d = o3d.geometry.PointCloud()
        pcd_pred_o3d.points = o3d.utility.Vector3dVector(pcd_pred)
        pcd_pred = pcd_pred_o3d
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt_o3d = o3d.geometry.PointCloud()
        pcd_trgt_o3d.points = o3d.utility.Vector3dVector(pcd_trgt)
        pcd_trgt = pcd_trgt_o3d
    
    # Downsample if requested
    if down_sample is not None and down_sample > 0:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    # Handle empty point clouds
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {
            "acc_da": float("inf"),
            "comp_da": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore_da": 0.0,
        }
    
    # Compute distances
    dist_pred_to_gt = nn_correspondance_da(verts_trgt, verts_pred)  # Accuracy
    dist_gt_to_pred = nn_correspondance_da(verts_pred, verts_trgt)  # Completeness
    
    # Compute metrics
    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    overall = (accuracy + completeness) / 2
    
    precision = float(np.mean((dist_pred_to_gt < threshold).astype(float)))
    recall = float(np.mean((dist_gt_to_pred < threshold).astype(float)))
    
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "acc_da": accuracy,
        "comp_da": completeness,
        "overall": overall,
        "precision": precision,
        "recall": recall,
        "fscore_da": fscore,
    }


def evaluate_pointcloud_reconstruction_da(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_conf: Optional[np.ndarray] = None,
    gt_valid_mask: Optional[np.ndarray] = None,
    use_proj: bool = True,
    use_icp: bool = True,
    max_points: Optional[int] = None,
    icp_threshold: float = 0.1,
    threshold: float = 0.05,
    down_sample: Optional[float] = None,
    debug: bool = False
) -> Dict[str, float]:
    """
    Evaluate point cloud reconstruction using Depth-Anything-3 metrics.
    
    This function evaluates point cloud quality using the Depth-Anything-3 benchmark
    metrics, which include accuracy (acc_da), completeness (comp_da), and F-score (fscore_da).
    
    Args:
        pred_points: Predicted point cloud (H, W, 3) or (B, H, W, 3)
        gt_points: Ground truth point cloud (H, W, 3) or (B, H, W, 3)
        pred_conf: Predicted confidence (H, W) or (B, H, W)
        gt_valid_mask: Ground truth valid mask (H, W) or (B, H, W)
        use_proj: Whether to use Umeyama alignment
        use_icp: Whether to use ICP refinement
        max_points: Maximum number of points
        icp_threshold: ICP threshold
        threshold: Distance threshold for precision/recall (meters)
        down_sample: Voxel size for downsampling (None to skip)
        debug: Whether to print debug information
    
    Returns:
        Dict with metrics: acc_da, comp_da, overall, precision, recall, fscore_da
    """
    # Handle batch dimension
    if pred_points.ndim == 4:
        pred_points = pred_points[0]
    if gt_points.ndim == 4:
        gt_points = gt_points[0]
    if pred_conf is not None and pred_conf.ndim == 3:
        pred_conf = pred_conf[0]
    if gt_valid_mask is not None and gt_valid_mask.ndim == 3:
        gt_valid_mask = gt_valid_mask[0]
    
    # Generate prediction mask
    pred_mask = None
    if gt_valid_mask is None and pred_conf is not None:
        pred_mask = pred_conf > 0.0
    
    # Get valid points
    pred_eval_mask = gt_valid_mask if gt_valid_mask is not None else pred_mask
    pred_points_valid = get_valid_points(pred_points, pred_eval_mask, max_points)
    gt_points_valid = get_valid_points(gt_points, gt_valid_mask, max_points)
    
    if debug:
        print(f"\n[DA DEBUG] Predict effective points in point cloud: {len(pred_points_valid)}")
        print(f"[DA DEBUG] GTPoint cloud valid points: {len(gt_points_valid)}")
    
    # Ensure point clouds are not empty
    if len(pred_points_valid) == 0 or len(gt_points_valid) == 0:
        return {
            'acc_da': float('inf'),
            'comp_da': float('inf'),
            'overall': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'fscore_da': 0.0,
        }
    
    # Point cloud alignment
    aligned_pred_points = pred_points_valid.copy()
    
    if use_proj:
        pred_corr, gt_corr = get_corresponding_points(
            pred_points,
            gt_points,
            pred_mask=pred_eval_mask,
            gt_valid_mask=gt_valid_mask,
            max_points=max_points,
        )
        if len(pred_corr) >= 3 and len(gt_corr) >= 3:
            s, R, t = umeyama_alignment(pred_corr, gt_corr, with_scale=True)
            aligned_pred_points = transform_points(pred_points_valid, s, R, t)
    
    if use_icp:
        # Use ICP for refinement
        transform = icp_refinement(aligned_pred_points, gt_points_valid, icp_threshold)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(aligned_pred_points)
        pcd_pred.transform(transform)
        aligned_pred_points = np.asarray(pcd_pred.points)
    
    # Evaluate using Depth-Anything-3 metrics
    metrics = evaluate_3d_reconstruction_da(
        aligned_pred_points,
        gt_points_valid,
        threshold=threshold,
        down_sample=down_sample
    )
    
    if debug:
        print(f"\n[DA DEBUG] Depth-Anything-3 index:")
        print(f"[DA DEBUG]   acc_da: {metrics['acc_da']:.6f}")
        print(f"[DA DEBUG]   comp_da: {metrics['comp_da']:.6f}")
        print(f"[DA DEBUG]   overall: {metrics['overall']:.6f}")
        print(f"[DA DEBUG]   fscore_da: {metrics['fscore_da']:.6f}")
    
    return metrics


if __name__ == "__main__":
    """
    Usage example
    """
    # Example1: Basic use
    print("=== VGGT Point Cloud Evaluator Example ===")
    
    # Generate random point cloud data as an example
    H, W = 224, 224
    
    # Predicted point cloud (B, H, W, 3)
    pred_points = np.random.rand(1, H, W, 3) * 10
    # Real point cloud (B, H, W, 3)
    gt_points = np.random.rand(1, H, W, 3) * 10
    # Prediction confidence (B, H, W)
    pred_conf = np.random.rand(1, H, W)
    # Real point cloud effective mask (B, H, W)
    gt_valid_mask = np.random.rand(1, H, W) > 0.3
    
    # Evaluate point cloud reconstruction
    metrics = evaluate_pointcloud_reconstruction(
        pred_points, gt_points, pred_conf, gt_valid_mask
    )
    
    print("Assessment results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== VGGTOutput format evaluation example ===")
    
    # Example2: useVGGTOutput format
    # simulationVGGTModel output
    vggt_output = {
        'world_points': torch.tensor(pred_points, dtype=torch.float32),
        'world_points_conf': torch.tensor(pred_conf, dtype=torch.float32)
    }
    
    # Simulate real data
    gt_data = {
        'pts3d': torch.tensor(gt_points, dtype=torch.float32),
        'valid_mask': torch.tensor(gt_valid_mask, dtype=torch.bool)
    }
    
    # Evaluate
    vggt_metrics = evaluate_vggt_output(vggt_output, gt_data)
    
    print("VGGTOutput evaluation results:")
    for key, value in vggt_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Evaluation example without projective transformation ===")
    
    # Example3: No projection transformation is used
    metrics_no_proj = evaluate_pointcloud_reconstruction(
        pred_points, gt_points, pred_conf, gt_valid_mask, use_proj=False
    )
    
    print("Evaluation results without using projection transformation:")
    print(f"acc: {metrics_no_proj['acc']:.4f}")
    print(f"comp: {metrics_no_proj['comp']:.4f}")
