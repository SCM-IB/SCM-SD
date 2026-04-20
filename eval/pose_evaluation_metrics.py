from copy import deepcopy
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import evo.tools.plot as plot
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from scipy.linalg import svd
from scipy.spatial.transform import Rotation


def closed_form_inverse_se3(se3):
    """
    Compute the inverse of each 4x4 SE3 matrix in a batch.
    Used to convert camera-from-world (w2c) to camera-to-world (c2w) matrix.
    
    Args:
        se3: SE3 matrices of shape (N, 4, 4) or (N, 3, 4)
        
    Returns:
        Inverted SE3 matrices of shape (N, 4, 4)
    """
    is_numpy = isinstance(se3, np.ndarray)
    
    if se3.shape[-2:] == (3, 4):
        se3_4x4 = np.eye(4)[None].repeat(len(se3), axis=0) if is_numpy else torch.eye(4)[None].repeat(len(se3), 1, 1)
        se3_4x4[:, :3, :4] = se3
        se3 = se3_4x4
    
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    
    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)
    
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    
    return inverted_matrix


def generate_valid_rotation_matrices(n):
    """
    Generate valid rotation matrices using scipy's Rotation class.
    
    Args:
        n: Number of rotation matrices to generate
        
    Returns:
        Rotation matrices of shape (n, 3, 3)
    """
    rotations = Rotation.random(n)
    return rotations.as_matrix()


def normalize_trajectory_to_first_frame(poses):
    """
    Normalize a trajectory so that the first frame becomes the world coordinate system origin.
    
    This function transforms all poses in the trajectory such that the first pose becomes
    the identity matrix (origin), and all other poses are relative to this first frame.
    
    Args:
        poses: Camera-to-world transformation matrices of shape (N, 4, 4)
        
    Returns:
        normalized_poses: Normalized camera-to-world transformation matrices of shape (N, 4, 4)
    """
    if len(poses) == 0:
        return poses
    
    # Get the first pose (will become the origin)
    first_pose = poses[0]
    
    # Compute the inverse of the first pose to get the transformation from first pose to origin
    first_pose_inv = np.linalg.inv(first_pose)
    
    # Apply this transformation to all poses
    normalized_poses = np.array([first_pose_inv @ pose for pose in poses])
    
    return normalized_poses


def camera_extrinsics_to_trajectory(extrinsics):
    """
    Convert camera extrinsic parameters (3x4 or 4x4 matrices) to trajectory format.
    
    This function converts camera extrinsics (which represent camera-from-world transformation)
    to camera-to-world transformation matrices, which is the standard format for trajectory evaluation.
    
    Args:
        extrinsics: Camera extrinsic parameters of shape (N, 3, 4) or (N, 4, 4)
                    In OpenCV coordinate system (x-right, y-down, z-forward)
                    Representing camera from world transformation: P_cam = [R | t] * P_world
    
    Returns:
        trajectory: Camera-to-world transformation matrices of shape (N, 4, 4)
                    This represents: P_world = [R^T | -R^T*t] * P_cam
    """
    if extrinsics.shape[-2:] == (3, 4):
        c2w_matrices = closed_form_inverse_se3(extrinsics)
    elif extrinsics.shape[-2:] == (4, 4):
        c2w_matrices = closed_form_inverse_se3(extrinsics)
    else:
        raise ValueError(f"Extrinsics must be shape (N, 3, 4) or (N, 4, 4), got {extrinsics.shape}")
    
    return c2w_matrices


def create_trajectory_from_camera_params(predicted_extrinsics, gt_extrinsics):
    """
    Create trajectory arrays from predicted and ground truth camera extrinsic parameters.
    
    This is the main function to use when you have camera parameters from your model.
    
    Args:
        predicted_extrinsics: Predicted camera extrinsics of shape (N, 3, 4) or (N, 4, 4)
        gt_extrinsics: Ground truth camera extrinsics of shape (N, 3, 4) or (N, 4, 4)
    
    Returns:
        poses_est: Estimated camera-to-world poses of shape (N, 4, 4)
        poses_gt: Ground truth camera-to-world poses of shape (N, 4, 4)
        frame_ids: Frame indices of shape (N,)
    """
    n_frames = len(predicted_extrinsics)
    
    poses_est = camera_extrinsics_to_trajectory(predicted_extrinsics)
    poses_gt = camera_extrinsics_to_trajectory(gt_extrinsics)
    
    # Normalize both trajectories to use the first frame as world origin
    poses_est = normalize_trajectory_to_first_frame(poses_est)
    poses_gt = normalize_trajectory_to_first_frame(poses_gt)
    
    frame_ids = np.arange(n_frames)
    
    return poses_est, poses_gt, frame_ids


def umeyama_alignment(src, dst, estimate_scale=True):
    """
    Compute the optimal rigid transformation (rotation, translation, and optionally scaling)
    that aligns the source point cloud to the target point cloud using the Umeyama algorithm.
    
    Args:
        src: Source point cloud of shape (3, N)
        dst: Target point cloud of shape (3, N)
        estimate_scale: Whether to estimate scaling factor
    
    Returns:
        scale: Scaling factor
        R: Rotation matrix of shape (3, 3)
        t: Translation vector of shape (3,)
    """
    assert src.shape == dst.shape, f"Input shapes don't match: src {src.shape}, dst {dst.shape}"
    assert src.shape[0] == 3, f"Expected point cloud dimension (3,N), got {src.shape}"

    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = dst_centered @ src_centered.T

    try:
        U, D, Vt = svd(cov)
        V = Vt.T

        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1

        R = U @ S @ V.T

        if estimate_scale:
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                scale = 1.0
            else:
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0

        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()

        return scale, R, t

    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, t


def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):
    """
    Evaluate trajectory by computing ATE, ARE, RPE-rot, and RPE-trans metrics.
    
    Args:
        poses_est: Estimated poses of shape (N, 4, 4)
        poses_gt: Ground truth poses of shape (N, 4, 4)
        frame_ids: Frame indices/timestamps of shape (N,)
        align: Whether to align the estimated trajectory with ground truth
    
    Returns:
        metrics: Dictionary containing 'ate', 'are', 'rpe_rot', 'rpe_trans'
        pillow_image: PIL Image of the trajectory plot
        transform: Transformation matrix used for alignment (4x4)
    """
    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(
            scalar_first=True
        ),
        timestamps=frame_ids,
    )

    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(
            scalar_first=True
        ),
        timestamps=frame_ids,
    )

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    ate = ate_result.stats["rmse"]

    transform = np.eye(4)
    if align:
        try:
            aligned_xyz = ate_result.trajectories["traj"].positions_xyz
            original_xyz = traj_est.positions_xyz

            if len(aligned_xyz) >= 3 and len(original_xyz) >= 3:
                min_points = min(len(aligned_xyz), len(original_xyz))
                aligned_xyz = aligned_xyz[:min_points]
                original_xyz = original_xyz[:min_points]

                try:
                    s, R, t = umeyama_alignment(
                        original_xyz.T,
                        aligned_xyz.T,
                        True,
                    )

                    transform = np.eye(4)
                    transform[:3, :3] = s * R
                    transform[:3, 3] = t

                except Exception as e:
                    print(f"umeyama_alignment failed: {e}")
            else:
                print("Insufficient points, cannot reliably compute transformation matrix")
        except Exception as e:
            print(f"Error computing transformation matrix: {e}")

        if np.array_equal(transform, np.eye(4)) and hasattr(ate_result, "trajectories"):
            try:
                orig_pos = traj_est.positions_xyz[0]
                aligned_pos = ate_result.trajectories["traj"].positions_xyz[0]

                translation = aligned_pos - orig_pos
                transform[:3, 3] = translation
                print(f"Fallback to simple translation transformation: {transform}")
            except Exception as e:
                print(f"Error building translation transformation: {e}")
                print("Will use identity matrix")

    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    are = are_result.stats["rmse"]

    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_rot = rpe_rots_result.stats["rmse"]

    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_trans = rpe_transs_result.stats["rmse"]

    plot_mode = plot.PlotMode.xz
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")

    plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    if align:
        traj_est_aligned = ate_result.trajectories["traj"]
        plot.traj_colormap(
            ax,
            traj_est_aligned,
            ate_result.np_arrays["error_array"],
            plot_mode,
            min_map=ate_result.stats["min"],
            max_map=ate_result.stats["max"],
        )
    else:
        plot.traj_colormap(
            ax,
            traj_est,
            ate_result.np_arrays["error_array"],
            plot_mode,
            min_map=ate_result.stats["min"],
            max_map=ate_result.stats["max"],
        )

    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return (
        {"ate": ate, "are": are, "rpe_rot": rpe_rot, "rpe_trans": rpe_trans},
        pillow_image,
        transform,
    )


# =============================================================================
# Usage examples and instructions
# =============================================================================


def example_usage():
    """
    Example：How to use this module to evaluate camera pose estimation results
    
    scene：
    - Your model inputs multi-view images
    - The model outputs camera parameters for each perspective（external reference）
    - You have the corresponding ground truth camera parameters
    - you want to calculateATE, ARE, RPE-rot, RPE-transThese four indicators
    """
    print("=" * 60)
    print("Camera pose evaluation index calculation - Usage example")
    print("=" * 60)
    
    # --------------------------
    # step1：Prepare data
    # --------------------------
    print("\n1. Prepare predicted and ground-truth camera parameters...")
    
    # Assume your model outputsNcamera extrinsic parameters
    # External parameter format：(N, 3, 4) or (N, 4, 4)
    # Represents the transformation from the world coordinate system to the camera coordinate system
    #
    # For example，N=5perspective：
    n_views = 5
    
    # Predicted camera extrinsics（from your model）
    # Generate a valid rotation matrix
    predicted_rotations = generate_valid_rotation_matrices(n_views)
    predicted_translations = np.random.rand(n_views, 3, 1) * 10
    predicted_extrinsics = np.concatenate([predicted_rotations, predicted_translations], axis=2)  # (n_views, 3, 4)
    # Notice：In actual use，This should be your model output
    # predicted_extrinsics = model(images)
    
    # True camera external parameters（from data set）
    # Generate a valid rotation matrix
    gt_rotations = generate_valid_rotation_matrices(n_views)
    gt_translations = np.random.rand(n_views, 3, 1) * 10
    gt_extrinsics = np.concatenate([gt_rotations, gt_translations], axis=2)
    # Notice：In actual use，This should be the true value provided by the data set
    # gt_extrinsics = dataset.get_ground_truth_extrinsics()
    
    print(f"   - Number of views: {n_views}")
    print(f"   - Predict external parameter shape: {predicted_extrinsics.shape}")
    print(f"   - truth value external parameter shape: {gt_extrinsics.shape}")
    
    # --------------------------
    # step2：Convert to track format
    # --------------------------
    print("\n2. Convert camera parameters to trajectory format...")
    
    # usecreate_trajectory_from_camera_paramsfunction
    # Will be external reference（world-to-camera）Convert to camera pose（camera-to-world）
    # And set the first pose as the origin of the world coordinate system
    poses_est, poses_gt, frame_ids = create_trajectory_from_camera_params(
        predicted_extrinsics, 
        gt_extrinsics
    )
    
    print(f"   - Estimated pose shape: {poses_est.shape}")
    print(f"   - Ground truth pose shape: {poses_gt.shape}")
    print(f"   - frameIDshape: {frame_ids.shape}")
    
    # --------------------------
    # step3：Calculate evaluation metrics
    # --------------------------
    print("\n3. Calculate evaluation metrics...")
    
    # calleval_trajectoryfunction calculation indicator
    # align=Truemeans aligning the estimated trajectory with the ground truth trajectory（recommend）
    metrics, trajectory_plot, transform = eval_trajectory(
        poses_est, 
        poses_gt, 
        frame_ids, 
        align=True
    )
    
    # --------------------------
    # step4：View results
    # --------------------------
    print("\n4. Assessment results:")
    print("-" * 40)
    print(f"   ATE (absolute trajectory error): {metrics['ate']:.4f}")
    print(f"   ARE (absolute rotation error): {metrics['are']:.4f}°")
    print(f"   RPE-rot (Relative rotation error): {metrics['rpe_rot']:.4f}°")
    print(f"   RPE-trans (relative translation error): {metrics['rpe_trans']:.4f}")
    print("-" * 40)
    
    # --------------------------
    # step5：Save or visualize
    # --------------------------
    print("\n5. Save results...")
    
    # Save track image
    trajectory_plot.save("trajectory_comparison.png")
    print("   - Track comparison chart saved: trajectory_comparison.png")
    
    # Save indicator to file
    import json
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("   - Evaluation metrics saved: evaluation_metrics.json")
    
    print("\n" + "=" * 60)
    print("Sample execution completed！")
    print("=" * 60)


# =============================================================================
# Explanation of key concepts
# =============================================================================

"""

Explanation of key concepts
============

1. Camera external parameters（Extrinsics）
   - Format：3x4 or 4x4 matrix
   - meaning：Transformation from world coordinate system to camera coordinate system
   - mathematical representation：P_cam = [R | t] * P_world
   - inRyes3x3rotation matrix，tyes3x1translation vector

2. camera pose（Camera Pose）
   - Format：4x4 matrix
   - meaning：Transformation from camera coordinate system to world coordinate system（The camera's position and orientation in the world）
   - mathematical representation：P_world = [R^T | -R^T*t] * P_cam
   - This is the standard format for trajectory assessment

3. Why conversion is needed？
   - The model usually outputs external parameters（world-to-camera）
   - Trajectory evaluation requires pose（camera-to-world）
   - of this modulecamera_extrinsics_to_trajectoryThe function completes this conversion

4. Four evaluation indicators
   - ATE (Absolute Trajectory Error)：absolute trajectory error
     * measure：Global consistency of the entire trajectory
     * calculate：The estimated position and the true position at each momentRMSE
     * unit：rice（or data set units）
     
   - ARE (Absolute Rotation Error)：absolute rotation error
     * measure：Accuracy of rotation estimation
     * calculate：The angle difference between the estimated rotation and the true rotation at each momentRMSE
     * unit：Spend（°）
     
   - RPE-rot (Relative Pose Error - rotation)：Relative rotation error
     * measure：Stability of rotation estimation between adjacent frames
     * calculate：relative rotation errorRMSE
     * unit：Spend（°）
     
   - RPE-trans (Relative Pose Error - translation)：relative translation error
     * measure：Stability of translation estimation between adjacent frames
     * calculate：relative translation errorRMSE
     * unit：rice（or data set units）

5. track alignment（Align）
   - effect：Spatially align estimated trajectories with ground truth trajectories
   - method：useUmeyamaAlgorithm to calculate optimal rigid transformation（rotate、Pan、Zoom）
   - recommend：set at evaluation timealign=True
   - reason：Eliminate coordinate system differences，More accurate assessment of trajectory shape

"""


def load_gt_poses_from_files(pose_dir):
    """
    Load ground truth pose from file（Format compatible with original project）
    
    Ground truth pose file format：
    - One from each perspective .txt document
    - The file name is a number（like 0.txt, 1.txt, 2.txt...）
    - The file content is16numbers，express4x4transformation matrix
    - Matrix representation：camera-to-world transform（The camera's position in the world coordinate system）
    
    Args:
        pose_dir: Directory path to store pose files
        
    Returns:
        c2ws: camera-to-world transformation matrix，shape (N, 4, 4)
        first_gt_pose: Original pose from first perspective（Used for subsequent point cloud transformation）
        available_frame_ids: available framesIDlist
    """
    from pathlib import Path
    
    pose_dir = Path(pose_dir)
    pose_files = sorted(
        pose_dir.glob("*.txt"), key=lambda x: int(x.stem)
    )
    
    if len(pose_files) == 0:
        print(f"Warning: No pose files (.txt) found in directory {pose_dir}")
        return None, None, None
    
    c2ws = []
    available_frame_ids = []
    
    for pose_file in pose_files:
        try:
            with open(pose_file, "r") as f:
                nums = [float(x) for x in f.read().strip().split()]
                pose = np.array(nums).reshape(4, 4)
                if not (np.isinf(pose).any() or np.isnan(pose).any()):
                    c2ws.append(pose)
                    available_frame_ids.append(int(pose_file.stem))
        except Exception as e:
            print(f"Error reading pose file {pose_file}: {e}")
            continue
    
    if len(c2ws) == 0:
        print(f"Warning: No valid pose files found in directory {pose_dir}")
        return None, None, None
    
    c2ws = np.stack(c2ws)
    available_frame_ids = np.array(available_frame_ids)
    
    first_gt_pose = c2ws[0].copy()
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    
    return c2ws, first_gt_pose, available_frame_ids


def evaluate_pose_estimation(predicted_extrinsics, gt_poses_c2w, frame_ids=None):
    """
    Complete posture assessment process（Recommended）
    
    This is a high-level encapsulation function，Simplify the assessment process。
    
    Args:
        predicted_extrinsics: Predicted camera extrinsics，shape (N, 3, 4) or (N, 4, 4)
                             express：world-to-camera transform
        gt_poses_c2w: ground truth pose，shape (N, 4, 4)
                     express：camera-to-world transform
        frame_ids: frameIDlist，if forNoneis automatically generated
        
    Returns:
        metrics: Evaluation Metrics Dictionary
        trajectory_plot: Trajectory comparison image
        transform: Alignment transformation matrix
    """
    if frame_ids is None:
        frame_ids = np.arange(len(predicted_extrinsics))
    
    poses_est = camera_extrinsics_to_trajectory(predicted_extrinsics)
    
    gt_poses_w2c = np.linalg.inv(gt_poses_c2w)
    
    metrics, trajectory_plot, transform = eval_trajectory(
        poses_est, gt_poses_w2c, frame_ids, align=True
    )
    
    return metrics, trajectory_plot, transform


def example_with_ground_truth_files():
    """
    Example：Use truth files for evaluation（Consistent with the original project workflow）
    
    scene：
    - Your model outputs multi-view camera extrinsic parameters
    - Ground truth poses are stored in files
    - Need to compare predicted results with true values
    """
    print("=" * 60)
    print("Camera pose assessment - Example of using a truth file")
    print("=" * 60)
    
    # --------------------------
    # step1：Prepare prediction results
    # --------------------------
    print("\n1. Prepare predicted camera extrinsic parameters...")
    
    n_views = 5
    predicted_rotations = generate_valid_rotation_matrices(n_views)
    predicted_translations = np.random.rand(n_views, 3, 1) * 10
    predicted_extrinsics = np.concatenate([predicted_rotations, predicted_translations], axis=2)
    print(f"   - Predict external parameter shape: {predicted_extrinsics.shape}")
    
    # --------------------------
    # step2：Load ground truth pose
    # --------------------------
    print("\n2. Load ground truth pose file...")
    
    # Notice：In actual use，This should be your ground truth pose file directory
    # Each file is a4x4matrix，expresscamera-to-worldtransform
    # pose_dir = "/path/to/your/gt_poses"
    # c2ws, first_gt_pose, frame_ids = load_gt_poses_from_files(pose_dir)
    
    # Here we use simulated data to demonstrate
    c2ws = np.random.rand(n_views, 4, 4)
    for i in range(n_views):
        c2ws[i, :3, :3] = Rotation.random().as_matrix()
        c2ws[i, 3, 3] = 1.0
    frame_ids = np.arange(n_views)
    
    print(f"   - Number of ground truth poses: {len(c2ws)}")
    print(f"   - frameID: {frame_ids}")
    
    # --------------------------
    # step3：Convert prediction result format
    # --------------------------
    print("\n3. Convert prediction extrinsic parameters to pose format...")
    
    poses_est = camera_extrinsics_to_trajectory(predicted_extrinsics)
    print(f"   - Estimated pose shape: {poses_est.shape}")
    
    # --------------------------
    # step4：Processing ground truth poses
    # --------------------------
    print("\n4. Processing ground truth poses...")
    
    # The truth value is alreadyc2wFormat，No conversion required
    gt_poses_c2w = c2ws
    print(f"   - truth valuec2wshape: {gt_poses_c2w.shape}")
    
    # normalized trajectory，Use the first frame as the origin of the world coordinate system
    poses_est = normalize_trajectory_to_first_frame(poses_est)
    gt_poses_c2w = normalize_trajectory_to_first_frame(gt_poses_c2w)
    
    # --------------------------
    # step5：Calculate evaluation metrics
    # --------------------------
    print("\n5. Calculate evaluation metrics...")
    
    metrics, trajectory_plot, transform = eval_trajectory(
        poses_est, gt_poses_c2w, frame_ids, align=True
    )
    
    print("\n6. Assessment results:")
    print("-" * 40)
    print(f"   ATE: {metrics['ate']:.4f}")
    print(f"   ARE: {metrics['are']:.4f}°")
    print(f"   RPE-rot: {metrics['rpe_rot']:.4f}°")
    print(f"   RPE-trans: {metrics['rpe_trans']:.4f}")
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Sample execution completed！")
    print("=" * 60)


if __name__ == "__main__":
    # Run the example
    example_usage()
    print("\n" * 3)
    example_with_ground_truth_files()
