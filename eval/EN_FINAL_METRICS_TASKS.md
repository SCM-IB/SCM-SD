# Final Evaluation Metrics

This file documents the final evaluation metrics kept in `scm-IB/eval_vggt`.

## 1. Camera Pose

Source: Pi3

Metrics:

- `ate`
  - Absolute Trajectory Error after Sim(3) alignment. Measures the average distance between estimated and ground truth camera positions along the trajectory.
- `rpe_trans`
  - Relative Pose Error for translation after Sim(3) alignment. Measures the average translation error between consecutive frames.
- `rpe_rot`
  - Relative Pose Error for rotation after Sim(3) alignment. Measures the average rotation error between consecutive frames.
- `rra_3`, `rra_5`, `rra_15`, `rra_30`
  - Relative Rotation Accuracy under threshold 3/5/15/30 degrees. Percentage of frame pairs where rotation error is below the specified threshold.
- `rta_3`, `rta_5`, `rta_15`, `rta_30`
  - Relative Translation Accuracy under threshold 3/5/15/30 degrees. Percentage of frame pairs where translation error is below the specified threshold.
- `auc_3`, `auc_5`, `auc_15`, `auc_30`
  - Area under the curve of the minimum relative rotation / translation accuracy up to threshold 3/5/15/30 degrees. Comprehensive metric that evaluates performance across all thresholds up to the specified value.

Task:

- Multi-view camera pose estimation / trajectory evaluation.

## 2. Multi-View Depth

Source: Pi3

Metrics:

- `abs_rel_scale`
  - Absolute Relative Error with sequence-level scale-only alignment. Measures the average relative difference between predicted and ground truth depths, with scale alignment applied to the entire sequence.
- `delta_1.25_scale`
  - Accuracy under `delta < 1.25` with sequence-level scale-only alignment. Percentage of pixels where the predicted depth is within a factor of 1.25 of the ground truth.
- `abs_rel_scale_shift`
  - Absolute Relative Error with sequence-level scale-and-shift alignment. Measures the average relative difference between predicted and ground truth depths, with both scale and shift alignment applied to the entire sequence.
- `delta_1.25_scale_shift`
  - Accuracy under `delta < 1.25` with sequence-level scale-and-shift alignment. Percentage of pixels where the predicted depth is within a factor of 1.25 of the ground truth, after scale and shift alignment.

Task:

- Multi-view depth estimation.

## 3. Monocular Depth

Source: Pi3

Metrics:

- `abs_rel_mono`
  - Absolute Relative Error with per-frame median-scale alignment. Measures the average relative difference between predicted and ground truth depths, with median scale alignment applied independently to each frame.
- `delta_1.25_mono`
  - Accuracy under `delta < 1.25` with per-frame median-scale alignment. Percentage of pixels where the predicted depth is within a factor of 1.25 of the ground truth, after per-frame median scale alignment.

Task:

- Monocular depth evaluation computed alongside the multi-view input setting.

## 4. Point Cloud Reconstruction

Source: Pi3

Metrics:

- `acc`
  - Mean distance from predicted points to ground-truth surface. Measures the average accuracy of predicted points compared to the ground truth surface.
- `acc_med`
  - Median distance from predicted points to ground-truth surface. Median value of the distances from predicted points to the ground truth surface, less sensitive to outliers.
- `comp`
  - Mean distance from ground-truth points to predicted surface. Measures how well the predicted surface captures the ground truth points (completeness).
- `comp_med`
  - Median distance from ground-truth points to predicted surface. Median value of the distances from ground truth points to the predicted surface.
- `nc1`
  - Mean normal consistency from predicted to ground truth. Measures the average consistency of surface normals between predicted and ground truth points.
- `nc1_med`
  - Median normal consistency from predicted to ground truth. Median value of the normal consistency from predicted to ground truth.
- `nc2`
  - Mean normal consistency from ground truth to predicted. Measures the average consistency of surface normals between ground truth and predicted points.
- `nc2_med`
  - Median normal consistency from ground truth to predicted. Median value of the normal consistency from ground truth to predicted.
- `nc`
  - Mean bidirectional normal consistency. Average of `nc1` and `nc2`, providing a balanced measure of normal consistency.
- `nc_med`
  - Median bidirectional normal consistency. Median value of the bidirectional normal consistency.

Task:

- Multi-view point cloud reconstruction quality evaluation.

## 5. Point Cloud F-score

Source: Depth-Anything-3

Metrics:

- `acc_da`
  - DA-3 accuracy term used in F-score evaluation. Average distance from predicted points to the nearest ground truth points within the threshold.
- `comp_da`
  - DA-3 completeness term used in F-score evaluation. Average distance from ground truth points to the nearest predicted points within the threshold.
- `overall`
  - DA-3 overall reconstruction error, `(acc_da + comp_da) / 2`. Average of accuracy and completeness terms.
- `precision`
  - Fraction of predicted points within the DA-3 distance threshold. Percentage of predicted points that are within the threshold distance of a ground truth point.
- `recall`
  - Fraction of ground-truth points within the DA-3 distance threshold. Percentage of ground truth points that are within the threshold distance of a predicted point.
- `fscore_da`
  - Harmonic mean of `precision` and `recall`. Balanced measure of precision and recall, highlighting cases where both are high.

Task:

- Threshold-based point cloud reconstruction evaluation.

## Removed Legacy Project-Only Fields

These fields are no longer part of the final evaluation outputs:

- `are`
- `chamfer_distance`
- `scale_factor`
- `inference_time_ms`
- `abs_rel`
- `delta_1.25`
- `depth_valid_pixels`
- `depth_valid_pixels_scale`
- `depth_valid_pixels_scale_shift`
- `depth_valid_pixels_mono`

These fields were removed to keep the final output focused on Pi3 and DA-3 aligned metrics only.
