# Structural Distillation Training Guide

## Overview

This guide explains how to train the VGGT model using structural distillation for RGB-to-thermal (infrared) domain adaptation. The student network learns structural information from the teacher network through feature decomposition and conditional compression.

## Architecture

### Teacher-Student Network

- **Teacher Network**: Pre-trained VGGT model taking RGB images as input (visible branch)
- **Student Network**: VGGT model taking thermal/infrared images as input (infrared branch)
- **Structural Distillation**: Knowledge transfer through structural feature decomposition and conditional compression

### Loss Functions

The total loss consists of four parts:

1. **Ground Truth Loss** (on student predictions):
   - Camera Loss: L1 loss on camera pose (weight: 5.0)
   - Depth Loss: Gradient-based depth loss (weight: 1.0)

2. **Structural Distillation Loss** (three components with configurable weights):
   - **Structure Preservation Loss**: MSE loss between teacher and student structural features (weight: a)
   - **Conditional Compression Loss**: Uniformization conditional contrastive loss (weight: b)
   - **Uncertainty Distillation Loss**: L1 loss between log uncertainties (weight: c, only when uncertainty is enabled)

**Total Loss Formula:**
```
L_total = L_camera * 5.0 + L_depth * 1.0 + (a*L_structure_preservation + b*L_conditional_compression + c*L_uncertainty_distillation)
```

3. **Uncertainty-Aware Adaptive Information Truncation**:
   - **Uncertainty-Aware Structure Preservation Loss**: Weighted MSE loss using teacher confidence map
   - **Uncertainty-Aware Conditional Compression Loss**: Weighted contrastive loss using teacher confidence map
   - **Uncertainty Distillation Loss**: L1 loss between log uncertainties

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── Dimsum/
│   ├── rgb/              # RGB images (.jpg)
│   ├── thermal/         # Thermal/infrared images (.jpg)
│   ├── depth_aligned/   # Aligned depth maps (.png)
│   ├── colmap/
│   │   └── sparse/0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── depth_params.json
├── Ebike/
│   ├── rgb/
│   ├── thermal/
│   ├── depth_aligned/
│   ├── colmap/
│   │   └── sparse/0/
│   └── depth_params.json
└── ...
```

## Configuration

The main configuration file is `training/config/colmap_dataset_scmib.yaml`.

### Required Parameters

You need to modify the following paths in `colmap_dataset_scmib.yaml`:

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - COLMAP_DIR: e:\Paper2\1\vggt-our\dataset  # Your dataset path
  val:
    dataset:
      dataset_configs:
        - COLMAP_DIR: e:\Paper2\1\vggt-our\dataset  # Your dataset path

model:
  visible_model_path: model.pt  # Pre-trained VGGT weights

checkpoint:
  resume_checkpoint_path: model.pt  # Resume checkpoint path
```

## Parameter Freezing Strategy

### Automatic Freezing by Config

The model automatically freezes parameters based on the configuration:

| Component | Freezing Condition |
|-----------|-------------------|
| **Teacher (visible branch)** | Always fully frozen |
| **Student DINO encoder (patch_embed)** | Always frozen |
| **Student aggregator (excluding patch_embed)** | Trainable for structural learning |
| **Student prediction heads (camera/depth)** | Trainable for task learning |

### Manual Freezing in Config

Freezing is specified in `colmap_dataset_scmib.yaml`:

```yaml
optim:
  frozen_module_names:
      - "*visible*"  # Always freeze teacher branch
      - "*infrared_aggregator.patch_embed*"  # Freeze student's DINO encoder
```

## Student Network Initialization Modes

The student network (infrared branch) can be initialized in three different modes:

### Mode 1: Initialize from Teacher Network (Recommended)

This is the default and recommended mode for knowledge distillation. The student network starts with the same weights as the teacher network.

```yaml
model:
  visible_model_path: ckpt/model.pt  # Teacher network pre-trained weights
  infrared_model_path: null  # Do not load separate student weights
  init_infrared_from_visible: true  # Initialize from teacher network
```

**Advantages:**
- Better starting point for knowledge distillation
- Faster convergence
- Common practice in KD training

### Mode 2: Train from Scratch (Random Initialization)

The student network starts with random weights, allowing it to learn from scratch.

```yaml
model:
  visible_model_path: ckpt/model.pt  # Teacher network pre-trained weights
  infrared_model_path: null  # Do not load separate student weights
  init_infrared_from_visible: false  # Initialize from scratch
```

**Advantages:**
- No bias from teacher network
- Can explore different feature representations
- Useful for ablation studies

### Mode 3: Load Pre-trained Student Weights

Load student network weights from a specific checkpoint file.

```yaml
model:
  visible_model_path: ckpt/model.pt  # Teacher network pre-trained weights
  infrared_model_path: ckpt/student_pretrained.pt  # Student network pre-trained weights
  init_infrared_from_visible: false  # Not needed when loading weights
```

**Advantages:**
- Resume training from a previous checkpoint
- Use pre-trained student weights from other experiments
- Fine-tune existing student models

## Training

### Single-GPU Training

Run training with the structural distillation configuration:

```bash
cd training
python launch.py --config colmap_dataset_scmib
```

### Multi-GPU Training

For multi-GPU training, use `torchrun`:

**Linux:**
```bash
CUDA_VISIBLE_DEVICES=1,5,6,7 torchrun --nproc_per_node=4 training/launch.py --config colmap_dataset_scmib
```

**Windows (PowerShell):**
```powershell
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nproc_per_node=4 training/launch.py --config colmap_dataset_scmib
```

**Windows (CMD):**
```cmd
set CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 training/launch.py --config colmap_dataset_scmib
```

Parameters:
- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `--nproc_per_node`: Number of GPUs (should match the number of GPUs in `CUDA_VISIBLE_DEVICES`)

## Key Hyperparameters

### Structural Distillation Configuration

Modify these in `colmap_dataset_scmib.yaml`:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `loss.structural_distillation.weight` | Overall structural distillation loss weight | 1.0 | 0.1 - 10.0 |
| `loss.structural_distillation.delta` | Diagonal suppression parameter | 0.1 | 0.0 - 1.0 |
| `loss.structural_distillation.tau_g` | Temperature for structure-aware adjacency | 0.1 | 0.01 - 1.0 |
| `loss.structural_distillation.tau` | Temperature for critic | 0.1 | 0.01 - 1.0 |
| `loss.structural_distillation.topk` | Number of top elements for structure summary | 32 | 8 - 64 |
| `loss.structural_distillation.candidate_strategy` | Candidate set construction strategy | "A+B" | "A", "B", "A+B" |
| `loss.structural_distillation.max_candidates` | Maximum number of candidates | 16 | 8 - 64 |
| `loss.structural_distillation.embed_dim` | Embedding dimension | 2048 | Should match aggregator output |
| `loss.structural_distillation.critic_dim` | Critic network output dimension | 256 | 64 - 512 |
| `loss.structural_distillation.use_uncertainty` | Enable uncertainty-aware loss | true | true/false |
| `loss.structural_distillation.uncertainty_weight` | Weight for uncertainty distillation loss | 1.0 | 0.1 - 10.0 |

#### Sub-Loss Weight Configuration

Each sub-loss in structural distillation can be configured independently:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `loss.structural_distillation.structure_preservation_weight` | Weight for structure preservation loss (a) | 1.0 | 0.0 - 10.0 |
| `loss.structural_distillation.conditional_compression_weight` | Weight for conditional compression loss (b) | 1.0 | 0.0 - 10.0 |
| `loss.structural_distillation.uncertainty_distillation_weight` | Weight for uncertainty distillation loss (c) | 1.0 | 0.0 - 10.0 |

**Example configurations:**

1. **Emphasize structure preservation:**
```yaml
loss:
  structural_distillation:
    structure_preservation_weight: 2.0
    conditional_compression_weight: 0.5
    uncertainty_distillation_weight: 0.5
```

2. **Emphasize conditional compression:**
```yaml
loss:
  structural_distillation:
    structure_preservation_weight: 0.5
    conditional_compression_weight: 2.0
    uncertainty_distillation_weight: 0.5
```

3. **Disable uncertainty distillation:**
```yaml
loss:
  structural_distillation:
    structure_preservation_weight: 1.0
    conditional_compression_weight: 1.0
    uncertainty_distillation_weight: 0.0
```

4. **Equal weighting (default):**
```yaml
loss:
  structural_distillation:
    structure_preservation_weight: 1.0
    conditional_compression_weight: 1.0
    uncertainty_distillation_weight: 1.0
```

### Ground Truth Loss Weights

| Parameter | Description | Default |
|-----------|-------------|---------|
| `loss.camera.weight` | Camera pose GT loss weight | 5.0 |
| `loss.depth.weight` | Depth GT loss weight | 1.0 |

### Optimization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `optim.optimizer.lr` | Learning rate | 5e-5 |
| `optim.optimizer.weight_decay` | Weight decay | 0.05 |
| `max_img_per_gpu` | Batch size per GPU | 12 |
| `accum_steps` | Gradient accumulation steps | 2 |

### Learning Rate Scheduler

The configuration uses a composite scheduler with linear warmup and cosine decay:

- **Warmup phase (5%)**: Linear increase from 1e-8 to 5e-5
- **Decay phase (95%)**: Cosine decay from 5e-5 to 1e-8

### Gradient Clipping

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gradient_clip.configs[].max_norm` | Max gradient norm | 1.0 |

Example - Reduce gradient clipping:
```yaml
gradient_clip:
  configs:
    - module_name: ["infrared_aggregator"]
      max_norm: 0.5
    - module_name: ["infrared_camera"]
      max_norm: 0.5
    - module_name: ["infrared_depth"]
      max_norm: 0.5
```

### Mixed Precision Training

Mixed precision training is enabled by default:

```yaml
optim:
  amp:
    enabled: True
    amp_dtype: bfloat16
```

### DDP Settings

Distributed Data Parallel (DDP) settings:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `distributed.backend` | DDP backend | nccl |
| `distributed.find_unused_parameters` | Find unused parameters | False |
| `distributed.gradient_as_bucket_view` | Use gradient as bucket view | True |
| `distributed.bucket_cap_mb` | Bucket capacity (MB) | 25 |

### CUDA Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cuda.cudnn_deterministic` | Deterministic CuDNN | False |
| `cuda.cudnn_benchmark` | CuDNN benchmark | False |
| `cuda.allow_tf32` | Allow TF32 | True |

## Training Metrics

All losses are logged to TensorBoard for monitoring:

| Metric | Description |
|--------|-------------|
| `loss_objective` | Total weighted loss |
| `infrared_loss_camera` | Camera pose loss (GT) |
| `infrared_loss_T` | Translation loss |
| `infrared_loss_R` | Rotation loss |
| `infrared_loss_FL` | Focal length loss |
| `infrared_loss_conf_depth` | Depth confidence loss |
| `infrared_loss_reg_depth` | Depth regularization loss |
| `infrared_loss_grad_depth` | Depth gradient loss |
| `structural_structure_preservation` | Structure preservation loss (weighted by a) |
| `structural_conditional_compression` | Conditional compression loss (weighted by b) |
| `structural_uncertainty_distillation` | Uncertainty distillation loss (weighted by c) |

**Note:** All structural distillation sub-losses are already weighted by their respective weights (a, b, c) as configured in the YAML file.

## Monitoring

### TensorBoard

Training logs are saved to `logs/<exp_name>/tensorboard/`.

View training progress:

```bash
tensorboard --logdir logs/colmap_dataset_structural_distillation/tensorboard
```

### Checkpoints

Model checkpoints are saved to `logs/<exp_name>/ckpts/`.

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size:
```yaml
max_img_per_gpu: 4
```

2. Enable gradient accumulation:
```yaml
accum_steps: 4
```

### Training Instability

1. Reduce learning rate:
```yaml
optim:
  optimizer:
    lr: 1e-5
```

2. Reduce structural distillation weight:
```yaml
loss:
  structural_distillation:
    weight: 0.5
```

3. Adjust temperature parameters:
```yaml
loss:
  structural_distillation:
    tau_g: 0.2
    tau: 0.2
```

### Teacher Weights Not Loading

Check that:
1. The path in `visible_model_path` is correct
2. The pre-trained model file exists
3. The model format is correct (.pt)

## Notes

- The dataset loader expects both `rgb/` and `thermal/` directories
- Image extensions supported: .jpg, .JPG, .png, .PNG
- Depth maps can be in either .png or .npy format
- Teacher branch is automatically frozen when using VGGTDualBranch
- DINO encoder (patch_embed) is always frozen for efficient training
- Structural distillation focuses on learning structural information from the teacher
- The conditional compression loss helps the student learn a more uniform representation
- Uncertainty-aware adaptive information truncation uses teacher's depth uncertainty to weight the distillation losses
- Confidence map is computed as C_t = clip(1 - 1/U_t, 0, 1) from teacher's uncertainty map
- Uncertainty distillation uses L1 loss between log uncertainties to ensure the student learns to estimate uncertainty similarly to the teacher

## Quick Reference

### Common Configurations

**For limited GPU memory (OOM):**
```yaml
max_img_per_gpu: 4
accum_steps: 4
```

**For faster convergence:**
```yaml
optim:
  optimizer:
    lr: 1e-4
```

**For stronger structural guidance:**
```yaml
loss:
  structural_distillation:
    weight: 2.0
    topk: 64
```

## Additional Resources

- Main configuration: `training/config/colmap_dataset_scmib.yaml`
- Structural distillation loss implementation: `training/structural_distillation_loss.py`
- Model implementation: `vggt/models/vggt.py`
- Parameter freezing logic: `vggt/models/vggt.py:set_freeze_parameters()`
