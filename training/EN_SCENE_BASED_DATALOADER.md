# scene level DataLoader Refactor documentation

## Overview

This document illustrates the refactored scenario level DataLoader Architecture。New schema samples by scene folder index，each batch Contains multiple scenes，Sample a specified number of images per scene。

## Architecture design

### data flow

```
1. SceneBasedBatchSampler.__iter__()
   ├─ random sampling image_num（For example：8images）
   ├─ random sampling aspect_ratio（For example：0.5）
   ├─ calculate batch_size = 24 / 8 = 3
   └─ sampling3scene index
      output：[(scene_idx_0, 8, 0.5), (scene_idx_1, 8, 0.5), (scene_idx_2, 8, 0.5)]

2. DataLoader.collate_fn()
   └─ Will3tuples combined intobatch
      call3Second-rate dataset.__getitem__()

3. ComposedDataset.__getitem__((scene_idx, 8, 0.5))
   └─ Call the underlying data set to obtain data

4. ColmapDataset.get_data(scene_idx, 8, 0.5)
   ├─ according to scene_idx Get scene name（For example："Dimsum"）
   ├─ Load the camera parameters of the scene（53camera）
   ├─ random sampling 8 imagesID（For example：[12, 45, 78, 23, 56, 89, 34, 67]）
   ├─ load this8image ofRGB、thermal infrared、depth、Camera parameters
   └─ Return contains8dictionary of images

5. finalbatch
   ├─ No.1Group：scene0（Dimsum）of8images
   ├─ No.2Group：scene1（Ebike）of8images
   └─ No.3Group：scene2（Truck）of8images
```

### key components

#### 1. SceneBasedDynamicDataset

Main data set class，Responsible：
- Initialize the underlying data set（ComposedDataset）
- Create a scene-level sampler（SceneBasedDistributedSampler）
- Create a batch sampler（SceneBasedBatchSampler）
- supply DataLoader Create interface

**Configuration parameters：**
```yaml
data:
  train:
    _target_: data.scene_based_dataloader.SceneBasedDynamicDataset
    num_workers: 8
    max_img_per_gpu: 24  # each GPU Most processed 24 images
    common_config:
      img_size: 518
      patch_size: 14
      img_nums: [2, 24]  # Sampling per scene 2-24 images
      augs:
        aspects: [0.33, 1.0]  # aspect ratio range
```

#### 2. SceneBasedBatchSampler

batch sampler，Responsible：
- Number of randomly sampled images（image_num）
- Randomly sampled aspect ratio（aspect_ratio）
- calculate batch_size = max_img_per_gpu / image_num
- sampling batch_size scene index

**Working principle：**
```python
# each batch sampling process
random_image_num = random.choice([2, 3, ..., 24])  # For example：8
random_aspect_ratio = random.uniform(0.33, 1.0)  # For example：0.5
batch_size = max_img_per_gpu / random_image_num  # For example：24 / 8 = 3

# sampling 3 scenes
batch = [
    (scene_idx_0, 8, 0.5),
    (scene_idx_1, 8, 0.5),
    (scene_idx_2, 8, 0.5),
]
```

#### 3. SceneBasedDistributedSampler

Distributed sampler，Responsible：
- Support many GPU Distributed training
- Evenly distribute scenes to different GPU
- Provides scene index iterator

**Distributed mode：**
- GPU 0: scene index [0, 4, 8, ...]
- GPU 1: scene index [1, 5, 9, ...]
- GPU 2: scene index [2, 6, 10, ...]
- GPU 3: scene index [3, 7, 11, ...]

#### 4. ColmapDataset·

COLMAP Dataset class，Responsible：
- according to scene_idx Get scene folder path
- Load the camera intrinsic and extrinsic parameters of the scene
- Randomly samples a specified number of images
- load RGB image、thermal infrared image、Depth map
- Process images、depth、Camera parameters

**data structure：**
```
dataset/
├── Dimsum/          # scene 0
│   ├── rgb/          # RGB image
│   ├── thermal/      # thermal infrared image
│   ├── colmap/
│   │   └── sparse/0/ # Camera parameters
│   └── depth_aligned/ # Depth map
├── Ebike/           # scene 1
└── Truck/           # scene 2
```

## Configuration file

### training configuration

```yaml
# colmap_dataset.yaml
max_img_per_gpu: 24  # each GPU Most processed 24 images

data:
  train:
    _target_: data.scene_based_dataloader.SceneBasedDynamicDataset
    num_workers: 8
    max_img_per_gpu: ${max_img_per_gpu}
    common_config:
      img_size: 518
      patch_size: 14
      img_nums: [2, 24]  # Sampling per scene 2-24 images
      augs:
        aspects: [0.33, 1.0]  # aspect ratio range
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.colmapdata.ColmapDataset
          split: train
          COLMAP_DIR: e:\Paper2\1\vggt-our\dataset
          min_num_images: 12  # The scene requires at least 12 images
          len_train: 100000
          expand_ratio: 1
          rgb_dir: rgb
          thermal_dir: thermal
          camera_dir: colmap/sparse/0
          depth_gt_dir: depth_aligned
```

### Batch Composition example

hypothesis `max_img_per_gpu = 24`，Randomly sampled `image_num = 8`：

```
Batch size = 24 / 8 = 3

Batch Include 3 scenes：
- scene 0 (Dimsum): 8 images
- scene 1 (Ebike): 8 images  
- scene 2 (Truck): 8 images

final batch shape:
- images: [3, 8, 3, H, W]  # (batch_size, num_images, channels, height, width)
- depths: [3, 8, H, W]
- extrinsics: [3, 8, 4, 4]
- intrinsics: [3, 8, 3, 3]
```

## How to use

### 1. Basic use

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Load configuration
cfg = compose(config_name="colmap_dataset")

# Create a dataset
train_dataset = instantiate(cfg.data.train, _recursive_=False)
train_dataset.seed = cfg.seed_value

# create DataLoader
train_loader = train_dataset.get_loader(epoch=0)

# Iterate batch
for batch in train_loader:
    images = batch['images']  # [batch_size, num_images, 3, H, W]
    depths = batch['depths']  # [batch_size, num_images, H, W]
    extrinsics = batch['extrinsics']  # [batch_size, num_images, 4, 4]
    intrinsics = batch['intrinsics']  # [batch_size, num_images, 3, 3]
    
    # forward propagation
    outputs = model(images, depths, extrinsics, intrinsics)
    
    # Calculate losses
    loss = criterion(outputs, batch)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
```

### 2. test DataLoader

Run test script to verify DataLoader Is it working properly>：

```bash
cd training
python test_scene_based_dataloader.py
```

## Differences from the original architecture

| characteristic | Original architecture (DynamicTorchDataset) | new architecture (SceneBasedDynamicDataset) |
|------|------------------------------|----------------------------------|
| **sampling unit** | Single image index | Scene folder index |
| **Batch composition** | Multiple images of the same scene | Multiple samples for different scenarios |
| **Number of images** | each batch fixed | each batch Dynamic changes |
| **Data diversity** | lower（same scene） | higher（different scenes） |
| **GPU Utilization** | May be uneven | more uniform |

## Advantages

1. **Better data diversity**：each batch Contains different scenarios，Increase data diversity
2. **more uniform GPU Utilization**：Dynamic adjustment batch size to accommodate different image quantities
3. **More in line with multi-perspective learning**：Independently sample multiple images per scene
4. **Better generalization ability**：Model learns across different scenarios

## Things to note

1. **Number of scenes**：Make sure the dataset has enough scenarios（suggestion > 10 indivual）
2. **Image quantity range**：`img_nums` The range should not be too large，suggestion [2, 24]
3. **GPU Memory**：according to GPU memory adjustment `max_img_per_gpu`
4. **Distributed training**：Ensure the number of scenes >= GPU quantity

## troubleshooting

### question 1: CUDA OOM

**reason**：`max_img_per_gpu` Setting too large

**solution**：reduce `max_img_per_gpu` value

```yaml
max_img_per_gpu: 16  # from 24 reduced to 16
```

### question 2: Insufficient scene images

**reason**：Number of scene images < `img_nums` minimum value

**solution**：Adjustment `min_num_images` or `img_nums`

```yaml
min_num_images: 8  # from 12 reduced to 8
img_nums: [2, 12]  # from [2, 24] reduced to [2, 12]
```

### question 3: Distributed training data duplication

**reason**：Number of scenes < GPU quantity

**solution**：Increase the number of scenes or reduce GPU quantity

## Performance optimization suggestions

1. **Adjustment num_workers**：according to CPU Core number adjustment
   ```yaml
   num_workers: 16  # for 8 nuclear CPU
   ```

2. **enable persistent_workers**：reduce worker startup overhead
   ```yaml
   persistent_workers: True
   ```

3. **Adjustment batch_size**：according to GPU Memory and training speed adjustments
   ```yaml
   max_img_per_gpu: 32  # increase to 32（If memory allows）
   ```

4. **use pin_memory**：accelerate CPU arrive GPU data transfer
   ```yaml
   pin_memory: True
   ```
