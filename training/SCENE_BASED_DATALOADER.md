# 场景级别 DataLoader 重构文档

## 概述

本文档说明了重构后的场景级别 DataLoader 架构。新的架构按场景文件夹索引进行采样，每个 batch 包含多个场景，每个场景采样指定数量的图像。

## 架构设计

### 数据流程

```
1. SceneBasedBatchSampler.__iter__()
   ├─ 随机采样 image_num（例如：8张图像）
   ├─ 随机采样 aspect_ratio（例如：0.5）
   ├─ 计算 batch_size = 24 / 8 = 3
   └─ 采样3个场景索引
      输出：[(scene_idx_0, 8, 0.5), (scene_idx_1, 8, 0.5), (scene_idx_2, 8, 0.5)]

2. DataLoader.collate_fn()
   └─ 将3个元组组合成batch
      调用3次 dataset.__getitem__()

3. ComposedDataset.__getitem__((scene_idx, 8, 0.5))
   └─ 调用底层数据集获取数据

4. ColmapDataset.get_data(scene_idx, 8, 0.5)
   ├─ 根据 scene_idx 获取场景名称（例如："Dimsum"）
   ├─ 加载场景的相机参数（53个相机）
   ├─ 随机采样 8 张图像ID（例如：[12, 45, 78, 23, 56, 89, 34, 67]）
   ├─ 加载这8张图像的RGB、热红外、深度、相机参数
   └─ 返回包含8张图像的字典

5. 最终batch
   ├─ 第1组：场景0（Dimsum）的8张图像
   ├─ 第2组：场景1（Ebike）的8张图像
   └─ 第3组：场景2（Truck）的8张图像
```

### 关键组件

#### 1. SceneBasedDynamicDataset

主数据集类，负责：
- 初始化底层数据集（ComposedDataset）
- 创建场景级别采样器（SceneBasedDistributedSampler）
- 创建批次采样器（SceneBasedBatchSampler）
- 提供 DataLoader 创建接口

**配置参数：**
```yaml
data:
  train:
    _target_: data.scene_based_dataloader.SceneBasedDynamicDataset
    num_workers: 8
    max_img_per_gpu: 24  # 每个 GPU 最多处理 24 张图像
    common_config:
      img_size: 518
      patch_size: 14
      img_nums: [2, 24]  # 每个场景采样 2-24 张图像
      augs:
        aspects: [0.33, 1.0]  # 宽高比范围
```

#### 2. SceneBasedBatchSampler

批次采样器，负责：
- 随机采样图像数量（image_num）
- 随机采样宽高比（aspect_ratio）
- 计算 batch_size = max_img_per_gpu / image_num
- 采样 batch_size 个场景索引

**工作原理：**
```python
# 每个 batch 的采样过程
random_image_num = random.choice([2, 3, ..., 24])  # 例如：8
random_aspect_ratio = random.uniform(0.33, 1.0)  # 例如：0.5
batch_size = max_img_per_gpu / random_image_num  # 例如：24 / 8 = 3

# 采样 3 个场景
batch = [
    (scene_idx_0, 8, 0.5),
    (scene_idx_1, 8, 0.5),
    (scene_idx_2, 8, 0.5),
]
```

#### 3. SceneBasedDistributedSampler

分布式采样器，负责：
- 支持多 GPU 分布式训练
- 将场景均匀分配到不同 GPU
- 提供场景索引迭代器

**分布式模式：**
- GPU 0: 场景索引 [0, 4, 8, ...]
- GPU 1: 场景索引 [1, 5, 9, ...]
- GPU 2: 场景索引 [2, 6, 10, ...]
- GPU 3: 场景索引 [3, 7, 11, ...]

#### 4. ColmapDataset·

COLMAP 数据集类，负责：
- 根据 scene_idx 获取场景文件夹路径
- 加载场景的相机内参和外参
- 随机采样指定数量的图像
- 加载 RGB 图像、热红外图像、深度图
- 处理图像、深度、相机参数

**数据结构：**
```
dataset/
├── Dimsum/          # 场景 0
│   ├── rgb/          # RGB 图像
│   ├── thermal/      # 热红外图像
│   ├── colmap/
│   │   └── sparse/0/ # 相机参数
│   └── depth_aligned/ # 深度图
├── Ebike/           # 场景 1
└── Truck/           # 场景 2
```

## 配置文件

### 训练配置

```yaml
# colmap_dataset.yaml
max_img_per_gpu: 24  # 每个 GPU 最多处理 24 张图像

data:
  train:
    _target_: data.scene_based_dataloader.SceneBasedDynamicDataset
    num_workers: 8
    max_img_per_gpu: ${max_img_per_gpu}
    common_config:
      img_size: 518
      patch_size: 14
      img_nums: [2, 24]  # 每个场景采样 2-24 张图像
      augs:
        aspects: [0.33, 1.0]  # 宽高比范围
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.colmapdata.ColmapDataset
          split: train
          COLMAP_DIR: e:\Paper2\1\vggt-our\dataset
          min_num_images: 12  # 场景至少需要 12 张图像
          len_train: 100000
          expand_ratio: 1
          rgb_dir: rgb
          thermal_dir: thermal
          camera_dir: colmap/sparse/0
          depth_gt_dir: depth_aligned
```

### Batch 组成示例

假设 `max_img_per_gpu = 24`，随机采样得到 `image_num = 8`：

```
Batch size = 24 / 8 = 3

Batch 包含 3 个场景：
- 场景 0 (Dimsum): 8 张图像
- 场景 1 (Ebike): 8 张图像  
- 场景 2 (Truck): 8 张图像

最终 batch shape:
- images: [3, 8, 3, H, W]  # (batch_size, num_images, channels, height, width)
- depths: [3, 8, H, W]
- extrinsics: [3, 8, 4, 4]
- intrinsics: [3, 8, 3, 3]
```

## 使用方法

### 1. 基本使用

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# 加载配置
cfg = compose(config_name="colmap_dataset")

# 创建数据集
train_dataset = instantiate(cfg.data.train, _recursive_=False)
train_dataset.seed = cfg.seed_value

# 创建 DataLoader
train_loader = train_dataset.get_loader(epoch=0)

# 迭代 batch
for batch in train_loader:
    images = batch['images']  # [batch_size, num_images, 3, H, W]
    depths = batch['depths']  # [batch_size, num_images, H, W]
    extrinsics = batch['extrinsics']  # [batch_size, num_images, 4, 4]
    intrinsics = batch['intrinsics']  # [batch_size, num_images, 3, 3]
    
    # 前向传播
    outputs = model(images, depths, extrinsics, intrinsics)
    
    # 计算损失
    loss = criterion(outputs, batch)
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### 2. 测试 DataLoader

运行测试脚本验证 DataLoader 是否正常工作：

```bash
cd training
python test_scene_based_dataloader.py
```

## 与原架构的区别

| 特性 | 原架构 (DynamicTorchDataset) | 新架构 (SceneBasedDynamicDataset) |
|------|------------------------------|----------------------------------|
| **采样单位** | 单张图像索引 | 场景文件夹索引 |
| **Batch 组成** | 同一场景的多张图像 | 不同场景的多个样本 |
| **图像数量** | 每个 batch 固定 | 每个 batch 动态变化 |
| **数据多样性** | 较低（同一场景） | 较高（不同场景） |
| **GPU 利用率** | 可能不均匀 | 更均匀 |

## 优势

1. **更好的数据多样性**：每个 batch 包含不同场景，增加数据多样性
2. **更均匀的 GPU 利用率**：动态调整 batch size 以适应不同图像数量
3. **更符合多视角学习**：每个场景独立采样多张图像
4. **更好的泛化能力**：模型在不同场景间学习

## 注意事项

1. **场景数量**：确保数据集有足够的场景（建议 > 10 个）
2. **图像数量范围**：`img_nums` 范围不宜过大，建议 [2, 24]
3. **GPU 内存**：根据 GPU 内存调整 `max_img_per_gpu`
4. **分布式训练**：确保场景数量 >= GPU 数量

## 故障排除

### 问题 1: CUDA OOM

**原因**：`max_img_per_gpu` 设置过大

**解决方案**：减小 `max_img_per_gpu` 值

```yaml
max_img_per_gpu: 16  # 从 24 减小到 16
```

### 问题 2: 场景图像不足

**原因**：场景图像数量 < `img_nums` 最小值

**解决方案**：调整 `min_num_images` 或 `img_nums`

```yaml
min_num_images: 8  # 从 12 减小到 8
img_nums: [2, 12]  # 从 [2, 24] 减小到 [2, 12]
```

### 问题 3: 分布式训练数据重复

**原因**：场景数量 < GPU 数量

**解决方案**：增加场景数量或减少 GPU 数量

## 性能优化建议

1. **调整 num_workers**：根据 CPU 核心数调整
   ```yaml
   num_workers: 16  # 对于 8 核 CPU
   ```

2. **启用 persistent_workers**：减少 worker 启动开销
   ```yaml
   persistent_workers: True
   ```

3. **调整 batch_size**：根据 GPU 内存和训练速度调整
   ```yaml
   max_img_per_gpu: 32  # 增加到 32（如果内存允许）
   ```

4. **使用 pin_memory**：加速 CPU 到 GPU 的数据传输
   ```yaml
   pin_memory: True
   ```
