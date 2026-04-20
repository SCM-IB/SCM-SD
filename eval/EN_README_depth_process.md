# Depth map processing script

Back-project multi-view depth map to world coordinate system，Support scale alignment for monocular depth estimation。

## Features

- **Depth map backprojection**：Back-project the multi-view depth map to the world coordinate system centered on the first frame
- **scale alignment**：Support scale alignment of monocular depth estimation models（likeDepthAnything）
- **Standard format output**：The output depth map format is the same asScanNet/7-ScenesBe consistent
  - 16single channelPNG
  - unit：mm（mm）
  - Invalid value：0
  - maximum depth：65.535rice（65535mm）

## Enter requirements

### Directory structure
```
data_root/
├── rgb/                    # RGBimage directory
├── depth_rgb/              # Monocular depth map directory（inverse depth/Disparity map format）
└── colmap/
    └── sparse/
        └── 0/
            ├── cameras.bin  # COLMAPCamera parameters
            ├── images.bin   # COLMAPimage pose
            └── points3D.bin # COLMAPsparse3Dpoint
```

### Depth map format
- **Format**：16single channelPNG
- **type**：inverse depth/disparity map（inverse depth）
- **normalization**：value range [0, 1]，Multiply when storing 2^16

## Output results

### 1. Aligned depth map
- **Table of contents**：`data_root/depth_aligned/`
- **Format**：16BitPNG，Unit mm
- **name**：Same name as the input image

### 2. Depth parameter file
- **document**：`data_root/depth_params.json`
- **content**：Scale parameter of each image
  ```json
  {
    "image_name": {
      "scale": 1.234,
      "offset": 0.567,
      "num_valid_points": 1234
    }
  }
  ```

### 3. point cloud file
- **document**：`output_path/pointcloud.ply`
- **coordinate system**：World coordinate system centered on the first frame

## How to use

### Basic usage

```bash
python depth_process.py --data_root /path/to/dataset --output_path /path/to/output
```

### Complete parameters

```bash
python depth_process.py \
    --data_root /path/to/dataset \
    --output_path /path/to/output \
    --downsample 1 \
    --save_aligned_depth \
    --max_depth 65.535 \
    --depth_dir_name depth_rgb \
    --n_jobs -1 \
    --target_height 480 \
    --target_width 640
```

### Parameter description

| parameter | illustrate | default value |
|------|------|--------|
| `--data_root` | Dataset root directory | Required |
| `--output_path` | Output point cloud path | Required |
| `--downsample` | Downsampling factor | 1 |
| `--save_aligned_depth` | Save the aligned depth map | True |
| `--max_depth` | maximum depth value（rice） | 65.535 |
| `--depth_dir_name` | Depth map directory name | depth_rgb |
| `--n_jobs` | Number of parallel tasks（-1means using allCPU） | -1 |
| `--target_height` | target depth map height | None |
| `--target_width` | Target depth map width | None |

## Processing flow

1. **readCOLMAPdata**
   - Load camera internal parameters（cameras.bin）
   - Load image pose（images.bin）
   - Loading sparsely3Dpoint（points3D.bin）

2. **Set reference frame**
   - Use first frame as reference frame
   - Convert all point clouds to reference frame coordinate system

3. **Calculate depth scale parameters**
   - for each image，useCOLMAPSparse points for scale alignment
   - Use the median andMAD（Mean Absolute Deviation）Perform robust alignment
   - Supports parallel processing acceleration

4. **Save aligned depth map**
   - Apply scale parameter to align depth map
   - Convert toScanNet/7-Scenesstandard format
   - save as16BitPNG，Unit mm

5. **Generate point cloud**
   - Back-project the aligned depth map to3Dspace
   - Convert to world coordinate system
   - save asPLYFormat

## scale alignment principle

### Inverse depth alignment
Monocular depth estimation model（likeDepthAnything）Usually output inverse depth/disparity map：

```
aligned_inv_depth = scale * inv_depth_mono + offset
aligned_depth = 1.0 / aligned_inv_depth
```

### Robust Alignment Method
Use the median andMADPerform robust alignment，Avoid the impact of outliers：

```python
# COLMAPInverse depth statistics
t_colmap = median(inv_depth_colmap)
s_colmap = mean(|inv_depth_colmap - t_colmap|)

# Statistics of monocular inverse depth
t_mono = median(inv_depth_mono)
s_mono = mean(|inv_depth_mono - t_mono|)

# Calculate scale parameters
scale = s_colmap / s_mono
offset = t_colmap - t_mono * scale
```

## Dependencies

```python
numpy
opencv-python
open3d
tqdm
joblib
```

## Things to note

1. **Depth map format**：Make sure the input depth map is inverse depth/Disparity map format
2. **COLMAPdata**：must contain the completecameras.bin、images.binandpoints3D.bin
3. **memory usage**：Be aware of memory usage when processing large data sets
4. **parallel processing**：use`--n_jobs`Parameters controlling the degree of parallelism，Avoid memory overflow

## Example

### deal with7-ScenesDataset

```bash
python depth_process.py \
    --data_root ./data/7Scenes/chess \
    --output_path ./output/chess_pointcloud.ply \
    --depth_dir_name depth_rgb \
    --n_jobs 8
```

### deal withScanNetDataset

```bash
python depth_process.py \
    --data_root ./data/scannet/scene0000_00 \
    --output_path ./output/scannet_pointcloud.ply \
    --depth_dir_name depth \
    --downsample 2
```

## troubleshooting

### question：Depth map not found
**solution**：examine`--depth_dir_name`Are the parameters correct>，Make sure the depth map directory exists

### question：Scale alignment failed
**solution**：
- make sureCOLMAPofpoints3D.binContains enough sparse points
- Check if the depth map is in inverse depth format
- Checkdepth_params.jsoninnum_valid_pointsis it enough

### question：memory overflow
**solution**：
- reduce`--n_jobs`Parameter value
- Increase`--downsample`Parameter downsampling
- Processing datasets in batches
