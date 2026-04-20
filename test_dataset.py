import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """
    Test script to verify COLMAP dataset loading works correctly.
    """
    logger.info("=" * 80)
    logger.info("Testing COLMAP Dataset Loading")
    logger.info("=" * 80)
    
    try:
        with initialize(version_base=None, config_path="training/config"):
            cfg = compose(config_name="colmap_dataset")
        
        logger.info("\nConfiguration loaded successfully:")
        logger.info(f"Experiment name: {cfg.exp_name}")
        logger.info(f"Image size: {cfg.img_size}")
        logger.info(f"Patch size: {cfg.patch_size}")
        logger.info(f"Max images per GPU: {cfg.max_img_per_gpu}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Configuration:")
        logger.info("=" * 80)
        
        train_cfg = cfg.data.train
        logger.info(f"\nTraining dataset:")
        logger.info(f"  Target: {train_cfg._target_}")  
        logger.info(f"  Num workers: {train_cfg.num_workers}")
        logger.info(f"  Max images per GPU: {train_cfg.max_img_per_gpu}")
        
        if 'dataset' in train_cfg:
            dataset_cfg = train_cfg.dataset
            logger.info(f"  Dataset target: {dataset_cfg._target_}")
            
            if 'dataset_configs' in dataset_cfg:
                for i, ds_config in enumerate(dataset_cfg.dataset_configs):
                    logger.info(f"\n  Sub-dataset {i+1}:")
                    logger.info(f"    Target: {ds_config._target_}")
                    logger.info(f"    Split: {ds_config.split}")
                    logger.info(f"    COLMAP_DIR: {ds_config.COLMAP_DIR}")
                    logger.info(f"    Min images: {ds_config.min_num_images}")
                    logger.info(f"    RGB dir: {ds_config.rgb_dir}")
                    logger.info(f"    Thermal dir: {ds_config.thermal_dir}")
                    logger.info(f"    Depth dir: {ds_config.depth_gt_dir}")
                    logger.info(f"    Camera dir: {ds_config.camera_dir}")
        
        val_cfg = cfg.data.val
        logger.info(f"\nValidation dataset:")
        logger.info(f"  Target: {val_cfg._target_}")
        logger.info(f"  Num workers: {val_cfg.num_workers}")
        
        if 'dataset' in val_cfg:
            dataset_cfg = val_cfg.dataset
            logger.info(f"  Dataset target: {dataset_cfg._target_}")
            
            if 'dataset_configs' in dataset_cfg:
                for i, ds_config in enumerate(dataset_cfg.dataset_configs):
                    logger.info(f"\n  Sub-dataset {i+1}:")
                    logger.info(f"    Target: {ds_config._target_}")
                    logger.info(f"    Split: {ds_config.split}")
                    logger.info(f"    COLMAP_DIR: {ds_config.COLMAP_DIR}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Checking Dataset Directory Structure:")
        logger.info("=" * 80)
        
        colmap_dir = cfg.data.train.dataset.dataset_configs[0].COLMAP_DIR
        if os.path.exists(colmap_dir):
            logger.info(f"\nDataset directory exists: {colmap_dir}")
            scenes = sorted(os.listdir(colmap_dir))
            logger.info(f"Found {len(scenes)} scenes:")
            for scene in scenes:
                scene_path = os.path.join(colmap_dir, scene)
                if os.path.isdir(scene_path):
                    logger.info(f"\n  Scene: {scene}")
                    
                    rgb_dir = os.path.join(scene_path, cfg.data.train.dataset.dataset_configs[0].rgb_dir)
                    thermal_dir = os.path.join(scene_path, cfg.data.train.dataset.dataset_configs[0].thermal_dir)
                    depth_dir = os.path.join(scene_path, cfg.data.train.dataset.dataset_configs[0].depth_gt_dir)
                    camera_dir = os.path.join(scene_path, cfg.data.train.dataset.dataset_configs[0].camera_dir)
                    
                    if os.path.exists(rgb_dir):
                        rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))]
                        logger.info(f"    RGB images: {len(rgb_files)} files")
                    else:
                        logger.warning(f"    RGB directory not found: {rgb_dir}")
                    
                    if os.path.exists(thermal_dir):
                        thermal_files = [f for f in os.listdir(thermal_dir) if f.lower().endswith(('.jpg', '.png'))]
                        logger.info(f"    Thermal images: {len(thermal_files)} files")
                    else:
                        logger.warning(f"    Thermal directory not found: {thermal_dir}")
                    
                    if os.path.exists(depth_dir):
                        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.npy'))]
                        logger.info(f"    Depth maps: {len(depth_files)} files")
                    else:
                        logger.warning(f"    Depth directory not found: {depth_dir}")
                    
                    if os.path.exists(camera_dir):
                        logger.info(f"    Camera directory exists")
                    else:
                        logger.warning(f"    Camera directory not found: {camera_dir}")
        else:
            logger.error(f"Dataset directory not found: {colmap_dir}")
            return False
        
        logger.info("\n" + "=" * 80)
        logger.info("All checks passed! Dataset is ready for training.")
        logger.info("=" * 80)
        
        logger.info("\nTo start training, run:")
        logger.info("  python training/launch.py --config colmap_dataset")
        logger.info("  or")
        logger.info("  train_colmap.bat")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during dataset testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
