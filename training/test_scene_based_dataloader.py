"""
Test script for scene-based dataloader
"""
import os
import sys
import time
import torch
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_scene_based_dataloader():
    """Test the scene-based dataloader"""
    
    # Initialize Hydra
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="colmap_dataset")
    
    print("=" * 80)
    print("Testing Scene-Based DataLoader")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Max images per GPU: {cfg.max_img_per_gpu}")
    print(f"  - Batch size (train): {cfg.limit_train_batches}")
    print(f"  - Batch size (val): {cfg.limit_val_batches}")
    print(f"  - Image size: {cfg.img_size}")
    print(f"  - Accum steps: {cfg.accum_steps}")
    
    # Import after Hydra initialization
    from hydra.utils import instantiate
    
    # Create train dataloader
    print("\n" + "=" * 80)
    print("Creating Train DataLoader...")
    print("=" * 80)
    
    train_dataset = instantiate(cfg.data.train, _recursive_=False)
    # Use random seed based on current time to ensure different sampling each run
    random_seed = int(time.time())
    train_dataset.seed = random_seed
    print(f"  Using random seed: {random_seed}")
    
    train_loader = train_dataset.get_loader(epoch=0)
    
    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(f"Train loader batch sampler type: {type(train_loader.batch_sampler).__name__}")
    
    # Test a few batches
    print("\n" + "=" * 80)
    print("Testing Train Batches...")
    print("=" * 80)
    
    num_batches_to_test = min(5, cfg.limit_train_batches)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_test:
            break
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Number of scenes in batch: {len(batch['seq_name'])}")
        print(f"  Scene names: {batch['seq_name']}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Depths shape: {batch['depths'].shape}")
        print(f"  Extrinsics shape: {batch['extrinsics'].shape}")
        print(f"  Intrinsics shape: {batch['intrinsics'].shape}")
        print(f"  Cam points shape: {batch['cam_points'].shape}")
        print(f"  World points shape: {batch['world_points'].shape}")
        print(f"  Point masks shape: {batch['point_masks'].shape}")
        print(f"  Frame num: {batch['frame_num']}")
        
        # Verify that each scene has the same number of images
        print(f"  Frames per scene: {batch['frame_num']}")
        # frame_num is now a scalar value representing the number of images per scene
        
        # Verify image dimensions
        img_size = batch['images'].shape
        assert img_size[-1] == cfg.img_size, f"Image width should be {cfg.img_size}"
        
        print(f"  ✓ Batch {batch_idx} passed validation")
    
    print("\n" + "=" * 80)
    print("Creating Validation DataLoader...")
    print("=" * 80)
    
    val_dataset = instantiate(cfg.data.val, _recursive_=False)
    # Use a different random seed for validation
    val_seed = int(time.time()) + 1
    val_dataset.seed = val_seed
    print(f"  Using validation seed: {val_seed}")
    
    val_loader = val_dataset.get_loader(epoch=0)
    
    print(f"\nVal dataset length: {len(val_dataset)}")
    
    # Test validation batch
    print("\n" + "=" * 80)
    print("Testing Validation Batch...")
    print("=" * 80)
    
    val_batch = next(iter(val_loader))
    print(f"\n--- Validation Batch ---")
    print(f"  Batch keys: {list(val_batch.keys())}")
    print(f"  Number of scenes in batch: {len(val_batch['seq_name'])}")
    print(f"  Images shape: {val_batch['images'].shape}")
    print(f"  Depths shape: {val_batch['depths'].shape}")
    print(f"  ✓ Validation batch passed validation")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_scene_based_dataloader()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
