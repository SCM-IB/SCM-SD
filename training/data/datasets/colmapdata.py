import os
import os.path as osp
import logging
import random
import glob
import time

import cv2
import numpy as np

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset



class ColmapDataset(BaseDataset):
    def __init__(self,
                common_conf,
                 split: str = "train",
                 COLMAP_DIR: str = "/...",
                 min_num_images: int = 24,
                 len_train: int = 100_000,
                 len_test: int = 10_000,
                 expand_ratio: int = 8,   #expand_range=int(img_per_seq(len(ids)) * expand_ratio)
                 expand_range: int = None,
                 rgb_dir: str = "rgb",
                 thermal_dir: str = "thermal",
                 camera_dir: str = "colmap/sparse/0",
                 depth_gt_dir: str = "depth_aligned",
                 ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.COLMAP_DIR = COLMAP_DIR
        self.expand_ratio = expand_ratio
        self.expand_range = expand_range

        self.min_num_images = min_num_images
        self.depth_max = 65.535
        self.len_train = len_train if split == "train" else len_test
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.camera_dir = camera_dir
        self.depth_gt_dir = depth_gt_dir

        logging.info(f"Colmap_DIR is {self.COLMAP_DIR}")
        logging.info(f"RGB directory is {self.rgb_dir}")
        logging.info(f"Thermal directory is {self.thermal_dir}")
        logging.info(f"Camera directory is {self.camera_dir}")
        logging.info(f"Depth GT directory is {self.depth_gt_dir}")

        scene_name_list = sorted(os.listdir(COLMAP_DIR))


        self.sequence_list = scene_name_list
        self.sequence_list_len = len(self.sequence_list)
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: COLMAP Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: COLMAP Data dataset length: {len(self)}")

        self.reading_dir = "images"

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.sequence_list_len

    def get_data(self,
                 seq_index=None,
                 img_per_seq=None,
                 seq_name=None,
                 ids=None,
                 aspect_ratio=1.0,
                 )-> dict:
        """
        Retrieve data for a specific Scene from custom dataset.
        
        Args:
            seq_index: Index of the scene in self.sequence_list
            img_per_seq: Number of images to sample from the scene
            seq_name: Name of the scene (if None, use seq_index)
            ids: Specific image IDs to use (if None, sample randomly)
            aspect_ratio: Target aspect ratio for images
        
        Returns:
            dict: Batch containing images, depths, camera parameters, etc.
        """
        seed = int(time.time())
        np.random.seed(seed)
        if self.inside_random and self.training:
            #random.seed(int(time.time()))
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        scene_path = osp.join(self.COLMAP_DIR, seq_name)

        # load camera parameters for a specific scene.
        try:
            camera_extrinsics_unsorted = read_camera_extrinsics(scene_path)  
            # camera_extrinsics_unsorted = read_extrinsics_test(scene_path)  
            camera_intrinsics = read_camera_intrinsics(scene_path)  
        except Exception as e:
            logging.error(f"Error loading camera extrinsics or intrinsics for {scene_path}: {e}")
            raise

        num_images = len(camera_extrinsics_unsorted)

        if ids == None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio, expand_range=self.expand_range)

        target_image_shape = self.get_target_shape(aspect_ratio)

        camera_extrinsics = sorted(camera_extrinsics_unsorted.copy(), key = lambda x : x.name)
       


        # for extr in camera_extrinsics:
        #     print(f"Image name: {extr.name:<50} image ID: {extr.id}")
        """
        Containers
        """
        images = []
        infrared_images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        # Track which images failed to load for debugging
        failed_rgb_images = []
        failed_ir_images = []
        failed_depth_images = []

        """
        Preprocess every images refer to ids(np.array)
        """
        for idx in ids:
            extr = camera_extrinsics[idx]
            image_name = extr.name
          
            # Load infrared image from thermal directory
            # Try to find the image with the correct extension
            thermal_filepath = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                test_path = osp.join(scene_path, self.thermal_dir, os.path.splitext(os.path.basename(image_name))[0] + ext)
                if os.path.exists(test_path):
                    thermal_filepath = test_path
                    break
            
            if thermal_filepath and os.path.exists(thermal_filepath):
                    ir_image = read_image_cv2(thermal_filepath)
            else:
                    failed_ir_images.append(image_name)
                    logging.warning(f"Infrared image not found for {image_name}, skipping this frame")
                    continue
            
            _name = os.path.basename(thermal_filepath).split(".")[0] if thermal_filepath else ""

            # Load visible image from rgb directory
            # Try to find the image with the correct extension
            rgb_filepath = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                test_path = osp.join(scene_path, self.rgb_dir, os.path.splitext(os.path.basename(image_name))[0] + ext)
                if os.path.exists(test_path):
                    rgb_filepath = test_path
                    break

            if rgb_filepath:
                image = read_image_cv2(rgb_filepath)
            else:
                failed_rgb_images.append(image_name)
                logging.warning(f"RGB image not found for {image_name}, skipping this frame")
                continue
            
            # Load depth ground truth from depth_gt directory
            # Try PNG format first (our dataset format)
            depth_gt_filepath = None
            for ext in ['.png', '.PNG', '.npy']:
                test_path = osp.join(scene_path, self.depth_gt_dir, os.path.splitext(os.path.basename(image_name))[0] + ext)
                if os.path.exists(test_path):
                    depth_gt_filepath = test_path
                    break
            
            if depth_gt_filepath:
                if depth_gt_filepath.endswith('.npy'):
                    depth_map = np.load(depth_gt_filepath).astype(np.float32)
                else:
                    depth_map = cv2.imread(depth_gt_filepath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    if depth_map.ndim == 3:
                        depth_map = depth_map[:, :, 0]
            else:
                # Fallback to placeholder if depth GT is not available
                failed_depth_images.append(image_name)
                logging.warning(f"Depth map not found for {image_name}, using zeros")
                depth_temp = np.zeros_like(image[:,:,0], dtype=np.float32)
                depth_map = np.zeros_like(depth_temp, dtype=np.float32)

            assert image.shape[:2] == depth_map.shape[:2], f"Image and depth map shapes don't match: {image.shape[:2]} vs {depth_map.shape[:2]}"
            assert image.shape[:2] == ir_image.shape[:2], f"Visible and infrared image shapes don't match: {image.shape[:2]} vs {ir_image.shape[:2]}"
            original_size = np.array(image.shape[:2])

            # Extract intrinsic and extrinsic for current image (camera_extrinsics[idx])
           
            intr = camera_intrinsics[extr.camera_id]
            # print(f"Intrinsic belongs to which camera_id: {intr.id}")
            # print(f"Extrinsic belongs to which camera_id: {extr.camera_id}")

            extri_opencv, intri_opencv = colmap2opencv(intr, extr)
            # print(f"cx, cy,Intrinsic matrix: {intr.params[2]} {intr.params[3]}")

            (   
                ir_image,
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                ir_image,
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=rgb_filepath,
            )
            '''
            Network input has strict size requirements：
                1. fixed width ：Must be configured img_size:=518
                2. height alignment ：must be patch_size an integer multiple of
                3. batch consistency ：Images in the same batch must be of the same size
            '''
            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            infrared_images.append(ir_image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "colmap"
        if len(images) == 0 or len(infrared_images) == 0:
            logging.error(f"[SKIP] Sequence {seq_name}: loaded {len(images)} RGB, {len(infrared_images)} IR from {len(ids)} images.")
            logging.error(f"[SKIP] Failed to load - RGB: {failed_rgb_images}, IR: {failed_ir_images}, Depth: {failed_depth_images}")
            logging.error(f"[SKIP] Paths: COLMAP_DIR={self.COLMAP_DIR}, rgb={self.rgb_dir}, thermal={self.thermal_dir}, depth={self.depth_gt_dir}")
            logging.error(f"[SKIP] IDs attempted: {list(ids)}")
            raise ValueError(f"No valid images for sequence {seq_name}")
        
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "infrared_images": infrared_images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch