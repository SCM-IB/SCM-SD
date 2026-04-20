# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        ir_image,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        ir_image = np.copy(ir_image)
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size


        # Move principal point to the image center and crop if necessary
        # According to the principal point coordinates in the internal parameter matrix (cx, cy) Calculate cropping area，The cropping area is centered on the main point，The size is limited by the distance from the principal point to the edge of the image
        # Update internal parameter matrix at the same time，Adjust the principal point coordinates to the center of the new cropping area
        ir_image, image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            ir_image, image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        original_size = np.array(ir_image.shape[:2])
        target_shape = target_image_shape
        
        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize images and update intrinsics
        # Scale the image to the approximate size of the target shape，while retaining safety boundaries
        if self.rescale:
            ir_image, image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                ir_image,image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug
            )
        else:
            print("Not rescaling the images")
        # print(f"2-STEP size: {np.array(ir_image.shape[:2])}")
        # Ensure final crop to target shape
        # Ensure that the final output image size exactly fits the target shape
        # Also crop with the main point as the center
        # If the cropping area is smaller than the target shape，Will be zero padded
        # Update the internal parameter matrix again
        ir_image, image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            ir_image, image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )
        # print(f"3-STEP size: {np.array(ir_image.shape[:2])}")
        # print(f"Depth map shape: {depth_map.shape}")

        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            ir_image, image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                ir_image,
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        return (
            ir_image,
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        Sample IDs around a center index using range-based sampling.

        This method samples IDs in a range around a center index, similar to
        the evaluation code's range mode. The center index is always included.

        Args:
            ids (list): Initial list of IDs. The first element is used as the center.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which to expand the sampling range.
                The range size is calculated as: int(len(ids) * expand_ratio * 2).
                Default is 1.0 if neither expand_ratio nor expand_range is provided.
            expand_range (int, optional): Fixed range size (total range, not half-range).
                If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original center index. Contains len(ids) elements.

        Examples:
            # Using expand_ratio=1.0 (default)
            # If ids=[6] and full_seq_num=100, with expand_ratio=1.0,
            # range_size = int(1 * 1.0 * 2) = 2, so IDs sampled from [5,6,7].
            
            # If ids=[6] and full_seq_num=100, need 4 images, expand_ratio=1.0,
            # range_size = int(4 * 1.0 * 2) = 8, so IDs sampled from [2...10].
            
            # If ids=[6] and full_seq_num=100, need 8 images, expand_ratio=1.0,
            # range_size = int(8 * 1.0 * 2) = 16, so IDs sampled from [0...14] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 1.0  # Default behavior

        total_ids = len(ids)
        center_idx = ids[0]

        # Determine the actual range_size (total range, not half-range)
        if expand_range is None:
            # Use ratio to determine range size: total_ids * expand_ratio * 2
            range_size = int(total_ids * expand_ratio * 2)
        else:
            range_size = expand_range

        # Calculate half range
        half_range = range_size // 2

        # Generate candidate indices (supporting circular/wrap-around)
        candidate_indices = []
        for i in range(-half_range, half_range + 1):
            idx = (center_idx + i) % full_seq_num
            candidate_indices.append(idx)

        candidate_indices = np.array(candidate_indices)

        # Sample (total_ids - 1) items from candidate indices, then add center_idx
        if len(candidate_indices) >= total_ids:
            # Randomly sample (total_ids - 1) items from candidates
            # Ensure center_idx is included by inserting it later
            candidates_without_center = candidate_indices[candidate_indices != center_idx]
            sampled = np.random.choice(
                candidates_without_center,
                size=(total_ids - 1),
                replace=False
            )
            result_ids = np.insert(sampled, 0, center_idx)
        else:
            # If not enough candidates, use all candidates and pad with center_idx
            result_ids = candidate_indices
            # Pad with center_idx if needed
            while len(result_ids) < total_ids:
                result_ids = np.append(result_ids, center_idx)
            # Truncate to exact size if needed
            result_ids = result_ids[:total_ids]

        result_ids = np.sort(result_ids)
        return result_ids
