# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from hydra.utils import instantiate
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler, RandomSampler
from abc import ABC, abstractmethod
import torch.distributed as dist
import time


from .worker_fn import get_worker_init_fn


class SceneBasedDynamicDataset(ABC):
    """
    A scene-based dynamic dataset that samples batches from different scenes.
    
    Each batch consists of multiple samples, where each sample comes from a different scene.
    For each scene, we sample num_images images and process them together.
    """
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = int(time.time()),
        max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset (e.g., data.composed_dataset.ComposedDataset)
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects
        self.image_num_range = common_config.img_nums

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        self.sampler = SceneBasedDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = SceneBasedBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu
        )

    def __len__(self):
        """
        Returns the length of the underlying dataset.
        """
        return len(self.dataset)

    def get_loader(self, epoch):
        """
        Creates a DataLoader for the given epoch.
        
        Each batch contains samples from different scenes. For each scene,
        we sample num_images images and process them together.
        """
        print(f"Building scene-based dynamic dataloader with epoch: {epoch}")

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )


class SceneBasedBatchSampler(Sampler):
    """
    A scene-based batch sampler that samples batches from different scenes.
    
    Each batch contains multiple samples, where each sample comes from a different scene.
    For each scene, we sample num_images images and process them together.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=int(time.time()),
                 max_img_per_gpu=48):
        """
        Initializes the scene-based batch sampler.

        Args:
            sampler: Instance of SceneBasedDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per scene.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples from different scenes.

        Each batch contains samples from different scenes. For each scene,
        we sample num_images images and process them together.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio for this batch
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)
                        current_batch.append(item)
                    except StopIteration:
                        # Sampler exhausted, create a new iterator to continue
                        sampler_iterator = iter(self.sampler)
                        try:
                            item = next(sampler_iterator)
                            current_batch.append(item)
                        except StopIteration:
                            break

                if not current_batch:
                    break

                # Add the same num_images and aspect_ratio to all items in the batch
                batch_with_params = [(scene_idx, random_image_num, random_aspect_ratio) for scene_idx, _, _ in current_batch]
                yield batch_with_params

            except StopIteration:
                break

    def __len__(self):
        """
        Returns the length of the sampler.
        """
        # Return a large dummy length
        return 1000000


class SceneBasedDistributedSampler:
    """
    A sampler that samples scene indices for distributed training.
    
    Each sample corresponds to a scene (scene_idx), and we will sample
    num_images images from that scene.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = int(time.time()),
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Check if distributed is available and initialized
        self.is_distributed = dist.is_available() and dist.is_initialized()

        if self.is_distributed:
            # Use DistributedSampler for distributed training
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last
            )
        else:
            # Use RandomSampler for single GPU training
            self.sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        if self.is_distributed:
            self.sampler.set_epoch(epoch)
        else:
            # For RandomSampler, we need to recreate it with a new seed
            self.sampler = RandomSampler(
                self.dataset,
                generator=torch.Generator().manual_seed(self.seed + epoch)
            )

    def __iter__(self):
        """
        Yields a sequence of (scene_idx, None, None).
        The num_images and aspect_ratio will be set by the batch sampler.
        """
        for scene_idx in self.sampler:
            yield (scene_idx, None, None)


    def update_parameters(self, aspect_ratio, image_num):
        """
        This method is not used in scene-based sampling,
        but kept for compatibility with the original interface.
        """
        pass

