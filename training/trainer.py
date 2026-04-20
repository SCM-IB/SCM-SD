# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from click.core import F


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json

import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from train_utils.tb_writer import *
from training.structural_distillation_loss import StructuralDistillationLoss


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """
    """
    one for DDP Universal trainer for training。It should naturally support multi-node training。
    This class is responsible for the coordination of the entire training and validation process，include：
    - Set up a distributed environment（DDP）。
    - Initialize model、optimizer、Loss function and data loader。
    - Handle checkpoints to resume training。
    - Execute the main training and validation loop。
    - Log metrics and visualizations to TensorBoard middle。
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging_conf: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging_conf: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        """
            data：for datasets and data loaders Hydra Configuration。
            model：for models Hydra Configuration。
            logging_conf：for logging（TensorBoard、Log frequency）of Hydra Configuration。
            checkpoint：for checkpointing Hydra Configuration。
            max_epochs：Total number of training rounds。
            mode：“train”for training and validation，“val”Only for verification。
            device：“cuda”or“cpu”。
            seed_value：Random seed for reproducibility。
            val_epoch_freq：How often to run verification（in rounds）。
            distributed：used for DDP set Hydra Configuration。
            cuda：used for CUDA specific settings（For example cuDNN）of Hydra Configuration。
            limit_train_batches：Limits on training batches per round（for debugging）。
            limit_val_batches：Limits on validation batches per round（for debugging）。
            optim：for optimizers and schedulers Hydra Configuration。
            loss：for the loss function Hydra Configuration。
            env_variables：A dictionary of environment variables to set。
            accum_steps：Number of steps to accumulate gradients before the optimizer step。
        """

        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging_conf
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)  # Set up training equipment（CPU or CUDA)
        self._setup_torch_dist_and_backend(cuda, distributed)  # set up torch Distribution and backend
        
        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        print(f"Logging directory: {self.logging_conf.log_dir}")

        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        print(f"============>>>>>>>>")
        # Check if distributed training is enabled
        if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
            assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."
        else:
            logging.info(f"Single GPU mode - distributed training not required")

        # Instantiate components (model, loss, etc.)
        self._setup_components() # Initialize all core training components（Model、loss、Logger etc.）
        self._setup_dataloaders() # Initialize training and validation data sets anddataloader

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")
        
        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)
        
        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)
            
        # Wrap the model with DDP use DDP packaging model
        self._setup_ddp_distributed_training(distributed, device)
        # Barrier to ensure all processes are synchronized before starting (only in distributed mode)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Check if distributed training is enabled (RANK and LOCAL_RANK environment variables are set)
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            # Single GPU mode - skip distributed initialization
            print(f"Single GPU mode detected - skipping distributed initialization")
            self.rank = 0
            return
        
        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """
        Loads a checkpoint from the given path to resume training.
        """
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Check if we're loading into a dual-branch model
        if hasattr(self.model, 'visible_aggregator'):
            # For dual-branch model, we only want to load into the infrared branch
            # Create a filtered state dict that only includes infrared branch parameters
            infrared_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('infrared_'):
                    infrared_state_dict[key] = value
            
            if infrared_state_dict:
                missing, unexpected = self.model.load_state_dict(
                    infrared_state_dict, strict=False
                )
                if self.rank == 0:
                    logging.info(f"Infrared branch state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
            else:
                # If no infrared-specific keys, try loading as usual
                missing, unexpected = self.model.load_state_dict(
                    model_state_dict, strict=self.checkpoint_conf.strict
                )
                if self.rank == 0:
                    logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
        else:
            # For single-branch model, load as usual
            missing, unexpected = self.model.load_state_dict(
                model_state_dict, strict=self.checkpoint_conf.strict
            )
            if self.rank == 0:
                logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            # Handle both list and single optimizer formats
            if isinstance(checkpoint["optimizer"], list):
                # If checkpoint has list of optimizers, load each one
                for i, optim_state in enumerate(checkpoint["optimizer"]):
                    if i < len(self.optims):
                        self.optims[i].optimizer.load_state_dict(optim_state)
            else:
                # If checkpoint has single optimizer, load to first optimizer
                if len(self.optims) > 0:
                    self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """
        Initializes all core training components using Hydra configs.
        """
        """
        Initialize training components（modeland frozen information、tb_writer、loss, etc.）
        Hydravia configuration fileyamlControl program behavior，instantite yes Hydra A factory function provided，
        According to the configuration object（usually dict or OmegaConf object）Automatically call the corresponding class or function，and pass in parameters。
        For example, 
        model.yaml
            model:
              _target_: torch.nn.Linear
              in_features: 784
              out_features: 10
        Usage:
            model = instantiate(cfg.model)  equal to 
            model = torch.nn.Linear(in_features=784, out_features=10)
        """
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)
        
        # Set up KD configuration and freeze parameters accordingly
        if hasattr(self.model, 'set_freeze_parameters'):
            # For structural distillation, we only need feature KD and trainable prediction heads
            kd_config = {
                'use_logit_kd': False,  # Disable logit KD
                'use_feature_kd': True,  # Enable feature KD for structural distillation
                'use_task_heads': True  # Enable trainable task heads for structural distillation
            }
            logging.info(f"Setting up model with KD configuration: {kd_config}")
            self.model.set_freeze_parameters(kd_config)
        
        # Initialize data path tracker for logging training data paths
        self.data_path_tracker = DataPathTracker(
            log_dir=self.logging_conf.log_dir,
            max_samples_to_log=100
        )

        # # Load pre-trained weights for visible branch if using dual-branch model
        # if hasattr(self.model, 'load_visible_weights') and hasattr(self.model_conf, 'visible_model_path') and self.model_conf.visible_model_path:
        #     logging.info(f"Loading pre-trained weights for visible branch from {self.model_conf.visible_model_path}")
        #     self.model.load_visible_weights(self.model_conf.visible_model_path)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value
                
        if self.mode in ["train"]:
            # Instantiate data.train._target_
            # (i.e data.dynamic_dataloader.DynamicTorchDataset) in .yaml file
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        """use DDP packaging model"""
        assert isinstance(self.model, torch.nn.Module)

        # Only wrap with DDP if distributed training is initialized
        if not (dist.is_available() and dist.is_initialized()):
            logging.info(f"Skipping DDP wrapping - not in distributed mode")
            return

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch) # Mainly locate the coordinate origin to the camera center of the first image
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True


    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys (phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Start training !!!
        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            # If gradient clipping is enabled，Register gradient hook（hooks）
            # to the specified module（like aggregator、depth），for subsequent calculation of the gradient norm。
            self.gradient_clipper.setup_clipping(self.model)

        #main training loop（Per-batch Loop）
        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            # Move coordinate system one to the camera center of the first image
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps
            # like accum_steps > 1，make a big batch split into accum_steps small chunk；
            # Subsequently for each chunk Execute forward+reverse，but does not update parameters immediately，Instead, the gradient is accumulated。
            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            ) # forward+reverse。 Notice Not called at this time optimizer.step()，gradients are accumulated

            # compute gradient and do SGD step
            # Learning rate scheduling（Scheduler Update）
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            # use Normalized training progress where ∈ [0,1] driver scheduler（like warmup + cosine）；
            # Only updated when training is not over（where < 1.0）。
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            # Key steps：Must be before cutting unscale_（AMP Require）；
            # gradient_clipper Return the gradient of each module L2 norm（like {"aggregator": 1.23}）；
            # renew Grad/... meters，Used to monitor gradient explosions/disappear。
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer) # Automatic processing AMP
            self.scaler.update() # # Update scaling factor

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)  # Periodically print training progress（Including all meters the average of）

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1 and hasattr(self.model, 'no_sync')
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    # The gateway to true forward propagation
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )

                # Skip if empty batch was detected
                if not loss_dict or "objective" not in loss_dict:
                    continue

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward() #  # AMP Reverse after zooming
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """

        # Trace data path information
        if phase == "train":
            self.data_path_tracker.log_iteration(
                batch=batch,
                iteration=self.steps[phase],
                epoch=self.epoch
            )
        
        # Check for empty batch and skip if necessary
        if batch["images"].shape[0] == 0:
            logging.warning(f"Empty batch detected at step {self.steps[phase]}, skipping...")
            self.steps[phase] += 1
            return {}

        # Check if we're using dual-branch model
        model_to_check = model.module if hasattr(model, 'module') else model
        if hasattr(model_to_check, 'visible_aggregator') and hasattr(model_to_check, 'infrared_aggregator'):
            # For dual-branch model (VGGTDualBranch)
            visible_images = batch["images"]
            infrared_images = batch["infrared_images"]
            
            # Prepare KD configuration
            # Check if structural_distillation config exists and has the required attributes
            if hasattr(self.loss_conf, 'structural_distillation'):
                kd_config = {
                    'use_logit_kd': False,  # Not used in structural distillation
                    'use_feature_kd': True   # Always use feature distillation for structural distillation
                }
            elif hasattr(self.loss_conf, 'distillation'):
                kd_config = {
                    'use_logit_kd': self.loss_conf.distillation.use_logit_kd if hasattr(self.loss_conf.distillation, 'use_logit_kd') else False,
                    'use_feature_kd': self.loss_conf.distillation.use_feature_kd if hasattr(self.loss_conf.distillation, 'use_feature_kd') else False
                }
            else:
                kd_config = {
                    'use_logit_kd': False,
                    'use_feature_kd': True
                }
            
            y_hat = model(visible_images=visible_images, infrared_images=infrared_images, kd_config=kd_config)
        else:
            # For single-branch model (original VGGT)
            y_hat = model(images=batch["images"])
            # y_hat = model(images=batch["infrared_images"])
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation.

    Fix:
    1. Avoid empty chunks when local batch size < accum_steps.
    2. Avoid silently dropping tail samples with the current floor-division slicing
       in get_chunk_from_data().
    """
    if accum_steps <= 1:
        return [batch]

    def _infer_batch_size(data):
        if isinstance(data, torch.Tensor):
            return data.shape[0]
        if is_sequence_of_primitives(data):
            return len(data)
        if isinstance(data, Mapping):
            for _, value in data.items():
                size = _infer_batch_size(value)
                if size is not None:
                    return size
        elif isinstance(data, Sequence) and not isinstance(data, str):
            for value in data:
                size = _infer_batch_size(value)
                if size is not None:
                    return size
        return None

    batch_size = _infer_batch_size(batch)
    if batch_size is None:
        raise ValueError("Unable to infer batch size in chunk_batch_for_accum_steps.")
    if batch_size == 0:
        return [batch]

    # num_chunks must not exceed batch_size, otherwise empty chunks appear.
    max_valid_chunks = min(accum_steps, batch_size)

    # get_chunk_from_data() uses floor-division slicing, so choose a chunk count
    # that exactly divides batch_size to avoid dropping tail samples.
    num_chunks = max_valid_chunks
    while num_chunks > 1 and batch_size % num_chunks != 0:
        num_chunks -= 1

    return [get_chunk_from_data(batch, i, num_chunks) for i in range(num_chunks)]

# def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
#     """Splits a batch into smaller chunks for gradient accumulation."""
#     if accum_steps == 1:
#         return [batch]
#     return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data


class DataPathTracker:
    """
    Track data path information used in each iteration of training
    
    Record the values ​​used in each iteration：
    - scene name
    - imageID
    - RGBimage path
    - Thermal infrared image path
    - Depth map path
    - Camera parameter path
    """
    
    def __init__(self, log_dir: str, max_samples_to_log: int = 100):
        """
        Initialize data path tracer
        
        Args:
            log_dir: Log saving directory
            max_samples_to_log: Maximum number of recorded samples（Avoid excessively large logs）
        """
        self.log_dir = log_dir
        self.max_samples_to_log = max_samples_to_log
        self.iteration_data = []
        self.current_iteration = 0
        self.enabled = True
        
        # Create log file
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "data_paths_log.txt")
        
        # Write log header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Training data path tracking log\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Configuration information:\n")
            f.write(f"  - Log directory: {log_dir}\n")
            f.write(f"  - Maximum number of recorded samples: {max_samples_to_log}\n")
            f.write(f"  - start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
    
    def log_iteration(self, batch: dict, iteration: int, epoch: int):
        """
        Record the data path information of the current iteration
        
        Args:
            batch: currentbatchdata
            iteration: Current number of iterations
            epoch: currentepoch
        """
        if not self.enabled:
            return
        
        # Check if the maximum number of records is exceeded
        if len(self.iteration_data) >= self.max_samples_to_log:
            return
        
        try:
            # extractbatchkey information in
            seq_name = batch.get("seq_name", "unknown")
            ids = batch.get("ids", [])
            frame_num = batch.get("frame_num", 0)
            
            # Build log entry
            log_entry = {
                "iteration": iteration,
                "epoch": epoch,
                "seq_name": seq_name,
                "frame_num": frame_num,
                "image_ids": ids.tolist() if hasattr(ids, 'tolist') else list(ids),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.iteration_data.append(log_entry)
            
            # Write log files in real time
            self._write_log_entry(log_entry)
            
        except Exception as e:
            print(f"Error recording data path: {e}")
    
    def _write_log_entry(self, log_entry: dict):
        """
        Write log entries to file
        
        Args:
            log_entry: Dictionary of log entries
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("-" * 80 + "\n")
            f.write(f"Iterate #{log_entry['iteration']} (Epoch {log_entry['epoch']})\n")
            f.write(f"time: {log_entry['timestamp']}\n")
            f.write(f"scene name: {log_entry['seq_name']}\n")
            f.write(f"number of frames: {log_entry['frame_num']}\n")
            f.write(f"imageID: {log_entry['image_ids']}\n")
            f.write("-" * 80 + "\n\n")
    
    def get_summary(self) -> str:
        """
        Get summary information of tracking data
        
        Returns:
            summary string
        """
        if not self.iteration_data:
            return "No data recorded"
        
        # Statistical scenario usage
        scene_counts = {}
        for entry in self.iteration_data:
            scene = entry['seq_name']
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        summary = f"\n{'='*80}\n"
        summary += f"Data path trace summary\n"
        summary += f"{'='*80}\n"
        summary += f"Total number of recorded samples: {len(self.iteration_data)}\n"
        summary += f"Number of different scenarios used: {len(scene_counts)}\n"
        summary += f"\nScenario usage statistics:\n"
        summary += f"{'-'*80}\n"
        
        # Sort by usage
        sorted_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)
        for scene, count in sorted_scenes:
            summary += f"  {scene}: {count} Second-rate\n"
        
        summary += f"{'='*80}\n"
        
        return summary
    
    def save_summary(self):
        """
        Save summary information to log file
        """
        summary = self.get_summary()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(summary)
    
    def disable(self):
        """Disable tracking"""
        self.enabled = False
    
    def enable(self):
        """Enable tracking"""
        self.enabled = True

