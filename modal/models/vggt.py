# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=False, enable_depth=True, enable_track=False):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def update_patch_dimensions(self, patch_width: int, patch_height: int):
        """
        Update patch dimensions for all attention layers in the model

        Args:
            patch_width: Patch width (typically 37)
            patch_height: Patch height (typically 28)
        """

        def update_attention_in_module(module):
            for name, child in module.named_children():
                update_attention_in_module(child)
                if hasattr(child, "patch_width") and hasattr(child, "patch_height"):
                    child.patch_width = patch_width
                    child.patch_height = patch_height

        update_attention_in_module(self.aggregator)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


class VGGTDualBranch(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 visible_model_path=None, infrared_model_path=None, init_infrared_from_visible=True):
        super().__init__()

        # Visible light branch (teacher)
        self.visible_aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.visible_camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.visible_point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.visible_depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.visible_track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # Infrared branch (student)
        self.infrared_aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.infrared_camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.infrared_point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.infrared_depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.infrared_track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # Load visible (teacher) weights if path is provided
        if visible_model_path is not None and visible_model_path != "":
            self.load_visible_weights(visible_model_path)
        
        # Load infrared (student) weights if path is provided
        if infrared_model_path is not None and infrared_model_path != "":
            self.load_infrared_weights(infrared_model_path)
        elif init_infrared_from_visible:
            # Initialize infrared branch with visible weights (common KD practice)
            self._init_infrared_from_visible()
        else:
            # Initialize infrared branch from scratch (random weights)
            print("Initializing infrared branch from scratch (random weights).")
 
    def forward(self, visible_images: torch.Tensor, infrared_images: torch.Tensor, query_points: torch.Tensor = None, kd_config: dict = None):
        """
        Forward pass of the dual-branch VGGT model.

        Args:
            visible_images (torch.Tensor): Visible light input images with shape [B, S, 3, H, W], in range [0, 1].
            infrared_images (torch.Tensor): Infrared input images with shape [B, S, 3, H, W], in range [0, 1].
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            kd_config (dict, optional): Configuration for knowledge distillation,
                including 'use_logit_kd' (bool) and 'use_feature_kd' (bool).
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - visible_predictions: Predictions from the visible light branch
                - infrared_predictions: Predictions from the infrared branch
        """
        # Default KD configuration if not provided
        if kd_config is None:
            kd_config = {
                'use_logit_kd': False,
                'use_feature_kd': False
            }
        
        # Ensure batch dimension exists
        if len(visible_images.shape) == 4:
            visible_images = visible_images.unsqueeze(0)
        if len(infrared_images.shape) == 4:
            infrared_images = infrared_images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Debug: check input shapes
        if visible_images.shape[0] == 0 or infrared_images.shape[0] == 0:
            raise ValueError(f"Empty input! visible_images: {visible_images.shape}, infrared_images: {infrared_images.shape}")

        # Forward pass through visible light branch (teacher)
        visible_aggregated_tokens_list, visible_patch_start_idx = self.visible_aggregator(visible_images)
        visible_predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.visible_camera_head is not None:
                if kd_config.get('use_logit_kd', False):
                    visible_pose_enc_list, visible_pose_enc_logits_list = self.visible_camera_head(visible_aggregated_tokens_list, return_logits=True)
                    visible_predictions["pose_enc_logits_list"] = visible_pose_enc_logits_list
                else:
                    visible_pose_enc_list = self.visible_camera_head(visible_aggregated_tokens_list)
                visible_predictions["pose_enc"] = visible_pose_enc_list[-1]
                # Only store the last pose encoding to save memory
                if self.training:
                    visible_predictions["pose_enc_list"] = visible_pose_enc_list
                
            if self.visible_depth_head is not None:
                if kd_config.get('use_logit_kd', False):
                    visible_depth, visible_depth_conf, visible_depth_logits = self.visible_depth_head(
                        visible_aggregated_tokens_list, images=visible_images, patch_start_idx=visible_patch_start_idx, return_logits=True
                    )
                    visible_predictions["depth_logits"] = visible_depth_logits
                else:
                    visible_depth, visible_depth_conf = self.visible_depth_head(
                        visible_aggregated_tokens_list, images=visible_images, patch_start_idx=visible_patch_start_idx
                    )
                visible_predictions["depth"] = visible_depth
                visible_predictions["depth_conf"] = visible_depth_conf

            if self.visible_point_head is not None:
                visible_pts3d, visible_pts3d_conf = self.visible_point_head(
                    visible_aggregated_tokens_list, images=visible_images, patch_start_idx=visible_patch_start_idx
                )
                visible_predictions["world_points"] = visible_pts3d
                visible_predictions["world_points_conf"] = visible_pts3d_conf

        if self.visible_track_head is not None and query_points is not None:
            visible_track_list, visible_vis, visible_conf = self.visible_track_head(
                visible_aggregated_tokens_list, images=visible_images, patch_start_idx=visible_patch_start_idx, query_points=query_points
            )
            visible_predictions["track"] = visible_track_list[-1]
            visible_predictions["vis"] = visible_vis
            visible_predictions["conf"] = visible_conf

        if not self.training:
            visible_predictions["images"] = visible_images

        # Forward pass through infrared branch (student)
        infrared_aggregated_tokens_list, infrared_patch_start_idx = self.infrared_aggregator(infrared_images)
        infrared_predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.infrared_camera_head is not None:
                if kd_config.get('use_logit_kd', False):
                    infrared_pose_enc_list, infrared_pose_enc_logits_list = self.infrared_camera_head(infrared_aggregated_tokens_list, return_logits=True)
                    infrared_predictions["pose_enc_logits_list"] = infrared_pose_enc_logits_list
                else:
                    infrared_pose_enc_list = self.infrared_camera_head(infrared_aggregated_tokens_list)
                infrared_predictions["pose_enc"] = infrared_pose_enc_list[-1]
                # Only store the last pose encoding to save memory
                if self.training:
                    infrared_predictions["pose_enc_list"] = infrared_pose_enc_list
                
            if self.infrared_depth_head is not None:
                if kd_config.get('use_logit_kd', False):
                    infrared_depth, infrared_depth_conf, infrared_depth_logits = self.infrared_depth_head(
                        infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx, return_logits=True
                    )
                    infrared_predictions["depth_logits"] = infrared_depth_logits
                else:
                    infrared_depth, infrared_depth_conf = self.infrared_depth_head(
                        infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx
                    )
                infrared_predictions["depth"] = infrared_depth
                infrared_predictions["depth_conf"] = infrared_depth_conf

            if self.infrared_point_head is not None:
                infrared_pts3d, infrared_pts3d_conf = self.infrared_point_head(
                    infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx
                )
                infrared_predictions["world_points"] = infrared_pts3d
                infrared_predictions["world_points_conf"] = infrared_pts3d_conf

        if self.infrared_track_head is not None and query_points is not None:
            infrared_track_list, infrared_vis, infrared_conf = self.infrared_track_head(
                infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx, query_points=query_points
            )
            infrared_predictions["track"] = infrared_track_list[-1]
            infrared_predictions["vis"] = infrared_vis
            infrared_predictions["conf"] = infrared_conf

        if not self.training:
            infrared_predictions["images"] = infrared_images

        # Store aggregator features for knowledge distillation if feature KD is enabled
        if self.training and kd_config.get('use_feature_kd', False):
            visible_predictions["aggregator_features"] = visible_aggregated_tokens_list
            infrared_predictions["aggregator_features"] = infrared_aggregated_tokens_list

        # Clean up intermediate variables to save memory
        del visible_aggregated_tokens_list
        del infrared_aggregated_tokens_list

        return {
            "visible_predictions": visible_predictions,
            "infrared_predictions": infrared_predictions,
            "kd_config": kd_config
        }

    def infrared_only_forward(self, infrared_images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass using only the infrared branch (for deployment/inference).

        Args:
            infrared_images (torch.Tensor): Infrared input images with shape [B, S, 3, H, W], in range [0, 1].
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the predictions from the infrared branch.
        """
        # Ensure batch dimension exists
        if len(infrared_images.shape) == 4:
            infrared_images = infrared_images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Forward pass through infrared branch (student)
        infrared_aggregated_tokens_list, infrared_patch_start_idx = self.infrared_aggregator(infrared_images)
        infrared_predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.infrared_camera_head is not None:
                infrared_pose_enc_list = self.infrared_camera_head(infrared_aggregated_tokens_list)
                infrared_predictions["pose_enc"] = infrared_pose_enc_list[-1]
                
            if self.infrared_depth_head is not None:
                infrared_depth, infrared_depth_conf = self.infrared_depth_head(
                    infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx
                )
                infrared_predictions["depth"] = infrared_depth
                infrared_predictions["depth_conf"] = infrared_depth_conf

            if self.infrared_point_head is not None:
                infrared_pts3d, infrared_pts3d_conf = self.infrared_point_head(
                    infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx
                )
                infrared_predictions["world_points"] = infrared_pts3d
                infrared_predictions["world_points_conf"] = infrared_pts3d_conf

        if self.infrared_track_head is not None and query_points is not None:
            infrared_track_list, infrared_vis, infrared_conf = self.infrared_track_head(
                infrared_aggregated_tokens_list, images=infrared_images, patch_start_idx=infrared_patch_start_idx, query_points=query_points
            )
            infrared_predictions["track"] = infrared_track_list[-1]
            infrared_predictions["vis"] = infrared_vis
            infrared_predictions["conf"] = infrared_conf

        if not self.training:
            infrared_predictions["images"] = infrared_images

        # Clean up intermediate variables to save memory
        del infrared_aggregated_tokens_list

        return infrared_predictions

    def load_visible_weights(self, visible_model_path):
        """
        Load pre-trained weights for the visible light branch.
        
        Args:
            visible_model_path: Path to the pre-trained visible light model weights.
        """
        if visible_model_path is None or visible_model_path == "":
            print("No visible model path provided, skipping loading visible branch weights.")
            return
            
        # Load pre-trained visible model
        visible_model = VGGT()
        visible_model.load_state_dict(torch.load(visible_model_path), strict=False)
        
        # Copy weights to visible branch
        self.visible_aggregator.load_state_dict(visible_model.aggregator.state_dict())
        if self.visible_camera_head is not None:
            self.visible_camera_head.load_state_dict(visible_model.camera_head.state_dict())
        if self.visible_point_head is not None:
            self.visible_point_head.load_state_dict(visible_model.point_head.state_dict())
        if self.visible_depth_head is not None:
            self.visible_depth_head.load_state_dict(visible_model.depth_head.state_dict())
        if self.visible_track_head is not None:
            self.visible_track_head.load_state_dict(visible_model.track_head.state_dict())
        
        # Freeze visible branch parameters (teacher network is fully frozen)
        for param in self.visible_aggregator.parameters():
            param.requires_grad = False
        if self.visible_camera_head is not None:
            for param in self.visible_camera_head.parameters():
                param.requires_grad = False
        if self.visible_point_head is not None:
            for param in self.visible_point_head.parameters():
                param.requires_grad = False
        if self.visible_depth_head is not None:
            for param in self.visible_depth_head.parameters():
                param.requires_grad = False
        if self.visible_track_head is not None:
            for param in self.visible_track_head.parameters():
                param.requires_grad = False
        
        print("Loaded pre-trained weights for visible light branch and froze its parameters.")
    
    def set_freeze_parameters(self, kd_config: dict = None):
        """
        Set freezing parameters for the dual-branch model based on KD configuration:
        - Teacher network (visible branch): always fully frozen
        - Student network (infrared branch):
            - DINO encoding: always frozen
            - Prediction heads: trainable if use_logit_kd or use_task_heads is True
            - Aggregator: trainable if use_feature_kd is True

        Args:
            kd_config (dict, optional): Configuration for knowledge distillation,
                including 'use_logit_kd' (bool), 'use_feature_kd' (bool), and 'use_task_heads' (bool).
                Default: None
        """
        # Default KD configuration if not provided
        if kd_config is None:
            kd_config = {
                'use_logit_kd': False,
                'use_feature_kd': False,
                'use_task_heads': False
            }
        
        # Ensure teacher network is fully frozen
        for param in self.visible_aggregator.parameters():
            param.requires_grad = False
        if self.visible_camera_head is not None:
            for param in self.visible_camera_head.parameters():
                param.requires_grad = False
        if self.visible_point_head is not None:
            for param in self.visible_point_head.parameters():
                param.requires_grad = False
        if self.visible_depth_head is not None:
            for param in self.visible_depth_head.parameters():
                param.requires_grad = False
        if self.visible_track_head is not None:
            for param in self.visible_track_head.parameters():
                param.requires_grad = False
        
        # Freeze student's DINO encoding (patch_embed)
        if hasattr(self.infrared_aggregator, "patch_embed"):
            for param in self.infrared_aggregator.patch_embed.parameters():
                param.requires_grad = False
        
        # Freeze student's prediction heads by default
        if self.infrared_camera_head is not None:
            for param in self.infrared_camera_head.parameters():
                param.requires_grad = False
        if self.infrared_point_head is not None:
            for param in self.infrared_point_head.parameters():
                param.requires_grad = False
        if self.infrared_depth_head is not None:
            for param in self.infrared_depth_head.parameters():
                param.requires_grad = False
        if self.infrared_track_head is not None:
            for param in self.infrared_track_head.parameters():
                param.requires_grad = False
        
        # Freeze student's aggregator by default
        for name, param in self.infrared_aggregator.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False
        
        # Adjust trainable parts based on KD config
        if kd_config.get('use_feature_kd', False):
            # Make student's aggregator trainable (excluding patch_embed)
            for name, param in self.infrared_aggregator.named_parameters():
                if "patch_embed" not in name:
                    param.requires_grad = True
        
        if kd_config.get('use_logit_kd', False) or kd_config.get('use_task_heads', False):
            # Make student's prediction heads trainable
            if self.infrared_camera_head is not None:
                for param in self.infrared_camera_head.parameters():
                    param.requires_grad = True
            if self.infrared_point_head is not None:
                for param in self.infrared_point_head.parameters():
                    param.requires_grad = True
            if self.infrared_depth_head is not None:
                for param in self.infrared_depth_head.parameters():
                    param.requires_grad = True
            if self.infrared_track_head is not None:
                for param in self.infrared_track_head.parameters():
                    param.requires_grad = True
        
        print("Set freezing parameters:")
        print("- Teacher network: fully frozen")
        print("- Student network:")
        print("  - DINO encoding: frozen")
        print(f"  - Prediction heads: {'trainable' if (kd_config.get('use_logit_kd', False) or kd_config.get('use_task_heads', False)) else 'frozen'}")
        print(f"  - Aggregator: {'trainable (feature KD)' if kd_config.get('use_feature_kd', False) else 'frozen'}")

    def _init_infrared_from_visible(self):
        """Initialize infrared branch with visible branch weights (common KD practice)."""
        self.infrared_aggregator.load_state_dict(self.visible_aggregator.state_dict())
        if self.infrared_camera_head is not None and self.visible_camera_head is not None:
            self.infrared_camera_head.load_state_dict(self.visible_camera_head.state_dict())
        if self.infrared_point_head is not None and self.visible_point_head is not None:
            self.infrared_point_head.load_state_dict(self.visible_point_head.state_dict())
        if self.infrared_depth_head is not None and self.visible_depth_head is not None:
            self.infrared_depth_head.load_state_dict(self.visible_depth_head.state_dict())
        if self.infrared_track_head is not None and self.visible_track_head is not None:
            self.infrared_track_head.load_state_dict(self.visible_track_head.state_dict())
        print("Initialized infrared branch weights from visible branch.")

    def load_infrared_weights(self, infrared_model_path):
        """Load pre-trained weights for the infrared branch."""
        if infrared_model_path is None or infrared_model_path == "":
            print("No infrared model path provided, skipping.")
            return
        
        infrared_model = VGGT()
        infrared_model.load_state_dict(torch.load(infrared_model_path), strict=False)
        
        self.infrared_aggregator.load_state_dict(infrared_model.aggregator.state_dict())
        if self.infrared_camera_head is not None:
            self.infrared_camera_head.load_state_dict(infrared_model.camera_head.state_dict())
        if self.infrared_point_head is not None:
            self.infrared_point_head.load_state_dict(infrared_model.point_head.state_dict())
        if self.infrared_depth_head is not None:
            self.infrared_depth_head.load_state_dict(infrared_model.depth_head.state_dict())
        if self.infrared_track_head is not None:
            self.infrared_track_head.load_state_dict(infrared_model.track_head.state_dict())
        
        print(f"Loaded pre-trained weights for infrared branch from {infrared_model_path}")

