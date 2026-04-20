import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass(eq=False)
class StructuralDistillationLoss(torch.nn.Module):
    """
    Structural distillation loss module for dual-branch VGGT.
    
    Supports:
    - Teacher-induced structural operator and feature decomposition
    - Conditional compression surrogate (uniformization conditional contrastive loss)
    - Structure preservation surrogate
    - Task sufficiency surrogate (camera and depth loss)
    """
    def __init__(self, camera=None, depth=None, structural_distillation=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.structural_distillation = structural_distillation
        
        # Initialize critic networks
        if structural_distillation is not None:
            self.query_critic = CriticNetwork(
                dim_in=structural_distillation.get('embed_dim', 1024),
                dim_out=structural_distillation.get('critic_dim', 256),
                topk=structural_distillation.get('topk', 16)
            )
            self.key_critic = CriticNetwork(
                dim_in=structural_distillation.get('embed_dim', 1024),
                dim_out=structural_distillation.get('critic_dim', 256),
                topk=structural_distillation.get('topk', 16)
            )
            
        # Uncertainty distillation parameters
        self.use_uncertainty = structural_distillation.get('use_uncertainty', False) if structural_distillation else False
        self.uncertainty_weight = structural_distillation.get('uncertainty_weight', 1.0) if structural_distillation else 1.0
    
    def forward(self, predictions, batch) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss with structural distillation.
        
        Args:
            predictions: Dict containing model predictions for different tasks
                        from both visible and infrared branches
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Extract predictions from both branches
        visible_predictions = predictions["visible_predictions"]
        infrared_predictions = predictions["infrared_predictions"]
        
        # Camera pose loss for infrared branch (student)
        if "pose_enc_list" in infrared_predictions:
            from .loss import compute_camera_loss
            camera_loss_dict = compute_camera_loss(infrared_predictions, batch, **self.camera)
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_loss
            loss_dict.update({f"infrared_{k}": v for k, v in camera_loss_dict.items()})
        
        # Depth estimation loss for infrared branch (student)
        if "depth" in infrared_predictions:
            from .loss import compute_depth_loss
            depth_loss_dict = compute_depth_loss(infrared_predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update({f"infrared_{k}": v for k, v in depth_loss_dict.items()})
        
        # Structural distillation loss
        if self.structural_distillation is not None:
            structural_loss_dict = self.compute_structural_distillation_loss(
                visible_predictions, infrared_predictions, **self.structural_distillation
            )
            # Each sub-loss is already weighted by its individual weight in compute_structural_distillation_loss
            # We sum all weighted sub-losses to get the total structural distillation loss
            structural_loss = sum(structural_loss_dict.values())
            total_loss = total_loss + structural_loss
            loss_dict.update({f"structural_{k}": v for k, v in structural_loss_dict.items()})
        
        loss_dict["objective"] = total_loss
        
        return loss_dict
    
    def compute_structural_distillation_loss(
        self,
        teacher_predictions,
        student_predictions,
        delta=0.1,
        tau_g=0.1,
        tau=0.1,
        topk=16,
        candidate_strategy="A+B", 
        max_candidates=64,
        use_uncertainty=False,
        uncertainty_weight=1.0,
        structure_preservation_weight=1.0,
        conditional_compression_weight=1.0,
        uncertainty_distillation_weight=1.0,
        **kwargs
    ):
        """
        Compute structural distillation loss.
        
        Args:
            teacher_predictions: Dict containing predictions from visible branch (teacher)
            student_predictions: Dict containing predictions from infrared branch (student)
            delta: Diagonal suppression parameter
            tau_g: Temperature for softmax in structure-aware adjacency
            tau: Temperature for critic
            topk: Number of top elements to keep in structure summary
            candidate_strategy: Candidate set construction strategy
            max_candidates: Maximum number of candidates
            use_uncertainty: Whether to use uncertainty-aware loss
            uncertainty_weight: Weight for uncertainty distillation loss
            
        Returns:
            Dict containing individual structural distillation losses
        """
        loss_dict = {}
        
        # Get aggregator features from both teacher and student
        teacher_features = teacher_predictions.get("aggregator_features", None)
        student_features = student_predictions.get("aggregator_features", None)
        
        if teacher_features is not None and student_features is not None:
            # Use the last stage features
            teacher_feat = teacher_features[-1]  # Shape: [B, V, N, D]
            student_feat = student_features[-1]  # Shape: [B, V, N, D]
            B, V, N, D = teacher_feat.shape
            
            # Get uncertainty maps if available
            teacher_uncertainty = teacher_predictions.get("depth_uncertainty", None)
            student_uncertainty = student_predictions.get("depth_uncertainty", None)
            
            # Compute confidence map if uncertainty is available
            confidence_map = None
            if use_uncertainty and teacher_uncertainty is not None:
                # Convert uncertainty to confidence map: C_t = clip(1 - 1/U_t, 0, 1)
                teacher_uncertainty = torch.clamp(teacher_uncertainty, min=1e-6)  # Avoid division by zero
                confidence_map = torch.clamp(1 - 1/teacher_uncertainty, min=0, max=1)
                
                # Get token grid size (based on patch size)
                token_h = H // 14  # patch_size=14
                token_w = W // 14
                
                # Resize confidence map to token grid using adaptive average pooling
                # This ensures each token gets the average confidence of its corresponding patch
                confidence_map = F.adaptive_avg_pool2d(
                    confidence_map,
                    output_size=(token_h, token_w)
                )
                
                # Flatten to [B, V, N, 1]
                N = token_h * token_w
                confidence_map = confidence_map.view(B, V, N, 1)
            
            # Teacher-induced structural operator and feature decomposition
            teacher_struct, teacher_res, student_struct, student_res, S_t = self.compute_structural_decomposition(
                teacher_feat, student_feat, delta, tau_g
            )
            
            # Structure preservation loss
            if use_uncertainty and confidence_map is not None:
                struct_preservation_loss = self.compute_uncertainty_aware_structure_preservation_loss(
                    student_struct, teacher_struct, confidence_map
                )
            else:
                struct_preservation_loss = self.compute_structure_preservation_loss(
                    student_struct, teacher_struct
                )
            loss_dict["structure_preservation"] = struct_preservation_loss * structure_preservation_weight
            
            # Conditional compression loss
            if use_uncertainty and confidence_map is not None:
                cond_compression_loss = self.compute_uncertainty_aware_conditional_compression_loss(
                    student_res, teacher_res, S_t, confidence_map, tau, topk, candidate_strategy, max_candidates
                )
            else:
                cond_compression_loss = self.compute_conditional_compression_loss(
                    student_res, teacher_res, S_t, tau, topk, candidate_strategy, max_candidates
                )
            loss_dict["conditional_compression"] = cond_compression_loss * conditional_compression_weight
            
            # Uncertainty distillation loss
            if use_uncertainty and teacher_uncertainty is not None and student_uncertainty is not None:
                uncertainty_loss = self.compute_uncertainty_distillation_loss(
                    student_uncertainty, teacher_uncertainty
                )
                loss_dict["uncertainty_distillation"] = uncertainty_loss * uncertainty_distillation_weight
        
        return loss_dict
    
    def compute_structural_decomposition(
        self, 
        teacher_feat, 
        student_feat, 
        delta, 
        tau_g
    ):
        """
        Compute structural decomposition for teacher and student features.
        
        Args:
            teacher_feat: Teacher features with shape [B, V, N, D]
            student_feat: Student features with shape [B, V, N, D]
            delta: Diagonal suppression parameter
            tau_g: Temperature for softmax in structure-aware adjacency
            
        Returns:
            teacher_struct: Teacher structural features
            teacher_res: Teacher residual features
            student_struct: Student structural features
            student_res: Student residual features
            S_t: Teacher structure-aware adjacency
        """
        B, V, N, D = teacher_feat.shape
        
        # Teacher token self-similarity using matrix multiplication for better performance
        # Reshape to [B*V, N, D] for batch matrix multiplication
        teacher_feat_reshaped = teacher_feat.view(B*V, N, D)
        A_t = torch.matmul(teacher_feat_reshaped, teacher_feat_reshaped.transpose(1, 2)) / torch.sqrt(torch.tensor(D, device=teacher_feat.device))
        A_t = A_t.view(B, V, N, N)
        
        # Structure-aware adjacency
        # Subtract delta from diagonal
        eye = torch.eye(N, device=teacher_feat.device).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        A_t = A_t - delta * eye
        
        # Row-wise softmax
        S_t = F.softmax(A_t / tau_g, dim=-1)
        
        # Teacher structural / residual decomposition
        # Use matrix multiplication instead of einsum
        S_t_reshaped = S_t.view(B*V, N, N)
        teacher_struct = torch.matmul(S_t_reshaped, teacher_feat_reshaped)
        teacher_struct = teacher_struct.view(B, V, N, D)
        teacher_res = teacher_feat - teacher_struct
        teacher_res = teacher_res.detach()  # Stop gradient
        
        # Student structural / residual decomposition
        student_feat_reshaped = student_feat.view(B*V, N, D)
        student_struct = torch.matmul(S_t_reshaped, student_feat_reshaped)
        student_struct = student_struct.view(B, V, N, D)
        student_res = student_feat - student_struct
        
        # Clean up intermediate variables to save memory
        del teacher_feat_reshaped, student_feat_reshaped, S_t_reshaped, A_t, eye
        
        return teacher_struct, teacher_res, student_struct, student_res, S_t
    
    def compute_structure_preservation_loss(
        self, 
        student_struct, 
        teacher_struct
    ):
        """
        Compute structure preservation loss.
        
        Args:
            student_struct: Student structural features
            teacher_struct: Teacher structural features (stop-grad)
            
        Returns:
            Structure preservation loss
        """
        # Stop gradient for teacher structural features
        teacher_struct = teacher_struct.detach()
        
        # Token-wise MSE loss
        loss = F.mse_loss(student_struct, teacher_struct)
        
        return loss
    
    def compute_conditional_compression_loss(
        self, 
        student_res, 
        teacher_res, 
        S_t, 
        tau, 
        topk, 
        candidate_strategy, 
        max_candidates
    ):
        """
        Compute conditional compression loss.
        
        Args:
            student_res: Student residual features
            teacher_res: Teacher residual features
            S_t: Teacher structure-aware adjacency
            tau: Temperature for critic
            topk: Number of top elements to keep in structure summary
            candidate_strategy: Candidate set construction strategy
            max_candidates: Maximum number of candidates
            
        Returns:
            Conditional compression loss
        """
        B, V, N, D = student_res.shape
        total_tokens = B * V * N
        
        # Get structure summary (TopK)
        # For each token m, get topk values from S_t[b, v, m, :]
        S_t_topk, _ = torch.topk(S_t, k=topk, dim=-1, largest=True, sorted=True)
        
        # Flatten for batch processing
        student_res_flat = student_res.view(-1, D)  # [B*V*N, D]
        teacher_res_flat = teacher_res.view(-1, D)  # [B*V*N, D]
        S_t_topk_flat = S_t_topk.view(-1, topk)  # [B*V*N, K]
        
        # Ensure critic networks are on the same device as input data
        device = student_res.device
        if self.query_critic.proj.weight.device != device:
            self.query_critic = self.query_critic.to(device)
            self.key_critic = self.key_critic.to(device)
        
        # Compute query and key embeddings
        q = self.query_critic(student_res_flat, S_t_topk_flat)
        k = self.key_critic(teacher_res_flat, S_t_topk_flat)
        
        # Compute similarities
        sim = torch.matmul(q, k.t()) / tau
        
        # Generate candidate sets for all tokens in batch using batch processing
        indices = torch.arange(total_tokens, device=device)
        padded_candidates, mask = self.construct_candidates_batch(
            indices, B, V, N, candidate_strategy, max_candidates, device
        )
        
        # Filter out tokens with no valid candidates
        valid_mask = mask.sum(dim=1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)
        
        padded_candidates = padded_candidates[valid_mask]
        mask = mask[valid_mask]
        valid_indices = indices[valid_mask]
        
        # Get similarities for all candidates
        candidate_sims = sim[valid_indices.unsqueeze(1), padded_candidates]
        
        # Apply mask to ignore padding
        candidate_sims = candidate_sims.masked_fill(~mask, -float('inf'))
        
        # Compute log softmax
        log_prob = F.log_softmax(candidate_sims, dim=1)
        
        # Compute loss: -mean(log_prob) for each token, then average
        loss = -log_prob.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
        loss = loss.mean()
        
        return loss
    
    def compute_uncertainty_aware_structure_preservation_loss(
        self, 
        student_struct, 
        teacher_struct, 
        confidence_map
    ):
        """
        Compute uncertainty-aware structure preservation loss.
        
        Args:
            student_struct: Student structural features
            teacher_struct: Teacher structural features (stop-grad)
            confidence_map: Confidence map from teacher uncertainty [B, V, N, 1]
            
        Returns:
            Uncertainty-aware structure preservation loss
        """
        # Stop gradient for teacher structural features
        teacher_struct = teacher_struct.detach()
        
        # Compute token-wise MSE loss
        loss = F.mse_loss(student_struct, teacher_struct, reduction='none')
        
        # Apply confidence weighting
        loss = loss * confidence_map
        
        # Normalize by sum of confidence
        sum_confidence = confidence_map.sum()
        if sum_confidence > 0:
            loss = loss.sum() / sum_confidence
        else:
            loss = torch.tensor(0.0, device=student_struct.device)
        
        return loss
    
    def compute_uncertainty_aware_conditional_compression_loss(
        self, 
        student_res, 
        teacher_res, 
        S_t, 
        confidence_map, 
        tau, 
        topk, 
        candidate_strategy, 
        max_candidates
    ):
        """
        Compute uncertainty-aware conditional compression loss.
        
        Args:
            student_res: Student residual features
            teacher_res: Teacher residual features
            S_t: Teacher structure-aware adjacency
            confidence_map: Confidence map from teacher uncertainty [B, V, N, 1]
            tau: Temperature for critic
            topk: Number of top elements to keep in structure summary
            candidate_strategy: Candidate set construction strategy
            max_candidates: Maximum number of candidates
            
        Returns:
            Uncertainty-aware conditional compression loss
        """
        B, V, N, D = student_res.shape
        total_tokens = B * V * N
        
        # Get structure summary (TopK)
        # For each token m, get topk values from S_t[b, v, m, :]
        S_t_topk, _ = torch.topk(S_t, k=topk, dim=-1, largest=True, sorted=True)
        
        # Flatten for batch processing
        student_res_flat = student_res.view(-1, D)  # [B*V*N, D]
        teacher_res_flat = teacher_res.view(-1, D)  # [B*V*N, D]
        S_t_topk_flat = S_t_topk.view(-1, topk)  # [B*V*N, K]
        confidence_flat = confidence_map.view(-1, 1)  # [B*V*N, 1]
        
        # Ensure critic networks are on the same device as input data
        device = student_res.device
        if self.query_critic.proj.weight.device != device:
            self.query_critic = self.query_critic.to(device)
            self.key_critic = self.key_critic.to(device)
        
        # Compute query and key embeddings
        q = self.query_critic(student_res_flat, S_t_topk_flat)
        k = self.key_critic(teacher_res_flat, S_t_topk_flat)
        
        # Compute similarities
        sim = torch.matmul(q, k.t()) / tau
        
        # Generate candidate sets for all tokens in batch using batch processing
        indices = torch.arange(total_tokens, device=device)
        padded_candidates, mask = self.construct_candidates_batch(
            indices, B, V, N, candidate_strategy, max_candidates, device
        )
        
        # Filter out tokens with no valid candidates
        valid_mask = mask.sum(dim=1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)
        
        padded_candidates = padded_candidates[valid_mask]
        mask = mask[valid_mask]
        valid_indices = indices[valid_mask]
        valid_confidences = confidence_flat[valid_indices].squeeze()
        
        # Get similarities for all candidates
        candidate_sims = sim[valid_indices.unsqueeze(1), padded_candidates]
        
        # Apply mask to ignore padding
        candidate_sims = candidate_sims.masked_fill(~mask, -float('inf'))
        
        # Compute log softmax
        log_prob = F.log_softmax(candidate_sims, dim=1)
        
        # Compute loss: -mean(log_prob) for each token, weighted by confidence
        token_loss = -log_prob.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
        weighted_loss = token_loss * valid_confidences
        total_confidence = valid_confidences.sum()
        
        if total_confidence > 0:
            loss = weighted_loss.sum() / total_confidence
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def compute_uncertainty_distillation_loss(
        self, 
        student_uncertainty, 
        teacher_uncertainty
    ):
        """
        Compute uncertainty distillation loss.
        
        Args:
            student_uncertainty: Student uncertainty map
            teacher_uncertainty: Teacher uncertainty map
            
        Returns:
            Uncertainty distillation loss
        """
        # Clamp uncertainty to avoid log(0)
        student_uncertainty = torch.clamp(student_uncertainty, min=1e-6)
        teacher_uncertainty = torch.clamp(teacher_uncertainty, min=1e-6)
        
        # Compute log uncertainty
        log_student_uncertainty = torch.log(student_uncertainty)
        log_teacher_uncertainty = torch.log(teacher_uncertainty).detach()  # Stop gradient
        
        # L1 loss between log uncertainties
        loss = F.l1_loss(log_student_uncertainty, log_teacher_uncertainty)
        
        return loss
    
    def construct_candidates_batch(
        self, 
        indices, 
        B, 
        V, 
        N, 
        candidate_strategy, 
        max_candidates, 
        device
    ):
        """
        Construct candidate sets for multiple tokens in batch.
        
        Args:
            indices: Tensor of token indices (flattened) [M]
            B: Batch size
            V: Number of views
            N: Number of tokens
            candidate_strategy: Candidate set construction strategy
            max_candidates: Maximum number of candidates per token
            device: Device to use for computations
            
        Returns:
            padded_candidates: Padded tensor of candidate indices [M, max_candidate_size]
            mask: Mask tensor indicating valid candidates [M, max_candidate_size]
        """
        M = len(indices)
        
        # Convert indices to (b, v, n)
        b = indices // (V * N)
        v = (indices % (V * N)) // N
        n = indices % N
        
        # Initialize candidate sets
        all_candidates = []
        
        for i in range(M):
            candidates = set()
            idx = indices[i].item()
            current_b = b[i].item()
            current_v = v[i].item()
            current_n = n[i].item()
            
            # Add positive sample
            candidates.add(idx)
            
            if "A" in candidate_strategy:
                # Strategy A: same sample cross-view neighborhood
                for v_prime in range(V):
                    if v_prime != current_v:
                        candidates.add(current_b * V * N + v_prime * N + current_n)
            
            if "B" in candidate_strategy:
                # Strategy B: structure-similar negatives
                # Generate random candidates instead of iterating all possibilities
                # This is more efficient and avoids excessive candidates
                num_candidates_needed = max_candidates - len(candidates)
                if num_candidates_needed > 0:
                    # Generate random indices
                    random_indices = torch.randint(0, B*V*N, (num_candidates_needed * 2,), device=device)
                    # Filter out current token
                    random_indices = random_indices[random_indices != idx]
                    # Add to candidates
                    for r_idx in random_indices[:num_candidates_needed]:
                        candidates.add(r_idx.item())
            
            # Limit to max_candidates
            candidates = list(candidates)[:max_candidates]
            all_candidates.append(candidates)
        
        # Find maximum candidate set size
        max_candidate_size = max(len(c) for c in all_candidates) if all_candidates else 0
        
        # Create padded candidate tensor
        padded_candidates = torch.zeros(M, max_candidate_size, dtype=torch.long, device=device)
        mask = torch.zeros(M, max_candidate_size, dtype=torch.bool, device=device)
        
        for i, candidates in enumerate(all_candidates):
            padded_candidates[i, :len(candidates)] = torch.tensor(candidates, device=device)
            mask[i, :len(candidates)] = True
        
        return padded_candidates, mask
    
    def construct_candidates(
        self, 
        idx, 
        B, 
        V, 
        N, 
        candidate_strategy, 
        max_candidates
    ):
        """
        Construct candidate set for a given token.
        
        Args:
            idx: Current token index (flattened)
            B: Batch size
            V: Number of views
            N: Number of tokens
            candidate_strategy: Candidate set construction strategy
            max_candidates: Maximum number of candidates
            
        Returns:
            List of candidate indices
        """
        # Convert idx to (b, v, n)
        b = idx // (V * N)
        v = (idx % (V * N)) // N
        n = idx % N
        
        candidates = set()
        
        # Add positive sample (same sample, same view, same token position)
        candidates.add(idx)
        
        if "A" in candidate_strategy:
            # Strategy A: same sample cross-view neighborhood
            for v_prime in range(V):
                if v_prime != v:
                    # Same position
                    candidates.add(b * V * N + v_prime * N + n)
        
        if "B" in candidate_strategy:
            # Strategy B: structure-similar negatives by TopKNN
            # For simplicity, we'll just sample random tokens from the same batch
            # In practice, this should be replaced with actual KNN
            # Generate random candidates instead of iterating all possibilities
            num_candidates_needed = max_candidates - len(candidates)
            if num_candidates_needed > 0:
                # Generate random indices
                random_indices = torch.randint(0, B*V*N, (num_candidates_needed * 2,))
                # Filter out current token
                random_indices = random_indices[random_indices != idx]
                # Add to candidates
                for r_idx in random_indices[:num_candidates_needed]:
                    candidates.add(r_idx.item())
        
        # Limit to max_candidates
        candidates = list(candidates)[:max_candidates]
        
        return candidates


class CriticNetwork(nn.Module):
    """
    Critic network for conditional compression loss.
    """
    def __init__(self, dim_in, dim_out, topk):
        super().__init__()
        
        # Linear projection for token features
        self.proj = nn.Linear(dim_in, dim_out)
        
        # MLP for combining token features and structure summary
        self.mlp = nn.Sequential(
            nn.Linear(dim_out + topk, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def forward(self, x, s):
        """
        Forward pass of the critic network.
        
        Args:
            x: Token features [B*V*N, D]
            s: Structure summary [B*V*N, K]
            
        Returns:
            Embedded features [B*V*N, C]
        """
        # Project token features
        x = self.proj(x)
        
        # Concatenate with structure summary
        x = torch.cat([x, s], dim=1)
        
        # Pass through MLP
        x = self.mlp(x)
        
        return x
