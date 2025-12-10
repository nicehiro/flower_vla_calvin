"""
FlowerSubgoalVLA: Extension of FLOWERVLA with proprio history injection and subgoal prediction head.

Architecture:
- Proprio history tokens injected into Florence-2 encoder for bidirectional attention
- Separate 12-layer DiT predicts future proprio states (subgoals)
- Subgoals condition the action head via AdaLN
- Two-stage training: Stage 1 uses GT subgoals, Stage 2 uses predicted
"""

import logging
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from flower.models.flower import FLOWERVLA
from flower.models.networks.transformers import (
    FlowBlock,
    TimestepEmbedder,
    RmsNorm,
    SharedAdaLNController,
    stateless_norm,
)

logger = logging.getLogger(__name__)


class ProprioHistoryEncoder(nn.Module):
    """Encode proprio history into tokens for VLM injection."""

    def __init__(self, proprio_dims: int, token_dim: int, history_len: int):
        super().__init__()
        self.history_len = history_len
        self.frame_proj = Mlp(
            in_features=proprio_dims,
            hidden_features=token_dim,
            out_features=token_dim,
            bias=True
        )

    def forward(self, proprio_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio_history: (B, N, proprio_dims) - N frames of proprio state
        Returns:
            (B, N, token_dim) - N tokens for VLM injection
        """
        B, N, D = proprio_history.shape
        flat = proprio_history.view(B * N, D)
        projected = self.frame_proj(flat)
        return projected.view(B, N, -1)


class SubgoalCondEncoder(nn.Module):
    """Encode subgoal trace for action head conditioning via AdaLN."""

    def __init__(self, proprio_dims: int, dit_dim: int):
        super().__init__()
        self.proj = Mlp(
            in_features=proprio_dims,
            hidden_features=dit_dim,
            out_features=dit_dim,
            bias=True
        )

    def forward(self, subgoal_trace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subgoal_trace: (B, subgoal_horizon, proprio_dims)
        Returns:
            (B, subgoal_horizon, dit_dim)
        """
        B, H, D = subgoal_trace.shape
        flat = subgoal_trace.view(B * H, D)
        projected = self.proj(flat)
        return projected.view(B, H, -1)


class SubgoalDiT(nn.Module):
    """Separate DiT for predicting future proprio states (subgoals)."""

    def __init__(
        self,
        proprio_dims: int,
        dit_dim: int,
        n_heads: int,
        n_layers: int,
        subgoal_horizon: int,
        vlm_hidden_dim: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        use_cross_attn: bool = True,
        use_rope: bool = True,
        query_seq_len: int = 100,
        rope_theta: float = 32.0,
    ):
        super().__init__()
        self.proprio_dims = proprio_dims
        self.dit_dim = dit_dim
        self.subgoal_horizon = subgoal_horizon

        # Encoder/decoder for proprio space
        self.subgoal_encoder = Mlp(
            in_features=proprio_dims,
            hidden_features=dit_dim,
            out_features=dit_dim,
            bias=True
        )
        self.subgoal_decoder = nn.Linear(dit_dim, proprio_dims)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(dit_dim)

        # Conditioning projection
        self.cond_linear = nn.Linear(vlm_hidden_dim, dit_dim, bias=False)
        self.cond_norm = RmsNorm(vlm_hidden_dim)

        # AdaLN controller
        self.adaln = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

        # Positional encoding
        self.use_rope = use_rope
        if not use_rope:
            self.positional_encoding = nn.Parameter(torch.randn(1, subgoal_horizon, dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_pdrop=mlp_pdrop,
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                rope_theta=rope_theta,
            ) for _ in range(n_layers)
        ])

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: Noisy subgoal trace (B, subgoal_horizon, proprio_dims)
            t: Timestep (B,)
            cond: VLM features (B, seq_len, vlm_hidden_dim)
        Returns:
            Predicted velocity field (B, subgoal_horizon, proprio_dims)
        """
        B = z.shape[0]

        # Encode to latent space
        z_flat = z.view(B * self.subgoal_horizon, self.proprio_dims)
        z_encoded = self.subgoal_encoder(z_flat).view(B, self.subgoal_horizon, self.dit_dim)

        # Add positional encoding if not using RoPE
        if not self.use_rope:
            z_encoded = z_encoded + self.positional_encoding

        # Process conditioning
        cond = self.cond_linear(self.cond_norm(cond))

        # Timestep embedding
        t_emb = self.t_embedder(t)

        # Global conditioning (mean pool VLM features + timestep)
        vlm_token = cond.mean(dim=1)
        global_cond = vlm_token + t_emb

        # Get AdaLN signals
        global_adaln = self.adaln(global_cond)

        # Process through DiT blocks
        cx = z_encoded
        for layer in self.dit:
            cx = layer(
                cx,
                global_cond,
                context=cond,
                is_causal=True,
                global_adaln=global_adaln
            )

        # Decode to proprio space
        cx_flat = cx.view(B * self.subgoal_horizon, self.dit_dim)
        output = self.subgoal_decoder(cx_flat).view(B, self.subgoal_horizon, self.proprio_dims)

        return output


class FlowerSubgoalVLA(FLOWERVLA):
    """
    FLOWERVLA with proprio history injection and subgoal prediction head.

    Key additions:
    - ProprioHistoryEncoder: Encodes past proprio states as tokens for VLM
    - SubgoalDiT: Predicts future proprio states (subgoals) from VLM features
    - SubgoalCondEncoder: Encodes subgoals for action head conditioning
    """

    def __init__(
        self,
        # Proprio History Configuration
        use_proprio_history: bool = True,
        proprio_history_len: int = 5,
        proprio_dims: int = 7,

        # Subgoal Configuration
        use_subgoal_head: bool = True,
        subgoal_horizon: int = 3,
        subgoal_interval: int = 5,
        subgoal_dit_dim: int = 1024,
        subgoal_n_layers: int = 12,
        subgoal_n_heads: int = 16,

        # Training Configuration
        training_stage: int = 1,
        subgoal_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,

        # Parent class args
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(**kwargs)

        # Store subgoal config
        self.use_proprio_history = use_proprio_history
        self.proprio_history_len = proprio_history_len
        self.proprio_dims = proprio_dims
        self.use_subgoal_head = use_subgoal_head
        self.subgoal_horizon = subgoal_horizon
        self.subgoal_interval = subgoal_interval
        self.training_stage = training_stage
        self.subgoal_loss_weight = subgoal_loss_weight
        self.action_loss_weight = action_loss_weight

        # Get VLM hidden dim
        hidden_dim = self.vlm.config.text_config.d_model

        # Setup proprio history encoder
        if use_proprio_history:
            self.proprio_history_encoder = ProprioHistoryEncoder(
                proprio_dims=proprio_dims,
                token_dim=hidden_dim,
                history_len=proprio_history_len,
            )

        # Setup subgoal head
        if use_subgoal_head:
            self.subgoal_dit = SubgoalDiT(
                proprio_dims=proprio_dims,
                dit_dim=subgoal_dit_dim,
                n_heads=subgoal_n_heads,
                n_layers=subgoal_n_layers,
                subgoal_horizon=subgoal_horizon,
                vlm_hidden_dim=hidden_dim,
                attn_pdrop=kwargs.get('attn_pdrop', 0.1),
                resid_pdrop=kwargs.get('resid_pdrop', 0.1),
                mlp_pdrop=kwargs.get('mlp_pdrop', 0.1),
                use_cross_attn=kwargs.get('use_cross_attn', True),
                use_rope=kwargs.get('use_rope', True),
                query_seq_len=kwargs.get('query_seq_len', 100),
                rope_theta=kwargs.get('rope_theta', 32.0),
            )
            self.subgoal_cond_encoder = SubgoalCondEncoder(
                proprio_dims=proprio_dims,
                dit_dim=self.dit_dim,
            )

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Encode observations with proprio history injection into VLM."""
        device = self.device
        default_type = next(self.parameters()).dtype

        embed_tensor = torch.zeros(len(batch["rgb_obs"]['rgb_static']), 1, 1)
        action_type_tensor = torch.ones(len(batch["rgb_obs"]['rgb_static']), self.act_window_size, 7)

        # Process primary image
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape

        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        # Get text embeddings
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)

        # Add task prompt
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)

        # Encode proprio history tokens if enabled
        if self.use_proprio_history and 'proprio_history' in batch:
            proprio_history = batch['proprio_history'].to(device).to(default_type)
            proprio_tokens = self.proprio_history_encoder(proprio_history)
        else:
            proprio_tokens = None

        # Merge sequence - proprio tokens FIRST (robot knows self, then observes)
        if proprio_tokens is not None:
            merged_embeds = torch.cat([
                proprio_tokens,
                image_features,
                task_prompt,
                text_embeds.to(image_features.device)
            ], dim=1)
        else:
            merged_embeds = torch.cat([
                image_features,
                task_prompt,
                text_embeds.to(image_features.device)
            ], dim=1)

        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)

        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout
        features = self.vlm_token_dropout(features)

        # Prepare frequency embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )

        # Get proprioception if enabled
        proprio = None
        if self.use_proprio and 'proprio' in batch:
            proprio = batch['proprio'].to(device).to(default_type)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': torch.ones_like(action_type_tensor),
            'proprio': proprio,
            'attention_mask': attention_mask,
        }

    def rf_loss_subgoal(self, cond: Dict, gt_subgoal_trace: torch.Tensor) -> torch.Tensor:
        """
        Compute rectified flow loss for subgoal prediction.

        Args:
            cond: Conditioning dict from encode_observations
            gt_subgoal_trace: Ground truth future proprio states (B, subgoal_horizon, proprio_dims)
        """
        default_dtype = next(self.parameters()).dtype

        subgoals = gt_subgoal_trace.to(self.device).to(default_dtype)
        b = subgoals.size(0)
        device = subgoals.device

        # Sample time (same strategy as action head)
        if self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)

        # Interpolate between subgoals and noise
        texp = t.view([b] + [1] * (subgoals.dim() - 1))
        z1 = torch.randn_like(subgoals, device=device).to(default_dtype)
        zt = (1 - texp) * subgoals + texp * z1

        # Forward through subgoal DiT
        vtheta = self.subgoal_dit(zt, t, cond['features'])

        # Compute loss
        diff = (z1 - subgoals) - vtheta
        loss = (diff ** 2).mean()

        return loss

    def sample_subgoals(self, z: torch.Tensor, cond: Dict, num_steps: int = 4) -> torch.Tensor:
        """
        Sample subgoals using Euler method.

        Args:
            z: Initial noise (B, subgoal_horizon, proprio_dims)
            cond: Conditioning dict
            num_steps: Number of denoising steps
        Returns:
            Predicted subgoal trace (B, subgoal_horizon, proprio_dims)
        """
        device = z.device
        b = z.size(0)
        dt = 1.0 / num_steps

        for i in range(num_steps, 0, -1):
            t_val = i / num_steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Predict velocity
            v = self.subgoal_dit(z, t_tensor, cond['features'])

            # Euler step
            z = z - dt * v

        return z.clamp(-1, 1)

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict,
                    subgoal_trace: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the DiT blocks with subgoal conditioning.
        """
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)

        # Handle proprioception
        if self.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        # Handle subgoal conditioning
        if self.use_subgoal_head and subgoal_trace is not None:
            subgoal_encoded = self.subgoal_cond_encoder(subgoal_trace.to(default_dtype))
            subgoal_emb = subgoal_encoded.mean(dim=1)  # Pool over horizon
        else:
            subgoal_emb = torch.zeros_like(frequency_embeds).squeeze(1)

        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)

        # Add positional encoding if not using ROPE/NOPE
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        # Process embeddings - now includes subgoal
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1) + \
                stateless_norm(subgoal_emb)  # NEW: subgoal conditioning

        cond = self.cond_linear(self.cond_norm(cond))

        # Set up conditioning
        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        # Setup context
        cx = z
        context = cond if self.use_cross_attn else None

        # Get adaln signals
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        # Process through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx,
                global_cond,
                context=context,
                is_causal=True,
                global_adaln=global_adaln
            )

        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims)

    def rf_loss(self, cond, actions, dataset_idx=None, subgoal_trace: torch.Tensor = None):
        """
        Compute the rectified flow loss with subgoal conditioning.
        """
        default_dtype = next(self.parameters()).dtype

        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time based on sampling strategy
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)

        # Interpolate
        zt = (1 - texp) * actions + texp * z1

        # Forward pass with subgoal conditioning
        vtheta = self.dit_forward(zt, t, cond, subgoal_trace=subgoal_trace)

        # Compute loss
        diff = (z1 - actions) - vtheta
        loss = (diff ** 2).mean()

        losses_dict = {
            "diff_min": diff.min().item(),
            "diff_max": diff.max().item(),
            "diff_mean": diff.mean().item(),
            "loss": loss.item(),
        }

        return loss, losses_dict

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step with two-stage subgoal training."""
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        subgoal_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)

            # Get ground truth subgoal trace if using subgoal head
            gt_subgoal_trace = None
            pred_subgoal_trace = None

            if self.use_subgoal_head and 'subgoal_trace' in dataset_batch:
                gt_subgoal_trace = dataset_batch['subgoal_trace'].to(self.device)

                # Compute subgoal loss
                sg_loss = self.rf_loss_subgoal(obs_features, gt_subgoal_trace)
                subgoal_loss = subgoal_loss + sg_loss

                # For stage 2, predict subgoals
                if self.training_stage == 2:
                    noise = torch.randn_like(gt_subgoal_trace)
                    pred_subgoal_trace = self.sample_subgoals(noise, obs_features)

            # Determine which subgoal trace to use for action head
            if self.use_subgoal_head:
                if self.training_stage == 1:
                    # Stage 1: use ground truth subgoals
                    subgoal_for_action = gt_subgoal_trace
                else:
                    # Stage 2: use predicted subgoals (gradients flow through)
                    subgoal_for_action = pred_subgoal_trace
            else:
                subgoal_for_action = None

            # Compute action loss
            act_loss, losses_dict = self.rf_loss(
                obs_features,
                dataset_batch["actions"],
                subgoal_trace=subgoal_for_action
            )
            action_loss = action_loss + act_loss
            total_bs = total_bs + len(dataset_batch["actions"])

        # Combine losses
        total_loss = self.action_loss_weight * action_loss
        if self.use_subgoal_head:
            total_loss = total_loss + self.subgoal_loss_weight * subgoal_loss

        total_loss = total_loss / len(batch)

        # Log metrics
        self._log_training_metrics(total_loss, action_loss, total_bs)
        if self.use_subgoal_head:
            self.log("train/subgoal_loss", subgoal_loss / len(batch), on_step=True, prog_bar=True)

        return total_loss

    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor],
                       inference: bool = False, subgoal_trace: torch.Tensor = None):
        """
        Sample actions using Euler method with subgoal conditioning.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        # Integration
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1]*(z.dim()-1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Predict velocity field with subgoal conditioning
            vc = self.dit_forward(z, t_tensor, cond, subgoal_trace=subgoal_trace)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """Forward pass for inference with subgoal prediction."""
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        # Create batch for observation encoding
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }

        # Add proprio history if available
        if self.use_proprio_history and 'proprio_history' in obs:
            batch['proprio_history'] = obs['proprio_history']

        features = self.encode_observations(batch)

        # Generate initial noise for action sampling
        noise = torch.randn(
            len(features['features']),
            self.act_window_size,
            self.action_dim,
            device=features['features'].device
        )

        # Predict subgoals if using subgoal head
        subgoal_trace = None
        if self.use_subgoal_head:
            subgoal_noise = torch.randn(
                (1, self.subgoal_horizon, self.proprio_dims),
                device=self.device
            )
            subgoal_trace = self.sample_subgoals(subgoal_noise, features)

        # Sample actions with subgoal conditioning
        return self.sample_actions(noise, features, inference=True, subgoal_trace=subgoal_trace)
