"""
Proprioceptive-Guided Vision-Language Token Selection.

This module provides mechanisms for selecting the most relevant VL tokens
using proprioceptive (robot state) information as guidance.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformers import FlowerCrossAttention


class ProprioHistoryEncoder(nn.Module):
    """Encodes proprio history frames into rich query tokens for VL selection.

    Each frame is encoded independently via MLP, then optional temporal self-attention
    is applied to capture dynamics (velocity, acceleration patterns). The resulting
    tokens serve as queries for selecting relevant VL tokens.

    Args:
        state_dim: Dimension of state per frame (e.g., 15 for CALVIN)
        output_dim: Output embedding dimension (e.g., 1024 = dit_dim)
        n_heads: Number of attention heads for temporal attention
        dropout: Dropout probability
        use_temporal_attn: If True, apply self-attention across history frames
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_attn: bool = True,
    ):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn

        # Per-frame encoder: state_dim -> output_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Optional: temporal self-attention across frames for dynamics awareness
        if use_temporal_attn:
            self.temporal_attn = nn.MultiheadAttention(
                output_dim, num_heads=n_heads, dropout=dropout, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: [B, H, state_dim] - H history frames of robot state

        Returns:
            proprio_tokens: [B, H, output_dim] - encoded query tokens for VL selection
        """
        # Encode each frame independently
        frame_embeds = self.frame_encoder(state_history)  # [B, H, output_dim]

        # Optional: temporal attention for dynamics awareness
        if self.use_temporal_attn:
            attn_out, _ = self.temporal_attn(frame_embeds, frame_embeds, frame_embeds)
            frame_embeds = self.temporal_norm(frame_embeds + attn_out)

        return frame_embeds


class ProprioGuidedVLSelector(nn.Module):
    """Selects most relevant VL tokens using proprio history as query.

    Uses cross-attention to compute relevance scores between proprio history
    and VL tokens, then selects Top-K tokens plus an optional global residual
    token to preserve information.

    This enables efficient DiT cross-attention by reducing context from ~650
    VL tokens to K+1 selected tokens, while using proprio state to guide
    which visual/language features are most relevant for action prediction.

    Supports multiple selection modes for ablation studies:
        - proprio_topk: Proprio-guided top-K selection (default)
        - random: Random K tokens per sample (ablation baseline)
        - mean_pool: Mean pool all VL tokens to single token
        - max_pool: Select top-K tokens based on max feature values
        - soft_topk_ste: Straight-Through Estimator for differentiable selection
        - soft_topk_gumbel: Gumbel-Softmax for differentiable selection

    Args:
        dim: Model dimension (dit_dim, e.g., 1024)
        n_heads: Number of attention heads for cross-attention
        top_k: Number of VL tokens to select
        use_residual: If True, append a global mean-pooled token to preserve info
        attn_dropout: Dropout probability for attention
        mode: Selection mode (see above)
    """

    VALID_MODES = [
        "proprio_topk", "random", "mean_pool", "max_pool",
        "soft_topk_ste", "soft_topk_gumbel", "proprio_weighted_mean",
        "max_pool_proprio_ctx",   # Config A: max selection + proprio as context tokens
        "proprio_max_hybrid",     # Config B: hybrid max+proprio scoring + gated injection
    ]

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        top_k: int = 64,
        use_residual: bool = True,
        attn_dropout: float = 0.1,
        mode: str = "proprio_topk",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid vl_selection_mode: '{mode}'. Must be one of {self.VALID_MODES}")

        self.dim = dim
        self.top_k = top_k
        self.use_residual = use_residual
        self.mode = mode

        # Cross-attention: proprio queries attend to VL tokens
        # Only needed for proprio-based modes
        if mode in ["proprio_topk", "soft_topk_ste", "soft_topk_gumbel", "proprio_weighted_mean", "proprio_max_hybrid"]:
            self.cross_attn = FlowerCrossAttention(
                dim=dim,
                n_heads=n_heads,
                attn_pdrop=attn_dropout,
                resid_pdrop=attn_dropout,
                use_rope=False,
            )

        # Global context projection for residual connection
        # Used by proprio_topk, max_pool, soft_topk_ste, soft_topk_gumbel, and new modes
        if use_residual and mode not in ["mean_pool", "random"]:
            self.global_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )

        # Config A: max_pool_proprio_ctx - alignment layer to project proprio to VL space
        if mode == "max_pool_proprio_ctx":
            self.proprio_to_vl_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )

        # Config B: proprio_max_hybrid - additional layers for gated injection and score combination
        if mode == "proprio_max_hybrid":
            # Gated addition for proprio injection into selected VL tokens
            self.proprio_gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            # Learnable weight for combining max and proprio scores
            # sigmoid(0.0) = 0.5, giving equal initial weight to both
            self.score_alpha = nn.Parameter(torch.tensor(0.0))

        # LightVLA-style noise schedule parameters for soft_topk_gumbel mode
        if mode == "soft_topk_gumbel":
            # Noise schedule: high exploration initially, near-deterministic at end
            self.gumbel_noise_start = 1.0
            self.gumbel_noise_end = 0.01
            self.gumbel_temperature = 1.0
            self.min_tokens = max(1, top_k // 4)  # Minimum unique tokens to keep
            self.register_buffer('_training_progress', torch.tensor(0.0))

    def set_training_progress(self, progress: float):
        """
        Update training progress for noise schedule (LightVLA-style).
        Call from training loop: selector.set_training_progress(global_step / max_steps)

        Args:
            progress: float in [0, 1], where 0 = start, 1 = end of training
        """
        if hasattr(self, '_training_progress'):
            self._training_progress.fill_(min(1.0, max(0.0, progress)))

    @property
    def current_noise_scale(self) -> float:
        """
        Cosine annealing from noise_start to noise_end.

        LightVLA insight: High noise early encourages exploration of different
        token selections. Low noise late stabilizes to optimal selection.
        """
        if not hasattr(self, '_training_progress'):
            return 0.0  # No noise for non-gumbel modes
        progress = self._training_progress.item()
        # Cosine annealing: starts at noise_start, ends at noise_end
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.gumbel_noise_end + (self.gumbel_noise_start - self.gumbel_noise_end) * cosine_decay

    def _generate_per_token_queries(self, proprio_tokens: torch.Tensor, vl_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate N queries (one per VL token) via proprio-VL cross-attention.

        LightVLA-style: Q = softmax(H_v @ H_l^T / sqrt(D)) @ H_l
        Adapted for proprio: Q = softmax(vl_tokens @ proprio_tokens^T / sqrt(D)) @ proprio_tokens

        Each query Q[i] represents "what proprio information is relevant for VL position i".
        This allows each VL token to have a personalized query based on its content
        and the current robot state, enabling position-aware token selection.

        Args:
            proprio_tokens: [B, H, D] - encoded proprio history (H frames)
            vl_tokens: [B, N, D] - VL features (N tokens)

        Returns:
            queries: [B, N, D] - one query per VL token position
        """
        B, N, D = vl_tokens.shape

        # Cross-attention weights: each VL token attends to proprio history
        # [B, N, D] @ [B, D, H] -> [B, N, H]
        attn_logits = torch.einsum('bnd,bhd->bnh', vl_tokens, proprio_tokens) / (D ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, N, H]

        # Generate queries: weighted sum of proprio tokens for each VL position
        # [B, N, H] @ [B, H, D] -> [B, N, D]
        queries = torch.einsum('bnh,bhd->bnd', attn_weights, proprio_tokens)

        return queries

    def forward(
        self,
        proprio_tokens: Optional[torch.Tensor],
        vl_tokens: torch.Tensor,
        return_scores: bool = False,
    ):
        """
        Args:
            proprio_tokens: [B, H, dim] - encoded proprio history (queries)
                           Can be None for modes that don't use proprio (random, mean_pool, max_pool)
            vl_tokens: [B, N, dim] - VLM features (keys/values), N ~ 650
            return_scores: If True, also return attention scores for visualization

        Returns:
            selected_context: [B, K+1, dim], [B, K, dim], or [B, 1, dim] depending on mode
            (optional) scores: [B, N] softmax attention scores if return_scores=True
        """
        B, N, D = vl_tokens.shape
        K = min(self.top_k, N)  # Handle edge case where N < top_k

        if self.mode == "proprio_topk":
            return self._proprio_topk(proprio_tokens, vl_tokens, K, return_scores)
        elif self.mode == "random":
            return self._random_select(vl_tokens, K, return_scores)
        elif self.mode == "mean_pool":
            return self._mean_pool(vl_tokens, return_scores)
        elif self.mode == "max_pool":
            return self._max_pool(vl_tokens, K, return_scores)
        elif self.mode == "soft_topk_ste":
            return self._soft_topk_ste(proprio_tokens, vl_tokens, K, return_scores)
        elif self.mode == "soft_topk_gumbel":
            return self._soft_topk_gumbel(proprio_tokens, vl_tokens, K, return_scores)
        elif self.mode == "proprio_weighted_mean":
            return self._proprio_weighted_mean(proprio_tokens, vl_tokens, return_scores)
        elif self.mode == "max_pool_proprio_ctx":
            return self._max_pool_proprio_ctx(proprio_tokens, vl_tokens, K, return_scores)
        elif self.mode == "proprio_max_hybrid":
            return self._proprio_max_hybrid(proprio_tokens, vl_tokens, K, return_scores)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _proprio_topk(self, proprio_tokens, vl_tokens, K, return_scores):
        """Proprio-guided top-K selection (default behavior)."""
        B, N, D = vl_tokens.shape

        # Cross-attention: proprio tokens attend to all VL tokens
        attended = self.cross_attn(proprio_tokens, vl_tokens)  # [B, H, D]

        # Compute relevance scores
        proprio_summary = attended.mean(dim=1)  # [B, D]
        scores = torch.einsum('bd,bnd->bn', proprio_summary, vl_tokens)  # [B, N]
        scores = scores / (D ** 0.5)
        scores_softmax = F.softmax(scores, dim=-1)

        # Top-K selection
        _, topk_indices = scores_softmax.topk(K, dim=-1)  # [B, K]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(vl_tokens, dim=1, index=topk_indices_expanded)

        # Add residual
        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))
            selected_context = torch.cat([selected_tokens, global_ctx], dim=1)
        else:
            selected_context = selected_tokens

        if return_scores:
            return selected_context, scores_softmax
        return selected_context

    def _random_select(self, vl_tokens, K, return_scores):
        """Random selection - different random tokens for each sample in batch."""
        B, N, D = vl_tokens.shape

        # Generate random indices per sample
        random_indices = torch.stack([
            torch.randperm(N, device=vl_tokens.device)[:K] for _ in range(B)
        ])  # [B, K]
        random_indices_expanded = random_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(vl_tokens, dim=1, index=random_indices_expanded)

        # No residual for random mode (doesn't make sense to add learned projection)
        if return_scores:
            # Return uniform scores for compatibility
            uniform_scores = torch.ones(B, N, device=vl_tokens.device) / N
            return selected_tokens, uniform_scores
        return selected_tokens

    def _mean_pool(self, vl_tokens, return_scores):
        """Mean pool all VL tokens to single token."""
        B, N, D = vl_tokens.shape

        # Simple mean pooling
        pooled = vl_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]

        if return_scores:
            uniform_scores = torch.ones(B, N, device=vl_tokens.device) / N
            return pooled, uniform_scores
        return pooled

    def _proprio_weighted_mean(self, proprio_tokens, vl_tokens, return_scores):
        """Proprio-guided weighted mean - quality over quantity.

        Instead of hard top-K selection, compute attention-weighted mean of all
        VL tokens using proprio-guided relevance scores. Outputs a single token
        that captures proprio-relevant information from the entire VL context.

        Args:
            proprio_tokens: [B, H, D] - encoded proprio history
            vl_tokens: [B, N, D] - VL features
            return_scores: If True, also return attention weights

        Returns:
            weighted_mean: [B, 1, D] - single proprio-guided context token
            (optional) attn_weights: [B, N] - attention weights over VL tokens
        """
        B, N, D = vl_tokens.shape

        # Cross-attention: proprio tokens attend to all VL tokens
        attended = self.cross_attn(proprio_tokens, vl_tokens)  # [B, H, D]

        # Compute relevance scores
        proprio_summary = attended.mean(dim=1)  # [B, D]
        scores = torch.einsum('bd,bnd->bn', proprio_summary, vl_tokens)  # [B, N]
        scores = scores / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, N]

        # Weighted mean instead of hard selection
        weighted_mean = torch.einsum('bn,bnd->bd', attn_weights, vl_tokens)  # [B, D]
        weighted_mean = weighted_mean.unsqueeze(1)  # [B, 1, D]

        if return_scores:
            return weighted_mean, attn_weights
        return weighted_mean

    def _max_pool(self, vl_tokens, K, return_scores):
        """Select top-K tokens based on max feature values (L-inf norm as score)."""
        B, N, D = vl_tokens.shape

        # Compute score as max absolute value per token (L-inf norm)
        scores = vl_tokens.abs().max(dim=-1)[0]  # [B, N]
        scores_softmax = F.softmax(scores, dim=-1)

        # Top-K selection based on max scores
        _, topk_indices = scores.topk(K, dim=-1)  # [B, K]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(vl_tokens, dim=1, index=topk_indices_expanded)

        # Add residual
        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))
            selected_context = torch.cat([selected_tokens, global_ctx], dim=1)
        else:
            selected_context = selected_tokens

        if return_scores:
            return selected_context, scores_softmax
        return selected_context

    def _soft_topk_ste(self, proprio_tokens, vl_tokens, K, return_scores):
        """Straight-Through Estimator: hard forward, soft backward.

        Proper STE implementation:
        - Forward: hard top-K selection (unscaled tokens)
        - Backward: gradients flow through soft attention weights to ALL N tokens

        The key insight: we need gradients to flow to all N token scores, not just
        the K selected ones. This allows the model to learn which tokens SHOULD
        have been selected (increase their scores) vs which shouldn't (decrease).

        We achieve this by computing a soft weighted sum over ALL tokens for the
        backward pass, while using hard top-K selection for the forward pass.
        """
        B, N, D = vl_tokens.shape

        # Compute scores (same as proprio_topk)
        attended = self.cross_attn(proprio_tokens, vl_tokens)
        proprio_summary = attended.mean(dim=1)
        scores = torch.einsum('bd,bnd->bn', proprio_summary, vl_tokens)
        scores = scores / (D ** 0.5)
        scores_softmax = F.softmax(scores, dim=-1)  # [B, N]

        if self.training:
            temperature = 1.0
            mask_value = torch.finfo(scores.dtype).min
            logits = scores  # [B, N]

            selected = []
            logits_work = logits
            for _ in range(K):
                probs_soft = F.softmax(logits_work / temperature, dim=-1)  # [B, N]
                idx = probs_soft.argmax(dim=-1)  # [B]
                onehot = F.one_hot(idx, num_classes=N).type_as(probs_soft)  # [B, N]
                st_probs = onehot + (probs_soft - probs_soft.detach())  # [B, N]

                token = torch.einsum("bn,bnd->bd", st_probs, vl_tokens)  # [B, D]
                selected.append(token)

                logits_work = logits_work.masked_fill(onehot.bool(), mask_value)

            selected_tokens = torch.stack(selected, dim=1)  # [B, K, D]
        else:
            # Inference: standard hard top-K selection
            _, topk_indices = scores_softmax.topk(K, dim=-1)  # [B, K]
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            selected_tokens = torch.gather(vl_tokens, dim=1, index=topk_indices_expanded)  # [B, K, D]

        # Add residual
        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))
            selected_context = torch.cat([selected_tokens, global_ctx], dim=1)
        else:
            selected_context = selected_tokens

        if return_scores:
            return selected_context, scores_softmax
        return selected_context

    def _soft_topk_gumbel(self, proprio_tokens, vl_tokens, K, return_scores):
        """Differentiable binary gating with Gumbel-Softmax (LightVLA-inspired).

        Each query selects one token via argmax. Tokens selected by any query get
        indicator=1, others get 0. Uses STE for gradient flow through soft attention.

        Args:
            proprio_tokens: [B, T, D] - Projected proprio states
            vl_tokens: [B, L, D] - Fused VL tokens
            K: Unused (kept for API compatibility)
            return_scores: Whether to return attention scores

        Returns:
            selected_tokens: [B, L, D] - All tokens with unselected zeroed out
        """
        _ = K  # Unused
        B, L, D = vl_tokens.shape

        # Query generation & scoring
        queries = self._generate_per_token_queries(proprio_tokens, vl_tokens)  # [B, L, D]
        scores = torch.einsum('bqd,bkd->bqk', queries, vl_tokens) / (D ** 0.5)  # [B, L, L]

        if self.training:
            # Add noise for exploration (decays during training)
            noise_scale = self.current_noise_scale
            noisy_scores = scores + torch.rand_like(scores) * noise_scale
            soft = F.softmax(noisy_scores, dim=-1)  # [B, L, L]
            hard_indices = noisy_scores.argmax(dim=-1)  # [B, L]

            # Count-based binary indicator
            device = vl_tokens.device
            selection_counts = torch.zeros(B, L, device=device, dtype=vl_tokens.dtype)
            selection_counts.scatter_add_(1, hard_indices, torch.ones(B, L, device=device, dtype=vl_tokens.dtype))
            hard_indicator = (selection_counts > 0).float()  # [B, L]

            # STE: hard forward, soft backward
            soft_indicator = soft.sum(dim=1) / L
            indicator = hard_indicator + soft_indicator - soft_indicator.detach()

            selected_tokens = vl_tokens * indicator.unsqueeze(-1)  # [B, L, D]
            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)

        else:
            # Inference: deterministic binary gating
            hard_indices = scores.argmax(dim=-1)
            device = vl_tokens.device
            selection_counts = torch.zeros(B, L, device=device, dtype=vl_tokens.dtype)
            selection_counts.scatter_add_(1, hard_indices, torch.ones(B, L, device=device, dtype=vl_tokens.dtype))
            indicator = (selection_counts > 0).float()

            selected_tokens = vl_tokens * indicator.unsqueeze(-1)
            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)

        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]
            selected_tokens = torch.cat([selected_tokens, global_ctx], dim=1)

        if return_scores:
            return selected_tokens, soft_weights
        return selected_tokens

    def _max_pool_proprio_ctx(self, proprio_tokens, vl_tokens, K, return_scores):
        """
        Config A: Max-pool selection + proprio as separate context tokens.

        - Uses L-inf norm for VL token selection (proven effective)
        - Projects proprio tokens to VL space for better alignment
        - Adds global residual for completeness

        Output: [B, K + H + 1, D] where H is proprio history length
        """
        B, N, D = vl_tokens.shape
        H = proprio_tokens.shape[1]  # proprio history length

        # === Step 1: Max-pool selection (same as _max_pool) ===
        scores = vl_tokens.abs().max(dim=-1)[0]  # [B, N]
        scores_softmax = F.softmax(scores, dim=-1)

        _, topk_indices = scores.topk(K, dim=-1)  # [B, K]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(vl_tokens, dim=1, index=topk_indices_expanded)  # [B, K, D]

        # === Step 2: Project proprio tokens to VL space for alignment ===
        # proprio_tokens: [B, H, D] - encoded by ProprioHistoryEncoder
        # Project to align with VL feature distribution
        proprio_aligned = self.proprio_to_vl_proj(proprio_tokens)  # [B, H, D]

        # === Step 3: Add global residual ===
        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]
            # Concatenate: [selected_vl, proprio_aligned, global_residual]
            selected_context = torch.cat([selected_tokens, proprio_aligned, global_ctx], dim=1)  # [B, K+H+1, D]
        else:
            selected_context = torch.cat([selected_tokens, proprio_aligned], dim=1)  # [B, K+H, D]

        if return_scores:
            return selected_context, scores_softmax
        return selected_context

    def _proprio_max_hybrid(self, proprio_tokens, vl_tokens, K, return_scores):
        """
        Config B: Hybrid scoring (max + proprio) + gated proprio injection.

        - Combines L-inf norm scores with proprio-guided attention scores
        - Uses gated addition to inject proprio info into selected features
        - Learnable alpha balances the two scoring methods

        Output: [B, K + 1, D]
        """
        B, N, D = vl_tokens.shape

        # === Step 1: Compute max-pool scores (L-inf norm) ===
        max_scores = vl_tokens.abs().max(dim=-1)[0]  # [B, N]
        # Normalize to [0, 1] range for fair combination
        max_scores_norm = max_scores / (max_scores.max(dim=-1, keepdim=True)[0] + 1e-8)

        # === Step 2: Compute proprio-guided scores ===
        attended = self.cross_attn(proprio_tokens, vl_tokens)  # [B, H, D]
        proprio_summary = attended.mean(dim=1)  # [B, D]
        proprio_scores = torch.einsum('bd,bnd->bn', proprio_summary, vl_tokens)  # [B, N]
        proprio_scores = proprio_scores / (D ** 0.5)
        proprio_scores_norm = F.softmax(proprio_scores, dim=-1)  # [B, N]

        # === Step 3: Combine scores with learnable alpha ===
        alpha = self.score_alpha.sigmoid()  # constrain to [0, 1]
        combined_scores = (1 - alpha) * max_scores_norm + alpha * proprio_scores_norm

        # === Step 4: Top-K selection based on combined scores ===
        _, topk_indices = combined_scores.topk(K, dim=-1)  # [B, K]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(vl_tokens, dim=1, index=topk_indices_expanded)  # [B, K, D]

        # === Step 5: Gated proprio injection ===
        proprio_broadcast = proprio_summary.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        gate = self.proprio_gate(proprio_broadcast)  # [B, K, D], values in [0, 1]
        fused = selected_tokens + gate * proprio_broadcast  # gated addition

        # === Step 6: Add global residual ===
        if self.use_residual:
            global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]
            selected_context = torch.cat([fused, global_ctx], dim=1)  # [B, K+1, D]
        else:
            selected_context = fused

        if return_scores:
            return selected_context, combined_scores
        return selected_context
