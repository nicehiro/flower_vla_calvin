"""
Pre-VLM Vision Token Selection using Text-Tokenized Proprioceptive State.

This module implements LightVLA-style vision token selection that happens BEFORE
the VLM encoder processes the tokens. It uses text-tokenized proprioceptive state
(discretized to VLM vocabulary tokens) as the query for selecting relevant vision patches.

Key differences from post-VLM selection (proprio_vl_selector.py):
- Operates on raw vision patches from vision encoder (~576 tokens)
- Uses text-tokenized proprio (VLM embeddings) instead of MLP-encoded
- Selection happens before VLM language model, not after
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProprioTextTokenizer(nn.Module):
    """Discretizes proprio state into bins and uses VLM text embeddings.

    Following RT-2/OpenVLA approach: discretize continuous values into bins
    and reuse the last N tokens of the VLM vocabulary.
    """

    def __init__(
        self,
        vlm_embeddings: nn.Module,
        vocab_size: int,
        num_bins: int = 256,
        min_value: float = -3.0,
        max_value: float = 3.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.vlm_embeddings = vlm_embeddings
        self.vocab_size = vocab_size

        logger.info(
            f"[ProprioTextTokenizer] bins={num_bins}, range=[{min_value}, {max_value}]"
        )

    def discretize(self, values: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to bin indices [0, num_bins-1]."""
        clipped = torch.clamp(values, self.min_value, self.max_value)
        normalized = (clipped - self.min_value) / (self.max_value - self.min_value)
        bin_indices = (normalized * (self.num_bins - 1)).long()
        return bin_indices

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """Convert proprio history to VLM text embeddings.

        Args:
            proprio: [B, history_len, state_dim] - proprio history

        Returns:
            embeddings: [B, history_len * state_dim, vlm_dim] - VLM embeddings
        """
        B, T, D = proprio.shape

        # Discretize to bin indices [0, num_bins-1]
        bin_indices = self.discretize(proprio)  # [B, T, D]

        # Map to last num_bins tokens of vocabulary (like OpenVLA)
        # Higher bin index -> token closer to end of vocab
        token_ids = self.vocab_size - 1 - bin_indices  # [B, T, D]

        # Get embeddings from VLM (shared embedding layer)
        embeddings = self.vlm_embeddings(token_ids)  # [B, T, D, vlm_dim]

        # Reshape to [B, T*D, vlm_dim]
        vlm_dim = embeddings.shape[-1]
        return embeddings.view(B, T * D, vlm_dim)


class PreVLMVisionSelector(nn.Module):
    """LightVLA-style pre-VLM vision token selection.

    Uses text-tokenized proprioceptive state as the query to select relevant
    vision patches BEFORE they enter the VLM encoder. Uses binary gating where
    tokens selected by any query are kept, others are zeroed.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        use_residual: bool = True,
        gumbel_noise_start: float = 1.0,
        gumbel_noise_end: float = 0.01,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.gumbel_noise_start = gumbel_noise_start
        self.gumbel_noise_end = gumbel_noise_end
        self.register_buffer('_training_progress', torch.tensor(0.0))

        if use_residual:
            self.global_proj = nn.Linear(vision_dim, vision_dim, bias=False)

        self.query_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self._logged_compression = False

        logger.info(f"[PreVLMVisionSelector] residual={use_residual}")

    def set_training_progress(self, progress: float):
        """Update training progress for noise annealing (called from training loop)."""
        self._training_progress.fill_(min(1.0, max(0.0, progress)))

    @property
    def current_noise_scale(self) -> float:
        """Get current noise scale based on training progress."""
        progress = self._training_progress.item()
        # Cosine annealing: starts at noise_start, ends at noise_end
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.gumbel_noise_end + (self.gumbel_noise_start - self.gumbel_noise_end) * cosine_decay

    def _generate_per_vision_queries(
        self,
        proprio_embeds: torch.Tensor,
        vision_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one query per vision token via proprio-vision cross-attention.

        LightVLA-style: Q = softmax(H_vision @ H_proprio^T / sqrt(D)) @ H_proprio

        Each query Q[i] represents "what proprio information is relevant for
        vision position i", enabling position-aware token selection.

        Args:
            proprio_embeds: [B, P, D] - text-tokenized proprio embeddings (P = H*state_dim)
            vision_tokens: [B, V, D] - vision patches from vision encoder

        Returns:
            queries: [B, V, D] - one query per vision token position
        """
        B, V, D = vision_tokens.shape

        # Project proprio to vision space for better alignment
        proprio_proj = self.query_proj(proprio_embeds)  # [B, P, D]

        # Cross-attention: each vision token attends to proprio embeddings
        # [B, V, D] @ [B, D, P] -> [B, V, P]
        attn_logits = torch.einsum('bvd,bpd->bvp', vision_tokens, proprio_proj) / (D ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, V, P]

        # Generate queries: weighted sum of proprio embeddings for each vision position
        # [B, V, P] @ [B, P, D] -> [B, V, D]
        queries = torch.einsum('bvp,bpd->bvd', attn_weights, proprio_proj)

        return queries

    def forward(
        self,
        vision_tokens: torch.Tensor,
        proprio_embeds: torch.Tensor,
        return_scores: bool = False,
    ) -> torch.Tensor:
        """Select vision tokens using proprio-derived queries.

        Uses Gumbel-softmax with noise annealing for differentiable training.
        Each query selects one token via argmax; tokens selected by any query
        get indicator=1, others get 0. STE enables gradient flow.

        Args:
            vision_tokens: [B, V, D] - vision patches from _encode_image()
            proprio_embeds: [B, P, D] - text-tokenized proprio from ProprioTextTokenizer
            return_scores: If True, also return attention weights

        Returns:
            selected_tokens: [B, V+1, D] if residual else [B, V, D]
                           All tokens with unselected ones zeroed out
        """
        B, V, D = vision_tokens.shape

        # Generate per-vision-position queries
        queries = self._generate_per_vision_queries(proprio_embeds, vision_tokens)  # [B, V, D]

        # Score each vision token against its query
        # Each query q[i] scores all vision tokens to select the most relevant one
        scores = torch.einsum('bqd,bkd->bqk', queries, vision_tokens) / (D ** 0.5)  # [B, V, V]

        if self.training:
            # Add noise for exploration (decays during training via cosine schedule)
            noise_scale = self.current_noise_scale
            noisy_scores = scores + torch.rand_like(scores) * noise_scale
            soft = F.softmax(noisy_scores, dim=-1)  # [B, V, V]
            hard_indices = noisy_scores.argmax(dim=-1)  # [B, V]

            # Count-based binary indicator: which tokens were selected by any query
            device = vision_tokens.device
            selection_counts = torch.zeros(B, V, device=device, dtype=vision_tokens.dtype)
            selection_counts.scatter_add_(
                1, hard_indices,
                torch.ones(B, V, device=device, dtype=vision_tokens.dtype)
            )
            hard_indicator = (selection_counts > 0).float()  # [B, V]

            # STE: hard forward, soft backward for gradient flow
            soft_indicator = soft.sum(dim=1) / V
            indicator = hard_indicator + soft_indicator - soft_indicator.detach()

            # Apply binary gating: selected tokens pass through, others zeroed
            selected_tokens = vision_tokens * indicator.unsqueeze(-1)  # [B, V, D]
            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)

        else:
            # Inference: deterministic selection (no noise)
            hard_indices = scores.argmax(dim=-1)  # [B, V]
            device = vision_tokens.device
            selection_counts = torch.zeros(B, V, device=device, dtype=vision_tokens.dtype)
            selection_counts.scatter_add_(
                1, hard_indices,
                torch.ones(B, V, device=device, dtype=vision_tokens.dtype)
            )
            indicator = (selection_counts > 0).float()

            selected_tokens = vision_tokens * indicator.unsqueeze(-1)
            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)

        # Log compression ratio (once per run)
        if not self._logged_compression:
            num_selected = int(indicator[0].sum().item())
            logger.info(
                f"[PreVLMVisionSelector] Vision: {V} -> Selected: {num_selected} "
                f"(compression: {V / max(num_selected, 1):.1f}x)"
            )
            self._logged_compression = True

        # Add global residual to preserve complete context
        if self.use_residual:
            global_ctx = self.global_proj(vision_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]
            selected_tokens = torch.cat([selected_tokens, global_ctx], dim=1)

        if return_scores:
            return selected_tokens, soft_weights
        return selected_tokens
