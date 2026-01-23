"""
Pre-VLM Vision Token Selection using Text-Tokenized Queries.

This module implements LightVLA-style vision token selection that happens BEFORE
the VLM encoder processes the tokens. Supports multiple query sources:
- text: uses task/instruction text embeddings (original LightVLA)
- proprio: uses text-tokenized proprioceptive state
- text_proprio: concatenates both query sources

Key differences from post-VLM selection (proprio_vl_selector.py):
- Operates on raw vision patches from vision encoder (~576 tokens)
- Uses text-tokenized queries (VLM embeddings) instead of MLP-encoded
- Selection happens before VLM language model, not after
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm

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

    Selects relevant vision patches BEFORE they enter the VLM encoder using
    text-tokenized queries. Supports multiple query sources for ablation:
    - text: task/instruction text embeddings (original LightVLA)
    - proprio: text-tokenized proprioceptive state
    - text_proprio: concatenates both query sources

    Implements exact LightVLA equations:
        H_v = RMSNorm(vision_tokens)
        H_l = RMSNorm(query_embeds)
        Q = softmax(H_v @ H_l^T / √D) @ H_l
        Q = RMSNorm(Q)
        Score = Q @ H_v^T / √D

    Selected tokens are kept, unselected are removed (not zeroed).
    """

    VALID_MODES = ("text", "proprio", "text_proprio")

    def __init__(
        self,
        vision_dim: int = 1024,
        selection_mode: str = "proprio",
        use_residual: bool = True,
        gumbel_noise_start: float = 1.0,
        gumbel_noise_end: float = 0.01,
    ):
        super().__init__()
        if selection_mode not in self.VALID_MODES:
            raise ValueError(
                f"selection_mode must be one of {self.VALID_MODES}, got '{selection_mode}'"
            )
        self.selection_mode = selection_mode
        self.use_residual = use_residual
        self.gumbel_noise_start = gumbel_noise_start
        self.gumbel_noise_end = gumbel_noise_end
        self.register_buffer('_training_progress', torch.tensor(0.0))

        if use_residual:
            self.global_proj = nn.Linear(vision_dim, vision_dim, bias=False)

        # LightVLA-style normalization
        self.vision_norm = RmsNorm(vision_dim, eps=1e-6)
        self.query_norm = RmsNorm(vision_dim, eps=1e-6)
        self.query_out_norm = RmsNorm(vision_dim, eps=1e-6)  # Norm after query generation
        self._logged_compression = False
        self._step_counter = 0
        self._diagnostics = {}
        self._last_selection_counts = None  # Cache for visualization (normalized selection histogram)
        self._last_hard_mask = None  # Binary mask for visualization (exactly which tokens were selected)
        self._last_soft_weights = None  # Soft attention weights for smooth heatmap visualization

        logger.info(
            f"[PreVLMVisionSelector] mode={selection_mode}, residual={use_residual}"
        )

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

    def get_visualization_data(self) -> dict:
        """Return data for attention visualization (call after forward).

        Returns a dictionary with:
        - soft_weights: [B, V] continuous attention weights for smooth heatmap visualization
        - selection_mask: [B, V] binary mask (True = token was selected)
        - selection_counts: [B, V] normalized selection histogram (how many times each token was selected)
        - num_selected: number of tokens selected
        - num_total: total number of vision tokens
        - compression_ratio: num_selected / num_total
        """
        return {
            "soft_weights": self._last_soft_weights,  # Continuous weights [B, V]
            "selection_mask": self._last_hard_mask,  # Binary mask [B, V]
            "selection_counts": self._last_selection_counts,  # Keep for debugging
            "num_selected": getattr(self, "_last_num_selected", 0),
            "num_total": getattr(self, "_last_num_total", 0),
            "compression_ratio": getattr(self, "_last_compression_ratio", 1.0),
        }

    def _compute_diagnostics(
        self,
        queries: torch.Tensor,
        scores: torch.Tensor,
        selection_counts: torch.Tensor,
        query_embeds: torch.Tensor,
        V: int,
        num_selected: int,
    ) -> None:
        """Compute diagnostic metrics for debugging selection behavior.

        Called every 100 steps during training to track:
        - Query similarity (are generated queries too similar?)
        - Selection distribution (are selections concentrated on few tokens?)
        - Score distribution (are scores too peaked?)
        """
        with torch.no_grad():
            diag = {}

            # Basic stats
            diag["pre_vlm/V"] = V
            diag["pre_vlm/P"] = query_embeds.shape[1]
            diag["pre_vlm/num_selected"] = num_selected
            diag["pre_vlm/selection_ratio"] = num_selected / V
            diag["pre_vlm/noise_scale"] = self.current_noise_scale

            # Selection distribution analysis
            # selection_counts: [B, V] - how many times each token was selected
            selection_counts_0 = selection_counts[0].float()
            max_count = selection_counts_0.max().item()
            mean_count = selection_counts_0.mean().item()
            std_count = selection_counts_0.std().item()
            diag["pre_vlm/max_selection_count"] = max_count
            diag["pre_vlm/mean_selection_count"] = mean_count
            diag["pre_vlm/std_selection_count"] = std_count

            # Input query_embeds similarity (upstream: is proprio tokenization diverse?)
            P = query_embeds.shape[1]
            input_embeds_flat = query_embeds[0]  # [P, D]
            input_normed = F.normalize(input_embeds_flat, dim=-1)
            input_cosine_sim = input_normed @ input_normed.T  # [P, P]
            input_mask = ~torch.eye(P, device=query_embeds.device, dtype=torch.bool)
            diag["pre_vlm/input_mean_cosine_sim"] = input_cosine_sim[input_mask].mean().item()
            diag["pre_vlm/input_max_cosine_sim"] = input_cosine_sim[input_mask].max().item()

            # Output query similarity (downstream: are generated queries too similar?)
            # queries: [B, V, D]
            queries_flat = queries[0]  # [V, D]
            queries_normed = F.normalize(queries_flat, dim=-1)
            cosine_sim = queries_normed @ queries_normed.T  # [V, V]
            # Exclude diagonal (self-similarity = 1.0)
            mask = ~torch.eye(V, device=queries.device, dtype=torch.bool)
            mean_cosine_sim = cosine_sim[mask].mean().item()
            max_cosine_sim = cosine_sim[mask].max().item()
            diag["pre_vlm/output_mean_cosine_sim"] = mean_cosine_sim
            diag["pre_vlm/output_max_cosine_sim"] = max_cosine_sim

            # Score distribution analysis
            # scores: [B, V, V]
            scores_float = scores[0].float()  # Cast to float32 for median (not supported in BFloat16)
            score_probs = F.softmax(scores_float, dim=-1)
            score_log_probs = F.log_softmax(scores_float, dim=-1)
            score_entropy = -(score_probs * score_log_probs).sum(-1).mean().item()
            max_score_gap = (scores_float.max(dim=-1).values - scores_float.median(dim=-1).values).mean().item()
            diag["pre_vlm/score_entropy"] = score_entropy
            diag["pre_vlm/max_score_gap"] = max_score_gap

            self._diagnostics = diag

    def _generate_per_vision_queries(
        self,
        query_embeds: torch.Tensor,
        vision_tokens: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> tuple:
        """Generate one query per vision token via cross-attention.

        LightVLA equations:
            H_v = RMSNorm(vision_tokens)
            H_l = RMSNorm(query_embeds)
            Q = softmax(H_v @ H_l^T / √D) @ H_l
            Q = RMSNorm(Q)

        Each query Q[i] represents "what query information is relevant for
        vision position i", enabling position-aware token selection.

        Args:
            query_embeds: [B, P, D] - text-tokenized query embeddings
                          (text, proprio, or concatenated depending on mode)
            vision_tokens: [B, V, D] - vision patches from vision encoder
            query_attention_mask: Optional [B, P] mask where 1=valid, 0=pad.
                If provided, padded query tokens are masked out of the
                vision->query attention.
            return_attn_weights: If True, also return vision->query attention
                weights of shape [B, V, P].

        Returns:
            queries: [B, V, D] - per-vision queries
            vision_normed: [B, V, D] - normalized vision tokens (for scoring)
            (optional) attn_weights: [B, V, P] - vision->query attention weights
        """
        B, V, D = vision_tokens.shape

        # Step 1 & 2: Normalize inputs (LightVLA-style)
        vision_normed = self.vision_norm(vision_tokens)  # [B, V, D]
        query_normed = self.query_norm(query_embeds)  # [B, P, D]

        # Step 3: Cross-attention to generate queries
        # Q = softmax(H_v @ H_l^T / √D) @ H_l
        attn_logits = torch.einsum('bvd,bpd->bvp', vision_normed, query_normed) / (D ** 0.5)

        if query_attention_mask is not None:
            mask = query_attention_mask.to(device=attn_logits.device, dtype=torch.bool).unsqueeze(1)  # [B, 1, P]
            mask_value = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, V, P]
        queries = torch.einsum('bvp,bpd->bvd', attn_weights, query_normed)  # [B, V, D]

        # Step 4: Normalize queries (LightVLA does this!)
        queries = self.query_out_norm(queries)  # [B, V, D]

        if return_attn_weights:
            return queries, vision_normed, attn_weights

        return queries, vision_normed

    def forward(
        self,
        vision_tokens: torch.Tensor,
        proprio_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        return_mask: bool = False,
        return_scores: bool = False,
    ) -> torch.Tensor:
        B, V, D = vision_tokens.shape

        if self.selection_mode == "text":
            if text_embeds is None:
                raise ValueError("text_embeds required for selection_mode='text'")
            query_embeds = text_embeds
        elif self.selection_mode == "proprio":
            if proprio_embeds is None:
                raise ValueError("proprio_embeds required for selection_mode='proprio'")
            query_embeds = proprio_embeds
        elif self.selection_mode == "text_proprio":
            if text_embeds is None or proprio_embeds is None:
                raise ValueError(
                    "Both text_embeds and proprio_embeds required for selection_mode='text_proprio'"
                )
            query_embeds = torch.cat([text_embeds, proprio_embeds], dim=1)

        # LightVLA: generate queries and get normalized vision for scoring
        queries, vision_normed = self._generate_per_vision_queries(
            query_embeds,
            vision_tokens,
            query_attention_mask=query_attention_mask,
            return_attn_weights=False,
        )

        # Score = Q @ H_v^T / √D (against normalized vision, not original!)
        scores = torch.einsum('bqd,bkd->bqk', queries, vision_normed) / (D ** 0.5)

        # Store soft attention weights for visualization (average distribution across queries)
        # Softmax per query creates distribution, sum and normalize gives expected selection probability
        # self._last_soft_weights = F.softmax(scores, dim=-1).sum(dim=1) / scores.shape[1]  # [B, V]
        self._last_soft_weights = F.softmax(scores.sum(dim=1), dim=-1)  # [B, V]

        if self.training:
            noise_scale = self.current_noise_scale
            # Use Gumbel noise for proper categorical exploration (Gumbel-max trick)
            # uniform = torch.rand_like(scores).clamp(1e-10, 1 - 1e-10)
            # gumbel_noise = -torch.log(-torch.log(uniform))
            # noisy_scores = scores + gumbel_noise * noise_scale
            noisy_scores = scores + torch.rand_like(scores) * noise_scale
            soft = F.softmax(noisy_scores, dim=-1)
            hard_indices = noisy_scores.argmax(dim=-1)

            device = vision_tokens.device
            selection_counts = torch.zeros(B, V, device=device, dtype=torch.int32)
            selection_counts.scatter_add_(
                1,
                hard_indices,
                torch.ones(B, V, device=device, dtype=torch.int32),
            )
            hard_mask = selection_counts > 0

            # STE for gradient flow
            soft_indicator_noisy = soft.sum(dim=1) / V
            hard_indicator = hard_mask.float()
            indicator = hard_indicator + soft_indicator_noisy - soft_indicator_noisy.detach()

            # Actually remove unselected tokens (LightVLA-style)
            num_selected = int(hard_mask[0].sum().item())
            if num_selected > 0:
                selected_tokens, selected_attention_mask = self._gather_selected(
                    vision_tokens, hard_mask, indicator
                )
            else:
                selected_tokens = vision_tokens[:, :1, :]
                selected_attention_mask = torch.ones(
                    (B, 1), device=device, dtype=torch.long
                )
                num_selected = 1

        else:
            hard_indices = scores.argmax(dim=-1)
            device = vision_tokens.device
            selection_counts = torch.zeros(B, V, device=device, dtype=torch.int32)
            selection_counts.scatter_add_(
                1,
                hard_indices,
                torch.ones(B, V, device=device, dtype=torch.int32),
            )
            hard_mask = selection_counts > 0

            num_selected = int(hard_mask[0].sum().item())
            if num_selected > 0:
                selected_tokens, selected_attention_mask = self._gather_selected_inference(
                    vision_tokens, hard_mask
                )
            else:
                selected_tokens = vision_tokens[:, :1, :]
                selected_attention_mask = torch.ones(
                    (B, 1), device=device, dtype=torch.long
                )
                num_selected = 1

        self._last_num_selected = num_selected
        self._last_num_total = V
        self._last_compression_ratio = num_selected / V

        # Store normalized selection counts for visualization (how many queries selected each token)
        # selection_counts: [B, V] - higher count = more queries selected this token
        max_count = selection_counts.max().clamp(min=1)
        self._last_selection_counts = selection_counts.float() / max_count
        # Store binary mask for crisp visualization (exactly which tokens were selected)
        self._last_hard_mask = hard_mask

        if self.training:
            self._step_counter += 1
            if self._step_counter % 100 == 0:
                self._compute_diagnostics(
                    queries=queries,
                    scores=scores,
                    selection_counts=selection_counts,
                    query_embeds=query_embeds,
                    V=V,
                    num_selected=num_selected,
                )

        if not self._logged_compression:
            logger.info(
                f"[PreVLMVisionSelector] Vision: {V} -> Selected: {num_selected} "
                f"(compression: {V / max(num_selected, 1):.1f}x)"
            )
            self._logged_compression = True

        if self.use_residual:
            global_ctx = self.global_proj(vision_tokens.mean(dim=1, keepdim=True))
            selected_tokens = torch.cat([selected_tokens, global_ctx], dim=1)
            selected_attention_mask = torch.cat(
                [
                    selected_attention_mask,
                    torch.ones((B, 1), device=device, dtype=torch.long),
                ],
                dim=1,
            )

        if return_scores:
            # Return normalized selection counts for visualization
            if return_mask:
                return selected_tokens, self._last_selection_counts, selected_attention_mask
            return selected_tokens, self._last_selection_counts

        if return_mask:
            return selected_tokens, selected_attention_mask
        return selected_tokens

    def _gather_selected(
        self,
        vision_tokens: torch.Tensor,
        hard_mask: torch.Tensor,
        indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected tokens with STE gradient flow.

        Training: applies indicator weights to maintain gradient flow through
        soft_indicator while using hard selection for forward pass.
        """
        B, V, D = vision_tokens.shape

        selected_list = []
        for b in range(B):
            mask_b = hard_mask[b]
            tokens_b = vision_tokens[b][mask_b]
            weights_b = indicator[b][mask_b].unsqueeze(-1)
            selected_list.append(tokens_b * weights_b)

        max_len = max(s.shape[0] for s in selected_list)
        padded = torch.zeros(
            B, max_len, D, device=vision_tokens.device, dtype=vision_tokens.dtype
        )
        attention_mask = torch.zeros(
            B, max_len, device=vision_tokens.device, dtype=torch.long
        )
        for b, sel in enumerate(selected_list):
            padded[b, : sel.shape[0]] = sel
            attention_mask[b, : sel.shape[0]] = 1

        return padded, attention_mask

    def _gather_selected_inference(
        self,
        vision_tokens: torch.Tensor,
        hard_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected tokens for inference (no gradient needed)."""
        B, V, D = vision_tokens.shape

        selected_list = []
        for b in range(B):
            mask_b = hard_mask[b]
            selected_list.append(vision_tokens[b][mask_b])

        max_len = max(s.shape[0] for s in selected_list)
        padded = torch.zeros(
            B, max_len, D, device=vision_tokens.device, dtype=vision_tokens.dtype
        )
        attention_mask = torch.zeros(
            B, max_len, device=vision_tokens.device, dtype=torch.long
        )
        for b, sel in enumerate(selected_list):
            padded[b, : sel.shape[0]] = sel
            attention_mask[b, : sel.shape[0]] = 1

        return padded, attention_mask
