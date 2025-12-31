import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm

###############################################################################
# Utility Functions
###############################################################################

def find_multiple(n: int, k: int) -> int:
    """
    Returns the smallest number greater than or equal to n that is a multiple of k.
    """
    return n if n % k == 0 else n + k - (n % k)

def stateless_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes x without maintaining running statistics.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-6)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies a modulation to x given shift and scale signals.
    The modulation formula: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


###############################################################################
# SwiGlu MLP
###############################################################################

class SwiGlu(nn.Module):
    """
    An implementation of the SwiGlu MLP activation as used in transformer feedforward layers.

    Args:
        dim: Input dimension.
        hidden_dim: Dimension of the hidden layer. If None, defaults to 4 * dim.
        dropout: Dropout probability.
        output_dim: Output dimension. Defaults to dim.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, output_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        # Following the original design: use 2/3 of hidden_dim (rounded to a multiple of 256)
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        if output_dim is None:
            output_dim = dim
        self.fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGlu MLP.
        """
        x1 = F.silu(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.dropout(x)
        x = self.proj(x)
        return x

###############################################################################
# Rotary Positional Embedding Helpers
###############################################################################

def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency matrices for 1D rotary embeddings.
    Returns:
        (cosine, sine): Tensors of shape [max_seq_len, dim/2].
    """
    freqs = torch.arange(0, dim, 2).float()  # [dim/2]
    freqs = theta ** (-freqs / dim)
    positions = torch.arange(max_seq_len).float()  # [max_seq_len]
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_seq_len, dim/2]
    return angles.cos(), angles.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension in half and rotates the halves.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to queries and keys.

    Args:
        q: Query tensor of shape [B, heads, seq_len, head_dim].
        k: Key tensor with the same shape as q.
        cos, sin: Cosine and sine frequency tensors of shape [max_seq_len, head_dim/2].
        position_ids: Optional tensor with position indices; if None, uses sequential positions.

    Returns:
        A tuple (q_rot, k_rot) with rotary embeddings applied.
    """
    seq_len = q.size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    cos = cos[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    sin = sin[position_ids].unsqueeze(0).unsqueeze(0)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

###############################################################################
# Attention Modules
###############################################################################

class FlowerAttention(nn.Module):
    """
    Multi-head self-attention module with optional rotary positional embeddings.

    Args:
        dim: Input dimension.
        n_heads: Number of attention heads.
        attn_pdrop: Dropout rate on the attention probabilities.
        resid_pdrop: Dropout rate on the output projection.
        use_rope: Whether to apply rotary embeddings.
        max_seq_len: Maximum sequence length for precomputed rotary frequencies.
        rope_theta: Theta value for rotary embeddings.
    """
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 use_rope: bool = False,
                 max_seq_len: int = 120,
                 rope_theta: float = 32) -> None:
        super().__init__()
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            self.rope_theta = rope_theta
            cos, sin = precompute_freqs_1d(self.head_dim, max_seq_len, theta=rope_theta)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor,
                custom_attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape [B, seq_len, dim].
            custom_attn_mask: Optional attention mask.
            is_causal: If True, applies causal masking.

        Returns:
            Tensor of shape [B, seq_len, dim] after attention and projection.
        """
        B, T, C = x.size()
        # Compute query, key, value and reshape for multi-head attention.
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        # Build causal mask if needed.
        if is_causal and custom_attn_mask is None:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif custom_attn_mask is not None:
            mask = custom_attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        else:
            mask = None
        # Use PyTorch's built-in scaled dot-product attention.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else ~mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
            is_causal=is_causal if custom_attn_mask is None else False
        )
        out = attn_output.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class FlowerCrossAttention(nn.Module):
    """
    Cross-attention module with optional rotary embeddings.

    Args:
        dim: Input and output dimension.
        n_heads: Number of attention heads.
        attn_pdrop: Dropout rate on the attention weights.
        resid_pdrop: Dropout rate on the output.
        use_rope: Whether to apply rotary embeddings.
        query_seq_len: Maximum length for queries.
        context_seq_len: Maximum length for context.
        rope_theta: Theta for query rotary embeddings.
        context_rope_theta: Theta for context rotary embeddings.
    """
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 use_rope: bool = False,
                 query_seq_len: int = 64,
                 context_seq_len: int = 384,
                 rope_theta: float = 32,
                 context_rope_theta: float = 1000.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            q_cos, q_sin = precompute_freqs_1d(self.head_dim, query_seq_len, theta=rope_theta)
            k_cos, k_sin = precompute_freqs_1d(self.head_dim, context_seq_len, theta=context_rope_theta)
            self.register_buffer("q_cos", q_cos)
            self.register_buffer("q_sin", q_sin)
            self.register_buffer("k_cos", k_cos)
            self.register_buffer("k_sin", k_sin)
            self.query_seq_len = query_seq_len
            self.context_seq_len = context_seq_len
            self.rope_theta = rope_theta
            self.context_rope_theta = context_rope_theta

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                custom_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies cross-attention between x (queries) and context (keys and values).

        Args:
            x: Query tensor of shape [B, seq_len, dim].
            context: Context tensor of shape [B, context_len, dim].
            custom_attn_mask: Optional attention mask.

        Returns:
            Tensor of shape [B, seq_len, dim].
        """
        B, T, C = x.size()
        _, S, _ = context.size()
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, _ = apply_rotary_pos_emb(q, q, self.q_cos, self.q_sin)
            k, _ = apply_rotary_pos_emb(k, k, self.k_cos, self.k_sin)
        if custom_attn_mask is not None:
            # First resh ape the mask to match q's sequence length
            mask = custom_attn_mask.unsqueeze(1).unsqueeze(2)  # [32, 1, 1, 101]
            mask = mask.expand(-1, self.n_heads, q.size(2), -1)  # [32, 16, 10, 101]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
                is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
                is_causal=False
            )
        out = attn_output.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out

###############################################################################
# Main FlowBlock
###############################################################################

class FlowBlock(nn.Module):
    """
    A transformer block for flow-based diffusion. Combines self-attention,
    (optional) cross-attention, and a SwiGlu MLP with adaptive layer normalization modulation.

    Args:
        dim: Input dimension.
        heads: Number of attention heads.
        attn_pdrop: Attention dropout rate.
        resid_pdrop: Residual dropout rate.
        mlp_pdrop: MLP dropout rate.
        use_cross_attn: Whether to include a cross-attention layer.
        use_rope: Whether to use rotary positional embeddings in self-attention.
        query_seq_len: Maximum query sequence length.
        rope_theta: Theta parameter for rotary embeddings.
        lora_dim: Intermediate dimension for adaptive normalization modulation.
        use_global_adaln: If True, combines global AdaLN modulation signals.
    """
    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 mlp_pdrop: float = 0.1,
                 use_cross_attn: bool = False,
                 use_rope: bool = False,
                 query_seq_len: int = 128,
                 rope_theta: float = 32,
                 lora_dim: int = 256,
                 use_global_adaln: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.use_cross_attn = use_cross_attn
        self.use_global_adaln = use_global_adaln

        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        self.norm3 = RmsNorm(dim, eps=1e-6) if use_cross_attn else None

        self.self_attn = FlowerAttention(dim=dim, n_heads=heads,
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                                           use_rope=use_rope, max_seq_len=query_seq_len, rope_theta=rope_theta)
        if use_cross_attn:
            self.cross_attn = FlowerCrossAttention(dim=dim, n_heads=heads,
                                                     attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                                                     use_rope=False)
        self.mlp = SwiGlu(dim, dropout=mlp_pdrop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, lora_dim),  # Down-project
            nn.Linear(lora_dim, 6 * dim)  # Up-project to produce 6 modulation signals
        )

    def forward(self, cx: torch.Tensor, c: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                custom_attn_mask: Optional[torch.Tensor] = None,
                custom_cross_attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                global_adaln: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through the FlowBlock.

        Args:
            cx: Input tensor for the block (e.g. action latent representations) of shape [B, L, D].
            c: Conditioning tensor (from external encoder).
            context: Optional context tensor for cross-attention.
            custom_attn_mask: Optional attention mask.
            is_causal: If True, uses causal self-attention.
            global_adaln: Optional list of global AdaLN modulation signals.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = cx.shape
        residual = cx

        # Compute modulation signals.
        modulation = self.adaLN_modulation(c)
        signals = modulation.chunk(6, dim=1)
        if self.use_global_adaln and global_adaln is not None:
            mod_signals = [signals[i] + global_adaln[i] for i in range(6)]
        else:
            mod_signals = signals
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_signals

        # Self-attention block with modulation.
        x_norm = self.norm1(cx)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        x_self = self.self_attn(x_mod, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        x_out = residual + gate_msa.unsqueeze(1) * x_self

        # Optionally apply cross-attention.
        if self.use_cross_attn:
            if context is None:
                raise ValueError("Context is required for cross-attention.")
            x_norm = self.norm2(x_out)
            x_cross = self.cross_attn(x_norm, context, custom_attn_mask=custom_cross_attn_mask)
            x_out = x_out + x_cross

        # MLP block with modulation.
        norm_layer = self.norm3 if self.use_cross_attn else self.norm2
        x_norm = norm_layer(x_out)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod)
        x_final = x_out + gate_mlp.unsqueeze(1) * mlp_out

        return x_final


###############################################################################
# Proprio-Guided VL Selection Modules
###############################################################################


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
        self._forward_logged = False  # For logging on first forward pass

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
        """
        Proprioceptive Token Selection with Gumbel-Softmax (Algorithm 1).

        This implements the differentiable token selection mechanism inspired by LightVLA,
        adapted for proprioceptive-guided selection of fused vision-language tokens.

        Algorithm:
            Step 0: P ← MLP(p)                           # Done in ProprioHistoryEncoder
            Step 1: Q ← softmax(H_fused · P^T / √D) · P  # Query Generation
            Step 2: S ← Q · H_fused^T / √D               # Token Scoring
            Step 3: Gumbel-Softmax with ε ~ U(0, α)      # Differentiable Selection
            Step 4: H'_fused ← I · H_fused               # Token Selection

        Key insight: Each of the L queries selects one token. When multiple queries
        select the same token, the effective token count is reduced adaptively.
        The noise schedule (α decays during training) enables exploration early
        and exploitation late.

        Args:
            proprio_tokens: [B, T, D] - Projected proprio states P (T timesteps)
            vl_tokens: [B, L, D] - Fused vision-language tokens H_fused
            K: Maximum number of tokens (used for inference deduplication)
            return_scores: Whether to return attention scores

        Returns:
            selected_tokens: [B, L, D] (training) or [B, K', D] (inference, K' ≤ K unique)
        """
        B, L, D = vl_tokens.shape  # L = number of fused tokens

        # === Step 1: Query Generation ===
        # Q ← softmax(H_fused · P^T / √D) · P
        # Each fused token generates a query by attending to proprio history
        queries = self._generate_per_token_queries(proprio_tokens, vl_tokens)  # [B, L, D]

        # === Step 2: Token Scoring ===
        # S ← Q · H_fused^T / √D
        # Each query scores all fused tokens
        scores = torch.einsum('bqd,bkd->bqk', queries, vl_tokens) / (D ** 0.5)  # [B, L, L]

        if self.training:
            # === Step 3: Gumbel-Softmax Selection ===
            # ε ~ U(0, α) where α decays during training
            noise_scale = self.current_noise_scale
            noise = torch.rand_like(scores) * noise_scale  # ε ~ U(0, α)
            noisy_scores = scores + noise  # S' ← S + ε

            # S_soft ← softmax_j(S')
            soft = F.softmax(noisy_scores, dim=-1)  # [B, L, L]

            # S_hard ← one_hot(argmax_j(S'))
            hard_indices = soft.argmax(dim=-1)  # [B, L]
            hard = F.one_hot(hard_indices, num_classes=L).float()  # [B, L, L]

            # I ← S_hard + S_soft - sg(S_soft)  (straight-through estimator)
            indicator = hard + soft - soft.detach()  # [B, L, L]

            # === Step 4: Token Selection ===
            # H'_fused ← I · H_fused
            selected_tokens = torch.einsum('bqk,bkd->bqd', indicator, vl_tokens)  # [B, L, D]

            # Log unique token count (for monitoring adaptive behavior)
            if not self._forward_logged:
                # Count unique tokens selected (where at least one query selected it)
                selection_counts = hard.sum(dim=1)  # [B, L]
                n_unique = (selection_counts > 0).sum(dim=-1).float().mean().item()
                import logging
                logging.getLogger(__name__).info(
                    f"[soft_topk_gumbel] Noise scale: {noise_scale:.4f}, "
                    f"Avg unique tokens: {n_unique:.1f}/{L}"
                )
                self._forward_logged = True

            # Soft weights for return_scores (importance per token)
            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)  # [B, L]

        else:
            # === Inference: Hard selection, deduplicate to unique tokens ===
            # No noise at inference
            hard_indices = scores.argmax(dim=-1)  # [B, L] - which token each query selects

            # Find unique selected tokens and gather them
            # For inference (typically B=1), we deduplicate to save computation
            unique_tokens_list = []
            for b in range(B):
                unique_indices = hard_indices[b].unique()

                # Sort by total score and limit to K if needed
                if len(unique_indices) > K:
                    token_scores = scores[b].sum(dim=0)  # [L]
                    idx_scores = token_scores[unique_indices]
                    _, top_idx = idx_scores.topk(K)
                    unique_indices = unique_indices[top_idx]

                unique_tokens = vl_tokens[b, unique_indices]  # [n_unique, D]
                unique_tokens_list.append(unique_tokens)

            # For inference, return variable-length unique tokens (no padding needed for B=1)
            if B == 1:
                selected_tokens = unique_tokens_list[0].unsqueeze(0)  # [1, n_unique, D]
            else:
                # Pad to max length for batched inference (rare case)
                max_len = max(t.size(0) for t in unique_tokens_list)
                padded = []
                for t in unique_tokens_list:
                    if t.size(0) < max_len:
                        padding = torch.zeros(max_len - t.size(0), D, device=t.device, dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                selected_tokens = torch.stack(padded, dim=0)

            soft_weights = F.softmax(scores.sum(dim=1), dim=-1)  # [B, L]

        # === Optional: Add global residual ===
        # if self.use_residual:
        #     global_ctx = self.global_proj(vl_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]
        #     selected_tokens = torch.cat([selected_tokens, global_ctx], dim=1)

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


###############################################################################
# Encoder Classes
###############################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb




class SharedAdaLNController(nn.Module):
    """Shared Adaptive Layer Normalization controller for all DiT blocks"""
    def __init__(self, dim, global_conddim, use_cross_attn=False):
        super().__init__()
        # Number of modulation signals needed
        num_mod_signals = 9 if use_cross_attn else 6
        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, num_mod_signals * dim, bias=False),
        )
        self.use_cross_attn = use_cross_attn

        # Zero initialize the final linear layer
        nn.init.zeros_(self.modCX[-1].weight)
        self.use_cross_attn = use_cross_attn

    def forward(self, global_cond):
        mod_signals = self.modCX(global_cond)
        if self.use_cross_attn:
            # Split into 9 parts for cross-attention path
            return mod_signals.chunk(9, dim=-1)
        else:
            # Split into 6 parts for self-attention only path
            return mod_signals.chunk(6, dim=-1)




class FreqEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb





class ActionSpaceEmbedderParameter(nn.Module):
    """
    Embeds discrete action indices using direct learnable parameters.
    """
    def __init__(
        self,
        hidden_size,
        max_actions=11,  # 0-10 inclusive
        embedding_size=256,
    ):
        super().__init__()
        # Direct learnable parameters for each action
        self.action_embeddings = nn.Parameter(
            torch.randn(max_actions, embedding_size) * 0.02  # Small initialization
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.max_actions = max_actions

    def forward(self, action_indices):
        """
        Convert action indices to embeddings using parameter lookup.

        Args:
            action_indices: tensor of shape (batch_size,) containing integers in [0, max_actions-1]
        """
        # Index into the parameter matrix
        embeddings = self.action_embeddings[action_indices]

        # Process through MLP
        embeddings = embeddings
        output = self.mlp(embeddings)

        return output

    def get_all_embeddings(self):
        """Returns embeddings for all possible actions."""
        return self.mlp(self.action_embeddings)




class ZeroEncoder(nn.Module):
    def __init__(self, dit_dim, device):
        super(ZeroEncoder, self).__init__()
        self.dit_dim = dit_dim
        self.device = device

    def forward(self, x):
        return torch.zeros((x.shape[0], self.dit_dim), device=self.device)
