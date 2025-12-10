# Proprio Enhancement & Subgoal Head Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance Flower VLA with proprio history tokens in VLM and a separate subgoal prediction head.

**Architecture:** Inject proprio history as tokens into Florence-2 encoder for bidirectional attention with VL features. Add a separate 12-layer DiT to predict future proprio states (subgoals), which condition the action head via AdaLN.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra configs, Florence-2 (transformers), timm MLP

---

## Task 1: Add Config Parameters

**Files:**
- Modify: `conf/config_calvin.yaml`
- Modify: `conf/model/flower.yaml`

**Step 1: Add proprio history config to config_calvin.yaml**

Add after line 18 (`proprio_dims: 7`):

```yaml
# Proprio history (for VLM injection)
use_proprio_history: true
proprio_history_len: 5            # frames: t-4, t-3, t-2, t-1, t

# Subgoal head
use_subgoal_head: true
subgoal_horizon: 3                # predict 3 future states
subgoal_interval: 5               # at t+5, t+10, t+15

# Training stage
training_stage: 1                 # 1 = GT subgoals, 2 = predicted
subgoal_loss_weight: 1.0
action_loss_weight: 1.0
```

**Step 2: Add subgoal DiT config to flower.yaml**

Add after line 42 (`mlp_pdrop: 0.1`):

```yaml
# Subgoal DiT Configuration
subgoal_dit_dim: 1024
subgoal_n_layers: 12
subgoal_n_heads: 16
```

**Step 3: Verify config loads**

Run:
```bash
cd /root/flower_vla_calvin && python -c "from omegaconf import OmegaConf; import hydra; hydra.initialize(config_path='conf'); cfg = hydra.compose(config_name='config_calvin'); print('use_proprio_history:', cfg.get('use_proprio_history', 'NOT FOUND')); print('subgoal_horizon:', cfg.get('subgoal_horizon', 'NOT FOUND'))"
```

Expected: Shows `use_proprio_history: true` and `subgoal_horizon: 3`

**Step 4: Commit**

```bash
git add conf/config_calvin.yaml conf/model/flower.yaml
git commit -m "feat: add config params for proprio history and subgoal head"
```

---

## Task 2: Add ProprioHistoryEncoder Module

**Files:**
- Modify: `flower/models/flower.py:18-30` (imports)
- Modify: `flower/models/flower.py` (add class before FLOWERVLA)

**Step 1: Add ProprioHistoryEncoder class**

Add after the imports (around line 35, before `class FLOWERVLA`):

```python
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
        # Project each frame independently
        # Reshape to (B*N, D), project, reshape back
        flat = proprio_history.view(B * N, D)
        projected = self.frame_proj(flat)
        return projected.view(B, N, -1)
```

**Step 2: Verify module can be instantiated**

Run:
```bash
cd /root/flower_vla_calvin && python -c "
import torch
from flower.models.flower import ProprioHistoryEncoder
enc = ProprioHistoryEncoder(proprio_dims=7, token_dim=1024, history_len=5)
x = torch.randn(2, 5, 7)
out = enc(x)
print('Input shape:', x.shape)
print('Output shape:', out.shape)
assert out.shape == (2, 5, 1024), f'Expected (2, 5, 1024), got {out.shape}'
print('SUCCESS')
"
```

Expected: `Input shape: torch.Size([2, 5, 7])`, `Output shape: torch.Size([2, 5, 1024])`, `SUCCESS`

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add ProprioHistoryEncoder module"
```

---

## Task 3: Add SubgoalCondEncoder Module

**Files:**
- Modify: `flower/models/flower.py` (add class after ProprioHistoryEncoder)

**Step 1: Add SubgoalCondEncoder class**

Add after `ProprioHistoryEncoder` class:

```python
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
```

**Step 2: Verify module can be instantiated**

Run:
```bash
cd /root/flower_vla_calvin && python -c "
import torch
from flower.models.flower import SubgoalCondEncoder
enc = SubgoalCondEncoder(proprio_dims=7, dit_dim=1024)
x = torch.randn(2, 3, 7)  # 3 subgoals
out = enc(x)
print('Input shape:', x.shape)
print('Output shape:', out.shape)
assert out.shape == (2, 3, 1024), f'Expected (2, 3, 1024), got {out.shape}'
print('SUCCESS')
"
```

Expected: `SUCCESS`

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add SubgoalCondEncoder module"
```

---

## Task 4: Update FLOWERVLA Constructor for Proprio History

**Files:**
- Modify: `flower/models/flower.py:38-87` (FLOWERVLA.__init__ parameters)
- Modify: `flower/models/flower.py:93-109` (_init_flags call)

**Step 1: Add new constructor parameters**

In `FLOWERVLA.__init__`, add these parameters after `use_proprio: bool = False,` (around line 62):

```python
        # Proprio History Configuration
        use_proprio_history: bool = False,
        proprio_history_len: int = 5,
        proprio_dims: int = 7,

        # Subgoal Configuration
        use_subgoal_head: bool = False,
        subgoal_horizon: int = 3,
        subgoal_interval: int = 5,
        subgoal_dit_dim: int = 1024,
        subgoal_n_layers: int = 12,
        subgoal_n_heads: int = 16,

        # Training Configuration
        training_stage: int = 1,
        subgoal_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
```

**Step 2: Update _init_flags call**

Update the `_init_flags` call (around line 93) to include new parameters:

```python
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
            # New flags
            use_proprio_history=use_proprio_history,
            proprio_history_len=proprio_history_len,
            proprio_dims=proprio_dims,
            use_subgoal_head=use_subgoal_head,
            subgoal_horizon=subgoal_horizon,
            subgoal_interval=subgoal_interval,
            training_stage=training_stage,
            subgoal_loss_weight=subgoal_loss_weight,
            action_loss_weight=action_loss_weight,
        )
```

**Step 3: Update _init_flags method**

Find the `_init_flags` method and add the new flags. Add these lines inside the method:

```python
        # Proprio history flags
        self.use_proprio_history = use_proprio_history
        self.proprio_history_len = proprio_history_len
        self.proprio_dims = proprio_dims

        # Subgoal flags
        self.use_subgoal_head = use_subgoal_head
        self.subgoal_horizon = subgoal_horizon
        self.subgoal_interval = subgoal_interval
        self.training_stage = training_stage
        self.subgoal_loss_weight = subgoal_loss_weight
        self.action_loss_weight = action_loss_weight
```

**Step 4: Verify model instantiates**

Run:
```bash
cd /root/flower_vla_calvin && python -c "
import torch
from flower.models.flower import FLOWERVLA
# Just check that constructor accepts new params (will fail on VLM load but that's ok)
try:
    model = FLOWERVLA(
        use_proprio_history=True,
        proprio_history_len=5,
        proprio_dims=7,
        use_subgoal_head=True,
        subgoal_horizon=3,
    )
except Exception as e:
    if 'use_proprio_history' in str(e) or 'proprio_dims' in str(e):
        print('FAIL: Parameter not recognized')
        raise
    else:
        print('SUCCESS: Parameters accepted (other error expected)')
"
```

Expected: `SUCCESS: Parameters accepted`

**Step 5: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add proprio history and subgoal params to FLOWERVLA constructor"
```

---

## Task 5: Initialize ProprioHistoryEncoder in FLOWERVLA

**Files:**
- Modify: `flower/models/flower.py` (_setup_dit_components or after VLM setup)

**Step 1: Add ProprioHistoryEncoder initialization**

After `self._setup_vlm(...)` call (around line 123), add:

```python
        # Setup proprio history encoder
        if use_proprio_history:
            self.proprio_history_encoder = ProprioHistoryEncoder(
                proprio_dims=proprio_dims,
                token_dim=self.vlm.config.text_config.d_model,  # Match Florence-2 hidden dim
                history_len=proprio_history_len,
            )
```

**Step 2: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: initialize ProprioHistoryEncoder in FLOWERVLA"
```

---

## Task 6: Update encode_observations for Proprio History

**Files:**
- Modify: `flower/models/flower.py:690-761` (encode_observations method)

**Step 1: Modify encode_observations to inject proprio tokens**

Find the `encode_observations` method. After image encoding and before merging embeddings, add proprio token handling.

Find this section (around line 726):

```python
        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)
```

Replace with:

```python
        # NEW: Encode proprio history tokens
        if self.use_proprio_history and 'proprio_history' in batch:
            proprio_history = batch['proprio_history'].to(device).to(default_type)
            proprio_tokens = self.proprio_history_encoder(proprio_history)
        else:
            proprio_tokens = None

        # Merge sequence - proprio tokens FIRST (robot knows self, then observes)
        if proprio_tokens is not None:
            merged_embeds = torch.cat([
                proprio_tokens,          # NEW: proprio history tokens first
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
```

**Step 2: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: inject proprio history tokens into Florence-2 encoder"
```

---

## Task 7: Update Dataset to Return Proprio History

**Files:**
- Modify: `flower/datasets/disk_dataset.py:200-248` (ExtendedDiskDataset._load_episode)

**Step 1: Add proprio_history_len parameter to ExtendedDiskDataset**

In `ExtendedDiskDataset.__init__` (around line 163), add parameter:

```python
    def __init__(
        self,
        *args: Any,
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        use_extracted_rel_actions: bool = False,
        extracted_dir: str = 'extracted/',
        proprio_history_len: int = 5,      # NEW
        subgoal_horizon: int = 3,          # NEW
        subgoal_interval: int = 5,         # NEW
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range
        self.proprio_history_len = proprio_history_len  # NEW
        self.subgoal_horizon = subgoal_horizon          # NEW
        self.subgoal_interval = subgoal_interval        # NEW
        # ... rest of __init__
```

**Step 2: Modify _load_episode to include proprio_history and subgoal_trace**

At the end of `_load_episode` method (before `return episode`), add:

```python
        # NEW: Build proprio history (past N frames)
        # We need to load additional past frames for proprio history
        proprio_key = 'robot_obs'  # or appropriate key for proprio state
        if proprio_key in episode:
            current_proprio = episode[proprio_key]  # Shape: (obs_seq_len, proprio_dim)
            # For history, we need frames before the current observation
            # Load additional past frames if needed
            proprio_history = []
            for i in range(self.proprio_history_len):
                past_offset = self.proprio_history_len - 1 - i
                past_idx = max(0, start_idx - past_offset)
                if past_idx < start_idx:
                    past_ep = self.load_file(self._get_episode_name(past_idx))
                    proprio_history.append(past_ep[proprio_key])
                else:
                    # Use current frame's first proprio
                    proprio_history.append(current_proprio[0])
            episode['proprio_history'] = np.stack(proprio_history, axis=0)  # (proprio_history_len, proprio_dim)

        # NEW: Build subgoal trace (future proprio states)
        if proprio_key in episode:
            seq_start, seq_end = self.find_sequence_boundaries(self.episode_lookup[idx])
            subgoal_trace = []
            for i in range(self.subgoal_horizon):
                future_offset = (i + 1) * self.subgoal_interval
                future_idx = min(start_idx + future_offset, seq_end - 1)
                future_ep = self.load_file(self._get_episode_name(future_idx))
                subgoal_trace.append(future_ep[proprio_key])
            episode['subgoal_trace'] = np.stack(subgoal_trace, axis=0)  # (subgoal_horizon, proprio_dim)
```

**Step 3: Commit**

```bash
git add flower/datasets/disk_dataset.py
git commit -m "feat: add proprio_history and subgoal_trace to dataset"
```

---

## Task 8: Update DataModule to Pass New Config

**Files:**
- Modify: `flower/datasets/hulc_data_module.py`

**Step 1: Check current datamodule structure**

Read the file to understand how dataset params are passed:

```bash
cd /root/flower_vla_calvin && head -100 flower/datasets/hulc_data_module.py
```

**Step 2: Add new params to dataset instantiation**

Find where `ExtendedDiskDataset` is instantiated and add the new parameters:

```python
proprio_history_len=self.proprio_history_len,
subgoal_horizon=self.subgoal_horizon,
subgoal_interval=self.subgoal_interval,
```

**Step 3: Commit**

```bash
git add flower/datasets/hulc_data_module.py
git commit -m "feat: pass proprio history and subgoal params to dataset"
```

---

## Task 9: Add SubgoalDiT Module

**Files:**
- Modify: `flower/models/flower.py` (add SubgoalDiT class)

**Step 1: Add SubgoalDiT class**

Add after `SubgoalCondEncoder` class:

```python
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
        if not use_rope:
            self.positional_encoding = nn.Parameter(torch.randn(1, subgoal_horizon, dit_dim) * 0.1)
        self.use_rope = use_rope

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
```

**Step 2: Verify SubgoalDiT can be instantiated**

Run:
```bash
cd /root/flower_vla_calvin && python -c "
import torch
from flower.models.flower import SubgoalDiT
dit = SubgoalDiT(
    proprio_dims=7,
    dit_dim=1024,
    n_heads=16,
    n_layers=12,
    subgoal_horizon=3,
    vlm_hidden_dim=1024,
)
z = torch.randn(2, 3, 7)
t = torch.rand(2)
cond = torch.randn(2, 100, 1024)
out = dit(z, t, cond)
print('Output shape:', out.shape)
assert out.shape == (2, 3, 7), f'Expected (2, 3, 7), got {out.shape}'
print('SUCCESS')
"
```

Expected: `Output shape: torch.Size([2, 3, 7])`, `SUCCESS`

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add SubgoalDiT module for subgoal prediction"
```

---

## Task 10: Initialize SubgoalDiT and SubgoalCondEncoder in FLOWERVLA

**Files:**
- Modify: `flower/models/flower.py` (after _setup_dit_components)

**Step 1: Add initialization after proprio_history_encoder**

After the proprio_history_encoder initialization, add:

```python
        # Setup subgoal head
        if use_subgoal_head:
            self.subgoal_dit = SubgoalDiT(
                proprio_dims=proprio_dims,
                dit_dim=subgoal_dit_dim,
                n_heads=subgoal_n_heads,
                n_layers=subgoal_n_layers,
                subgoal_horizon=subgoal_horizon,
                vlm_hidden_dim=self.vlm.config.text_config.d_model,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_pdrop=mlp_pdrop,
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                rope_theta=rope_theta,
            )
            self.subgoal_cond_encoder = SubgoalCondEncoder(
                proprio_dims=proprio_dims,
                dit_dim=dit_dim,
            )
```

**Step 2: Store subgoal DiT params as instance variables**

Add before the initialization:

```python
        # Store subgoal params for later use
        self.subgoal_dit_dim = subgoal_dit_dim if use_subgoal_head else None
        self.subgoal_n_layers = subgoal_n_layers if use_subgoal_head else None
        self.subgoal_n_heads = subgoal_n_heads if use_subgoal_head else None
```

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: initialize SubgoalDiT and SubgoalCondEncoder in FLOWERVLA"
```

---

## Task 11: Add Subgoal RF Loss Method

**Files:**
- Modify: `flower/models/flower.py` (add rf_loss_subgoal method)

**Step 1: Add rf_loss_subgoal method**

Add after the existing `rf_loss` method:

```python
    def rf_loss_subgoal(self, cond: Dict, gt_subgoal_trace: torch.Tensor):
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
```

**Step 2: Add sample_subgoals method**

Add after rf_loss_subgoal:

```python
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
```

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add subgoal RF loss and sampling methods"
```

---

## Task 12: Update dit_forward for Subgoal Conditioning

**Files:**
- Modify: `flower/models/flower.py:568-630` (dit_forward method)

**Step 1: Add subgoal_trace parameter to dit_forward**

Change method signature:

```python
    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict, subgoal_trace: torch.Tensor = None) -> torch.Tensor:
```

**Step 2: Add subgoal conditioning to t_emb**

After proprio_embeds computation (around line 597), add:

```python
        # Handle subgoal conditioning
        if self.use_subgoal_head and subgoal_trace is not None:
            subgoal_encoded = self.subgoal_cond_encoder(subgoal_trace.to(default_dtype))
            subgoal_emb = subgoal_encoded.mean(dim=1)  # Pool over horizon
        else:
            subgoal_emb = torch.zeros_like(proprio_embeds)

        # Process embeddings - now includes subgoal
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1) + \
                stateless_norm(subgoal_emb)  # NEW: subgoal conditioning
```

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: add subgoal conditioning to action DiT forward"
```

---

## Task 13: Update rf_loss for Subgoal Conditioning

**Files:**
- Modify: `flower/models/flower.py:495-544` (rf_loss method)

**Step 1: Add subgoal_trace parameter**

Change method signature:

```python
    def rf_loss(self, cond, actions, dataset_idx=None, subgoal_trace: torch.Tensor = None):
```

**Step 2: Pass subgoal_trace to dit_forward**

Update the dit_forward call:

```python
        # Forward pass
        vtheta = self.dit_forward(zt, t, cond, subgoal_trace=subgoal_trace)
```

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: pass subgoal_trace through rf_loss to dit_forward"
```

---

## Task 14: Update training_step for Two-Stage Training

**Files:**
- Modify: `flower/models/flower.py:433-471` (training_step method)

**Step 1: Implement two-stage training logic**

Replace the training_step method body with:

```python
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
```

**Step 2: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: implement two-stage training with subgoal head"
```

---

## Task 15: Update forward and step Methods for Inference

**Files:**
- Modify: `flower/models/flower.py:798-861` (forward and step methods)

**Step 1: Update forward method**

Find the `forward` method and update to use subgoal prediction:

```python
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """Forward pass for inference."""
        # Encode observations
        obs_features = self.encode_observations(obs)

        # Generate noise for action sampling
        noise_shape = (1, self.act_window_size, self.action_dim)
        noise = torch.randn(noise_shape, device=self.device)

        # Predict subgoals if using subgoal head
        subgoal_trace = None
        if self.use_subgoal_head:
            subgoal_noise = torch.randn(
                (1, self.subgoal_horizon, self.proprio_dims),
                device=self.device
            )
            subgoal_trace = self.sample_subgoals(subgoal_noise, obs_features)

        # Sample actions with subgoal conditioning
        return self.sample_actions(noise, obs_features, inference=True, subgoal_trace=subgoal_trace)
```

**Step 2: Update sample_actions signature**

Add subgoal_trace parameter to sample_actions:

```python
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool=False, subgoal_trace: torch.Tensor = None):
```

And update the dit_forward call inside:

```python
            vc = self.dit_forward(z, t_tensor, cond, subgoal_trace=subgoal_trace)
```

**Step 3: Commit**

```bash
git add flower/models/flower.py
git commit -m "feat: update forward and sample_actions for subgoal-conditioned inference"
```

---

## Task 16: Update process_state for Proprio History

**Files:**
- Modify: `flower/datasets/utils/episode_utils.py`

**Step 1: Check current process_state implementation**

Read the file to understand how proprio state is processed.

**Step 2: Update to handle proprio_history**

Add handling for `proprio_history` key in the episode dict, converting to tensor.

**Step 3: Commit**

```bash
git add flower/datasets/utils/episode_utils.py
git commit -m "feat: handle proprio_history in process_state"
```

---

## Task 17: Integration Test

**Files:**
- None (verification only)

**Step 1: Run a quick training sanity check**

```bash
cd /root/flower_vla_calvin && python -c "
import torch
from omegaconf import OmegaConf
from flower.models.flower import FLOWERVLA

# Create model with new features enabled
model = FLOWERVLA(
    vlm_path='microsoft/Florence-2-base',  # Use base for faster test
    use_proprio_history=True,
    proprio_history_len=5,
    proprio_dims=7,
    use_subgoal_head=True,
    subgoal_horizon=3,
    subgoal_interval=5,
    training_stage=1,
)

# Create dummy batch
batch = {
    'rgb_obs': {
        'rgb_static': torch.randn(2, 1, 3, 224, 224),
    },
    'lang_text': ['pick up the red block', 'move the blue cube'],
    'actions': torch.randn(2, 10, 7),
    'proprio_history': torch.randn(2, 5, 7),
    'subgoal_trace': torch.randn(2, 3, 7),
}

# Test encode_observations
print('Testing encode_observations...')
obs_features = model.encode_observations(batch)
print('Features shape:', obs_features['features'].shape)

# Test subgoal loss
print('Testing rf_loss_subgoal...')
sg_loss = model.rf_loss_subgoal(obs_features, batch['subgoal_trace'])
print('Subgoal loss:', sg_loss.item())

# Test action loss with subgoal conditioning
print('Testing rf_loss with subgoal...')
act_loss, _ = model.rf_loss(obs_features, batch['actions'], subgoal_trace=batch['subgoal_trace'])
print('Action loss:', act_loss.item())

print('SUCCESS: All components working!')
"
```

Expected: `SUCCESS: All components working!`

**Step 2: Commit integration test results**

```bash
git add -A
git commit -m "feat: complete proprio history and subgoal head integration"
```

---

## Summary

This plan implements:

1. **Config changes** (Task 1): New YAML parameters for proprio history and subgoal head
2. **ProprioHistoryEncoder** (Tasks 2, 5, 6): Encodes proprio history into tokens for VLM
3. **SubgoalCondEncoder** (Task 3): Encodes subgoal trace for action head conditioning
4. **SubgoalDiT** (Tasks 9, 10): Separate 12-layer DiT for subgoal prediction
5. **Dataset changes** (Tasks 7, 8, 16): Return proprio_history and subgoal_trace
6. **Training changes** (Tasks 11-14): Two-stage training with subgoal loss
7. **Inference changes** (Task 15): Subgoal-conditioned action sampling

Total: 17 tasks with ~50 atomic steps.
