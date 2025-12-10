# Proprio Enhancement & Subgoal Head Design

## Overview

This document describes two architectural enhancements to Flower VLA:

1. **Proprio History Enhancement**: Inject proprioceptive state history as tokens into Florence-2's encoder for bidirectional attention with vision-language features
2. **Subgoal Head**: A separate DiT that predicts future proprio states, conditioning the action head

The core insight (inspired by Pi0.5) is that proprioceptive state should be a "first-class citizen" in attention rather than just an AdaLN modulation signal. This allows VL features to become "proprio-grounded."

## Motivation

**Current limitations in Flower:**
- Proprio is encoded via MLP and summed into timestep embedding
- Only influences AdaLN modulation - cannot guide attention/feature selection
- Uses single-frame proprio, no history
- VL features are not grounded to robot state

**Goals:**
- Enable bidirectional attention between proprio and VL tokens
- Use proprio history for richer state representation
- Predict future proprio states as subgoals to guide action generation
- Semantic ordering: "robot knows self first, then observes, then acts"

---

## Part 1: Proprio History Enhancement

### Data Flow

```
Proprio History (N frames × proprio_dims)
    ↓
Per-frame MLP projection
    ↓
N proprio tokens (N × 1024D)
    ↓
Inject into merged_embeds (proprio first):
[proprio_tokens | image_features | task_prompt | text_embeds]
    ↓
Florence-2 Encoder (bidirectional attention)
    ↓
Proprio-grounded VL features
```

### Config Parameters

```yaml
# In config_calvin.yaml
proprio_dims: 7                   # Existing - proprio state dimension
use_proprio_history: true         # New - enable history tokens in VLM
proprio_history_len: 5            # New - number of past frames (t-4 to t)
proprio_token_dim: 1024           # New - same as Florence-2 hidden dim
```

### New Module: ProprioHistoryEncoder

```python
class ProprioHistoryEncoder(nn.Module):
    """Encode proprio history into tokens for VLM injection."""

    def __init__(self, proprio_dims, token_dim, history_len):
        super().__init__()
        self.frame_proj = Mlp(proprio_dims, token_dim, out_features=token_dim)
        self.history_len = history_len

    def forward(self, proprio_history):
        # proprio_history: (B, N, proprio_dims)
        # Output: (B, N, token_dim)
        return self.frame_proj(proprio_history)
```

### Changes to encode_observations()

```python
def encode_observations(self, batch):
    # ... existing image encoding ...

    # NEW: Encode proprio history
    if self.use_proprio_history:
        proprio_history = batch['proprio_history']  # (B, N, proprio_dims)
        proprio_tokens = self.proprio_history_encoder(proprio_history)  # (B, N, 1024)

    # Merge sequence - proprio FIRST
    merged_embeds = torch.cat([
        proprio_tokens,      # NEW: proprio history tokens first
        image_features,
        task_prompt,
        text_embeds
    ], dim=1)

    # Update attention mask to include proprio tokens
    attention_mask = torch.ones(merged_embeds.shape[:2], device=device)

    # Process through Florence-2 encoder (bidirectional attention)
    features = self.vlm.get_encoder()(
        inputs_embeds=merged_embeds,
        attention_mask=attention_mask
    ).last_hidden_state

    # ... rest of method ...
```

### Dataset Changes

Return `proprio_history` of shape `(B, N, proprio_dims)`:

```python
# In dataset __getitem__:
proprio_history = []
for i in range(proprio_history_len):
    past_idx = max(0, current_idx - (proprio_history_len - 1 - i))
    proprio_history.append(proprio_state[past_idx])

batch['proprio_history'] = torch.stack(proprio_history, dim=0)  # (N, proprio_dims)
```

---

## Part 2: Subgoal Head Architecture

### Subgoal Trace Definition

```
Subgoal Trace = [proprio(t+k), proprio(t+2k), ..., proprio(t+n*k)]

Where:
- k = subgoal_interval (e.g., 5 steps)
- n = subgoal_horizon (e.g., 3 subgoals)

Example: subgoal_interval=5, subgoal_horizon=3
→ Predict proprio at t+5, t+10, t+15
```

### Config Parameters

```yaml
# In config_calvin.yaml
use_subgoal_head: true
subgoal_horizon: 3              # Number of future proprio states to predict
subgoal_interval: 5             # Steps between each subgoal

# Subgoal DiT architecture (smaller than action DiT)
subgoal_dit_dim: 1024           # Same dim for compatibility
subgoal_n_layers: 12            # Smaller: 12 vs 18 for action
subgoal_n_heads: 16             # Same heads
```

### Architecture

```
Proprio-grounded VL features (from Florence-2)
    ↓
┌─────────────────────────────────────┐
│         Subgoal DiT                 │
│  - 12 layers (vs 18 for action)     │
│  - Own timestep embedder            │
│  - Own AdaLN controller             │
│  - Cross-attention to VL features   │
│  - Causal self-attention            │
└─────────────────────────────────────┘
    ↓
Linear decoder → (B, subgoal_horizon, proprio_dims)
    ↓
Subgoal Trace
```

### Key Components

1. **SubgoalDiT**: Separate DiT module, similar structure to action DiT but:
   - 12 layers instead of 18
   - Own `t_embedder`, `adaln`, encoders/decoders
   - Completely separate parameters (no sharing)
   - Input: noise of shape `(B, subgoal_horizon, proprio_dims)`
   - Output: predicted proprio states `(B, subgoal_horizon, proprio_dims)`

2. **subgoal_encoder**: MLP to project `(proprio_dims) → (subgoal_dit_dim)`

3. **subgoal_decoder**: Linear to project `(subgoal_dit_dim) → (proprio_dims)`

4. **Loss**: Same rectified flow loss as action head (MSE between predicted and GT velocity field)

---

## Part 3: Subgoal Conditioning on Action Head

### Data Flow

```
Subgoal Trace (B, subgoal_horizon, proprio_dims)
    ↓
Subgoal Cond Encoder MLP → (B, subgoal_horizon, dit_dim)
    ↓
Mean pooling → (B, dit_dim)
    ↓
Add to global_cond:

global_cond = vlm_token + t_emb + subgoal_emb
                  ↓
            AdaLN modulation
                  ↓
            Action DiT layers
```

### New Module: SubgoalCondEncoder

```python
class SubgoalCondEncoder(nn.Module):
    """Encode subgoal trace for action head conditioning."""

    def __init__(self, proprio_dims, dit_dim):
        super().__init__()
        self.proj = Mlp(proprio_dims, dit_dim, out_features=dit_dim)

    def forward(self, subgoal_trace):
        # (B, subgoal_horizon, proprio_dims) → (B, subgoal_horizon, dit_dim)
        return self.proj(subgoal_trace)
```

### Changes to dit_forward() for Action Head

```python
def dit_forward(self, z, t, cond_dict, subgoal_trace=None):
    # ... existing code ...

    # NEW: Encode subgoal trace
    if self.use_subgoal_head and subgoal_trace is not None:
        subgoal_encoded = self.subgoal_cond_encoder(subgoal_trace)  # (B, H, dit_dim)
        subgoal_emb = subgoal_encoded.mean(dim=1)  # (B, dit_dim)
    else:
        subgoal_emb = torch.zeros_like(t_emb)

    # Updated global conditioning
    t_emb = stateless_norm(self.t_embedder(t)) + \
            stateless_norm(frequency_embeds) + \
            stateless_norm(proprio_embeds) + \
            stateless_norm(subgoal_emb)  # NEW

    # ... rest of method ...
```

---

## Part 4: Two-Stage Training

### Stage 1: Supervised Subgoal Training

Both heads train, but action head receives ground truth subgoals.

```
Ground Truth Subgoal Trace ──────────────────┐
        │                                    │
        ▼                                    ▼
   Subgoal Head                        Action Head
   (learns to predict)              (conditioned on GT)
        │                                    │
        ▼                                    ▼
   Subgoal Loss                        Action Loss
        │                                    │
        └──────────── Total Loss ────────────┘
```

### Stage 2: End-to-End Fine-tuning

Action head receives predicted subgoals. Gradients flow through both heads.

```
                    Predicted Subgoal Trace
                              │
                              ▼
   Subgoal Head ─────────► Action Head
        │                      │
        ▼                      ▼
   Subgoal Loss           Action Loss
        │                      │
        └────── Total Loss ────┘
              (gradients flow through)
```

### Config Parameters

```yaml
training_stage: 1                # 1 = GT subgoals, 2 = predicted
subgoal_loss_weight: 1.0         # Weight for subgoal loss
action_loss_weight: 1.0          # Weight for action loss
```

### Implementation

```python
def training_step(self, batch, batch_idx):
    cond = self.encode_observations(batch)
    gt_subgoal_trace = batch['subgoal_trace']  # (B, subgoal_horizon, proprio_dims)

    # Subgoal head forward (always predict from noise)
    pred_subgoal_trace = self.sample_subgoals(noise, cond)
    subgoal_loss = self.rf_loss_subgoal(cond, gt_subgoal_trace)

    # Action head forward
    if self.training_stage == 1:
        # Stage 1: use ground truth
        action_loss = self.rf_loss_action(cond, actions, subgoal_trace=gt_subgoal_trace)
    else:
        # Stage 2: use predicted (with gradient flow)
        action_loss = self.rf_loss_action(cond, actions, subgoal_trace=pred_subgoal_trace)

    total_loss = self.subgoal_loss_weight * subgoal_loss + \
                 self.action_loss_weight * action_loss
    return total_loss
```

### Dataset: Ground Truth Subgoal Trace

```python
# In dataset __getitem__:
subgoal_trace = []
for i in range(subgoal_horizon):
    future_idx = current_idx + (i + 1) * subgoal_interval
    future_idx = min(future_idx, episode_end_idx)  # Handle boundary
    subgoal_trace.append(proprio_state[future_idx])

batch['subgoal_trace'] = torch.stack(subgoal_trace, dim=0)  # (subgoal_horizon, proprio_dims)
```

---

## Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FLOWER VLA (Enhanced)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────────────┐    │
│  │ Proprio      │  │   RGB       │  │  Task     │  │    Language      │    │
│  │ History      │  │   Images    │  │  Prompt   │  │    Instruction   │    │
│  │ (N frames)   │  │             │  │  <Flow>   │  │                  │    │
│  └──────┬───────┘  └──────┬──────┘  └─────┬─────┘  └────────┬─────────┘    │
│         │                 │               │                  │              │
│         ▼                 │               │                  │              │
│  ┌──────────────┐         │               │                  │              │
│  │ ProprioHist  │         │               │                  │              │
│  │ Encoder      │         │               │                  │              │
│  └──────┬───────┘         │               │                  │              │
│         │                 │               │                  │              │
│         ▼                 ▼               ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  merged_embeds = [proprio_tokens | image | task_prompt | text]      │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                        │
│                    │   Florence-2 Encoder          │                        │
│                    │   (Bidirectional Attention)   │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│                                    ▼                                        │
│                     Proprio-Grounded VL Features                            │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │                               │                        │
│                    ▼                               ▼                        │
│         ┌─────────────────────┐         ┌─────────────────────┐            │
│         │   Subgoal DiT       │         │    Action DiT       │            │
│         │   (12 layers)       │         │    (18 layers)      │            │
│         └──────────┬──────────┘         └──────────┬──────────┘            │
│                    │                               ▲                        │
│                    ▼                               │                        │
│         ┌─────────────────────┐                    │                        │
│         │  Subgoal Trace      │────────────────────┘                        │
│         │  (t+5, t+10, t+15)  │      (AdaLN conditioning)                   │
│         └─────────────────────┘                                             │
│                                                                             │
│                                        │                                    │
│                                        ▼                                    │
│                              ┌─────────────────┐                            │
│                              │  Action Chunk   │                            │
│                              │  (10 × 7 dim)   │                            │
│                              └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `conf/config_calvin.yaml` | Add new config params (proprio history, subgoal head, training stage) |
| `conf/model/flower.yaml` | Add subgoal DiT architecture params |
| `flower/models/flower.py` | ProprioHistoryEncoder, SubgoalDiT, SubgoalCondEncoder, two-stage training |
| `flower/models/networks/transformers.py` | Potentially add SubgoalDiT class (or reuse FlowBlock) |
| `flower/datasets/base_dataset.py` | Return proprio_history and subgoal_trace |
| `flower/datasets/hulc_data_module.py` | Handle new data fields |

---

## Complete Config Additions

```yaml
# config_calvin.yaml additions:

# Proprio history (for VLM injection)
use_proprio_history: true
proprio_history_len: 5            # frames: t-4, t-3, t-2, t-1, t

# Subgoal head
use_subgoal_head: true
subgoal_horizon: 3                # predict 3 future states
subgoal_interval: 5               # at t+5, t+10, t+15

# Subgoal DiT architecture
subgoal_dit_dim: 1024
subgoal_n_layers: 12
subgoal_n_heads: 16

# Training
training_stage: 1                 # 1 = GT subgoals, 2 = predicted
subgoal_loss_weight: 1.0
action_loss_weight: 1.0
```

---

## New Modules Summary

1. **ProprioHistoryEncoder**: Per-frame MLP projection for history tokens → VLM injection
2. **SubgoalDiT**: Separate 12-layer DiT for subgoal prediction (same structure as action DiT)
3. **SubgoalCondEncoder**: Encode predicted subgoals for action head AdaLN conditioning

---

## Future Considerations

- **Shared DiT layers**: Could share early layers between subgoal and action DiT for efficiency
- **Learnable subgoal intervals**: Let model learn which future timesteps matter
- **Latent subgoals**: Use VAE to predict in latent space instead of proprio space
- **Subgoal tokens in cross-attention**: Add subgoal as context tokens instead of just AdaLN
