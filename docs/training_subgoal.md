# FlowerSubgoalVLA Training Guide

## Overview

FlowerSubgoalVLA extends FLOWERVLA with:
- **Proprio History Injection**: Past proprioceptive states encoded as tokens for VLM bidirectional attention
- **Subgoal Prediction Head**: Separate 12-layer DiT that predicts future proprio states
- **Two-Stage Training**: Stage 1 uses GT subgoals, Stage 2 uses predicted subgoals

## Quick Start

### Stage 1: Train with Ground Truth Subgoals

```bash
python train.py --config-name=config_calvin_subgoal \
    training_stage=1 \
    subgoal_loss_weight=1.0 \
    action_loss_weight=1.0 \
    logger.id=subgoal_stage1_run1
```

### Stage 2: Train with Predicted Subgoals

```bash
python train.py --config-name=config_calvin_subgoal \
    training_stage=2 \
    model.load_pretrained=True \
    model.pretrained_model_path=/path/to/stage1_checkpoint.pt \
    logger.id=subgoal_stage2_run1
```

## Configuration Reference

### Main Config: `conf/config_calvin_subgoal.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `proprio_dims` | 7 | Proprioceptive state dimension |
| `use_proprio_history` | true | Enable proprio history injection |
| `proprio_history_len` | 5 | Number of past frames (t-4 to t) |
| `use_subgoal_head` | true | Enable subgoal prediction head |
| `subgoal_horizon` | 3 | Number of future states to predict |
| `subgoal_interval` | 10 | Frame interval between subgoals (t+10, t+20, t+30) |
| `training_stage` | 1 | 1=GT subgoals, 2=predicted subgoals |
| `subgoal_loss_weight` | 1.0 | Weight for subgoal prediction loss |
| `action_loss_weight` | 1.0 | Weight for action prediction loss |

### Model Config: `conf/model/flower_subgoal_calvin.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subgoal_dit_dim` | 1024 | Subgoal DiT hidden dimension |
| `subgoal_n_layers` | 12 | Subgoal DiT transformer layers |
| `subgoal_n_heads` | 16 | Subgoal DiT attention heads |

### Datamodule Config: `conf/datamodule/calvin_subgoal.yaml`

Passes `proprio_history_len`, `subgoal_horizon`, `subgoal_interval` to `ExtendedDiskDataset`.

## Data Pipeline

The dataset returns additional fields:
- `proprio_history`: `(proprio_history_len, proprio_dim)` - past proprio states
- `subgoal_trace`: `(subgoal_horizon, proprio_dim)` - future proprio states (GT)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Florence-2 VLM            │
                    │  (with proprio history injection)   │
                    └─────────────────┬───────────────────┘
                                      │ VLM features
                    ┌─────────────────┴───────────────────┐
                    │                                     │
              ┌─────▼─────┐                       ┌───────▼───────┐
              │ Subgoal   │                       │   Action      │
              │   DiT     │                       │     DiT       │
              │(12 layers)│                       │ (18 layers)   │
              └─────┬─────┘                       └───────▲───────┘
                    │ predicted subgoals                  │
                    │         ┌───────────────────────────┘
                    │         │ subgoal conditioning (AdaLN)
                    └─────────┴───────────────────────────────────
```

## Training Tips

1. **Stage 1 Focus**: Ensure subgoal prediction converges well before moving to Stage 2
2. **Loss Weights**: Start with equal weights (1.0), adjust if one loss dominates
3. **Learning Rate**: Use same LR as original FLOWERVLA (2e-5)
4. **Batch Size**: 8 per GPU, 4 GPUs = 32 effective batch size

## Monitoring

Key metrics to watch in wandb:
- `train/subgoal_loss`: Subgoal prediction loss (rectified flow)
- `train/action_loss`: Action prediction loss (rectified flow)
- `train/total_loss`: Combined weighted loss

## File Structure

```
conf/
├── config_calvin_subgoal.yaml      # Main training config
├── datamodule/
│   └── calvin_subgoal.yaml         # Datamodule with subgoal params
└── model/
    └── flower_subgoal_calvin.yaml  # Model config for FlowerSubgoalVLA

flower/
├── models/
│   └── flower_subgoal.py           # FlowerSubgoalVLA + all subgoal modules
└── datasets/
    └── disk_dataset.py             # ExtendedDiskDataset with proprio/subgoal
```
