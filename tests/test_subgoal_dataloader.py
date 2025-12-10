"""Test script for subgoal dataloader with calvin_debug_dataset."""
import sys
sys.path.insert(0, '/root/flower_vla_calvin')

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

from flower.datasets.disk_dataset import ExtendedDiskDataset


def test_subgoal_dataloader():
    """Test that proprio_history and subgoal_trace are correctly loaded."""

    data_dir = Path("/data/fywang/Calvin/calvin_debug_dataset/training")
    lang_folder = "lang_clip_resnet50"

    # Check if lang folder exists
    if not (data_dir / lang_folder).exists():
        print(f"WARNING: {data_dir / lang_folder} not found, using training dir directly")
        lang_folder = None

    # Dataset config matching calvin_subgoal.yaml
    proprio_state = OmegaConf.create({
        'n_state_obs': 8,
        'keep_indices': [[0, 7], [14, 15]],
        'robot_orientation_idx': [3, 6],
        'normalize': True,
        'normalize_robot_orientation': True,
    })

    obs_space = OmegaConf.create({
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language'],
    })

    dataset = ExtendedDiskDataset(
        datasets_dir=data_dir,
        key="lang",
        save_format="npz",
        batch_size=4,
        min_window_size=10,
        max_window_size=10,
        proprio_state=proprio_state,
        obs_space=obs_space,
        skip_frames=1,
        pad=False,
        lang_folder=lang_folder,
        aux_lang_loss_window=8,
        num_workers=0,
        obs_seq_len=1,
        action_seq_len=10,
        future_range=1,
        use_extracted_rel_actions=True,
        # Subgoal params
        proprio_history_len=5,
        subgoal_horizon=3,
        subgoal_interval=10,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"proprio_history_len: {dataset.proprio_history_len}")
    print(f"subgoal_horizon: {dataset.subgoal_horizon}")
    print(f"subgoal_interval: {dataset.subgoal_interval}")

    # Test a few samples
    for i in [0, len(dataset) // 2, len(dataset) - 1]:
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]

        print(f"Sample keys: {list(sample.keys())}")

        # Check required keys exist
        assert 'proprio_history' in sample, f"Missing proprio_history in sample {i}"
        assert 'subgoal_trace' in sample, f"Missing subgoal_trace in sample {i}"

        proprio_history = sample['proprio_history']
        subgoal_trace = sample['subgoal_trace']

        print(f"proprio_history shape: {proprio_history.shape}")
        print(f"subgoal_trace shape: {subgoal_trace.shape}")

        # Verify shapes
        assert proprio_history.shape[0] == 5, f"Expected proprio_history_len=5, got {proprio_history.shape[0]}"
        assert subgoal_trace.shape[0] == 3, f"Expected subgoal_horizon=3, got {subgoal_trace.shape[0]}"
        assert proprio_history.shape[1] == 15, f"Expected proprio_dim=15, got {proprio_history.shape[1]}"
        assert subgoal_trace.shape[1] == 15, f"Expected proprio_dim=15, got {subgoal_trace.shape[1]}"

        # Print other keys
        print(f"Other keys: {[k for k in sample.keys() if k not in ['proprio_history', 'subgoal_trace']]}")

        if 'robot_obs' in sample:
            print(f"robot_obs shape: {sample['robot_obs'].shape}")
        if 'rel_actions' in sample:
            print(f"rel_actions shape: {sample['rel_actions'].shape}")

    print("\n✓ All tests passed!")
    return dataset


if __name__ == "__main__":
    test_subgoal_dataloader()
