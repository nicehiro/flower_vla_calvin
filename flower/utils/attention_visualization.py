"""
Attention visualization utilities for PreVLMVisionSelector.

Provides tools to visualize which vision tokens are being selected/attended to
during CALVIN evaluation, helping understand model attention patterns.

Supports dual-view configurations (rgb_static + rgb_gripper) where:
- 100 total tokens = 50 per view (Florence-2-large @ 224x224)
- Each view has 49 spatial tokens (7x7) + 1 CLS token
- Visualization shows side-by-side heatmaps for both views
"""

import cv2
import math
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict


@dataclass
class DualViewMasks:
    """Container for split dual-view mask data."""
    static_mask: np.ndarray  # [spatial_tokens] - excludes CLS
    gripper_mask: np.ndarray  # [spatial_tokens] - excludes CLS
    grid_size: Tuple[int, int]  # (H, W) for spatial grid
    tokens_per_view: int  # Total tokens per view (including CLS)
    has_cls: bool  # Whether CLS token was detected and excluded
    static_selected: int  # Number selected in static view
    gripper_selected: int  # Number selected in gripper view


def infer_grid_size(num_tokens: int) -> Tuple[Tuple[int, int], bool]:
    """Infer spatial grid dimensions from token count.

    Handles CLS token detection: if token count is not a perfect square,
    assumes last token is CLS and uses sqrt(n-1) for spatial grid.

    Args:
        num_tokens: Number of vision tokens (e.g., 49, 50, 196, 577)

    Returns:
        Tuple of ((height, width), has_cls):
        - (height, width) tuple for the spatial grid
        - has_cls: True if CLS token was detected and excluded
    """
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return (side, side), False

    # Check if n-1 is a perfect square (CLS token case)
    side_minus_one = int(math.sqrt(num_tokens - 1))
    if side_minus_one * side_minus_one == num_tokens - 1:
        return (side_minus_one, side_minus_one), True

    # Non-square: try common ratios
    for h in range(side + 1, 1, -1):
        if num_tokens % h == 0:
            return (h, num_tokens // h), False
    return (side, side), False  # Fallback to approximate square


def split_dual_view_mask(
    mask: torch.Tensor,
    tokens_per_view: Optional[int] = None,
) -> DualViewMasks:
    """Split a combined dual-view mask into per-view spatial masks.

    For Florence-2-large with dual views (100 tokens total):
    - Tokens 0-49: rgb_static (0-48 spatial + 49 CLS)
    - Tokens 50-99: rgb_gripper (50-98 spatial + 99 CLS)

    Args:
        mask: [V] or [B, V] binary mask or attention weights
        tokens_per_view: Tokens per view (auto-detected as len/2 if None)

    Returns:
        DualViewMasks containing split spatial masks and metadata
    """
    mask_np = mask.detach().cpu().numpy().flatten()
    total_tokens = len(mask_np)

    if tokens_per_view is None:
        tokens_per_view = total_tokens // 2

    # Detect grid size and CLS presence for single view
    (grid_h, grid_w), has_cls = infer_grid_size(tokens_per_view)
    spatial_tokens = grid_h * grid_w

    # Split into two views
    view1 = mask_np[:tokens_per_view]
    view2 = mask_np[tokens_per_view:tokens_per_view * 2]

    # Exclude CLS token (last token of each view) if detected
    if has_cls:
        static_mask = view1[:spatial_tokens]
        gripper_mask = view2[:spatial_tokens]
    else:
        static_mask = view1
        gripper_mask = view2

    # Compute selected count: for binary masks, sum works directly
    # For soft weights, count tokens above mean weight as "selected"
    def count_selected(weights: np.ndarray) -> int:
        if weights.max() <= 1.0 and weights.min() >= 0.0 and weights.sum() < len(weights) * 0.5:
            # Likely soft weights (values small, don't sum to large number)
            threshold = weights.mean()
            return int((weights > threshold).sum())
        else:
            # Binary mask or unnormalized
            return int(weights.sum())

    return DualViewMasks(
        static_mask=static_mask,
        gripper_mask=gripper_mask,
        grid_size=(grid_h, grid_w),
        tokens_per_view=tokens_per_view,
        has_cls=has_cls,
        static_selected=count_selected(static_mask),
        gripper_selected=count_selected(gripper_mask),
    )


def weights_to_heatmap(
    weights: np.ndarray,
    target_size: Tuple[int, int] = (200, 200),
    grid_size: Optional[Tuple[int, int]] = None,
    binary_mask: bool = False,
) -> np.ndarray:
    """Convert flat attention weights to resized heatmap image.

    Args:
        weights: numpy array of shape [V] with attention weights (spatial tokens only)
        target_size: (width, height) for output heatmap
        grid_size: Optional (height, width) for spatial grid, auto-detected if None
        binary_mask: If True, treat weights as binary (0/1) and use nearest neighbor
            interpolation for crisp edges. If False, use linear interpolation.

    Returns:
        Grayscale heatmap image as numpy array [H, W] with values 0-255
    """
    weights_np = weights.flatten()
    num_tokens = len(weights_np)

    # Auto-detect grid if not specified
    if grid_size is None:
        grid_size, _ = infer_grid_size(num_tokens)

    # Reshape to spatial grid (may truncate/pad if mismatch)
    h, w = grid_size
    target_len = h * w
    if num_tokens < target_len:
        weights_np = np.pad(weights_np, (0, target_len - num_tokens))
    elif num_tokens > target_len:
        weights_np = weights_np[:target_len]

    heatmap = weights_np.reshape(grid_size)

    if binary_mask:
        # Binary mask: 0 -> 0, 1 -> 255 (no normalization needed)
        heatmap = (heatmap * 255).astype(np.uint8)
        # Use NEAREST neighbor to keep crisp edges
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        # Continuous weights: normalize and use linear interpolation
        min_val, max_val = heatmap.min(), heatmap.max()
        if max_val - min_val > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return heatmap


def create_highlight_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),  # Red for selected
) -> np.ndarray:
    """Create overlay that highlights regions based on attention weights.

    For continuous weights, the highlight intensity is proportional to the weight.
    For binary masks, only selected regions are highlighted.

    Args:
        image: RGB image as numpy array [H, W, 3]
        heatmap: Grayscale heatmap [H, W] with values 0-255 (intensity = attention weight)
        alpha: Maximum blending factor for highlighted regions (0=image, 1=color)
        highlight_color: RGB color for highlighted regions

    Returns:
        Blended RGB image as numpy array [H, W, 3]
    """
    result = image.copy().astype(np.float32)

    # Use heatmap values as continuous blend weights (normalized to 0-1)
    # Higher attention = more highlight color
    weight = heatmap.astype(np.float32) / 255.0  # [H, W] in range [0, 1]

    # Scale by alpha to control maximum highlight intensity
    weight = weight * alpha

    # Expand to 3 channels
    weight_3ch = np.stack([weight, weight, weight], axis=-1)

    # Create highlight layer
    highlight = np.full_like(result, highlight_color, dtype=np.float32)

    # Blend: result = image * (1 - weight) + highlight * weight
    result = result * (1 - weight_3ch) + highlight * weight_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend heatmap onto image.

    Args:
        image: RGB image as numpy array [H, W, 3]
        heatmap: RGB heatmap as numpy array [H, W, 3]
        alpha: Blending factor (0=image only, 1=heatmap only)

    Returns:
        Blended RGB image as numpy array [H, W, 3]
    """
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 25),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Add text with background for visibility.

    Args:
        image: RGB image as numpy array [H, W, 3]
        text: Text to overlay
        position: (x, y) position for text
        font_scale: Font size scale
        color: RGB text color
        bg_color: RGB background color

    Returns:
        Image with text overlay as numpy array [H, W, 3]
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, 2)
    cv2.rectangle(
        img,
        (position[0] - 2, position[1] - h - 2),
        (position[0] + w + 2, position[1] + 4),
        bg_color,
        -1,
    )
    cv2.putText(img, text, position, font, font_scale, color, 2)
    return img


def apply_jet_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Apply JET colormap overlay with uniform blending.

    Shows full gradient from blue (low attention) to red (high attention).

    Args:
        image: RGB image as numpy array [H, W, 3]
        heatmap: Grayscale heatmap [H, W] with values 0-255
        alpha: Blending factor (0=original only, 1=heatmap only)

    Returns:
        Blended RGB image as numpy array [H, W, 3]
    """
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap_jet, alpha, 0)


class AttentionVisualizer:
    """Manages attention visualization during evaluation.

    Supports dual-view configurations (rgb_static + rgb_gripper) where:
    - 100 total tokens = 50 per view (Florence-2-large @ 224x224)
    - Each view has 49 spatial tokens (7x7) + 1 CLS token
    - Visualization shows side-by-side heatmaps for both views

    Output structure:
        {save_dir}/
          sequence_000/
            0_move_slider_right/
              video_original.mp4      # Original video (side-by-side views)
              video_attention.mp4     # Video with attention heatmap overlay
              frames/
                step_0000.png         # Attention heatmap overlay frames
                step_0010.png
                ...
              frames_original/        # Original frames without heatmap
                step_0000.png
                step_0010.png
                ...
              success (or failure)
            1_rotate_red_block_left/
              ...
          sequence_001/
            ...
    """

    def __init__(
        self,
        save_dir: Path,
        alpha: float = 0.4,
        fps: int = 30,
        highlight_color: Tuple[int, int, int] = (255, 0, 0),
    ):
        """Initialize visualizer.

        Args:
            save_dir: Base directory for saving visualizations
            alpha: Blending factor for heatmap overlay (0=image, 1=heatmap)
            fps: Frame rate for output videos
            highlight_color: RGB color for selected token highlighting
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.fps = fps
        self.highlight_color = highlight_color

        self.current_sequence_dir = None
        self.current_subtask_dir = None
        self.frames_original = []  # Original frames (side-by-side views)
        self.frames_attention = []  # Frames with attention heatmap overlay
        self.frame_idx = 0
        self._dual_view_info = None  # Cached DualViewMasks for grid size

    def start_sequence(self, seq_idx: int):
        """Start a new evaluation sequence.

        Args:
            seq_idx: Sequence index number
        """
        self.current_sequence_dir = self.save_dir / f"sequence_{seq_idx:03d}"
        self.current_sequence_dir.mkdir(parents=True, exist_ok=True)
        self._dual_view_info = None  # Reset for new sequence

    def start_subtask(self, subtask_idx: int, subtask_name: str):
        """Start a new subtask within current sequence.

        Args:
            subtask_idx: Subtask index within sequence
            subtask_name: Name of the subtask (e.g., "move_slider_right")
        """
        if self.current_sequence_dir is None:
            raise RuntimeError("Must call start_sequence() before start_subtask()")

        self.current_subtask_dir = (
            self.current_sequence_dir / f"{subtask_idx}_{subtask_name}"
        )
        self.current_subtask_dir.mkdir(exist_ok=True)
        (self.current_subtask_dir / "frames").mkdir(exist_ok=True)
        (self.current_subtask_dir / "frames_original").mkdir(exist_ok=True)

        self.frames_original = []
        self.frames_attention = []
        self.frame_idx = 0

    def _create_dual_view_visualization(
        self,
        rgb_static: np.ndarray,
        rgb_gripper: np.ndarray,
        attn_weights: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, DualViewMasks]:
        """Create side-by-side visualization for dual views.

        Args:
            rgb_static: RGB image from static camera [H, W, 3]
            rgb_gripper: RGB image from gripper camera [H, W, 3]
            attn_weights: Attention weights [B, V] or [V] with V=100 (dual view)

        Returns:
            Tuple of (original_sidebyside, attention_sidebyside, dual_view_masks)
        """
        # IMMEDIATELY preserve original inputs before ANY processing
        # This ensures originals are safe even if inputs are numpy views
        rgb_static_orig = rgb_static.copy()
        rgb_gripper_orig = rgb_gripper.copy()

        # Split mask into per-view components
        weights = attn_weights[0] if attn_weights.dim() > 1 else attn_weights
        dual_masks = split_dual_view_mask(weights)

        h_s, w_s = rgb_static.shape[:2]
        h_g, w_g = rgb_gripper.shape[:2]

        # Create heatmaps for each view using the correct spatial grid
        heatmap_static = weights_to_heatmap(
            dual_masks.static_mask,
            target_size=(w_s, h_s),
            grid_size=dual_masks.grid_size,
            binary_mask=False,
        )
        heatmap_gripper = weights_to_heatmap(
            dual_masks.gripper_mask,
            target_size=(w_g, h_g),
            grid_size=dual_masks.grid_size,
            binary_mask=False,
        )

        # Apply JET colormap with intensity-based blending
        # Pass copies to protect originals from in-place modification
        vis_static = apply_jet_heatmap_overlay(rgb_static.copy(), heatmap_static, self.alpha)
        vis_gripper = apply_jet_heatmap_overlay(rgb_gripper.copy(), heatmap_gripper, self.alpha)

        # Add per-view text overlays
        spatial_tokens = dual_masks.grid_size[0] * dual_masks.grid_size[1]
        vis_static = add_text_overlay(
            vis_static, f"Static: {dual_masks.static_selected}/{spatial_tokens}"
        )
        vis_gripper = add_text_overlay(
            vis_gripper, f"Gripper: {dual_masks.gripper_selected}/{spatial_tokens}"
        )

        # Create side-by-side layouts
        # Resize gripper view to match static view height if needed
        if h_s != h_g:
            scale = h_s / h_g
            new_w = int(w_g * scale)
            rgb_gripper_orig_resized = cv2.resize(rgb_gripper_orig, (new_w, h_s))
            vis_gripper_resized = cv2.resize(vis_gripper, (new_w, h_s))
        else:
            rgb_gripper_orig_resized = rgb_gripper_orig
            vis_gripper_resized = vis_gripper

        # Use preserved original copies for clean output
        original_sidebyside = np.concatenate(
            [rgb_static_orig, rgb_gripper_orig_resized], axis=1
        )
        attention_sidebyside = np.concatenate(
            [vis_static, vis_gripper_resized], axis=1
        )

        return original_sidebyside, attention_sidebyside, dual_masks

    def _create_single_view_visualization(
        self,
        rgb_image: np.ndarray,
        attn_weights: torch.Tensor,
        num_selected: int,
        num_total: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization for single view (legacy/fallback).

        Args:
            rgb_image: RGB image [H, W, 3]
            attn_weights: Attention weights [B, V] or [V]
            num_selected: Number of tokens selected
            num_total: Total number of vision tokens

        Returns:
            Tuple of (original_frame, attention_frame)
        """
        h, w = rgb_image.shape[:2]

        # Get weights and detect grid size
        weights = attn_weights[0] if attn_weights.dim() > 1 else attn_weights
        weights_np = weights.detach().cpu().numpy().flatten()
        grid_size, has_cls = infer_grid_size(len(weights_np))

        # Exclude CLS token if present
        spatial_tokens = grid_size[0] * grid_size[1]
        if has_cls and len(weights_np) > spatial_tokens:
            weights_np = weights_np[:spatial_tokens]

        heatmap = weights_to_heatmap(
            weights_np, target_size=(w, h), grid_size=grid_size, binary_mask=False
        )
        # Apply JET colormap with intensity-based blending
        vis_frame = apply_jet_heatmap_overlay(rgb_image, heatmap, self.alpha)
        vis_frame = add_text_overlay(vis_frame, f"Sel: {num_selected}/{num_total}")

        return rgb_image.copy(), vis_frame

    def add_frame(
        self,
        rgb_static: np.ndarray,
        attn_weights: Optional[torch.Tensor],
        num_selected: int,
        num_total: int,
        is_policy_step: bool,
        rgb_gripper: Optional[np.ndarray] = None,
    ):
        """Add a frame to the current subtask.

        Heatmap overlay is ONLY applied on policy steps (when model is called).
        Intermediate frames (action chunking) show raw RGB with selection count.

        Supports both single-view and dual-view modes:
        - Single-view: Only rgb_static provided
        - Dual-view: Both rgb_static and rgb_gripper provided (side-by-side output)

        Args:
            rgb_static: RGB image from static camera as numpy array [H, W, 3]
            attn_weights: Optional attention weights tensor [B, V] or [V]
            num_selected: Number of tokens selected
            num_total: Total number of vision tokens
            is_policy_step: Whether this is a policy step (model was called)
            rgb_gripper: Optional RGB image from gripper camera [H, W, 3]
        """
        if self.current_subtask_dir is None:
            raise RuntimeError("Must call start_subtask() before add_frame()")

        is_dual_view = rgb_gripper is not None

        if is_policy_step and attn_weights is not None:
            if is_dual_view:
                original_frame, vis_frame, dual_masks = self._create_dual_view_visualization(
                    rgb_static, rgb_gripper, attn_weights
                )
                self._dual_view_info = dual_masks  # Cache for intermediate frames
            else:
                original_frame, vis_frame = self._create_single_view_visualization(
                    rgb_static, attn_weights, num_selected, num_total
                )
        else:
            # Intermediate frames: use last known layout without heatmap
            if is_dual_view:
                # Create side-by-side without heatmap
                h_s, w_s = rgb_static.shape[:2]
                h_g, w_g = rgb_gripper.shape[:2]
                if h_s != h_g:
                    scale = h_s / h_g
                    new_w = int(w_g * scale)
                    rgb_gripper_resized = cv2.resize(rgb_gripper, (new_w, h_s))
                else:
                    rgb_gripper_resized = rgb_gripper

                original_frame = np.concatenate([rgb_static, rgb_gripper_resized], axis=1)

                # Add selection text from cached data if available
                if self._dual_view_info is not None:
                    dm = self._dual_view_info
                    spatial_tokens = dm.grid_size[0] * dm.grid_size[1]
                    vis_static = add_text_overlay(
                        rgb_static.copy(), f"Static: {dm.static_selected}/{spatial_tokens}"
                    )
                    vis_gripper = add_text_overlay(
                        rgb_gripper_resized.copy(), f"Gripper: {dm.gripper_selected}/{spatial_tokens}"
                    )
                    vis_frame = np.concatenate([vis_static, vis_gripper], axis=1)
                else:
                    vis_frame = original_frame.copy()
            else:
                original_frame = rgb_static.copy()
                vis_frame = add_text_overlay(rgb_static.copy(), f"Sel: {num_selected}/{num_total}")

        # Save frames on policy steps only
        if is_policy_step:
            # Save ORIGINAL frame (no heatmap, no text)
            orig_path = (
                self.current_subtask_dir / "frames_original" / f"step_{self.frame_idx:04d}.png"
            )
            cv2.imwrite(str(orig_path), cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))

            # Save attention-overlaid frame
            attn_path = (
                self.current_subtask_dir / "frames" / f"step_{self.frame_idx:04d}.png"
            )
            cv2.imwrite(str(attn_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

        # Store for video generation
        self.frames_original.append(original_frame)
        self.frames_attention.append(vis_frame)
        self.frame_idx += 1

    def finish_subtask(self, success: bool):
        """Finish current subtask and save videos.

        Args:
            success: Whether the subtask was completed successfully
        """
        if not self.frames_attention or self.current_subtask_dir is None:
            return

        h, w = self.frames_attention[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Save original video (without heatmap overlay)
        video_orig_path = self.current_subtask_dir / "video_original.mp4"
        out_orig = cv2.VideoWriter(str(video_orig_path), fourcc, self.fps, (w, h))
        for frame in self.frames_original:
            # Handle potential size mismatches during resize
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out_orig.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_orig.release()

        # Save attention video (with heatmap overlay)
        video_attn_path = self.current_subtask_dir / "video_attention.mp4"
        out_attn = cv2.VideoWriter(str(video_attn_path), fourcc, self.fps, (w, h))
        for frame in self.frames_attention:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out_attn.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_attn.release()

        self.frames_original = []
        self.frames_attention = []

        # Write success/failure marker file
        marker = self.current_subtask_dir / ("success" if success else "failure")
        marker.touch()

    def finish_sequence(self):
        """Finish current sequence (cleanup)."""
        self.current_subtask_dir = None
        self.frames_original = []
        self.frames_attention = []
        self.frame_idx = 0
        self._dual_view_info = None
