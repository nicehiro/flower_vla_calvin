"""
FLOWER VLA Profiling Utilities

Provides comprehensive profiling for timing, token counts, and VRAM usage
following pi0/FlowerVLA methodology for consistent benchmarking.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ProfilerMetrics:
    """Container for per-step profiling metrics."""

    # Timing metrics (ms)
    timings: Dict[str, float] = field(default_factory=dict)

    # Token counts
    tokens: Dict[str, int] = field(default_factory=dict)

    # Memory metrics (MB)
    memory: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timings": self.timings.copy(),
            "tokens": self.tokens.copy(),
            "memory": self.memory.copy(),
        }


class FlowerProfiler:
    """
    Comprehensive profiler for FLOWER VLA evaluation.

    Tracks:
    - Timing breakdown per component (with CUDA synchronization)
    - Visual token counts (before/after selection)
    - VRAM usage (peak, allocated, cached)
    - Derived metrics (control frequency, compression ratios)

    Usage:
        profiler = FlowerProfiler(warmup_steps=5)

        for step in range(num_steps):
            profiler.start_step()

            with profiler.profile("vision_encoder"):
                # vision encoding...

            profiler.record_tokens("vision_tokens_raw", 576)

            # ... rest of forward pass ...

            profiler.end_step()

        profiler.print_report()
    """

    def __init__(
        self,
        warmup_steps: int = 5,
        enabled: bool = True,
        log_to_wandb: bool = False,
        action_chunk_size: int = 1,
    ):
        """
        Initialize the profiler.

        Args:
            warmup_steps: Number of initial steps to skip for warmup
            enabled: Whether profiling is active
            log_to_wandb: Whether to log metrics to wandb
            action_chunk_size: Number of actions per forward pass (for computing effective Hz)
        """
        self.warmup_steps = warmup_steps
        self.enabled = enabled
        self.log_to_wandb = log_to_wandb
        self.action_chunk_size = action_chunk_size

        # Step tracking
        self.total_steps = 0
        self.profiled_steps = 0
        self.total_simulation_steps = 0  # Tracks actual env steps (for avg latency calc)

        # Current step state
        self._current_step: Optional[ProfilerMetrics] = None
        self._step_start_time: Optional[float] = None
        self._active_timers: Dict[str, float] = {}

        # Accumulated metrics across all steps
        self._all_metrics: List[ProfilerMetrics] = []

        # Flag to track if we're past warmup
        self._past_warmup = False

    @property
    def is_warmup(self) -> bool:
        """Check if we're still in warmup phase."""
        return self.total_steps < self.warmup_steps

    def record_simulation_step(self) -> None:
        """
        Record a simulation step (called every env.step).

        This tracks actual environment steps, which may differ from forward passes
        due to action chunking. Used to compute average latency per simulation step.
        """
        if not self.enabled or self.is_warmup:
            return
        self.total_simulation_steps += 1

    def start_step(self) -> None:
        """Start profiling a new step."""
        if not self.enabled:
            return

        self.total_steps += 1

        if self.is_warmup:
            return

        if not self._past_warmup:
            self._past_warmup = True
            # Reset peak memory stats after warmup
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            logger.info(f"[Profiler] Warmup complete ({self.warmup_steps} steps), starting profiling")

        # Initialize new step metrics
        self._current_step = ProfilerMetrics()

        # Sync and record start time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start_time = time.perf_counter()

        # Reset peak memory for this step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_step(self) -> None:
        """End profiling for the current step."""
        if not self.enabled or self.is_warmup or self._current_step is None:
            return

        # Sync and record total step time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self._step_start_time is not None:
            elapsed_ms = (time.perf_counter() - self._step_start_time) * 1000
            self._current_step.timings["step_total"] = elapsed_ms

        # Record memory metrics
        if torch.cuda.is_available():
            self._current_step.memory["vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self._current_step.memory["vram_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            self._current_step.memory["vram_cached_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)

        # Store the step metrics
        self._all_metrics.append(self._current_step)
        self.profiled_steps += 1

        # Reset current step
        self._current_step = None
        self._step_start_time = None

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for timing a code section with CUDA synchronization.

        Args:
            name: Name of the component being profiled

        Usage:
            with profiler.profile("vision_encoder"):
                features = model.encode_vision(images)
        """
        if not self.enabled or self.is_warmup or self._current_step is None:
            yield
            return

        # Sync before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        try:
            yield
        finally:
            # Sync after timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Accumulate if same name called multiple times per step
            if name in self._current_step.timings:
                self._current_step.timings[name] += elapsed_ms
            else:
                self._current_step.timings[name] = elapsed_ms

    def record_tokens(self, name: str, count: int) -> None:
        """
        Record token count for a component.

        Args:
            name: Name of the token count metric
            count: Number of tokens
        """
        if not self.enabled or self.is_warmup or self._current_step is None:
            return
        self._current_step.tokens[name] = count

    def record_metric(self, name: str, value: float, category: str = "timings") -> None:
        """
        Record a custom metric.

        Args:
            name: Name of the metric
            value: Metric value
            category: Category ("timings", "tokens", or "memory")
        """
        if not self.enabled or self.is_warmup or self._current_step is None:
            return

        if category == "timings":
            self._current_step.timings[name] = value
        elif category == "tokens":
            self._current_step.tokens[name] = int(value)
        elif category == "memory":
            self._current_step.memory[name] = value

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate aggregated profiling report with mean/std/min/max.

        Returns:
            Dictionary containing aggregated metrics
        """
        if not self._all_metrics:
            return {"error": "No profiled steps available"}

        report = {
            "metadata": {
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "profiled_steps": self.profiled_steps,
            },
            "timings": {},
            "tokens": {},
            "memory": {},
            "derived": {},
        }

        # Aggregate timing metrics
        timing_keys = set()
        for m in self._all_metrics:
            timing_keys.update(m.timings.keys())

        for key in timing_keys:
            values = [m.timings.get(key, 0) for m in self._all_metrics if key in m.timings]
            if values:
                report["timings"][key] = self._compute_stats(values)

        # Aggregate token metrics
        token_keys = set()
        for m in self._all_metrics:
            token_keys.update(m.tokens.keys())

        for key in token_keys:
            values = [m.tokens.get(key, 0) for m in self._all_metrics if key in m.tokens]
            if values:
                report["tokens"][key] = self._compute_stats(values)

        # Aggregate memory metrics
        memory_keys = set()
        for m in self._all_metrics:
            memory_keys.update(m.memory.keys())

        for key in memory_keys:
            values = [m.memory.get(key, 0) for m in self._all_metrics if key in m.memory]
            if values:
                report["memory"][key] = self._compute_stats(values)

        # Compute derived metrics
        if "step_total" in report["timings"]:
            avg_step_ms = report["timings"]["step_total"]["mean"]
            if avg_step_ms > 0:
                forward_pass_hz = 1000.0 / avg_step_ms
                report["derived"]["forward_pass_hz"] = forward_pass_hz
                report["derived"]["effective_action_hz"] = forward_pass_hz * self.action_chunk_size
                report["derived"]["action_chunk_size"] = self.action_chunk_size

        # Average latency per simulation step
        if "step_total" in report["timings"] and self.total_simulation_steps > 0:
            total_latency_ms = sum(m.timings.get("step_total", 0) for m in self._all_metrics)
            avg_step_latency_ms = total_latency_ms / self.total_simulation_steps
            report["derived"]["avg_step_latency_ms"] = avg_step_latency_ms
            report["derived"]["total_simulation_steps"] = self.total_simulation_steps
            report["derived"]["total_forward_passes"] = self.profiled_steps
            report["derived"]["total_model_latency_ms"] = total_latency_ms

            # Latency breakdown per simulation step (each component / total_simulation_steps)
            report["step_latency_breakdown"] = {}
            for key in report["timings"]:
                total_component_ms = sum(m.timings.get(key, 0) for m in self._all_metrics)
                report["step_latency_breakdown"][key] = total_component_ms / self.total_simulation_steps

        # Compression ratios
        if "vision_tokens_raw" in report["tokens"] and "vision_tokens_after_pre_vlm" in report["tokens"]:
            raw = report["tokens"]["vision_tokens_raw"]["mean"]
            after = report["tokens"]["vision_tokens_after_pre_vlm"]["mean"]
            if after > 0:
                report["derived"]["pre_vlm_compression_ratio"] = raw / after

        if "vl_tokens_after_vlm" in report["tokens"] and "vl_tokens_after_post_vlm" in report["tokens"]:
            before = report["tokens"]["vl_tokens_after_vlm"]["mean"]
            after = report["tokens"]["vl_tokens_after_post_vlm"]["mean"]
            if after > 0:
                report["derived"]["post_vlm_compression_ratio"] = before / after

        return report

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute mean, std, min, max for a list of values."""
        import numpy as np
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values),
        }

    def print_report(self) -> None:
        """Print formatted profiling report to console."""
        report = self.generate_report()

        if "error" in report:
            print(f"Profiling Error: {report['error']}")
            return

        print("\n" + "=" * 60)
        print("FLOWER VLA Profiling Report")
        print("=" * 60)

        meta = report["metadata"]
        print(f"Steps profiled: {meta['profiled_steps']} (after {meta['warmup_steps']} warmup)")

        # Timing breakdown
        if report["timings"]:
            print("\nTiming Breakdown (ms):")
            print("-" * 40)

            # Define preferred order for timing display
            timing_order = [
                "vision_encoder",
                "text_embed",
                "pre_vlm_selection",
                "vlm_encoder",
                "vl_projection",
                "post_vlm_selection",
                "dit_sampling_total",
                "dit_per_step_avg",
                "action_decode",
                "step_total",
            ]

            # Print in order, then any remaining
            printed = set()
            for key in timing_order:
                if key in report["timings"]:
                    stats = report["timings"][key]
                    print(f"  {key:30s}: {stats['mean']:8.2f} +/- {stats['std']:6.2f}")
                    printed.add(key)

            for key, stats in report["timings"].items():
                if key not in printed:
                    print(f"  {key:30s}: {stats['mean']:8.2f} +/- {stats['std']:6.2f}")

        # Step latency breakdown (per simulation step, accounting for action chunking)
        if report.get("step_latency_breakdown"):
            print("\nStep Latency Breakdown (ms per simulation step):")
            print("-" * 40)

            # Use same preferred order
            timing_order = [
                "vision_encoder",
                "text_embed",
                "pre_vlm_selection",
                "vlm_encoder",
                "vl_projection",
                "post_vlm_selection",
                "dit_sampling_total",
                "dit_per_step_avg",
                "action_decode",
                "step_total",
            ]

            breakdown = report["step_latency_breakdown"]
            printed = set()
            for key in timing_order:
                if key in breakdown:
                    print(f"  {key:30s}: {breakdown[key]:8.3f}")
                    printed.add(key)

            for key, value in breakdown.items():
                if key not in printed:
                    print(f"  {key:30s}: {value:8.3f}")

        # Token counts
        if report["tokens"]:
            print("\nToken Counts:")
            print("-" * 40)

            token_order = [
                "vision_tokens_raw",
                "vision_tokens_after_pre_vlm",
                "text_tokens",
                "vl_tokens_after_vlm",
                "vl_tokens_after_post_vlm",
            ]

            printed = set()
            for key in token_order:
                if key in report["tokens"]:
                    stats = report["tokens"][key]
                    print(f"  {key:30s}: {stats['mean']:8.1f}")
                    printed.add(key)

            for key, stats in report["tokens"].items():
                if key not in printed:
                    print(f"  {key:30s}: {stats['mean']:8.1f}")

        # Memory
        if report["memory"]:
            print("\nMemory (MB):")
            print("-" * 40)
            for key, stats in report["memory"].items():
                print(f"  {key:30s}: {stats['mean']:8.1f}")

        # Derived metrics
        if report["derived"]:
            print("\nDerived Metrics:")
            print("-" * 40)
            if "forward_pass_hz" in report["derived"]:
                print(f"  {'Forward Pass Frequency':30s}: {report['derived']['forward_pass_hz']:8.1f} Hz")
            if "effective_action_hz" in report["derived"]:
                chunk_size = report["derived"].get("action_chunk_size", 1)
                print(f"  {'Effective Action Rate':30s}: {report['derived']['effective_action_hz']:8.1f} Hz (chunk={chunk_size})")
            if "avg_step_latency_ms" in report["derived"]:
                sim_steps = report["derived"].get("total_simulation_steps", 0)
                fwd_passes = report["derived"].get("total_forward_passes", 0)
                print(f"  {'Avg Step Latency':30s}: {report['derived']['avg_step_latency_ms']:8.3f} ms")
                print(f"  {'Total Simulation Steps':30s}: {sim_steps:8d}")
                print(f"  {'Total Forward Passes':30s}: {fwd_passes:8d}")
            if "pre_vlm_compression_ratio" in report["derived"]:
                print(f"  {'Pre-VLM Compression':30s}: {report['derived']['pre_vlm_compression_ratio']:8.1f}x")
            if "post_vlm_compression_ratio" in report["derived"]:
                print(f"  {'Post-VLM Compression':30s}: {report['derived']['post_vlm_compression_ratio']:8.1f}x")

        print("=" * 60 + "\n")

    def save_report(self, path: Path) -> None:
        """
        Save profiling report to JSON file.

        Args:
            path: Path to save the report
        """
        report = self.generate_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Profiler] Report saved to {path}")

    def log_to_wandb_summary(self) -> None:
        """Log profiling metrics to wandb summary."""
        if not self.log_to_wandb:
            return

        try:
            import wandb
            if wandb.run is None:
                return

            report = self.generate_report()

            # Log timing metrics
            for key, stats in report.get("timings", {}).items():
                wandb.run.summary[f"profiling/timing/{key}_mean_ms"] = stats["mean"]
                wandb.run.summary[f"profiling/timing/{key}_std_ms"] = stats["std"]

            # Log token metrics
            for key, stats in report.get("tokens", {}).items():
                wandb.run.summary[f"profiling/tokens/{key}_mean"] = stats["mean"]

            # Log memory metrics
            for key, stats in report.get("memory", {}).items():
                wandb.run.summary[f"profiling/memory/{key}_mean_mb"] = stats["mean"]

            # Log derived metrics
            for key, value in report.get("derived", {}).items():
                wandb.run.summary[f"profiling/derived/{key}"] = value

        except ImportError:
            logger.warning("[Profiler] wandb not available for logging")

    def reset(self) -> None:
        """Reset all profiling state."""
        self.total_steps = 0
        self.profiled_steps = 0
        self.total_simulation_steps = 0
        self._current_step = None
        self._step_start_time = None
        self._active_timers = {}
        self._all_metrics = []
        self._past_warmup = False
