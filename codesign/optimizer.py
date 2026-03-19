"""
Co-design optimizer: searches the (architecture, hardware) design space
for Pareto-optimal tradeoffs between accuracy proxy, latency, and energy.

Pareto optimality: a design point D dominates D' iff D is at least as good
on all objectives and strictly better on at least one.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import itertools

import numpy as np

from .architecture import NNArchitecture, Layer, LayerType
from .hardware import HardwareSpec
from .estimator import PerformanceEstimator, PerformanceResult


@dataclass
class DesignPoint:
    """A single point in the co-design space."""
    arch: NNArchitecture
    hw: HardwareSpec
    result: PerformanceResult

    # Objectives (lower is better for latency/energy; higher for accuracy_proxy)
    accuracy_proxy: float   # e.g. total_params normalised — proxy for capacity
    latency_ms: float
    energy_mj: float
    memory_mb: float

    is_pareto_optimal: bool = False

    def dominates(self, other: "DesignPoint") -> bool:
        """Return True if self Pareto-dominates other (minimise latency/energy,
        maximise accuracy_proxy; we flip accuracy_proxy to -val for uniform min)."""
        # Objectives: [−accuracy_proxy, latency_ms, energy_mj]
        s = (-self.accuracy_proxy, self.latency_ms, self.energy_mj)
        o = (-other.accuracy_proxy, other.latency_ms, other.energy_mj)
        return all(si <= oi for si, oi in zip(s, o)) and any(si < oi for si, oi in zip(s, o))


def pareto_front(points: List[DesignPoint]) -> List[DesignPoint]:
    """Mark Pareto-optimal points (non-dominated set) in place and return them."""
    for p in points:
        p.is_pareto_optimal = True

    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and q.dominates(p):
                p.is_pareto_optimal = False
                break

    return [p for p in points if p.is_pareto_optimal]


# ---------------------------------------------------------------------------
# Architecture generator helpers
# ---------------------------------------------------------------------------

def _mobilenet_block(
    name: str,
    c_in: int, c_out: int, h: int, w: int,
    stride: int = 1, width_mult: float = 1.0
) -> List[Layer]:
    """MobileNetV1-style depthwise-separable block."""
    c_in_scaled = max(1, int(c_in * width_mult))
    c_out_scaled = max(1, int(c_out * width_mult))
    h_out = (h + stride - 1) // stride
    w_out = (w + stride - 1) // stride
    return [
        Layer(
            name=f"{name}/dw",
            layer_type=LayerType.DEPTHWISE_CONV2D,
            input_shape=(c_in_scaled, h, w),
            output_shape=(c_in_scaled, h_out, w_out),
            kernel_size=3,
            stride=stride,
            activation="relu6",
        ),
        Layer(
            name=f"{name}/pw",
            layer_type=LayerType.CONV2D,
            input_shape=(c_in_scaled, h_out, w_out),
            output_shape=(c_out_scaled, h_out, w_out),
            kernel_size=1,
            activation="relu6",
        ),
    ]


def build_mobilenet_style(
    width_mult: float = 1.0,
    resolution: int = 224,
    num_classes: int = 1000,
) -> NNArchitecture:
    """
    MobileNet-V1-style architecture parameterised by width multiplier and
    input resolution. width_mult ∈ {0.25, 0.5, 0.75, 1.0} gives the
    standard variants.
    """
    name = f"MobileNet-w{width_mult:.2f}-r{resolution}"
    arch = NNArchitecture(name=name, description=f"MobileNet-style, α={width_mult}, res={resolution}")
    h = w = resolution

    # Stem conv
    c0 = max(1, int(32 * width_mult))
    arch.add_layer(Layer(
        name="stem",
        layer_type=LayerType.CONV2D,
        input_shape=(3, h, w),
        output_shape=(c0, h // 2, w // 2),
        kernel_size=3, stride=2, activation="relu6",
    ))
    h = w = h // 2

    # DW-sep blocks: (c_in, c_out, stride)
    spec = [
        (32, 64, 1), (64, 128, 2), (128, 128, 1),
        (128, 256, 2), (256, 256, 1), (256, 512, 2),
        *[(512, 512, 1)] * 5,
        (512, 1024, 2), (1024, 1024, 1),
    ]
    for i, (ci, co, s) in enumerate(spec):
        for layer in _mobilenet_block(f"block{i}", ci, co, h, w, stride=s, width_mult=width_mult):
            arch.add_layer(layer)
        h = (h + s - 1) // s
        w = h

    # Global average pool + classifier (modelled as a Linear)
    c_final = max(1, int(1024 * width_mult))
    arch.add_layer(Layer(
        name="classifier",
        layer_type=LayerType.LINEAR,
        input_shape=(c_final,),
        output_shape=(num_classes,),
    ))
    return arch


# ---------------------------------------------------------------------------
# CoDesignOptimizer
# ---------------------------------------------------------------------------

class CoDesignOptimizer:
    """
    Searches over a grid of architecture configurations and hardware targets,
    estimates performance for each (arch, hw) pair, and returns the
    Pareto-optimal frontier.

    Parameters
    ----------
    estimator       PerformanceEstimator to use.
    arch_builder    Callable that maps a config dict → NNArchitecture.
    arch_configs    List of config dicts to try.
    hw_targets      List of HardwareSpec targets.
    accuracy_fn     Optional callable: NNArchitecture → float (accuracy proxy).
                    Defaults to normalised log10(params).
    """

    def __init__(
        self,
        estimator: Optional[PerformanceEstimator] = None,
        arch_builder: Optional[Callable] = None,
        arch_configs: Optional[List[dict]] = None,
        hw_targets: Optional[List[HardwareSpec]] = None,
        accuracy_fn: Optional[Callable] = None,
    ):
        self.estimator = estimator or PerformanceEstimator()
        self.arch_builder = arch_builder or (lambda cfg: build_mobilenet_style(**cfg))
        self.arch_configs = arch_configs or [
            {"width_mult": w, "resolution": r}
            for w in [0.25, 0.5, 0.75, 1.0]
            for r in [96, 128, 160, 192, 224]
        ]
        self.hw_targets = hw_targets or []
        self.accuracy_fn = accuracy_fn or self._default_accuracy

    @staticmethod
    def _default_accuracy(arch: NNArchitecture) -> float:
        """
        Accuracy proxy: normalised log(params).
        More parameters → higher capacity → higher proxy.
        Normalised to [0, 1] over reasonable range (100K – 100M params).
        """
        params = max(arch.total_params(), 1)
        lo, hi = np.log10(1e5), np.log10(1e8)
        val = (np.log10(params) - lo) / (hi - lo)
        return float(np.clip(val, 0.0, 1.0))

    def run(self) -> List[DesignPoint]:
        """Evaluate all (arch_config, hw) pairs and return all design points."""
        points: List[DesignPoint] = []

        for cfg, hw in itertools.product(self.arch_configs, self.hw_targets):
            arch = self.arch_builder(cfg)
            result = self.estimator.estimate(arch, hw)
            acc = self.accuracy_fn(arch)

            dp = DesignPoint(
                arch=arch,
                hw=hw,
                result=result,
                accuracy_proxy=acc,
                latency_ms=result.latency_ms,
                energy_mj=result.energy_mj,
                memory_mb=result.total_memory_mb,
            )
            points.append(dp)

        pareto_front(points)
        return points

    def pareto_points(self, all_points: Optional[List[DesignPoint]] = None) -> List[DesignPoint]:
        """Return only Pareto-optimal design points from a prior run()."""
        pts = all_points if all_points is not None else self.run()
        return [p for p in pts if p.is_pareto_optimal]

    def print_pareto_table(self, points: List[DesignPoint]) -> None:
        pareto = [p for p in points if p.is_pareto_optimal]
        print(f"\n{'Pareto-Optimal Design Points':^80}")
        print("=" * 80)
        hdr = f"{'Architecture':<32} {'Hardware':<22} {'Acc':>5} {'Lat(ms)':>9} {'Eng(mJ)':>9} {'Mem(MB)':>8}"
        print(hdr)
        print("-" * 80)
        pareto_sorted = sorted(pareto, key=lambda p: (-p.accuracy_proxy, p.latency_ms))
        for p in pareto_sorted:
            print(
                f"{p.arch.name:<32} {p.hw.name:<22} "
                f"{p.accuracy_proxy:>5.3f} {p.latency_ms:>9.3f} "
                f"{p.energy_mj:>9.4f} {p.memory_mb:>8.2f}"
            )
        print(f"\n{len(pareto)} / {len(points)} design points on Pareto front")
