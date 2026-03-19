"""
Performance estimator using the roofline model.

Given an NNArchitecture and a HardwareSpec, estimates:
  - Inference latency (ms)
  - Peak memory requirement (MB)
  - Total energy consumption (mJ)
  - Per-layer compute/memory boundedness
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from .architecture import NNArchitecture, Layer
from .hardware import HardwareSpec


@dataclass
class LayerResult:
    """Performance estimate for a single layer."""
    name: str
    flops: int
    arith_intensity: float       # FLOP/byte
    ridge_point: float           # FLOP/byte threshold for this HW
    is_compute_bound: bool
    attainable_flops: float      # actual FLOP/s achievable (roofline ceiling)
    latency_ms: float
    energy_mj: float
    weight_bytes: int
    activation_bytes: int
    fits_in_sram: bool


@dataclass
class PerformanceResult:
    """Aggregate performance estimate for a full network on a hardware target."""
    arch_name: str
    hw_name: str
    total_flops: int
    total_params: int
    weight_mb: float
    peak_activation_mb: float
    total_memory_mb: float
    latency_ms: float
    energy_mj: float
    throughput_fps: float       # frames per second (1 / latency)
    layer_results: List[LayerResult] = field(default_factory=list)

    # Roofline summary
    pct_compute_bound: float = 0.0
    pct_memory_bound: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  {self.arch_name}  →  {self.hw_name}",
            f"{'='*55}",
            f"  FLOPs:       {self.total_flops / 1e9:.3f} GFLOPs",
            f"  Params:      {self.total_params / 1e6:.3f} M",
            f"  Weights:     {self.weight_mb:.2f} MB",
            f"  Peak acts:   {self.peak_activation_mb:.2f} MB",
            f"  Total mem:   {self.total_memory_mb:.2f} MB",
            f"  Latency:     {self.latency_ms:.3f} ms",
            f"  Throughput:  {self.throughput_fps:.1f} FPS",
            f"  Energy:      {self.energy_mj:.3f} mJ",
            f"  Compute-bnd: {self.pct_compute_bound:.1f}%  "
            f"Mem-bnd: {self.pct_memory_bound:.1f}%",
        ]
        return "\n".join(lines)


class PerformanceEstimator:
    """
    Roofline-based performance estimator.

    The roofline model upper-bounds attainable performance as:
        P = min(peak_flops, mem_bandwidth × arithmetic_intensity)

    Hardware efficiency factor accounts for:
    - Instruction overhead, pipeline stalls
    - Data layout mismatches
    - Memory hierarchy effects not captured by simple BW

    We use a conservative hw_efficiency=0.5 by default (50% utilisation).
    """

    def __init__(self, hw_efficiency: float = 0.5, overhead_factor: float = 1.1):
        """
        Parameters
        ----------
        hw_efficiency   Fraction of peak FLOP/s typically achieved (0–1).
        overhead_factor  Multiplier on latency for framework / kernel launch overhead.
        """
        self.hw_efficiency = hw_efficiency
        self.overhead_factor = overhead_factor

    def estimate_layer(self, layer: Layer, hw: HardwareSpec) -> LayerResult:
        flops = layer.flops()
        ai = layer.arithmetic_intensity()
        ridge = hw.ridge_point

        # Roofline: attainable = min(compute_roof, memory_roof * AI)
        compute_roof = hw.peak_flops * self.hw_efficiency
        memory_roof = hw.mem_bandwidth * ai * self.hw_efficiency
        attainable = min(compute_roof, memory_roof)

        is_compute_bound = ai >= ridge

        if attainable > 0:
            latency_s = flops / attainable
        else:
            latency_s = 0.0

        latency_ms = latency_s * 1e3 * self.overhead_factor

        # Energy = power × time
        energy_mj = hw.power_watts * latency_s * 1e3

        weight_b = layer.weight_memory_bytes(hw.dtype_bytes)
        act_b = layer.activation_memory_bytes(hw.dtype_bytes)
        fits = (weight_b + act_b) <= hw.sram_bytes

        return LayerResult(
            name=layer.name,
            flops=flops,
            arith_intensity=ai,
            ridge_point=ridge,
            is_compute_bound=is_compute_bound,
            attainable_flops=attainable,
            latency_ms=latency_ms,
            energy_mj=energy_mj,
            weight_bytes=weight_b,
            activation_bytes=act_b,
            fits_in_sram=fits,
        )

    def estimate(self, arch: NNArchitecture, hw: HardwareSpec) -> PerformanceResult:
        layer_results = [self.estimate_layer(l, hw) for l in arch.layers]

        total_latency = sum(r.latency_ms for r in layer_results)
        total_energy = sum(r.energy_mj for r in layer_results)

        weight_mb = arch.total_weight_bytes(hw.dtype_bytes) / 1e6
        peak_act_mb = arch.peak_activation_bytes(hw.dtype_bytes) / 1e6
        total_mem_mb = weight_mb + peak_act_mb

        n_compute = sum(1 for r in layer_results if r.is_compute_bound)
        n_layers = len(layer_results)

        pct_compute = 100.0 * n_compute / n_layers if n_layers else 0.0
        pct_memory = 100.0 - pct_compute

        throughput = 1000.0 / total_latency if total_latency > 0 else float("inf")

        return PerformanceResult(
            arch_name=arch.name,
            hw_name=hw.name,
            total_flops=arch.total_flops(),
            total_params=arch.total_params(),
            weight_mb=weight_mb,
            peak_activation_mb=peak_act_mb,
            total_memory_mb=total_mem_mb,
            latency_ms=total_latency,
            energy_mj=total_energy,
            throughput_fps=throughput,
            layer_results=layer_results,
            pct_compute_bound=pct_compute,
            pct_memory_bound=pct_memory,
        )
