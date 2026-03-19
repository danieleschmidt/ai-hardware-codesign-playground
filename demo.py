#!/usr/bin/env python3
"""
AI Hardware Co-Design Demo
==========================

Co-designs a MobileNet-style architecture for three hardware targets:
  1. Cloud GPU (NVIDIA A100)
  2. Edge TPU (Google Coral)
  3. MCU (ARM Cortex-M7)

Shows:
  - Per-target performance estimates using the roofline model
  - Roofline analysis (compute vs memory bound layers)
  - Pareto-optimal frontier across architecture / hardware design space
"""

import sys
sys.path.insert(0, ".")

from codesign import (
    NNArchitecture, HardwareSpec,
    PerformanceEstimator,
    CoDesignOptimizer,
    RooflineAnalyzer,
)
from codesign.hardware import cloud_gpu, edge_tpu, mcu
from codesign.optimizer import build_mobilenet_style


def section(title: str) -> None:
    print(f"\n{'#' * 65}")
    print(f"#  {title}")
    print(f"{'#' * 65}\n")


# ---------------------------------------------------------------------------
# 1. Hardware targets
# ---------------------------------------------------------------------------
section("Hardware Targets")
targets = [cloud_gpu(), edge_tpu(), mcu()]
for hw in targets:
    print(hw.summary())
    print()

# ---------------------------------------------------------------------------
# 2. Reference architecture: MobileNet-1.0-224
# ---------------------------------------------------------------------------
section("Reference Architecture: MobileNet α=1.0, 224×224")
ref_arch = build_mobilenet_style(width_mult=1.0, resolution=224)
print(ref_arch.summary())

# ---------------------------------------------------------------------------
# 3. Per-target performance estimates
# ---------------------------------------------------------------------------
section("Performance Estimates (Roofline Model)")
estimator = PerformanceEstimator()
analyzer  = RooflineAnalyzer(estimator)

results = {}
for hw in targets:
    result = analyzer.analyze(ref_arch, hw)
    results[hw.name] = result
    analyzer.print_report(result)
    print()
    analyzer.print_ascii_chart(result)
    print()

# ---------------------------------------------------------------------------
# 4. Explore slimmer variants for constrained hardware
# ---------------------------------------------------------------------------
section("Slim Variants on Edge TPU & MCU")
slim_configs = [
    {"width_mult": 0.25, "resolution": 96},
    {"width_mult": 0.25, "resolution": 128},
    {"width_mult": 0.5,  "resolution": 96},
    {"width_mult": 0.5,  "resolution": 128},
    {"width_mult": 0.75, "resolution": 96},
    {"width_mult": 1.0,  "resolution": 128},
]

print(f"{'Architecture':<32} {'Hardware':<22} {'Lat(ms)':>9} {'Mem(MB)':>9} {'Eng(mJ)':>9}")
print("-" * 85)
for cfg in slim_configs:
    arch = build_mobilenet_style(**cfg)
    for hw in [edge_tpu(), mcu()]:
        r = estimator.estimate(arch, hw)
        print(f"{arch.name:<32} {hw.name:<22} {r.latency_ms:>9.3f} {r.total_memory_mb:>9.2f} {r.energy_mj:>9.4f}")

# ---------------------------------------------------------------------------
# 5. Pareto-optimal co-design
# ---------------------------------------------------------------------------
section("Pareto-Optimal Co-Design (all width × resolution × hardware)")

optimizer = CoDesignOptimizer(
    estimator=estimator,
    hw_targets=targets,
    arch_configs=[
        {"width_mult": w, "resolution": r}
        for w in [0.25, 0.5, 0.75, 1.0]
        for r in [96, 128, 160, 192, 224]
    ],
)

print("Searching design space (20 architectures × 3 hardware targets = 60 points)...")
all_points = optimizer.run()
optimizer.print_pareto_table(all_points)

# ---------------------------------------------------------------------------
# 6. Recommendations
# ---------------------------------------------------------------------------
section("Hardware-Specific Recommendations")
pareto = [p for p in all_points if p.is_pareto_optimal]

for hw in targets:
    hw_pareto = sorted(
        [p for p in pareto if p.hw.name == hw.name],
        key=lambda p: (-p.accuracy_proxy, p.latency_ms),
    )
    best = hw_pareto[0] if hw_pareto else None
    if best:
        print(f"  {hw.name}")
        print(f"    Best arch:  {best.arch.name}")
        print(f"    Latency:    {best.latency_ms:.3f} ms")
        print(f"    Energy:     {best.energy_mj:.4f} mJ")
        print(f"    Memory:     {best.memory_mb:.2f} MB")
        print(f"    Acc proxy:  {best.accuracy_proxy:.3f}")
        print()

print("Done.")
