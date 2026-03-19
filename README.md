# AI Hardware Co-Design Playground

A simulator that models how neural network architecture choices affect hardware performance.
Use it to find Pareto-optimal (accuracy, latency, energy) design points across hardware targets.

## What It Does

Neural network deployment is a joint optimization problem: the same architecture behaves
very differently on a cloud GPU, an edge TPU, and a microcontroller. This simulator models
that relationship using the **roofline model** — a principled way to reason about whether
each layer is compute-bound or memory-bandwidth-bound on a given chip.

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│  NNArchitecture │────▶│PerformanceEstim.│────▶│   PerformanceResult  │
│  - layers       │     │  roofline model │     │  - latency_ms        │
│  - FLOPs        │     │  hw_efficiency  │     │  - energy_mj         │
│  - params       │     │  overhead_fac.  │     │  - throughput_fps    │
└─────────────────┘     └────────┬────────┘     │  - compute/mem split │
                                 │               └──────────────────────┘
┌─────────────────┐              │
│  HardwareSpec   │──────────────┘
│  - peak_tflops  │
│  - mem_bw_gbps  │     ┌─────────────────┐     ┌──────────────────────┐
│  - sram_bytes   │     │CoDesignOptimizer│────▶│  Pareto Front        │
│  - power_watts  │     │  grid search    │     │  (accuracy proxy,    │
│  - ridge_point  │     │  Pareto filter  │     │   latency, energy)   │
└─────────────────┘     └─────────────────┘     └──────────────────────┘
```

## Roofline Model

A layer's **arithmetic intensity** (AI) determines its performance ceiling:

```
AI = FLOPs / bytes_of_memory_traffic

attainable_perf = min(peak_flops, mem_bandwidth × AI)

                    │ Compute roof (flat line)
 Attainable FLOP/s  │──────────────────────────
                    │           /
                    │          / ← memory roof (slope = bandwidth)
                    │         /
                    │        /
                    └────────┴──────────────────
                          ridge point       AI (FLOP/byte)
```

Layers to the left of the ridge point are **memory-bound** — throwing more compute cores
won't help. Layers to the right are **compute-bound** — you need faster arithmetic units.

## Modules

| Module | Purpose |
|--------|---------|
| `codesign/architecture.py` | `NNArchitecture`, `Layer`, `LayerType` — represents networks with precise FLOP/weight counts |
| `codesign/hardware.py` | `HardwareSpec` — models a hardware target; includes pre-built `cloud_gpu()`, `edge_tpu()`, `mcu()` profiles |
| `codesign/estimator.py` | `PerformanceEstimator` — roofline-based latency/energy/memory estimator |
| `codesign/optimizer.py` | `CoDesignOptimizer` — Pareto-optimal search over architecture × hardware space |
| `codesign/roofline.py` | `RooflineAnalyzer` — per-layer roofline analysis with ASCII chart and matplotlib plot |

## Quick Start

```bash
# Run the co-design demo (no extra deps needed)
python3 demo.py

# Run tests
python3 -m pytest tests/ -v
```

Requires Python 3.9+ and numpy. Matplotlib is optional (roofline plots).

## Demo: MobileNet Co-Design Across 3 Targets

```
Hardware: Cloud GPU (A100-80G)    312 TFLOP/s   2000 GB/s  ridge=156 FLOP/byte
Hardware: Edge TPU (Coral)          4 TOPS        26 GB/s   ridge=156 FLOP/byte
Hardware: MCU (Cortex-M7)         896 MFLOP/s    0.4 GB/s  ridge=  2 FLOP/byte

MobileNet-1.0 on each:
  Cloud GPU → 0.06 ms, 23 mJ   (all layers memory-bound — GPU vastly overpowered)
  Edge TPU  → 5.0 ms,  9 mJ   (all layers memory-bound at 8-bit precision)
  MCU       → 2831 ms, 721 mJ  (78% compute-bound — very compute-limited)
```

The optimizer searches 20 architecture variants × 3 hardware targets = 60 design points
and returns the Pareto-optimal subset: designs where no other point beats them on all
three objectives simultaneously (accuracy proxy, latency, energy).

## Extending

**Add a new hardware target:**
```python
from codesign.hardware import HardwareSpec

jetson = HardwareSpec(
    name="Jetson Orin (INT8)",
    peak_tflops=275.0,
    mem_bandwidth_gbps=204.8,
    sram_bytes=64 * 1024**2,
    power_watts=60.0,
    dtype_bytes=1,
)
```

**Define a custom architecture:**
```python
from codesign.architecture import NNArchitecture, Layer, LayerType

arch = NNArchitecture(name="my-net")
arch.add_layer(Layer(
    name="conv1",
    layer_type=LayerType.CONV2D,
    input_shape=(3, 224, 224),
    output_shape=(64, 112, 112),
    kernel_size=3, stride=2,
))
# ... add more layers
```

**Run Pareto-optimal co-design:**
```python
from codesign import CoDesignOptimizer, PerformanceEstimator

optimizer = CoDesignOptimizer(
    estimator=PerformanceEstimator(),
    arch_builder=lambda cfg: build_mobilenet_style(**cfg),
    arch_configs=[{"width_mult": w, "resolution": r} for w in [0.5, 1.0] for r in [128, 224]],
    hw_targets=[cloud_gpu(), edge_tpu(), mcu()],
)
points = optimizer.run()
optimizer.print_pareto_table(points)
```

## Limitations

- Latency estimates assume idealized roofline (no batching, no pipeline effects).
- The accuracy proxy is log(params) — not actual measured accuracy.
- Hardware efficiency factor (default 50%) is a rough constant; real utilization varies.
- Does not model memory hierarchy beyond a single SRAM/DRAM split.

## License

MIT
