"""
Hardware specification models.

Represents hardware accelerators with the parameters needed for
roofline-model performance estimation: peak FLOP/s, memory bandwidth,
on-chip SRAM, power budget.

Pre-built profiles: cloud_gpu, edge_tpu, mcu
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HardwareSpec:
    """
    Hardware accelerator specification.

    Attributes
    ----------
    name            Human-readable label.
    peak_tflops     Peak theoretical throughput (TFLOP/s, float32 unless noted).
    mem_bandwidth_gbps  Off-chip memory bandwidth (GB/s).
    sram_bytes      On-chip SRAM / cache (bytes). Used to judge whether
                    weights fit on-chip (cache-resident) or must stream.
    power_watts     Thermal design power (W).
    clock_mhz       Clock frequency (MHz) — used for latency modelling.
    compute_units   Number of parallel compute units (informational).
    dtype_bytes     Native compute dtype width (default 4 = fp32;
                    set to 2 for fp16/bf16, 1 for int8).
    """

    name: str

    # Performance envelope
    peak_tflops: float         # e.g. 312.0 for A100 fp16
    mem_bandwidth_gbps: float  # e.g. 2000 for HBM2e

    # On-chip resources
    sram_bytes: int            # e.g. 40 MB for A100 L2 + shared mem
    power_watts: float

    # Optional detail
    clock_mhz: float = 1000.0
    compute_units: int = 1
    dtype_bytes: int = 4        # fp32 by default

    @property
    def peak_flops(self) -> float:
        """Peak FLOP/s in float."""
        return self.peak_tflops * 1e12

    @property
    def mem_bandwidth(self) -> float:
        """Off-chip memory bandwidth in bytes/s."""
        return self.mem_bandwidth_gbps * 1e9

    @property
    def ridge_point(self) -> float:
        """
        Ridge point of the roofline model (FLOP/byte).
        Layers with arithmetic intensity above this are compute-bound;
        below → memory-bound.
        """
        return self.peak_flops / self.mem_bandwidth

    def efficiency_ratio(self) -> float:
        """TFLOPS per Watt."""
        return self.peak_tflops / self.power_watts

    def summary(self) -> str:
        lines = [f"Hardware: {self.name}"]
        lines.append(f"  Peak:       {self.peak_tflops:.1f} TFLOP/s (dtype {self.dtype_bytes*8}-bit)")
        lines.append(f"  Bandwidth:  {self.mem_bandwidth_gbps:.0f} GB/s")
        lines.append(f"  SRAM:       {self.sram_bytes / 1e6:.1f} MB on-chip")
        lines.append(f"  Power:      {self.power_watts:.0f} W  ({self.efficiency_ratio():.2f} TFLOPS/W)")
        lines.append(f"  Ridge pt:   {self.ridge_point:.1f} FLOP/byte")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built hardware profiles
# ---------------------------------------------------------------------------

def cloud_gpu() -> HardwareSpec:
    """
    NVIDIA A100-80GB SXM (fp16/bf16 tensor core peak).
    Represents a modern data-center GPU for cloud inference/training.
    """
    return HardwareSpec(
        name="Cloud GPU (A100-80G)",
        peak_tflops=312.0,        # fp16 tensor cores
        mem_bandwidth_gbps=2000.0,
        sram_bytes=40 * 1024**2,  # ~40 MB L2 + shared mem
        power_watts=400.0,
        clock_mhz=1410,
        compute_units=6912,       # CUDA cores
        dtype_bytes=2,            # fp16
    )


def edge_tpu() -> HardwareSpec:
    """
    Google Edge TPU (Coral USB / PCIe).
    Represents a purpose-built edge inference accelerator (int8).
    """
    return HardwareSpec(
        name="Edge TPU (Coral)",
        peak_tflops=4.0,          # 4 TOPS int8
        mem_bandwidth_gbps=25.6,  # LPDDR4x
        sram_bytes=8 * 1024**2,   # 8 MB on-chip SRAM
        power_watts=2.0,
        clock_mhz=500,
        compute_units=256,
        dtype_bytes=1,            # int8
    )


def mcu() -> HardwareSpec:
    """
    ARM Cortex-M7 microcontroller (STM32H7 class).
    Represents a deeply embedded MCU for ultra-low-power inference.
    """
    return HardwareSpec(
        name="MCU (Cortex-M7)",
        peak_tflops=0.000896,     # ~896 MFLOP/s @ 480 MHz with FPU
        mem_bandwidth_gbps=0.426, # ~426 MB/s AXI bus
        sram_bytes=512 * 1024,    # 512 KB SRAM
        power_watts=0.280,
        clock_mhz=480,
        compute_units=1,
        dtype_bytes=4,            # fp32 (no int8 native)
    )
