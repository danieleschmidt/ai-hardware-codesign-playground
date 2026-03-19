"""Tests for HardwareSpec and pre-built profiles."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from codesign.hardware import HardwareSpec, cloud_gpu, edge_tpu, mcu


def test_peak_flops_conversion():
    hw = HardwareSpec(
        name="test",
        peak_tflops=10.0,
        mem_bandwidth_gbps=100.0,
        sram_bytes=1024,
        power_watts=100.0,
    )
    assert hw.peak_flops == 10.0e12


def test_mem_bandwidth_conversion():
    hw = HardwareSpec(
        name="test",
        peak_tflops=10.0,
        mem_bandwidth_gbps=200.0,
        sram_bytes=1024,
        power_watts=100.0,
    )
    assert hw.mem_bandwidth == 200.0e9


def test_ridge_point():
    hw = HardwareSpec(
        name="test",
        peak_tflops=100.0,
        mem_bandwidth_gbps=1000.0,
        sram_bytes=1024,
        power_watts=100.0,
    )
    # ridge = peak_flops / mem_bandwidth = 1e14 / 1e12 = 100
    assert abs(hw.ridge_point - 100.0) < 1e-6


def test_efficiency_ratio():
    hw = HardwareSpec(
        name="test",
        peak_tflops=10.0,
        mem_bandwidth_gbps=100.0,
        sram_bytes=1024,
        power_watts=200.0,
    )
    assert abs(hw.efficiency_ratio() - 0.05) < 1e-9


# Pre-built profiles

def test_cloud_gpu_profile():
    hw = cloud_gpu()
    assert hw.peak_tflops > 100.0
    assert hw.mem_bandwidth_gbps > 1000.0
    assert hw.power_watts > 100.0


def test_edge_tpu_profile():
    hw = edge_tpu()
    assert 1.0 <= hw.peak_tflops <= 10.0
    assert hw.power_watts < 10.0


def test_mcu_profile():
    hw = mcu()
    assert hw.peak_tflops < 0.01
    assert hw.power_watts < 1.0


def test_ridge_ordering():
    # GPU should have higher ridge point than MCU (more compute vs bandwidth)
    gpu = cloud_gpu()
    micro = mcu()
    assert gpu.ridge_point > micro.ridge_point


def test_summary_runs():
    for hw in [cloud_gpu(), edge_tpu(), mcu()]:
        s = hw.summary()
        assert hw.name in s
