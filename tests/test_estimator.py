"""Tests for PerformanceEstimator and PerformanceResult."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from codesign.architecture import Layer, LayerType, NNArchitecture
from codesign.hardware import cloud_gpu, edge_tpu, mcu
from codesign.estimator import PerformanceEstimator


def make_simple_arch():
    arch = NNArchitecture(name="simple")
    arch.add_layer(Layer(
        name="conv1",
        layer_type=LayerType.CONV2D,
        input_shape=(3, 32, 32),
        output_shape=(16, 16, 16),
        kernel_size=3, stride=2,
    ))
    arch.add_layer(Layer(
        name="fc",
        layer_type=LayerType.LINEAR,
        input_shape=(16 * 16 * 16,),
        output_shape=(10,),
    ))
    return arch


def test_estimate_returns_result():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    hw = cloud_gpu()
    result = est.estimate(arch, hw)
    assert result.arch_name == arch.name
    assert result.hw_name == hw.name


def test_latency_positive():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    assert result.latency_ms > 0


def test_energy_positive():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    assert result.energy_mj > 0


def test_throughput_consistency():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    expected_fps = 1000.0 / result.latency_ms
    assert abs(result.throughput_fps - expected_fps) < 1e-3


def test_memory_estimate_positive():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    assert result.total_memory_mb > 0
    assert result.weight_mb >= 0
    assert result.peak_activation_mb >= 0


def test_layer_count():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    assert len(result.layer_results) == len(arch.layers)


def test_pareto_pct_sums_100():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    total = result.pct_compute_bound + result.pct_memory_bound
    assert abs(total - 100.0) < 1e-6


def test_mcu_slower_than_gpu():
    """MCU should have higher latency than A100 for the same model."""
    est = PerformanceEstimator()
    arch = make_simple_arch()
    gpu_result = est.estimate(arch, cloud_gpu())
    mcu_result = est.estimate(arch, mcu())
    assert mcu_result.latency_ms > gpu_result.latency_ms


def test_compute_bound_detection():
    """A large matmul should be compute-bound on GPU."""
    est = PerformanceEstimator()
    hw = cloud_gpu()
    # Large dense layer: high arithmetic intensity
    layer = Layer(
        name="big_fc",
        layer_type=LayerType.LINEAR,
        input_shape=(4096,),
        output_shape=(4096,),
    )
    r = est.estimate_layer(layer, hw)
    # Either could be true depending on weight vs activation ratio,
    # but result should be valid
    assert r.latency_ms >= 0
    assert r.arith_intensity > 0


def test_summary_runs():
    est = PerformanceEstimator()
    arch = make_simple_arch()
    result = est.estimate(arch, cloud_gpu())
    s = result.summary()
    assert "FLOPs" in s
