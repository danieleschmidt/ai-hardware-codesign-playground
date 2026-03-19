"""Tests for CoDesignOptimizer and Pareto frontier."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from codesign.hardware import cloud_gpu, edge_tpu, mcu
from codesign.estimator import PerformanceEstimator
from codesign.optimizer import (
    CoDesignOptimizer, DesignPoint, pareto_front,
    build_mobilenet_style,
)


def test_build_mobilenet_style():
    arch = build_mobilenet_style(width_mult=0.25, resolution=96)
    assert arch.total_params() > 0
    assert arch.total_flops() > 0


def test_mobilenet_width_scaling():
    """Wider MobileNet should have more params."""
    small = build_mobilenet_style(width_mult=0.25, resolution=96)
    large = build_mobilenet_style(width_mult=1.0, resolution=96)
    assert large.total_params() > small.total_params()


def test_optimizer_runs():
    opt = CoDesignOptimizer(
        estimator=PerformanceEstimator(),
        hw_targets=[cloud_gpu()],
        arch_configs=[
            {"width_mult": 0.25, "resolution": 96},
            {"width_mult": 0.5,  "resolution": 96},
        ],
    )
    points = opt.run()
    assert len(points) == 2   # 2 configs × 1 hw


def test_pareto_front_non_empty():
    opt = CoDesignOptimizer(
        estimator=PerformanceEstimator(),
        hw_targets=[cloud_gpu(), edge_tpu()],
        arch_configs=[
            {"width_mult": w, "resolution": r}
            for w in [0.25, 0.5, 1.0]
            for r in [96, 128]
        ],
    )
    points = opt.run()
    pareto = [p for p in points if p.is_pareto_optimal]
    assert len(pareto) > 0
    assert len(pareto) <= len(points)


def test_pareto_domination():
    """A point that is better on all axes must dominate another."""
    from codesign.architecture import NNArchitecture
    from codesign.hardware import HardwareSpec
    from codesign.estimator import PerformanceResult

    def make_point(acc, lat, eng):
        arch = NNArchitecture(name="dummy")
        hw = cloud_gpu()
        result = PerformanceResult(
            arch_name="dummy", hw_name=hw.name,
            total_flops=0, total_params=0,
            weight_mb=0, peak_activation_mb=0,
            total_memory_mb=0,
            latency_ms=lat, energy_mj=eng, throughput_fps=0,
        )
        return DesignPoint(arch=arch, hw=hw, result=result,
                           accuracy_proxy=acc, latency_ms=lat,
                           energy_mj=eng, memory_mb=0)

    good = make_point(acc=0.9, lat=1.0, eng=1.0)
    bad  = make_point(acc=0.5, lat=5.0, eng=5.0)
    assert good.dominates(bad)
    assert not bad.dominates(good)


def test_pareto_front_marks_correctly():
    from codesign.architecture import NNArchitecture
    from codesign.estimator import PerformanceResult

    def make_point(acc, lat, eng):
        arch = NNArchitecture(name="dummy")
        hw = cloud_gpu()
        result = PerformanceResult(
            arch_name="dummy", hw_name=hw.name,
            total_flops=0, total_params=0,
            weight_mb=0, peak_activation_mb=0, total_memory_mb=0,
            latency_ms=lat, energy_mj=eng, throughput_fps=0,
        )
        return DesignPoint(arch=arch, hw=hw, result=result,
                           accuracy_proxy=acc, latency_ms=lat,
                           energy_mj=eng, memory_mb=0)

    # p1 is on Pareto front (high acc, low lat)
    # p2 is dominated by p1
    p1 = make_point(acc=0.9, lat=1.0, eng=1.0)
    p2 = make_point(acc=0.5, lat=5.0, eng=5.0)
    p3 = make_point(acc=0.6, lat=2.0, eng=0.5)  # on front (good energy)

    front = pareto_front([p1, p2, p3])
    assert p1 in front
    assert p2 not in front
    assert p3 in front
