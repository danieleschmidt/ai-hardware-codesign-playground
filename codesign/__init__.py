"""
AI Hardware Co-Design Simulator

Models how neural network architecture choices affect hardware performance.
Uses the roofline model to estimate compute/memory boundedness and
runs Pareto-optimal search across architecture-hardware design spaces.
"""

from .architecture import NNArchitecture, Layer, LayerType
from .hardware import HardwareSpec
from .estimator import PerformanceEstimator, PerformanceResult
from .optimizer import CoDesignOptimizer, DesignPoint
from .roofline import RooflineAnalyzer

__all__ = [
    "NNArchitecture", "Layer", "LayerType",
    "HardwareSpec",
    "PerformanceEstimator", "PerformanceResult",
    "CoDesignOptimizer", "DesignPoint",
    "RooflineAnalyzer",
]
