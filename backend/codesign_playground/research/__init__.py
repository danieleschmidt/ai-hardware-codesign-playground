"""
Research module for AI Hardware Co-Design Playground.

This module contains experimental algorithms, research frameworks,
and novel optimization techniques for hardware-software co-design.
"""

from .novel_algorithms import (
    AlgorithmType,
    ExperimentConfig,
    ExperimentResult,
    QuantumInspiredOptimizer,
    NeuralArchitectureEvolution,
    get_quantum_optimizer,
    get_neural_evolution,
    run_comparative_study
)

__all__ = [
    "AlgorithmType",
    "ExperimentConfig", 
    "ExperimentResult",
    "QuantumInspiredOptimizer",
    "NeuralArchitectureEvolution",
    "get_quantum_optimizer",
    "get_neural_evolution",
    "run_comparative_study"
]