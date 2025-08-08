"""
AI Hardware Co-Design Playground

An interactive environment for co-optimizing neural networks and hardware accelerators.
"""

from .core import (
    AcceleratorDesigner,
    Accelerator,
    ModelProfile,
    ModelOptimizer,
    OptimizationResult,
    DesignSpaceExplorer,
    DesignSpaceResult,
    Workflow,
)

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragon-labs.com"

__all__ = [
    "AcceleratorDesigner",
    "Accelerator",
    "ModelProfile", 
    "ModelOptimizer",
    "OptimizationResult",
    "DesignSpaceExplorer",
    "DesignSpaceResult",
    "Workflow",
]