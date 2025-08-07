"""
AI Hardware Co-Design Playground with Sentiment Analysis

An interactive environment for co-optimizing neural networks and hardware accelerators,
now with integrated sentiment analysis capabilities.
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
from .sentiment_analyzer import (
    SentimentAnalyzerAPI,
    SentimentResult,
    SentimentLabel,
    SimpleSentimentAnalyzer,
)

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragon-labs.com"

__all__ = [
    # Hardware Co-Design
    "AcceleratorDesigner",
    "Accelerator",
    "ModelProfile", 
    "ModelOptimizer",
    "OptimizationResult",
    "DesignSpaceExplorer",
    "DesignSpaceResult",
    "Workflow",
    # Sentiment Analysis
    "SentimentAnalyzerAPI",
    "SentimentResult",
    "SentimentLabel",
    "SimpleSentimentAnalyzer",
]