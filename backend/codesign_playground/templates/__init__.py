"""
Hardware accelerator templates for the AI Hardware Co-Design Playground.

This package provides reusable, parameterizable hardware accelerator templates
for different AI workloads and optimization strategies.
"""

from .systolic_array import SystolicArray
from .vector_processor import VectorProcessor  
from .transformer_accelerator import TransformerAccelerator
from .custom_template import CustomTemplate

__all__ = [
    "SystolicArray",
    "VectorProcessor", 
    "TransformerAccelerator",
    "CustomTemplate"
]