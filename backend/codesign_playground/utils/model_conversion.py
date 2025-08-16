"""
Model conversion utilities for AI Hardware Co-Design Playground.

This module provides utilities for converting between different ML model formats
(PyTorch, TensorFlow, ONNX) and optimizing models for hardware acceleration.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelConverter:
    """Utility class for converting between different ML model formats."""
    
    def __init__(self):
        """Initialize model converter."""
        self.supported_formats = {
            "pytorch": [".pt", ".pth"],
            "tensorflow": [".pb", ".h5"],
            "onnx": [".onnx"],
            "tflite": [".tflite"],
        }
        
    def convert_to_onnx(
        self, 
        model: Any, 
        input_shape: Tuple[int, ...], 
        output_path: str,
        source_framework: str = "auto"
    ) -> str:
        """
        Convert a model to ONNX format.
        
        Args:
            model: Source model object or path
            input_shape: Input tensor shape for the model
            output_path: Path to save the ONNX model
            source_framework: Source framework ("pytorch", "tensorflow", "auto")
            
        Returns:
            Path to the converted ONNX model
        """
        if source_framework == "auto":
            source_framework = self._detect_framework(model)
        
        if source_framework == "pytorch":
            return self._pytorch_to_onnx(model, input_shape, output_path)
        elif source_framework == "tensorflow":
            return self._tensorflow_to_onnx(model, input_shape, output_path)
        else:
            raise ValueError(f"Conversion from {source_framework} to ONNX not supported")
    
    def convert_to_tflite(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        output_path: str,
        source_framework: str = "auto",
        quantize: bool = False
    ) -> str:
        """
        Convert a model to TensorFlow Lite format.
        
        Args:
            model: Source model object or path
            input_shape: Input tensor shape
            output_path: Path to save the TFLite model
            source_framework: Source framework
            quantize: Whether to apply quantization
            
        Returns:
            Path to the converted TFLite model
        """
        if source_framework == "auto":
            source_framework = self._detect_framework(model)
        
        if source_framework == "tensorflow":
            return self._tensorflow_to_tflite(model, input_shape, output_path, quantize)
        elif source_framework == "onnx":
            # Convert ONNX -> TensorFlow -> TFLite
            temp_tf_path = tempfile.mktemp(suffix=".pb")
            try:
                tf_model = self._onnx_to_tensorflow(model, temp_tf_path)
                return self._tensorflow_to_tflite(tf_model, input_shape, output_path, quantize)
            finally:
                if os.path.exists(temp_tf_path):
                    os.remove(temp_tf_path)
        else:
            raise ValueError(f"Conversion from {source_framework} to TFLite not supported")
    
    def optimize_model(
        self,
        model: Any,
        optimization_level: str = "moderate",
        target_hardware: str = "cpu",
        preserve_accuracy: bool = True
    ) -> Any:
        """
        Apply optimization to a model for hardware acceleration.
        
        Args:
            model: Model to optimize
            optimization_level: "conservative", "moderate", "aggressive"
            target_hardware: "cpu", "gpu", "tpu", "custom"
            preserve_accuracy: Whether to prioritize accuracy preservation
            
        Returns:
            Optimized model
        """
        framework = self._detect_framework(model)
        
        if framework == "pytorch":
            return self._optimize_pytorch_model(model, optimization_level, target_hardware)
        elif framework == "tensorflow":
            return self._optimize_tensorflow_model(model, optimization_level, target_hardware)
        elif framework == "onnx":
            return self._optimize_onnx_model(model, optimization_level, target_hardware)
        else:
            logger.warning(f"Optimization not supported for framework: {framework}")
            return model
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the ML framework of a model."""
        if isinstance(model, str):
            # File path
            path = Path(model)
            suffix = path.suffix.lower()
            
            for framework, extensions in self.supported_formats.items():
                if suffix in extensions:
                    return framework
            
            # Check content for additional hints
            if "torch" in path.name.lower():
                return "pytorch"
            elif "tensorflow" in path.name.lower():
                return "tensorflow"
        else:
            # Model object
            model_type = str(type(model)).lower()
            
            if "torch" in model_type:
                return "pytorch"
            elif "tensorflow" in model_type or "keras" in model_type:
                return "tensorflow"
            elif "onnx" in model_type:
                return "onnx"
        
        # Default fallback
        return "pytorch"
    
    def _pytorch_to_onnx(self, model: Any, input_shape: Tuple[int, ...], output_path: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch
            
            # Load model if it's a path
            if isinstance(model, str):
                model = torch.load(model, map_location='cpu')
            
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=11
            )
            
            logger.info(f"Successfully converted PyTorch model to ONNX: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("PyTorch not available for ONNX conversion")
        except Exception as e:
            raise RuntimeError(f"PyTorch to ONNX conversion failed: {e}")
    
    def _tensorflow_to_onnx(self, model: Any, input_shape: Tuple[int, ...], output_path: str) -> str:
        """Convert TensorFlow model to ONNX."""
        try:
            import tensorflow as tf
            import tf2onnx
            
            # Load model if it's a path
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Convert using tf2onnx
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            output_path, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            
            logger.info(f"Successfully converted TensorFlow model to ONNX: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("TensorFlow or tf2onnx not available for ONNX conversion")
        except Exception as e:
            raise RuntimeError(f"TensorFlow to ONNX conversion failed: {e}")
    
    def _tensorflow_to_tflite(
        self, 
        model: Any, 
        input_shape: Tuple[int, ...], 
        output_path: str, 
        quantize: bool
    ) -> str:
        """Convert TensorFlow model to TFLite."""
        try:
            import tensorflow as tf
            
            # Load model if it's a path
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Apply optimizations
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Successfully converted to TFLite: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("TensorFlow not available for TFLite conversion")
        except Exception as e:
            raise RuntimeError(f"TensorFlow to TFLite conversion failed: {e}")
    
    def _onnx_to_tensorflow(self, model: Any, output_path: str) -> Any:
        """Convert ONNX model to TensorFlow."""
        try:
            import onnx
            import onnx_tf
            
            # Load ONNX model if it's a path
            if isinstance(model, str):
                onnx_model = onnx.load(model)
            else:
                onnx_model = model
            
            # Convert to TensorFlow
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            tf_rep.export_graph(output_path)
            
            logger.info(f"Successfully converted ONNX to TensorFlow: {output_path}")
            return output_path
            
        except ImportError:
            raise RuntimeError("ONNX or onnx-tf not available for conversion")
        except Exception as e:
            raise RuntimeError(f"ONNX to TensorFlow conversion failed: {e}")
    
    def _optimize_pytorch_model(
        self, 
        model: Any, 
        optimization_level: str, 
        target_hardware: str
    ) -> Any:
        """Optimize PyTorch model."""
        try:
            import torch
            
            if isinstance(model, str):
                model = torch.load(model, map_location='cpu')
            
            model.eval()
            
            # Apply optimizations based on level
            if optimization_level == "moderate":
                # Apply basic optimizations
                if hasattr(torch.jit, 'optimize_for_inference'):
                    model = torch.jit.script(model)
                    model = torch.jit.optimize_for_inference(model)
            elif optimization_level == "aggressive":
                # Apply more aggressive optimizations
                model = torch.jit.script(model)
                if hasattr(torch.jit, 'optimize_for_inference'):
                    model = torch.jit.optimize_for_inference(model)
                # Could add quantization here
            
            return model
            
        except Exception as e:
            logger.warning(f"PyTorch optimization failed: {e}")
            return model
    
    def _optimize_tensorflow_model(
        self, 
        model: Any, 
        optimization_level: str, 
        target_hardware: str
    ) -> Any:
        """Optimize TensorFlow model."""
        try:
            import tensorflow as tf
            
            if isinstance(model, str):
                model = tf.keras.models.load_model(model)
            
            # Apply optimizations based on level
            if optimization_level in ["moderate", "aggressive"]:
                # Convert to concrete function for optimization
                full_model = tf.function(lambda x: model(x))
                concrete_func = full_model.get_concrete_function(
                    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
                )
                
                # Apply graph optimizations
                from tensorflow.python.tools import optimize_for_inference_lib
                # This is a simplified optimization - in practice would be more complex
                
            return model
            
        except Exception as e:
            logger.warning(f"TensorFlow optimization failed: {e}")
            return model
    
    def _optimize_onnx_model(
        self, 
        model: Any, 
        optimization_level: str, 
        target_hardware: str
    ) -> Any:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxoptimizer import optimize
            
            if isinstance(model, str):
                model = onnx.load(model)
            
            # Apply ONNX optimizations
            passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_transpose"]
            
            if optimization_level == "moderate":
                passes.extend(["fuse_consecutive_transposes", "fuse_transpose_into_gemm"])
            elif optimization_level == "aggressive":
                passes.extend([
                    "fuse_consecutive_transposes", 
                    "fuse_transpose_into_gemm",
                    "fuse_add_bias_into_conv",
                    "fuse_bn_into_conv"
                ])
            
            optimized_model = optimize(model, passes)
            return optimized_model
            
        except ImportError:
            logger.warning("ONNX optimizer not available")
            return model
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            return model


def convert_model_format(
    input_path: str,
    output_path: str,
    target_format: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    optimization_level: str = "moderate"
) -> str:
    """
    High-level function to convert between model formats.
    
    Args:
        input_path: Path to input model
        output_path: Path for output model
        target_format: Target format ("onnx", "tflite", "pytorch", "tensorflow")
        input_shape: Input tensor shape (required for some conversions)
        optimization_level: Optimization level to apply
        
    Returns:
        Path to converted model
    """
    converter = ModelConverter()
    
    if target_format == "onnx":
        if input_shape is None:
            raise ValueError("input_shape required for ONNX conversion")
        return converter.convert_to_onnx(input_path, input_shape, output_path)
    elif target_format == "tflite":
        if input_shape is None:
            raise ValueError("input_shape required for TFLite conversion")
        return converter.convert_to_tflite(input_path, input_shape, output_path)
    else:
        raise ValueError(f"Target format {target_format} not supported")


def optimize_for_hardware(
    model_path: str,
    target_hardware: str = "cpu",
    optimization_level: str = "moderate",
    output_path: Optional[str] = None
) -> str:
    """
    Optimize a model for specific hardware.
    
    Args:
        model_path: Path to model file
        target_hardware: Target hardware ("cpu", "gpu", "tpu", "custom")
        optimization_level: Optimization level ("conservative", "moderate", "aggressive")
        output_path: Output path (defaults to adding _optimized suffix)
        
    Returns:
        Path to optimized model
    """
    converter = ModelConverter()
    
    # Load and optimize model
    optimized_model = converter.optimize_model(
        model_path, 
        optimization_level, 
        target_hardware
    )
    
    # Save optimized model
    if output_path is None:
        path = Path(model_path)
        output_path = str(path.parent / f"{path.stem}_optimized{path.suffix}")
    
    # Save based on framework
    framework = converter._detect_framework(model_path)
    if framework == "pytorch":
        import torch
        torch.save(optimized_model, output_path)
    elif framework == "tensorflow":
        optimized_model.save(output_path)
    elif framework == "onnx":
        import onnx
        onnx.save(optimized_model, output_path)
    
    logger.info(f"Saved optimized model to: {output_path}")
    return output_path