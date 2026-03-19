"""
Neural network architecture representation.

Models layers with their compute and memory characteristics
for downstream hardware performance estimation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class LayerType(Enum):
    CONV2D = "conv2d"
    DEPTHWISE_CONV2D = "depthwise_conv2d"
    LINEAR = "linear"
    ATTENTION = "attention"
    BATCH_NORM = "batch_norm"
    ACTIVATION = "activation"
    POOLING = "pooling"


@dataclass
class Layer:
    """A single layer in a neural network."""

    name: str
    layer_type: LayerType

    # Spatial dims (for conv layers)
    input_shape: Tuple[int, ...]   # (C, H, W) or (seq_len, d_model)
    output_shape: Tuple[int, ...]  # (C_out, H_out, W_out) or (seq_len, d_out)

    # Conv params
    kernel_size: int = 1
    stride: int = 1
    groups: int = 1            # groups=in_channels → depthwise

    # Attention params
    num_heads: int = 1

    # Misc
    activation: Optional[str] = None
    use_bias: bool = True

    def num_weights(self) -> int:
        """Number of learnable parameters (weights) in this layer."""
        lt = self.layer_type
        if lt == LayerType.CONV2D:
            c_in = self.input_shape[0]
            c_out = self.output_shape[0]
            return (c_in // self.groups) * c_out * self.kernel_size**2 + (c_out if self.use_bias else 0)
        elif lt == LayerType.DEPTHWISE_CONV2D:
            c_in = self.input_shape[0]
            return c_in * self.kernel_size**2 + (c_in if self.use_bias else 0)
        elif lt == LayerType.LINEAR:
            in_feat = self.input_shape[-1]
            out_feat = self.output_shape[-1]
            return in_feat * out_feat + (out_feat if self.use_bias else 0)
        elif lt == LayerType.ATTENTION:
            d_model = self.input_shape[-1]
            # Q, K, V projections + output projection
            return 4 * d_model * d_model
        elif lt in (LayerType.BATCH_NORM,):
            return 2 * self.input_shape[0]   # gamma + beta per channel
        else:
            return 0

    def flops(self) -> int:
        """
        Multiply-accumulate operations (MACs * 2 = FLOPs).
        Counts only arithmetic ops, not activations separately.
        """
        lt = self.layer_type
        if lt == LayerType.CONV2D:
            c_in = self.input_shape[0]
            c_out, h_out, w_out = self.output_shape
            macs = (c_in // self.groups) * c_out * h_out * w_out * self.kernel_size**2
            return 2 * macs
        elif lt == LayerType.DEPTHWISE_CONV2D:
            c_in = self.input_shape[0]
            _, h_out, w_out = self.output_shape
            macs = c_in * h_out * w_out * self.kernel_size**2
            return 2 * macs
        elif lt == LayerType.LINEAR:
            in_feat = self.input_shape[-1]
            out_feat = self.output_shape[-1]
            return 2 * in_feat * out_feat
        elif lt == LayerType.ATTENTION:
            seq_len = self.input_shape[0]
            d_model = self.input_shape[-1]
            # QKV projections + attention scores + output proj
            qkv_proj = 3 * 2 * seq_len * d_model * d_model
            attn_scores = 2 * seq_len * seq_len * d_model
            out_proj = 2 * seq_len * d_model * d_model
            return qkv_proj + attn_scores + out_proj
        elif lt == LayerType.BATCH_NORM:
            # 4 ops per element (subtract mean, divide std, scale, shift)
            total = 1
            for d in self.input_shape:
                total *= d
            return 4 * total
        else:
            return 0

    def activation_memory_bytes(self, dtype_bytes: int = 4) -> int:
        """Memory needed to store activations (input + output) in bytes."""
        def prod(shape):
            r = 1
            for d in shape:
                r *= d
            return r
        return (prod(self.input_shape) + prod(self.output_shape)) * dtype_bytes

    def weight_memory_bytes(self, dtype_bytes: int = 4) -> int:
        """Memory needed to store weights in bytes."""
        return self.num_weights() * dtype_bytes

    def arithmetic_intensity(self) -> float:
        """
        Arithmetic intensity = FLOPs / bytes of memory traffic.
        Assumes weights are loaded once and activations are streamed.
        """
        flops = self.flops()
        bytes_traffic = self.activation_memory_bytes() + self.weight_memory_bytes()
        if bytes_traffic == 0:
            return float("inf")
        return flops / bytes_traffic


@dataclass
class NNArchitecture:
    """A complete neural network architecture."""

    name: str
    layers: List[Layer] = field(default_factory=list)
    description: str = ""

    def add_layer(self, layer: Layer) -> "NNArchitecture":
        self.layers.append(layer)
        return self

    def total_flops(self) -> int:
        return sum(l.flops() for l in self.layers)

    def total_params(self) -> int:
        return sum(l.num_weights() for l in self.layers)

    def total_weight_bytes(self, dtype_bytes: int = 4) -> int:
        return sum(l.weight_memory_bytes(dtype_bytes) for l in self.layers)

    def peak_activation_bytes(self, dtype_bytes: int = 4) -> int:
        """Peak memory for activations (max single-layer footprint)."""
        return max((l.activation_memory_bytes(dtype_bytes) for l in self.layers), default=0)

    def summary(self) -> str:
        lines = [f"Architecture: {self.name}"]
        lines.append(f"  Layers:     {len(self.layers)}")
        lines.append(f"  FLOPs:      {self.total_flops() / 1e9:.3f} GFLOPs")
        lines.append(f"  Params:     {self.total_params() / 1e6:.3f} M")
        lines.append(f"  Weights:    {self.total_weight_bytes() / 1e6:.2f} MB (fp32)")
        lines.append(f"  Peak acts:  {self.peak_activation_bytes() / 1e6:.2f} MB (fp32)")
        return "\n".join(lines)
