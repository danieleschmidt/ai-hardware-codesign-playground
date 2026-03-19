"""Tests for NNArchitecture and Layer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from codesign.architecture import Layer, LayerType, NNArchitecture


# ---------------------------------------------------------------------------
# Layer FLOP counting
# ---------------------------------------------------------------------------

def test_conv2d_flops():
    layer = Layer(
        name="conv",
        layer_type=LayerType.CONV2D,
        input_shape=(3, 224, 224),
        output_shape=(32, 112, 112),
        kernel_size=3,
        stride=2,
    )
    # MACs = c_in * c_out * h_out * w_out * k^2
    expected_macs = 3 * 32 * 112 * 112 * 9
    assert layer.flops() == 2 * expected_macs


def test_depthwise_conv2d_flops():
    layer = Layer(
        name="dw",
        layer_type=LayerType.DEPTHWISE_CONV2D,
        input_shape=(32, 112, 112),
        output_shape=(32, 112, 112),
        kernel_size=3,
        stride=1,
    )
    # MACs = c_in * h_out * w_out * k^2  (one filter per channel)
    expected_macs = 32 * 112 * 112 * 9
    assert layer.flops() == 2 * expected_macs


def test_linear_flops():
    layer = Layer(
        name="fc",
        layer_type=LayerType.LINEAR,
        input_shape=(1280,),
        output_shape=(1000,),
    )
    assert layer.flops() == 2 * 1280 * 1000


def test_attention_flops():
    layer = Layer(
        name="attn",
        layer_type=LayerType.ATTENTION,
        input_shape=(197, 768),
        output_shape=(197, 768),
        num_heads=12,
    )
    flops = layer.flops()
    assert flops > 0


# ---------------------------------------------------------------------------
# Weight counting
# ---------------------------------------------------------------------------

def test_conv2d_weights():
    layer = Layer(
        name="conv",
        layer_type=LayerType.CONV2D,
        input_shape=(3, 224, 224),
        output_shape=(32, 112, 112),
        kernel_size=3,
        use_bias=True,
    )
    # 3 * 32 * 9 + 32 (bias)
    assert layer.num_weights() == 3 * 32 * 9 + 32


def test_linear_weights():
    layer = Layer(
        name="fc",
        layer_type=LayerType.LINEAR,
        input_shape=(512,),
        output_shape=(256,),
        use_bias=True,
    )
    assert layer.num_weights() == 512 * 256 + 256


def test_batch_norm_weights():
    layer = Layer(
        name="bn",
        layer_type=LayerType.BATCH_NORM,
        input_shape=(64, 56, 56),
        output_shape=(64, 56, 56),
    )
    # gamma + beta per channel
    assert layer.num_weights() == 128


# ---------------------------------------------------------------------------
# Arithmetic intensity
# ---------------------------------------------------------------------------

def test_arithmetic_intensity_positive():
    layer = Layer(
        name="conv",
        layer_type=LayerType.CONV2D,
        input_shape=(3, 112, 112),
        output_shape=(16, 56, 56),
        kernel_size=3,
        stride=2,
    )
    ai = layer.arithmetic_intensity()
    assert ai > 0


# ---------------------------------------------------------------------------
# NNArchitecture aggregate stats
# ---------------------------------------------------------------------------

def test_architecture_total_flops():
    arch = NNArchitecture(name="tiny")
    arch.add_layer(Layer(
        name="l1",
        layer_type=LayerType.LINEAR,
        input_shape=(128,),
        output_shape=(64,),
    ))
    arch.add_layer(Layer(
        name="l2",
        layer_type=LayerType.LINEAR,
        input_shape=(64,),
        output_shape=(10,),
    ))
    expected = 2 * 128 * 64 + 2 * 64 * 10
    assert arch.total_flops() == expected


def test_architecture_total_params():
    arch = NNArchitecture(name="tiny")
    arch.add_layer(Layer(
        name="l1",
        layer_type=LayerType.LINEAR,
        input_shape=(128,),
        output_shape=(64,),
        use_bias=False,
    ))
    assert arch.total_params() == 128 * 64


def test_architecture_summary_runs():
    arch = NNArchitecture(name="test")
    arch.add_layer(Layer(
        name="conv",
        layer_type=LayerType.CONV2D,
        input_shape=(3, 32, 32),
        output_shape=(16, 16, 16),
        kernel_size=3,
        stride=2,
    ))
    s = arch.summary()
    assert "test" in s
    assert "FLOPs" in s
