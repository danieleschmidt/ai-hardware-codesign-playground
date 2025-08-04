"""
Transformer Accelerator hardware template specialized for attention mechanisms.

This module provides a hardware accelerator optimized for transformer models
with dedicated attention computation units and memory hierarchy.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""
    sequence_length: int
    embedding_dim: int
    num_heads: int
    head_dim: int
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate attention configuration."""
        if self.head_dim * self.num_heads != self.embedding_dim:
            raise ValueError("head_dim * num_heads must equal embedding_dim")


@dataclass
class PerformanceEstimate:
    """Performance estimation for transformer operations."""
    attention_gflops: float = 0.0
    feedforward_gflops: float = 0.0
    total_gflops: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    latency_ms: float = 0.0
    throughput_tokens_s: float = 0.0


class TransformerAccelerator:
    """
    Specialized accelerator for transformer attention mechanisms.
    
    Features:
    - Multi-head attention units
    - Optimized softmax computation
    - Configurable precision (FP16, BF16, INT8)
    - Memory hierarchy optimized for attention patterns
    """
    
    def __init__(
        self,
        max_sequence_length: int = 2048,
        embedding_dim: int = 768,
        num_heads: int = 12,
        precision: str = "fp16",
        num_attention_units: int = 4
    ):
        """
        Initialize transformer accelerator.
        
        Args:
            max_sequence_length: Maximum supported sequence length
            embedding_dim: Model embedding dimension
            num_heads: Number of attention heads
            precision: Numerical precision ("fp16", "bf16", "int8", "fp32")
            num_attention_units: Number of parallel attention units
        """
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.precision = precision
        self.num_attention_units = num_attention_units
        
        # Derived parameters
        self.head_dim = embedding_dim // num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        
        # Memory configuration
        self.attention_memory_kb = self._calculate_attention_memory()
        self.weight_memory_kb = self._calculate_weight_memory()
        
        # Performance model
        self.performance_model = {}
        
        # Hardware resources
        self.resource_estimate = {}
        
        logger.info(
            "Initialized TransformerAccelerator",
            max_seq_len=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            precision=precision,
            attention_units=num_attention_units
        )
    
    def optimize_for_model(self, model_name: str) -> None:
        """
        Optimize accelerator for specific transformer model.
        
        Args:
            model_name: Name of transformer model ("gpt2", "bert", "t5", etc.)
        """
        model_configs = {
            "gpt2": {
                "sequence_length": 1024,
                "embedding_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "vocab_size": 50257
            },
            "gpt2-medium": {
                "sequence_length": 1024,
                "embedding_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "vocab_size": 50257
            },
            "bert-base": {
                "sequence_length": 512,
                "embedding_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "vocab_size": 30522
            },
            "bert-large": {
                "sequence_length": 512,
                "embedding_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "vocab_size": 30522
            },
            "t5-base": {
                "sequence_length": 512,
                "embedding_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "vocab_size": 32128
            }
        }
        
        if model_name not in model_configs:
            logger.warning(f"Unknown model: {model_name}, using default configuration")
            return
        
        config = model_configs[model_name]
        
        # Update configuration
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.embedding_dim // self.num_heads
        
        # Optimize attention units for model
        optimal_units = min(self.num_heads, 8)  # Cap at 8 units
        self.num_attention_units = optimal_units
        
        # Update memory requirements
        self.attention_memory_kb = self._calculate_attention_memory()
        self.weight_memory_kb = self._calculate_weight_memory()
        
        # Model-specific optimizations
        if "gpt" in model_name:
            self._optimize_for_autoregressive()
        elif "bert" in model_name:
            self._optimize_for_bidirectional()
        elif "t5" in model_name:
            self._optimize_for_encoder_decoder()
        
        logger.info(f"Optimized accelerator for {model_name} model")
    
    def estimate_performance(
        self,
        batch_size: int = 1,
        sequence_length: int = 512,
        frequency_mhz: float = 800.0
    ) -> PerformanceEstimate:
        """
        Estimate performance for given configuration.
        
        Args:
            batch_size: Batch size
            sequence_length: Input sequence length
            frequency_mhz: Operating frequency
            
        Returns:
            Performance estimates
        """
        if sequence_length > self.max_sequence_length:
            logger.warning(f"Sequence length {sequence_length} exceeds maximum {self.max_sequence_length}")
            sequence_length = self.max_sequence_length
        
        # Calculate attention computation
        attention_ops = self._calculate_attention_ops(batch_size, sequence_length)
        feedforward_ops = self._calculate_feedforward_ops(batch_size, sequence_length)
        total_ops = attention_ops + feedforward_ops
        
        # Calculate throughput
        peak_ops_per_cycle = self.num_attention_units * 4  # 4 ops per attention unit
        peak_ops_per_second = peak_ops_per_cycle * frequency_mhz * 1e6
        
        # Estimate utilization
        utilization = self._estimate_utilization(sequence_length)
        actual_ops_per_second = peak_ops_per_second * utilization
        
        # Calculate latency
        cycles_needed = total_ops / (peak_ops_per_cycle * utilization)
        latency_ms = cycles_needed / (frequency_mhz * 1000)
        
        # Memory bandwidth requirements
        memory_bandwidth = self._estimate_memory_bandwidth(batch_size, sequence_length, frequency_mhz)
        
        return PerformanceEstimate(
            attention_gflops=attention_ops / (latency_ms / 1000) / 1e9 if latency_ms > 0 else 0,
            feedforward_gflops=feedforward_ops / (latency_ms / 1000) / 1e9 if latency_ms > 0 else 0,
            total_gflops=total_ops / (latency_ms / 1000) / 1e9 if latency_ms > 0 else 0,
            memory_bandwidth_gb_s=memory_bandwidth,
            latency_ms=latency_ms,
            throughput_tokens_s=sequence_length * batch_size * 1000 / latency_ms if latency_ms > 0 else 0
        )
    
    def generate_rtl(self, output_dir: str = "./rtl") -> str:
        """
        Generate RTL for transformer accelerator.
        
        Args:
            output_dir: Output directory for RTL files
            
        Returns:
            Path to generated RTL file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        rtl_file = output_path / f"transformer_accelerator_h{self.num_heads}_d{self.embedding_dim}.sv"
        
        # Generate RTL code
        rtl_code = self._generate_transformer_rtl()
        
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        logger.info(f"Generated transformer accelerator RTL: {rtl_file}")
        return str(rtl_file)
    
    def generate_hls(self, output_dir: str = "./hls") -> str:
        """
        Generate HLS C++ code for transformer accelerator.
        
        Args:
            output_dir: Output directory for HLS files
            
        Returns:
            Path to generated HLS file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hls_file = output_path / f"transformer_accelerator.cpp"
        
        # Generate HLS code
        hls_code = self._generate_transformer_hls()
        
        with open(hls_file, 'w') as f:
            f.write(hls_code)
        
        logger.info(f"Generated transformer accelerator HLS: {hls_file}")
        return str(hls_file)
    
    def _calculate_attention_memory(self) -> int:
        """Calculate memory required for attention computation."""
        # Memory for Q, K, V matrices and attention scores
        qkv_memory = 3 * self.max_sequence_length * self.embedding_dim
        attention_scores = self.num_heads * self.max_sequence_length * self.max_sequence_length
        
        # Add buffer for intermediate computations
        total_elements = qkv_memory + attention_scores
        
        # Convert to KB based on precision
        bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[self.precision]
        memory_kb = (total_elements * bytes_per_element) // 1024
        
        return memory_kb
    
    def _calculate_weight_memory(self) -> int:
        """Calculate memory required for model weights."""
        # Weight matrices: Q, K, V projections + output projection
        projection_weights = 4 * self.embedding_dim * self.embedding_dim
        
        # Feed-forward weights (assuming 4x expansion)
        ff_weights = 2 * self.embedding_dim * (4 * self.embedding_dim)
        
        total_weights = projection_weights + ff_weights
        
        # Convert to KB
        bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[self.precision]
        memory_kb = (total_weights * bytes_per_element) // 1024
        
        return memory_kb
    
    def _calculate_attention_ops(self, batch_size: int, sequence_length: int) -> int:
        """Calculate operations for attention computation."""
        # Q @ K^T
        qk_ops = batch_size * self.num_heads * sequence_length * sequence_length * self.head_dim
        
        # Softmax (approximated)
        softmax_ops = batch_size * self.num_heads * sequence_length * sequence_length * 4
        
        # Attention @ V
        av_ops = batch_size * self.num_heads * sequence_length * sequence_length * self.head_dim
        
        # Output projection
        proj_ops = batch_size * sequence_length * self.embedding_dim * self.embedding_dim
        
        return qk_ops + softmax_ops + av_ops + proj_ops
    
    def _calculate_feedforward_ops(self, batch_size: int, sequence_length: int) -> int:
        """Calculate operations for feed-forward layers."""
        # Assuming standard transformer feed-forward with 4x hidden dimension
        ff_hidden = 4 * self.embedding_dim
        
        # First linear layer
        ops1 = batch_size * sequence_length * self.embedding_dim * ff_hidden
        
        # Activation (GELU/ReLU) - approximated
        activation_ops = batch_size * sequence_length * ff_hidden * 2
        
        # Second linear layer
        ops2 = batch_size * sequence_length * ff_hidden * self.embedding_dim
        
        return ops1 + activation_ops + ops2
    
    def _estimate_utilization(self, sequence_length: int) -> float:
        """Estimate hardware utilization."""
        # Base utilization
        base_util = 0.85
        
        # Adjust for sequence length efficiency
        seq_efficiency = min(1.0, sequence_length / 512.0)
        
        # Adjust for number of attention units
        unit_efficiency = min(1.0, self.num_attention_units / self.num_heads)
        
        return base_util * seq_efficiency * unit_efficiency
    
    def _estimate_memory_bandwidth(self, batch_size: int, sequence_length: int, frequency_mhz: float) -> float:
        """Estimate required memory bandwidth in GB/s."""
        # Calculate data movement per operation
        bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[self.precision]
        
        # Attention matrix reads/writes
        attention_data = batch_size * self.num_heads * sequence_length * sequence_length * bytes_per_element
        
        # Weight matrix reads
        weight_data = self.embedding_dim * self.embedding_dim * bytes_per_element
        
        # Estimate cycles per operation
        cycles_per_token = 100  # Rough estimate
        total_cycles = batch_size * sequence_length * cycles_per_token
        
        # Calculate bandwidth
        total_data_bytes = attention_data + weight_data
        time_seconds = total_cycles / (frequency_mhz * 1e6)
        bandwidth_gb_s = (total_data_bytes / time_seconds) / 1e9 if time_seconds > 0 else 0
        
        return bandwidth_gb_s
    
    def _optimize_for_autoregressive(self) -> None:
        """Optimize for autoregressive models like GPT."""
        # Key-Value caching optimization
        self.kv_cache_enabled = True
        
        # Causal mask optimization
        self.causal_attention = True
        
        logger.info("Applied autoregressive optimizations")
    
    def _optimize_for_bidirectional(self) -> None:
        """Optimize for bidirectional models like BERT."""
        # Full attention matrix computation
        self.causal_attention = False
        
        # Position encoding optimization
        self.position_encoding_type = "learned"
        
        logger.info("Applied bidirectional optimizations")
    
    def _optimize_for_encoder_decoder(self) -> None:
        """Optimize for encoder-decoder models like T5."""
        # Cross-attention units
        self.cross_attention_units = self.num_attention_units // 2
        
        # Relative position encoding
        self.position_encoding_type = "relative"
        
        logger.info("Applied encoder-decoder optimizations")
    
    def _generate_transformer_rtl(self) -> str:
        """Generate SystemVerilog RTL for transformer accelerator."""
        precision_width = {"fp32": 32, "fp16": 16, "bf16": 16, "int8": 8}[self.precision]
        
        rtl_template = f'''
//==============================================================================
// Transformer Accelerator
// Embedding Dim: {self.embedding_dim}, Heads: {self.num_heads}, Precision: {self.precision}
// Generated by AI Hardware Co-Design Playground
//==============================================================================

`timescale 1ns/1ps

module transformer_accelerator #(
    parameter EMBEDDING_DIM = {self.embedding_dim},
    parameter NUM_HEADS = {self.num_heads},
    parameter HEAD_DIM = {self.head_dim},
    parameter MAX_SEQ_LEN = {self.max_sequence_length},
    parameter DATA_WIDTH = {precision_width},
    parameter NUM_ATTENTION_UNITS = {self.num_attention_units}
) (
    input wire clk,
    input wire rst_n,
    
    // Control interface
    input wire start,
    input wire [15:0] seq_length,
    input wire [7:0] batch_size,
    output wire done,
    output wire ready,
    
    // Memory interface for inputs
    output wire [31:0] input_addr,
    input wire [DATA_WIDTH*EMBEDDING_DIM-1:0] input_data,
    output wire input_req,
    input wire input_ack,
    
    // Memory interface for weights
    output wire [31:0] weight_addr,
    input wire [DATA_WIDTH*EMBEDDING_DIM-1:0] weight_data,
    output wire weight_req,
    input wire weight_ack,
    
    // Memory interface for outputs
    output wire [31:0] output_addr,
    output wire [DATA_WIDTH*EMBEDDING_DIM-1:0] output_data,
    output wire output_req,
    output wire output_we,
    input wire output_ack
);

    // Control state machine
    typedef enum logic [3:0] {{
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE_QKV,
        COMPUTE_ATTENTION,
        COMPUTE_OUTPUT,
        STORE_RESULT,
        DONE
    }} state_t;
    
    state_t current_state, next_state;
    
    // Attention computation units
    genvar i;
    generate
        for (i = 0; i < NUM_ATTENTION_UNITS; i++) begin : attention_units
            attention_unit #(
                .HEAD_DIM(HEAD_DIM),
                .DATA_WIDTH(DATA_WIDTH),
                .MAX_SEQ_LEN(MAX_SEQ_LEN),
                .UNIT_ID(i)
            ) attn_unit (
                .clk(clk),
                .rst_n(rst_n),
                .enable(current_state == COMPUTE_ATTENTION),
                .seq_length(seq_length),
                .query_data(query_data[i]),
                .key_data(key_data[i]),
                .value_data(value_data[i]),
                .attention_out(attention_out[i]),
                .valid_out(attention_valid[i])
            );
        end
    endgenerate
    
    // QKV projection matrices
    wire [DATA_WIDTH*EMBEDDING_DIM-1:0] query_data [0:NUM_ATTENTION_UNITS-1];
    wire [DATA_WIDTH*EMBEDDING_DIM-1:0] key_data [0:NUM_ATTENTION_UNITS-1];
    wire [DATA_WIDTH*EMBEDDING_DIM-1:0] value_data [0:NUM_ATTENTION_UNITS-1];
    wire [DATA_WIDTH*HEAD_DIM-1:0] attention_out [0:NUM_ATTENTION_UNITS-1];
    wire attention_valid [0:NUM_ATTENTION_UNITS-1];
    
    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: if (start) next_state = LOAD_WEIGHTS;
            LOAD_WEIGHTS: if (weight_ack) next_state = COMPUTE_QKV;
            COMPUTE_QKV: next_state = COMPUTE_ATTENTION;
            COMPUTE_ATTENTION: if (&attention_valid) next_state = COMPUTE_OUTPUT;
            COMPUTE_OUTPUT: next_state = STORE_RESULT;
            STORE_RESULT: if (output_ack) next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    // Output assignments
    assign done = (current_state == DONE);
    assign ready = (current_state == IDLE);
    assign input_req = (current_state == COMPUTE_QKV);
    assign weight_req = (current_state == LOAD_WEIGHTS);
    assign output_req = (current_state == STORE_RESULT);
    assign output_we = (current_state == STORE_RESULT);

endmodule

//==============================================================================
// Attention Unit
//==============================================================================

module attention_unit #(
    parameter HEAD_DIM = 64,
    parameter DATA_WIDTH = 16,
    parameter MAX_SEQ_LEN = 2048,
    parameter UNIT_ID = 0
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [15:0] seq_length,
    
    input wire [DATA_WIDTH*HEAD_DIM-1:0] query_data,
    input wire [DATA_WIDTH*HEAD_DIM-1:0] key_data,
    input wire [DATA_WIDTH*HEAD_DIM-1:0] value_data,
    
    output reg [DATA_WIDTH*HEAD_DIM-1:0] attention_out,
    output reg valid_out
);

    // Attention score computation
    reg [DATA_WIDTH*2-1:0] attention_scores [0:MAX_SEQ_LEN-1];
    reg [DATA_WIDTH-1:0] softmax_out [0:MAX_SEQ_LEN-1];
    
    // Simplified attention computation (Q * K^T)
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            attention_out <= 0;
            valid_out <= 0;
        end else if (enable) begin
            // Compute attention scores (simplified)
            for (i = 0; i < seq_length; i++) begin
                attention_scores[i] <= query_data[DATA_WIDTH-1:0] * key_data[DATA_WIDTH-1:0];
            end
            
            // Apply softmax (simplified - would need proper implementation)
            for (i = 0; i < seq_length; i++) begin
                softmax_out[i] <= attention_scores[i][DATA_WIDTH-1:0];
            end
            
            // Compute final attention output (Attention * V)
            attention_out <= softmax_out[0] * value_data[DATA_WIDTH-1:0];
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
'''
        
        return rtl_template.strip()
    
    def _generate_transformer_hls(self) -> str:
        """Generate HLS C++ code for transformer accelerator."""
        hls_template = f'''
//==============================================================================
// Transformer Accelerator HLS Implementation
// Generated by AI Hardware Co-Design Playground
//==============================================================================

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

// Configuration parameters
#define EMBEDDING_DIM {self.embedding_dim}
#define NUM_HEADS {self.num_heads}
#define HEAD_DIM {self.head_dim}
#define MAX_SEQ_LEN {self.max_sequence_length}
#define NUM_ATTENTION_UNITS {self.num_attention_units}

// Data types based on precision
'''
        
        if self.precision == "fp16":
            hls_template += '''
typedef ap_fixed<16,6> data_t;
typedef ap_fixed<32,12> acc_t;
'''
        elif self.precision == "fp32":
            hls_template += '''
typedef float data_t;
typedef float acc_t;
'''
        elif self.precision == "int8":
            hls_template += '''
typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;
'''
        
        hls_template += f'''

// Multi-head attention function
void multihead_attention(
    data_t input[MAX_SEQ_LEN][EMBEDDING_DIM],
    data_t q_weight[EMBEDDING_DIM][EMBEDDING_DIM],
    data_t k_weight[EMBEDDING_DIM][EMBEDDING_DIM], 
    data_t v_weight[EMBEDDING_DIM][EMBEDDING_DIM],
    data_t output[MAX_SEQ_LEN][EMBEDDING_DIM],
    int seq_length
) {{
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=input_mem
    #pragma HLS INTERFACE m_axi port=q_weight offset=slave bundle=weight_mem
    #pragma HLS INTERFACE m_axi port=k_weight offset=slave bundle=weight_mem
    #pragma HLS INTERFACE m_axi port=v_weight offset=slave bundle=weight_mem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=output_mem
    #pragma HLS INTERFACE s_axilite port=seq_length
    #pragma HLS INTERFACE s_axilite port=return
    
    // Local buffers
    data_t q_proj[MAX_SEQ_LEN][EMBEDDING_DIM];
    data_t k_proj[MAX_SEQ_LEN][EMBEDDING_DIM];
    data_t v_proj[MAX_SEQ_LEN][EMBEDDING_DIM];
    data_t attention_scores[NUM_HEADS][MAX_SEQ_LEN][MAX_SEQ_LEN];
    
    #pragma HLS ARRAY_PARTITION variable=q_proj cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=k_proj cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=v_proj cyclic factor=4 dim=2
    
    // Compute Q, K, V projections
    QKV_PROJECTION:
    for (int i = 0; i < seq_length; i++) {{
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < EMBEDDING_DIM; j++) {{
            #pragma HLS UNROLL factor=4
            acc_t q_sum = 0, k_sum = 0, v_sum = 0;
            
            for (int k = 0; k < EMBEDDING_DIM; k++) {{
                #pragma HLS UNROLL factor=8
                q_sum += input[i][k] * q_weight[k][j];
                k_sum += input[i][k] * k_weight[k][j];
                v_sum += input[i][k] * v_weight[k][j];
            }}
            
            q_proj[i][j] = q_sum;
            k_proj[i][j] = k_sum;
            v_proj[i][j] = v_sum;
        }}
    }}
    
    // Compute attention scores for each head
    ATTENTION_COMPUTATION:
    for (int h = 0; h < NUM_HEADS; h++) {{
        #pragma HLS UNROLL factor=2
        
        // Q * K^T
        for (int i = 0; i < seq_length; i++) {{
            for (int j = 0; j < seq_length; j++) {{
                #pragma HLS PIPELINE II=1
                acc_t score = 0;
                
                for (int d = 0; d < HEAD_DIM; d++) {{
                    #pragma HLS UNROLL factor=4
                    int q_idx = h * HEAD_DIM + d;
                    int k_idx = h * HEAD_DIM + d;
                    score += q_proj[i][q_idx] * k_proj[j][k_idx];
                }}
                
                // Scale by sqrt(head_dim)
                attention_scores[h][i][j] = score / hls::sqrt((float)HEAD_DIM);
            }}
        }}
        
        // Apply softmax (simplified)
        SOFTMAX:
        for (int i = 0; i < seq_length; i++) {{
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            
            // Find max for numerical stability
            data_t max_val = attention_scores[h][i][0];
            for (int j = 1; j < seq_length; j++) {{
                if (attention_scores[h][i][j] > max_val) {{
                    max_val = attention_scores[h][i][j];
                }}
            }}
            
            // Compute exp and sum
            for (int j = 0; j < seq_length; j++) {{
                attention_scores[h][i][j] = hls::exp(attention_scores[h][i][j] - max_val);
                sum += attention_scores[h][i][j];
            }}
            
            // Normalize
            for (int j = 0; j < seq_length; j++) {{
                attention_scores[h][i][j] /= sum;
            }}
        }}
    }}
    
    // Apply attention to values and combine heads
    OUTPUT_COMPUTATION:
    for (int i = 0; i < seq_length; i++) {{
        for (int j = 0; j < EMBEDDING_DIM; j++) {{
            #pragma HLS PIPELINE II=1
            acc_t result = 0;
            
            for (int h = 0; h < NUM_HEADS; h++) {{
                #pragma HLS UNROLL factor=2
                acc_t head_result = 0;
                
                for (int k = 0; k < seq_length; k++) {{
                    #pragma HLS UNROLL factor=4
                    int v_idx = h * HEAD_DIM + (j % HEAD_DIM);
                    head_result += attention_scores[h][i][k] * v_proj[k][v_idx];
                }}
                
                result += head_result;
            }}
            
            output[i][j] = result;
        }}
    }}
}}

// Feed-forward network
void feed_forward(
    data_t input[MAX_SEQ_LEN][EMBEDDING_DIM],
    data_t ff1_weight[EMBEDDING_DIM][EMBEDDING_DIM*4],
    data_t ff2_weight[EMBEDDING_DIM*4][EMBEDDING_DIM],
    data_t output[MAX_SEQ_LEN][EMBEDDING_DIM],
    int seq_length
) {{
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=input_mem
    #pragma HLS INTERFACE m_axi port=ff1_weight offset=slave bundle=weight_mem
    #pragma HLS INTERFACE m_axi port=ff2_weight offset=slave bundle=weight_mem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=output_mem
    #pragma HLS INTERFACE s_axilite port=seq_length
    #pragma HLS INTERFACE s_axilite port=return
    
    data_t intermediate[MAX_SEQ_LEN][EMBEDDING_DIM*4];
    #pragma HLS ARRAY_PARTITION variable=intermediate cyclic factor=8 dim=2
    
    // First linear layer + GELU activation
    FF1_LAYER:
    for (int i = 0; i < seq_length; i++) {{
        for (int j = 0; j < EMBEDDING_DIM*4; j++) {{
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            
            for (int k = 0; k < EMBEDDING_DIM; k++) {{
                #pragma HLS UNROLL factor=8
                sum += input[i][k] * ff1_weight[k][j];
            }}
            
            // GELU activation (approximated)
            data_t x = sum;
            intermediate[i][j] = 0.5f * x * (1.0f + hls::tanh(0.797885f * (x + 0.044715f * x * x * x)));
        }}
    }}
    
    // Second linear layer
    FF2_LAYER:
    for (int i = 0; i < seq_length; i++) {{
        for (int j = 0; j < EMBEDDING_DIM; j++) {{
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            
            for (int k = 0; k < EMBEDDING_DIM*4; k++) {{
                #pragma HLS UNROLL factor=8
                sum += intermediate[i][k] * ff2_weight[k][j];
            }}
            
            output[i][j] = sum;
        }}
    }}
}}

// Top-level transformer layer
extern "C" {{
void transformer_layer(
    data_t input[MAX_SEQ_LEN][EMBEDDING_DIM],
    data_t q_weight[EMBEDDING_DIM][EMBEDDING_DIM],
    data_t k_weight[EMBEDDING_DIM][EMBEDDING_DIM],
    data_t v_weight[EMBEDDING_DIM][EMBEDDING_DIM],
    data_t ff1_weight[EMBEDDING_DIM][EMBEDDING_DIM*4],
    data_t ff2_weight[EMBEDDING_DIM*4][EMBEDDING_DIM],
    data_t output[MAX_SEQ_LEN][EMBEDDING_DIM],
    int seq_length
) {{
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=q_weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=k_weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=v_weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=ff1_weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=ff2_weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=seq_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    data_t attention_output[MAX_SEQ_LEN][EMBEDDING_DIM];
    data_t ff_output[MAX_SEQ_LEN][EMBEDDING_DIM];
    
    #pragma HLS DATAFLOW
    
    // Multi-head attention
    multihead_attention(input, q_weight, k_weight, v_weight, attention_output, seq_length);
    
    // Feed-forward network
    feed_forward(attention_output, ff1_weight, ff2_weight, ff_output, seq_length);
    
    // Residual connection and layer norm (simplified)
    RESIDUAL_LAYERNORM:
    for (int i = 0; i < seq_length; i++) {{
        for (int j = 0; j < EMBEDDING_DIM; j++) {{
            #pragma HLS PIPELINE II=1
            output[i][j] = input[i][j] + ff_output[i][j];
        }}
    }}
}}
}}
'''
        
        return hls_template.strip()
    
    def __str__(self) -> str:
        """String representation of transformer accelerator."""
        return f"TransformerAccelerator(seq_len={self.max_sequence_length}, dim={self.embedding_dim}, heads={self.num_heads}, {self.precision})"