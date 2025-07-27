"""Hardware-related test fixtures for AI Hardware Co-Design Playground."""

import pytest
from typing import Dict, Any, List
from pathlib import Path
import tempfile
import json

@pytest.fixture
def systolic_array_spec() -> Dict[str, Any]:
    """Sample systolic array specification for testing."""
    return {
        "type": "systolic_array",
        "rows": 16,
        "cols": 16,
        "data_width": 8,
        "weight_width": 8,
        "accumulator_width": 32,
        "input_buffer_size": 1024,
        "weight_buffer_size": 4096,
        "output_buffer_size": 512,
        "clock_frequency_mhz": 200,
        "memory_interface": {
            "type": "axi4",
            "data_width": 512,
            "address_width": 32,
        },
        "optimizations": {
            "enable_pipelining": True,
            "enable_weight_reuse": True,
            "enable_partial_sums": True,
        }
    }

@pytest.fixture
def vector_processor_spec() -> Dict[str, Any]:
    """Sample vector processor specification for testing."""
    return {
        "type": "vector_processor",
        "vector_length": 256,
        "num_lanes": 8,
        "data_width": 16,
        "instruction_width": 32,
        "register_file_size": 32,
        "memory_banks": 4,
        "supported_operations": [
            "add", "sub", "mul", "mac", "div",
            "relu", "sigmoid", "tanh",
            "max_pool", "avg_pool",
            "conv1d", "conv2d"
        ],
        "custom_instructions": [
            {
                "name": "batch_norm",
                "latency": 3,
                "throughput": 8,
                "resource_usage": {"alu": 2, "memory": 1}
            }
        ]
    }

@pytest.fixture
def transformer_accelerator_spec() -> Dict[str, Any]:
    """Sample transformer accelerator specification for testing."""
    return {
        "type": "transformer_accelerator",
        "max_sequence_length": 512,
        "embedding_dim": 768,
        "num_attention_heads": 12,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "feedforward_dim": 3072,
        "precision": "fp16",
        "attention_mechanism": "multi_head_attention",
        "memory_hierarchy": {
            "l1_cache_kb": 64,
            "l2_cache_kb": 512,
            "embedding_cache_kb": 1024,
        },
        "optimizations": {
            "attention_sparsity": 0.9,
            "enable_kv_cache": True,
            "enable_flash_attention": True,
        }
    }

@pytest.fixture
def sample_hardware_constraints() -> Dict[str, Any]:
    """Sample hardware constraints for testing optimization."""
    return {
        "area_budget_mm2": 10.0,
        "power_budget_mw": 1000.0,
        "frequency_target_mhz": 200.0,
        "memory_bandwidth_gb_s": 100.0,
        "technology_node": "28nm",
        "supply_voltage": 1.0,
        "operating_temperature": 85,
        "reliability_target": {
            "mtbf_hours": 1000000,
            "error_rate": 1e-12,
        }
    }

@pytest.fixture
def sample_performance_targets() -> Dict[str, Any]:
    """Sample performance targets for testing."""
    return {
        "throughput_ops_per_sec": 1000000,
        "latency_max_ns": 10000,
        "energy_per_op_pj": 50,
        "utilization_min": 0.8,
        "accuracy_loss_max": 0.01,
        "memory_efficiency_min": 0.9,
    }

@pytest.fixture
def sample_rtl_code() -> str:
    """Sample RTL code for testing."""
    return """
// Simple 8-bit adder module for testing
module simple_adder #(
    parameter WIDTH = 8
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire             valid_in,
    output reg  [WIDTH:0]   sum,
    output reg              valid_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sum <= 0;
        valid_out <= 1'b0;
    end else begin
        if (valid_in) begin
            sum <= a + b;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
end

endmodule
"""

@pytest.fixture
def sample_testbench() -> str:
    """Sample testbench code for testing."""
    return """
// Testbench for simple_adder
module tb_simple_adder;
    parameter WIDTH = 8;
    parameter CLK_PERIOD = 10;

    reg             clk;
    reg             rst_n;
    reg [WIDTH-1:0] a;
    reg [WIDTH-1:0] b;
    reg             valid_in;
    wire [WIDTH:0]  sum;
    wire            valid_out;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT instantiation
    simple_adder #(.WIDTH(WIDTH)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .a(a),
        .b(b),
        .valid_in(valid_in),
        .sum(sum),
        .valid_out(valid_out)
    );

    // Test stimulus
    initial begin
        rst_n = 0;
        a = 0;
        b = 0;
        valid_in = 0;
        
        #(CLK_PERIOD * 2);
        rst_n = 1;
        
        #(CLK_PERIOD);
        a = 8'd15;
        b = 8'd25;
        valid_in = 1;
        
        #(CLK_PERIOD);
        valid_in = 0;
        
        #(CLK_PERIOD * 5);
        $finish;
    end

    // Monitor
    initial begin
        $monitor("Time=%0t, a=%d, b=%d, sum=%d, valid_out=%b", 
                 $time, a, b, sum, valid_out);
    end

endmodule
"""

@pytest.fixture
def mock_synthesis_results() -> Dict[str, Any]:
    """Mock synthesis results for testing."""
    return {
        "area": {
            "total_area_um2": 50000,
            "logic_area_um2": 30000,
            "memory_area_um2": 15000,
            "io_area_um2": 5000,
            "utilization": 0.75,
        },
        "timing": {
            "critical_path_ns": 4.5,
            "setup_slack_ns": 0.5,
            "hold_slack_ns": 0.2,
            "max_frequency_mhz": 222,
        },
        "power": {
            "total_power_mw": 450,
            "dynamic_power_mw": 350,
            "static_power_mw": 100,
            "power_density_mw_mm2": 45,
        },
        "resources": {
            "luts": 1500,
            "flip_flops": 800,
            "dsps": 16,
            "brams": 8,
            "carry_chains": 150,
        },
        "warnings": [
            "High fanout net: clk (fanout=1000)",
            "Unconstrained path found"
        ],
        "errors": []
    }

@pytest.fixture
def mock_place_route_results() -> Dict[str, Any]:
    """Mock place and route results for testing."""
    return {
        "routing": {
            "total_nets": 5000,
            "routed_nets": 4998,
            "unrouted_nets": 2,
            "congestion_level": 0.3,
        },
        "timing": {
            "wns_ns": 0.8,  # Worst negative slack
            "tns_ns": 0.0,  # Total negative slack
            "failing_paths": 0,
            "critical_path_delay_ns": 4.2,
        },
        "power": {
            "total_power_mw": 475,
            "ir_drop_max_mv": 50,
            "thermal_hotspots": 2,
        },
        "physical": {
            "die_area_mm2": 1.0,
            "core_utilization": 0.78,
            "routing_utilization": 0.65,
        },
        "drc_violations": 0,
        "lvs_status": "clean"
    }

@pytest.fixture
def hardware_template_library() -> List[Dict[str, Any]]:
    """Library of hardware templates for testing."""
    return [
        {
            "name": "systolic_array_8x8",
            "type": "systolic_array",
            "description": "8x8 systolic array for matrix multiplication",
            "parameters": {"rows": 8, "cols": 8, "data_width": 8},
            "resource_estimate": {"area_mm2": 0.5, "power_mw": 100},
        },
        {
            "name": "vector_proc_simd128",
            "type": "vector_processor", 
            "description": "128-element SIMD vector processor",
            "parameters": {"vector_length": 128, "num_lanes": 4},
            "resource_estimate": {"area_mm2": 2.0, "power_mw": 500},
        },
        {
            "name": "conv_engine_3x3",
            "type": "convolution_engine",
            "description": "Specialized 3x3 convolution engine",
            "parameters": {"kernel_size": 3, "channels": 256},
            "resource_estimate": {"area_mm2": 1.5, "power_mw": 300},
        }
    ]

@pytest.fixture
def temp_rtl_file(temp_dir) -> Path:
    """Create a temporary RTL file for testing."""
    rtl_file = temp_dir / "test_module.v"
    rtl_file.write_text("""
module test_module (
    input clk,
    input rst,
    input [7:0] data_in,
    output reg [7:0] data_out
);
    always @(posedge clk) begin
        if (rst)
            data_out <= 8'b0;
        else
            data_out <= data_in;
    end
endmodule
""")
    return rtl_file

@pytest.fixture
def mock_fpga_resources() -> Dict[str, Any]:
    """Mock FPGA resource information."""
    return {
        "device": "xcvu9p-flga2104-2-i",
        "family": "UltraScale+",
        "available_resources": {
            "luts": 1182240,
            "flip_flops": 2364480,
            "dsps": 6840,
            "brams_36k": 4320,
            "uram": 960,
        },
        "used_resources": {
            "luts": 50000,
            "flip_flops": 25000,
            "dsps": 100,
            "brams_36k": 50,
            "uram": 5,
        },
        "utilization": {
            "luts": 4.2,
            "flip_flops": 1.1,
            "dsps": 1.5,
            "brams_36k": 1.2,
            "uram": 0.5,
        }
    }