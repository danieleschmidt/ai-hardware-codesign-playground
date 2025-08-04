"""
Systolic Array hardware template for matrix multiplication accelerators.

This module provides a configurable systolic array template optimized
for matrix multiplication operations in neural networks.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceEstimate:
    """Hardware resource estimation."""
    luts: int = 0
    ffs: int = 0
    dsps: int = 0
    bram_kb: int = 0
    power_mw: float = 0.0
    frequency_mhz: float = 0.0


class SystolicArray:
    """
    Configurable systolic array template for matrix multiplication.
    
    Supports weight-stationary, output-stationary, and row-stationary dataflows
    with configurable dimensions and data widths.
    """
    
    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        data_width: int = 8,
        accumulator_width: int = 32,
        dataflow: str = "weight_stationary"
    ):
        """
        Initialize systolic array configuration.
        
        Args:
            rows: Number of processing element rows
            cols: Number of processing element columns
            data_width: Width of input data (bits)
            accumulator_width: Width of accumulator (bits)
            dataflow: Dataflow pattern ("weight_stationary", "output_stationary", "row_stationary")
        """
        self.rows = rows
        self.cols = cols
        self.data_width = data_width
        self.accumulator_width = accumulator_width
        self.dataflow = dataflow
        
        # Configuration state
        self.configured_for = None
        self.performance_model = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info(
            "Initialized SystolicArray",
            rows=rows,
            cols=cols,
            data_width=data_width,
            dataflow=dataflow
        )
    
    def configure_for_conv2d(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        input_height: int = 224,
        input_width: int = 224
    ) -> None:
        """
        Configure systolic array for 2D convolution operations.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Convolution kernel size
            input_height: Input feature map height
            input_width: Input feature map width
        """
        config = {
            "operation": "conv2d",
            "input_channels": input_channels,
            "output_channels": output_channels,
            "kernel_size": kernel_size,
            "input_height": input_height,
            "input_width": input_width,
        }
        
        # Calculate mapping efficiency
        total_weights = input_channels * output_channels * kernel_size * kernel_size
        pe_count = self.rows * self.cols
        
        # Calculate cycles needed
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1
        total_outputs = output_channels * output_height * output_width
        
        # Estimate cycles based on dataflow
        if self.dataflow == "weight_stationary":
            cycles_per_output = input_channels * kernel_size * kernel_size
            total_cycles = total_outputs * cycles_per_output // pe_count
        elif self.dataflow == "output_stationary":
            cycles_per_weight_load = total_outputs // pe_count
            total_cycles = total_weights * cycles_per_weight_load
        else:  # row_stationary
            total_cycles = total_outputs * input_channels * kernel_size * kernel_size // pe_count
        
        self.performance_model = {
            "total_cycles": total_cycles,
            "utilization": min(1.0, total_weights / pe_count),
            "throughput_ops_per_cycle": pe_count * self.performance_model.get("utilization", 0.8),
            "memory_accesses": total_weights + total_outputs,
        }
        
        self.configured_for = config
        
        logger.info(
            "Configured systolic array for Conv2D",
            input_channels=input_channels,
            output_channels=output_channels,
            estimated_cycles=total_cycles,
            utilization=self.performance_model["utilization"]
        )
    
    def configure_for_matmul(
        self,
        m: int,
        n: int, 
        k: int
    ) -> None:
        """
        Configure systolic array for matrix multiplication (M x K) * (K x N).
        
        Args:
            m: First matrix rows
            n: Second matrix columns  
            k: Inner dimension
        """
        config = {
            "operation": "matmul",
            "m": m,
            "n": n,
            "k": k,
        }
        
        # Calculate optimal tiling
        pe_count = self.rows * self.cols
        total_operations = m * n * k
        
        # Estimate cycles for different dataflows
        if self.dataflow == "weight_stationary":
            # Weights stay in PEs, stream activations and collect outputs
            cycles = (m * n * k) // pe_count
        elif self.dataflow == "output_stationary":
            # Outputs stay in PEs, stream weights and activations
            cycles = (m * n * k) // pe_count
        else:  # row_stationary
            cycles = (m * n * k) // pe_count
        
        utilization = min(1.0, total_operations / (pe_count * cycles))
        
        self.performance_model = {
            "total_cycles": cycles,
            "utilization": utilization,
            "throughput_ops_per_cycle": pe_count * utilization,
            "memory_accesses": m * k + k * n + m * n,  # A + B + C
        }
        
        self.configured_for = config
        
        logger.info(
            "Configured systolic array for GEMM",
            m=m, n=n, k=k,
            estimated_cycles=cycles,
            utilization=utilization
        )
    
    def generate_rtl(self, output_dir: str = "./rtl") -> str:
        """
        Generate SystemVerilog RTL for the systolic array.
        
        Args:
            output_dir: Output directory for RTL files
            
        Returns:
            Path to generated RTL file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        rtl_file = output_path / f"systolic_array_{self.rows}x{self.cols}.sv"
        
        # Generate RTL code
        rtl_code = self._generate_systolic_rtl()
        
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        logger.info(f"Generated systolic array RTL: {rtl_file}")
        return str(rtl_file)
    
    def estimate_resources(self) -> ResourceEstimate:
        """
        Estimate hardware resource requirements.
        
        Returns:
            Resource estimate for the systolic array
        """
        pe_count = self.rows * self.cols
        
        # Rough estimates based on PE complexity
        luts_per_pe = 100 + (self.data_width * 2)  # MAC + control
        ffs_per_pe = 50 + self.accumulator_width    # Registers
        dsps_per_pe = 1 if self.data_width <= 18 else 2  # DSP48E2 in Xilinx
        
        # Memory requirements (rough estimate)
        if self.dataflow == "weight_stationary":
            bram_kb = (pe_count * self.data_width * 256) // (8 * 1024)  # Weight storage
        else:
            bram_kb = (pe_count * self.accumulator_width * 64) // (8 * 1024)  # Buffer storage
        
        # Power estimation (very rough)
        power_per_pe = 5.0  # mW per PE
        total_power = pe_count * power_per_pe
        
        # Frequency estimate based on critical path
        base_freq = 300.0  # MHz
        if self.data_width > 16:
            frequency = base_freq * 0.8  # Higher precision reduces frequency
        else:
            frequency = base_freq
        
        return ResourceEstimate(
            luts=pe_count * luts_per_pe,
            ffs=pe_count * ffs_per_pe,
            dsps=pe_count * dsps_per_pe,
            bram_kb=bram_kb,
            power_mw=total_power,
            frequency_mhz=frequency
        )
    
    def get_performance_metrics(self, frequency_mhz: float = 200.0) -> Dict[str, float]:
        """
        Get performance metrics for configured operation.
        
        Args:
            frequency_mhz: Operating frequency
            
        Returns:
            Performance metrics dictionary
        """
        if not self.configured_for:
            logger.warning("Systolic array not configured for specific operation")
            return {}
        
        pe_count = self.rows * self.cols
        cycles = self.performance_model.get("total_cycles", 1000)
        utilization = self.performance_model.get("utilization", 0.8)
        
        # Calculate metrics
        latency_ms = cycles / (frequency_mhz * 1000)
        throughput_ops_s = pe_count * frequency_mhz * 1e6 * utilization
        
        if self.configured_for["operation"] == "conv2d":
            # Convolution-specific metrics
            config = self.configured_for
            output_size = (config["input_height"] - config["kernel_size"] + 1) ** 2
            images_per_second = 1000 / latency_ms if latency_ms > 0 else 0
            
            return {
                "latency_ms": latency_ms,
                "throughput_ops_s": throughput_ops_s,
                "images_per_second": images_per_second,
                "utilization": utilization,
                "peak_throughput_ops_s": pe_count * frequency_mhz * 1e6,
            }
        
        elif self.configured_for["operation"] == "matmul":
            # Matrix multiplication metrics
            config = self.configured_for
            total_ops = config["m"] * config["n"] * config["k"]
            gflops = total_ops / (latency_ms / 1000) / 1e9 if latency_ms > 0 else 0
            
            return {
                "latency_ms": latency_ms,
                "throughput_ops_s": throughput_ops_s,
                "gflops": gflops,
                "utilization": utilization,
                "peak_gflops": pe_count * frequency_mhz * 1e6 / 1e9 * 2,  # MAC = 2 ops
            }
        
        return {
            "latency_ms": latency_ms,
            "throughput_ops_s": throughput_ops_s,
            "utilization": utilization,
        }
    
    def _validate_config(self) -> None:
        """Validate systolic array configuration."""
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rows and columns must be positive")
        
        if self.data_width not in [4, 8, 16, 32]:
            logger.warning(f"Unusual data width: {self.data_width} bits")
        
        if self.dataflow not in ["weight_stationary", "output_stationary", "row_stationary"]:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")
        
        if self.accumulator_width < self.data_width * 2:
            logger.warning("Accumulator width may be too small for data width")
    
    def _generate_systolic_rtl(self) -> str:
        """Generate SystemVerilog RTL code for systolic array."""
        # Generate comprehensive RTL with proper dataflow implementation
        rtl_template = f'''
//==============================================================================
// Systolic Array {self.rows}x{self.cols} - {self.dataflow.upper()} Dataflow
// Generated by AI Hardware Co-Design Playground
//==============================================================================

`timescale 1ns/1ps

module systolic_array_{self.rows}x{self.cols} #(
    parameter DATA_WIDTH = {self.data_width},
    parameter ACC_WIDTH = {self.accumulator_width},
    parameter ROWS = {self.rows},
    parameter COLS = {self.cols}
) (
    input wire clk,
    input wire rst_n,
    
    // Control signals
    input wire enable,
    input wire clear_acc,
    
    // Data inputs
    input wire [DATA_WIDTH-1:0] a_data [0:ROWS-1],
    input wire [DATA_WIDTH-1:0] b_data [0:COLS-1],
    input wire a_valid [0:ROWS-1],
    input wire b_valid [0:COLS-1],
    
    // Output data
    output wire [ACC_WIDTH-1:0] c_data [0:ROWS-1][0:COLS-1],
    output wire c_valid [0:ROWS-1][0:COLS-1]
);

    // Internal interconnect arrays
    wire [DATA_WIDTH-1:0] a_internal [0:ROWS-1][0:COLS];
    wire [DATA_WIDTH-1:0] b_internal [0:ROWS][0:COLS-1];
    wire a_valid_internal [0:ROWS-1][0:COLS];
    wire b_valid_internal [0:ROWS][0:COLS-1];
    
    // Connect inputs to internal arrays
    genvar i, j;
    generate
        for (i = 0; i < ROWS; i++) begin : input_a_connections
            assign a_internal[i][0] = a_data[i];
            assign a_valid_internal[i][0] = a_valid[i];
        end
        
        for (j = 0; j < COLS; j++) begin : input_b_connections
            assign b_internal[0][j] = b_data[j];
            assign b_valid_internal[0][j] = b_valid[j];
        end
    endgenerate
    
    // Generate processing element array
    generate
        for (i = 0; i < ROWS; i++) begin : pe_rows
            for (j = 0; j < COLS; j++) begin : pe_cols
                processing_element #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH),
                    .DATAFLOW("{self.dataflow}")
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .clear_acc(clear_acc),
                    
                    // Inputs
                    .a_in(a_internal[i][j]),
                    .b_in(b_internal[i][j]),
                    .a_valid_in(a_valid_internal[i][j]),
                    .b_valid_in(b_valid_internal[i][j]),
                    
                    // Outputs
                    .a_out(a_internal[i][j+1]),
                    .b_out(b_internal[i+1][j]),
                    .a_valid_out(a_valid_internal[i][j+1]),
                    .b_valid_out(b_valid_internal[i+1][j]),
                    .c_out(c_data[i][j]),
                    .c_valid(c_valid[i][j])
                );
            end
        end
    endgenerate

endmodule

//==============================================================================
// Processing Element
//==============================================================================

module processing_element #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter DATAFLOW = "weight_stationary"
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire clear_acc,
    
    // Data inputs
    input wire [DATA_WIDTH-1:0] a_in,
    input wire [DATA_WIDTH-1:0] b_in,
    input wire a_valid_in,
    input wire b_valid_in,
    
    // Data outputs
    output reg [DATA_WIDTH-1:0] a_out,
    output reg [DATA_WIDTH-1:0] b_out,
    output reg a_valid_out,
    output reg b_valid_out,
    output reg [ACC_WIDTH-1:0] c_out,
    output reg c_valid
);

    // Internal registers
    reg [DATA_WIDTH-1:0] weight_reg;
    reg [ACC_WIDTH-1:0] accumulator;
    reg weight_loaded;
    
    // MAC operation
    wire [ACC_WIDTH-1:0] mult_result;
    assign mult_result = $signed(a_in) * $signed(b_in);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= 0;
            b_out <= 0;
            a_valid_out <= 0;
            b_valid_out <= 0;
            c_out <= 0;
            c_valid <= 0;
            weight_reg <= 0;
            accumulator <= 0;
            weight_loaded <= 0;
        end else if (enable) begin
            // Pipeline data through PE
            a_out <= a_in;
            b_out <= b_in;
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
            
            // Clear accumulator if requested
            if (clear_acc) begin
                accumulator <= 0;
                c_valid <= 0;
            end
            
            // Perform MAC operation based on dataflow
            if (DATAFLOW == "weight_stationary") begin
                // Load weight once, stream activations
                if (!weight_loaded && b_valid_in) begin
                    weight_reg <= b_in;
                    weight_loaded <= 1;
                end else if (weight_loaded && a_valid_in) begin
                    accumulator <= accumulator + ($signed(a_in) * $signed(weight_reg));
                    c_valid <= 1;
                end
            end else if (DATAFLOW == "output_stationary") begin
                // Accumulate in place, stream weights and activations
                if (a_valid_in && b_valid_in) begin
                    accumulator <= accumulator + mult_result;
                    c_valid <= 1;
                end
            end else begin // row_stationary
                // Hybrid approach
                if (a_valid_in && b_valid_in) begin
                    accumulator <= accumulator + mult_result;
                    c_valid <= 1;
                end
            end
            
            c_out <= accumulator;
        end
    end

endmodule
'''
        
        return rtl_template.strip()
    
    def __str__(self) -> str:
        """String representation of systolic array configuration."""
        return f"SystolicArray({self.rows}x{self.cols}, {self.data_width}b, {self.dataflow})"