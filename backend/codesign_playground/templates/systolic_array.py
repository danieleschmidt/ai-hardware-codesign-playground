"""
Systolic Array hardware template for matrix multiplication accelerators.

This module provides a configurable systolic array template optimized
for matrix multiplication operations in neural networks.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
# Optional dependency with fallback
try:
    import numpy as np
except ImportError:
    np = None
import math

from ..utils.logging import get_logger
from ..utils.monitoring import record_metric

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
        
        # Enhanced cycle estimation based on dataflow and memory constraints
        if self.dataflow == "weight_stationary":
            # Weights are pre-loaded, stream activations
            cycles_per_output = input_channels * kernel_size * kernel_size
            pipeline_stages = 3  # Fetch, compute, writeback
            memory_stall_factor = 1.2  # Account for memory access patterns
            total_cycles = int((total_outputs * cycles_per_output // pe_count) * memory_stall_factor + pipeline_stages)
        elif self.dataflow == "output_stationary":
            # Partial sums accumulate in place
            weight_fetch_cycles = total_weights // (pe_count * 4)  # Assume 4 weights per cycle fetch
            compute_cycles = total_outputs * input_channels * kernel_size * kernel_size // pe_count
            total_cycles = max(weight_fetch_cycles, compute_cycles) + 10  # Startup overhead
        else:  # row_stationary
            # Hybrid approach with row-wise weight reuse
            row_reuse_factor = min(self.rows, output_channels)
            effective_weight_loads = total_weights // row_reuse_factor
            total_cycles = effective_weight_loads + (total_outputs * input_channels * kernel_size * kernel_size // pe_count)
        
        # Enhanced utilization calculation considering dataflow efficiency
        theoretical_utilization = min(1.0, (total_outputs * input_channels * kernel_size * kernel_size) / (pe_count * total_cycles))
        
        # Apply dataflow-specific efficiency factors
        if self.dataflow == "weight_stationary":
            dataflow_efficiency = 0.95  # High efficiency due to weight reuse
        elif self.dataflow == "output_stationary":
            dataflow_efficiency = 0.85  # Medium efficiency due to partial sum handling
        else:  # row_stationary
            dataflow_efficiency = 0.90  # Good balance
        
        # Consider memory bandwidth limitations
        memory_intensity = (total_weights + total_outputs) / total_cycles
        memory_efficiency = min(1.0, 32 / memory_intensity)  # Assume 32 GB/s bandwidth
        
        actual_utilization = theoretical_utilization * dataflow_efficiency * memory_efficiency
        
        self.performance_model = {
            "total_cycles": total_cycles,
            "utilization": actual_utilization,
            "theoretical_utilization": theoretical_utilization,
            "dataflow_efficiency": dataflow_efficiency,
            "memory_efficiency": memory_efficiency,
            "throughput_ops_per_cycle": pe_count * actual_utilization,
            "memory_accesses": total_weights + total_outputs,
            "memory_intensity": memory_intensity,
        }
        
        self.configured_for = config
        
        # Record performance metrics
        record_metric("systolic_conv2d_config", 1, "counter")
        record_metric("systolic_conv2d_utilization", actual_utilization, "gauge")
        record_metric("systolic_conv2d_cycles", total_cycles, "gauge")
        
        logger.info(
            "Configured systolic array for Conv2D",
            input_channels=input_channels,
            output_channels=output_channels,
            estimated_cycles=total_cycles,
            utilization=actual_utilization,
            dataflow_efficiency=dataflow_efficiency,
            memory_efficiency=memory_efficiency
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
        
        # Enhanced cycle estimation for matrix multiplication
        if self.dataflow == "weight_stationary":
            # Weights pre-loaded, stream A and B matrices
            weight_load_cycles = (k * n) // pe_count  # Load B matrix into PEs
            compute_cycles = (m * n * k) // pe_count
            pipeline_overhead = max(self.rows, self.cols)
            cycles = weight_load_cycles + compute_cycles + pipeline_overhead
        elif self.dataflow == "output_stationary":
            # C matrix accumulated in place
            matrix_setup_cycles = (m * n) // pe_count  # Setup output locations
            compute_cycles = (m * n * k) // pe_count
            cycles = matrix_setup_cycles + compute_cycles
        else:  # row_stationary
            # Row-wise processing with partial weight reuse
            row_blocks = math.ceil(m / self.rows)
            col_blocks = math.ceil(n / self.cols)
            cycles_per_block = k + max(self.rows, self.cols)  # Computation + data movement
            cycles = row_blocks * col_blocks * cycles_per_block
        
        # Enhanced utilization calculation
        theoretical_utilization = min(1.0, total_operations / (pe_count * cycles))
        
        # Dataflow efficiency factors for GEMM
        if self.dataflow == "weight_stationary":
            dataflow_efficiency = 0.92
        elif self.dataflow == "output_stationary": 
            dataflow_efficiency = 0.88
        else:  # row_stationary
            dataflow_efficiency = 0.85
        
        # Memory efficiency based on matrix dimensions
        total_data = m * k + k * n + m * n  # A + B + C matrices
        memory_cycles = total_data // 8  # Assume 8 elements per cycle bandwidth
        memory_bound = cycles < memory_cycles
        memory_efficiency = 0.7 if memory_bound else 0.95
        
        utilization = theoretical_utilization * dataflow_efficiency * memory_efficiency
        
        self.performance_model = {
            "total_cycles": cycles,
            "utilization": utilization,
            "theoretical_utilization": theoretical_utilization,
            "dataflow_efficiency": dataflow_efficiency,
            "memory_efficiency": memory_efficiency,
            "throughput_ops_per_cycle": pe_count * utilization,
            "memory_accesses": total_data,
            "memory_bound": memory_bound,
            "arithmetic_intensity": total_operations / total_data,
        }
        
        self.configured_for = config
        
        # Record performance metrics
        record_metric("systolic_gemm_config", 1, "counter")
        record_metric("systolic_gemm_utilization", utilization, "gauge")
        record_metric("systolic_gemm_cycles", cycles, "gauge")
        record_metric("systolic_gemm_arithmetic_intensity", total_operations / total_data, "gauge")
        
        logger.info(
            "Configured systolic array for GEMM",
            m=m, n=n, k=k,
            estimated_cycles=cycles,
            utilization=utilization,
            dataflow_efficiency=dataflow_efficiency,
            memory_efficiency=memory_efficiency,
            arithmetic_intensity=total_operations / total_data
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
        
        # Enhanced resource estimation based on modern FPGA characteristics
        
        # LUT estimation: MAC unit + control logic + interconnect
        mac_luts = self.data_width * 6  # Multiplier LUTs
        acc_luts = self.accumulator_width * 2  # Accumulator logic
        control_luts = 25  # Control FSM and datapath control
        interconnect_luts = 15  # Local interconnect logic
        luts_per_pe = mac_luts + acc_luts + control_luts + interconnect_luts
        
        # Flip-flop estimation: Pipeline registers + accumulator + control
        pipeline_ffs = self.data_width * 4  # Input/output pipeline stages
        acc_ffs = self.accumulator_width  # Accumulator register
        control_ffs = 20  # Control state and valid signals
        ffs_per_pe = pipeline_ffs + acc_ffs + control_ffs
        
        # DSP estimation based on precision and multiply requirements
        if self.data_width <= 9:
            dsps_per_pe = 0.25  # Can pack 4 small multipliers per DSP
        elif self.data_width <= 18:
            dsps_per_pe = 1  # One DSP48E2 per PE
        else:
            dsps_per_pe = 2  # Need multiple DSPs for larger precision
        
        # Memory requirements based on dataflow and buffering needs
        if self.dataflow == "weight_stationary":
            # Need to store weights for each PE
            weight_storage_bits = pe_count * self.data_width * 512  # Assume 512 weights per PE
            activation_buffer_bits = self.rows * self.data_width * 128  # Input buffers
            total_memory_bits = weight_storage_bits + activation_buffer_bits
        elif self.dataflow == "output_stationary":
            # Need larger accumulator buffers
            output_buffer_bits = pe_count * self.accumulator_width * 64
            input_buffer_bits = (self.rows + self.cols) * self.data_width * 128
            total_memory_bits = output_buffer_bits + input_buffer_bits
        else:  # row_stationary
            # Balanced memory requirements
            total_memory_bits = pe_count * (self.data_width * 256 + self.accumulator_width * 32)
        
        bram_kb = total_memory_bits // (8 * 1024)
        
        # Enhanced power estimation
        # Static power: 0.5mW per PE, Dynamic power: based on utilization and frequency
        static_power_pe = 0.5  # mW
        dynamic_power_pe = 4.0 * (self.data_width / 8) ** 1.5  # Scale with precision
        total_power = pe_count * (static_power_pe + dynamic_power_pe)
        
        # Frequency estimation based on critical path analysis
        base_freq = 350.0  # MHz for modern FPGAs
        
        # Critical path factors
        multiplier_delay = 1.0 + (self.data_width - 8) * 0.05  # Scale with width
        accumulator_delay = 1.0 + (self.accumulator_width - 32) * 0.02
        interconnect_delay = 1.0 + math.log2(max(self.rows, self.cols)) * 0.1
        
        total_delay_factor = multiplier_delay * accumulator_delay * interconnect_delay
        frequency = base_freq / total_delay_factor
        
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