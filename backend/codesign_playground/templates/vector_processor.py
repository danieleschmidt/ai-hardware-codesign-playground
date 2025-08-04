"""
Vector Processor hardware template for AI workloads.

This module provides a RISC-V vector extension compatible processor
with custom instruction support for AI operations.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CustomInstruction:
    """Custom instruction definition."""
    name: str
    opcode: str
    latency: int
    throughput: int  # ops per cycle
    description: str = ""
    operands: List[str] = field(default_factory=list)


@dataclass
class InstructionSetArchitecture:
    """Generated ISA specification."""
    base_isa: str = "RV64V"
    custom_instructions: List[CustomInstruction] = field(default_factory=list)
    vector_extensions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ISA to dictionary."""
        return {
            "base_isa": self.base_isa,
            "custom_instructions": [
                {
                    "name": inst.name,
                    "opcode": inst.opcode,
                    "latency": inst.latency,
                    "throughput": inst.throughput,
                    "description": inst.description,
                    "operands": inst.operands
                }
                for inst in self.custom_instructions
            ],
            "vector_extensions": self.vector_extensions
        }


class VectorProcessor:
    """
    RISC-V vector extension based processor for AI workloads.
    
    Supports configurable vector length, multiple lanes, and custom
    instruction extensions for specialized AI operations.
    """
    
    def __init__(
        self,
        vector_length: int = 512,
        num_lanes: int = 8,
        data_width: int = 32,
        supported_ops: Optional[List[str]] = None
    ):
        """
        Initialize vector processor configuration.
        
        Args:
            vector_length: Maximum vector length in elements
            num_lanes: Number of parallel execution lanes
            data_width: Width of each vector element (bits)
            supported_ops: List of supported vector operations
        """
        self.vector_length = vector_length
        self.num_lanes = num_lanes
        self.data_width = data_width
        self.supported_ops = supported_ops or [
            "add", "sub", "mul", "mac", "relu", "sigmoid", "max", "min"
        ]
        
        # ISA and custom instructions
        self.isa = InstructionSetArchitecture()
        self.custom_instructions = []
        
        # Performance model
        self.performance_model = {}
        
        # Load default vector extensions
        self._initialize_vector_extensions()
        
        logger.info(
            "Initialized VectorProcessor",
            vector_length=vector_length,
            num_lanes=num_lanes,
            data_width=data_width,
            supported_ops=len(self.supported_ops)
        )
    
    def add_custom_instruction(
        self,
        name: str,
        latency: int,
        throughput: int,
        description: str = "",
        operands: Optional[List[str]] = None
    ) -> None:
        """
        Add custom instruction to the processor ISA.
        
        Args:
            name: Instruction name
            latency: Execution latency in cycles
            throughput: Operations per cycle
            description: Human-readable description
            operands: List of operand types
        """
        # Generate opcode (simplified)
        opcode = f"0x{len(self.custom_instructions) + 0x100:03x}"
        
        custom_inst = CustomInstruction(
            name=name,
            opcode=opcode,
            latency=latency,
            throughput=throughput,
            description=description,
            operands=operands or ["vector", "vector", "vector"]
        )
        
        self.custom_instructions.append(custom_inst)
        self.isa.custom_instructions.append(custom_inst)
        
        logger.info(
            "Added custom instruction",
            name=name,
            opcode=opcode,
            latency=latency,
            throughput=throughput
        )
    
    def configure_for_workload(
        self,
        workload_type: str,
        model_params: Dict[str, Any]
    ) -> None:
        """
        Configure processor for specific AI workload.
        
        Args:
            workload_type: Type of workload ("cnn", "transformer", "rnn", "mlp")
            model_params: Model-specific parameters
        """
        if workload_type == "cnn":
            self._configure_for_cnn(model_params)
        elif workload_type == "transformer":
            self._configure_for_transformer(model_params)
        elif workload_type == "rnn":
            self._configure_for_rnn(model_params)
        elif workload_type == "mlp":
            self._configure_for_mlp(model_params)
        else:
            logger.warning(f"Unknown workload type: {workload_type}")
        
        logger.info(f"Configured vector processor for {workload_type} workload")
    
    def generate_isa(self) -> InstructionSetArchitecture:
        """Generate complete ISA specification."""
        # Add standard vector extensions if not present
        standard_extensions = ["V", "Zve32f", "Zve64f"]
        for ext in standard_extensions:
            if ext not in self.isa.vector_extensions:
                self.isa.vector_extensions.append(ext)
        
        return self.isa
    
    def generate_rtl(self, output_dir: str = "./rtl") -> str:
        """
        Generate RTL for the vector processor.
        
        Args:
            output_dir: Output directory for RTL files
            
        Returns:
            Path to generated RTL file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        rtl_file = output_path / f"vector_processor_v{self.vector_length}_l{self.num_lanes}.sv"
        
        # Generate RTL code
        rtl_code = self._generate_vector_processor_rtl()
        
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        logger.info(f"Generated vector processor RTL: {rtl_file}")
        return str(rtl_file)
    
    def generate_compiler_support(self, output_dir: str = "./compiler") -> Dict[str, str]:
        """
        Generate compiler support files.
        
        Args:
            output_dir: Output directory for compiler files
            
        Returns:
            Dictionary of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Generate instruction definitions
        inst_def_file = output_path / "custom_instructions.json"
        with open(inst_def_file, 'w') as f:
            json.dump(self.isa.to_dict(), f, indent=2)
        generated_files["instruction_definitions"] = str(inst_def_file)
        
        # Generate intrinsics header
        intrinsics_file = output_path / "vector_intrinsics.h"
        intrinsics_code = self._generate_intrinsics_header()
        with open(intrinsics_file, 'w') as f:
            f.write(intrinsics_code)
        generated_files["intrinsics_header"] = str(intrinsics_file)
        
        # Generate assembly templates
        asm_file = output_path / "vector_templates.s"
        asm_code = self._generate_assembly_templates()
        with open(asm_file, 'w') as f:
            f.write(asm_code)
        generated_files["assembly_templates"] = str(asm_file)
        
        logger.info(f"Generated compiler support files in {output_dir}")
        return generated_files
    
    def estimate_performance(
        self,
        workload_ops: int,
        frequency_mhz: float = 800.0
    ) -> Dict[str, float]:
        """
        Estimate performance for given workload.
        
        Args:
            workload_ops: Number of operations in workload
            frequency_mhz: Operating frequency
            
        Returns:
            Performance metrics
        """
        # Calculate theoretical peak performance
        peak_ops_per_cycle = self.num_lanes
        peak_ops_per_second = peak_ops_per_cycle * frequency_mhz * 1e6
        
        # Estimate actual performance with utilization factor
        utilization = self._estimate_utilization()
        actual_ops_per_second = peak_ops_per_second * utilization
        
        # Calculate latency
        cycles_needed = workload_ops / (peak_ops_per_cycle * utilization)
        latency_ms = cycles_needed / (frequency_mhz * 1000)
        
        return {
            "peak_ops_per_second": peak_ops_per_second,
            "actual_ops_per_second": actual_ops_per_second,
            "utilization": utilization,
            "latency_ms": latency_ms,
            "throughput_fps": 1000 / latency_ms if latency_ms > 0 else 0,
            "vector_efficiency": self._calculate_vector_efficiency(),
        }
    
    def _initialize_vector_extensions(self) -> None:
        """Initialize standard vector extensions."""
        self.isa.vector_extensions = ["V", "Zve32f", "Zve64f"]
        
        # Add common AI operations as custom instructions
        ai_instructions = [
            ("relu", 1, self.num_lanes, "Rectified Linear Unit activation"),
            ("sigmoid", 3, self.num_lanes // 2, "Sigmoid activation function"),
            ("softmax", 5, self.num_lanes // 4, "Softmax normalization"),
            ("layernorm", 4, self.num_lanes // 2, "Layer normalization"),
            ("gelu", 4, self.num_lanes // 2, "Gaussian Error Linear Unit"),
        ]
        
        for name, latency, throughput, desc in ai_instructions:
            if name in self.supported_ops:
                self.add_custom_instruction(name, latency, throughput, desc)
    
    def _configure_for_cnn(self, params: Dict[str, Any]) -> None:
        """Configure for CNN workloads."""
        # Add CNN-specific custom instructions
        self.add_custom_instruction(
            "conv3x3", 4, self.num_lanes,
            "3x3 convolution operation",
            ["vector", "vector", "vector", "scalar"]
        )
        
        self.add_custom_instruction(
            "maxpool2x2", 1, self.num_lanes,
            "2x2 max pooling operation",
            ["vector", "vector"]
        )
        
        # Update performance model for CNN characteristics
        self.performance_model.update({
            "workload_type": "cnn",
            "compute_intensity": params.get("compute_intensity", 2.0),
            "memory_intensity": params.get("memory_intensity", 1.5),
        })
    
    def _configure_for_transformer(self, params: Dict[str, Any]) -> None:
        """Configure for Transformer workloads."""
        # Add attention-specific operations
        self.add_custom_instruction(
            "attention_qkv", 6, self.num_lanes // 2,
            "Query-Key-Value attention computation",
            ["vector", "vector", "vector", "vector"]
        )
        
        self.add_custom_instruction(
            "multihead_attn", 8, self.num_lanes // 4,
            "Multi-head attention block",
            ["vector", "vector", "vector", "vector", "scalar"]
        )
        
        self.performance_model.update({
            "workload_type": "transformer",
            "sequence_length": params.get("sequence_length", 512),
            "attention_heads": params.get("attention_heads", 8),
        })
    
    def _configure_for_rnn(self, params: Dict[str, Any]) -> None:
        """Configure for RNN workloads."""
        self.add_custom_instruction(
            "lstm_cell", 10, self.num_lanes // 2,
            "LSTM cell computation",
            ["vector", "vector", "vector", "vector"]
        )
        
        self.add_custom_instruction(
            "gru_cell", 8, self.num_lanes // 2,
            "GRU cell computation",
            ["vector", "vector", "vector"]
        )
    
    def _configure_for_mlp(self, params: Dict[str, Any]) -> None:
        """Configure for MLP workloads."""
        self.add_custom_instruction(
            "dense_layer", 3, self.num_lanes,
            "Dense layer computation with activation",
            ["vector", "vector", "vector", "scalar"]
        )
    
    def _estimate_utilization(self) -> float:
        """Estimate vector lane utilization."""
        # Simplified utilization model
        base_utilization = 0.8
        
        # Adjust based on vector length vs typical data sizes
        if self.vector_length >= 512:
            utilization = base_utilization * 0.95
        elif self.vector_length >= 256:
            utilization = base_utilization * 0.9
        else:
            utilization = base_utilization * 0.8
        
        return min(1.0, utilization)
    
    def _calculate_vector_efficiency(self) -> float:
        """Calculate vector processing efficiency."""
        # Factor in the number of custom instructions
        custom_inst_factor = 1.0 + (len(self.custom_instructions) * 0.1)
        
        # Factor in lane count efficiency
        lane_efficiency = min(1.0, self.num_lanes / 16.0)  # Optimal around 16 lanes
        
        return min(1.0, lane_efficiency * custom_inst_factor * 0.9)
    
    def _generate_vector_processor_rtl(self) -> str:
        """Generate SystemVerilog RTL for vector processor."""
        rtl_template = f'''
//==============================================================================
// Vector Processor - RISC-V Vector Extension Compatible
// Vector Length: {self.vector_length}, Lanes: {self.num_lanes}
// Generated by AI Hardware Co-Design Playground
//==============================================================================

`timescale 1ns/1ps

module vector_processor #(
    parameter VLEN = {self.vector_length},      // Vector length in elements
    parameter NLANES = {self.num_lanes},        // Number of execution lanes
    parameter ELEN = {self.data_width},         // Element width in bits
    parameter SLEN = VLEN,                      // Striping distance
    parameter CUSTOM_INST_COUNT = {len(self.custom_instructions)}
) (
    input wire clk,
    input wire rst_n,
    
    // Scalar processor interface
    input wire [31:0] scalar_pc,
    input wire [31:0] instruction,
    input wire inst_valid,
    output wire inst_ready,
    
    // Memory interface
    output wire [31:0] mem_addr,
    output wire [ELEN*NLANES-1:0] mem_wdata,
    input wire [ELEN*NLANES-1:0] mem_rdata,
    output wire mem_req,
    output wire mem_we,
    input wire mem_ack,
    
    // Vector register file interface
    output wire [4:0] vrd_addr,
    output wire [4:0] vrs1_addr,
    output wire [4:0] vrs2_addr,
    output wire [VLEN*ELEN-1:0] vrd_data,
    input wire [VLEN*ELEN-1:0] vrs1_data,
    input wire [VLEN*ELEN-1:0] vrs2_data,
    output wire vrf_we
);

    // Vector execution unit
    vector_execution_unit #(
        .VLEN(VLEN),
        .NLANES(NLANES),
        .ELEN(ELEN)
    ) vexu (
        .clk(clk),
        .rst_n(rst_n),
        .instruction(instruction),
        .inst_valid(inst_valid),
        .inst_ready(inst_ready),
        .vrs1_data(vrs1_data),
        .vrs2_data(vrs2_data),
        .vrd_data(vrd_data),
        .vrf_we(vrf_we),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_rdata(mem_rdata),
        .mem_req(mem_req),
        .mem_we(mem_we),
        .mem_ack(mem_ack)
    );
    
    // Instruction decode
    vector_decode vdecode (
        .instruction(instruction),
        .vrd_addr(vrd_addr),
        .vrs1_addr(vrs1_addr),
        .vrs2_addr(vrs2_addr)
    );

endmodule

//==============================================================================
// Vector Execution Unit
//==============================================================================

module vector_execution_unit #(
    parameter VLEN = 512,
    parameter NLANES = 8,
    parameter ELEN = 32
) (
    input wire clk,
    input wire rst_n,
    
    input wire [31:0] instruction,
    input wire inst_valid,
    output reg inst_ready,
    
    input wire [VLEN*ELEN-1:0] vrs1_data,
    input wire [VLEN*ELEN-1:0] vrs2_data,
    output reg [VLEN*ELEN-1:0] vrd_data,
    output reg vrf_we,
    
    output reg [31:0] mem_addr,
    output reg [ELEN*NLANES-1:0] mem_wdata,
    input wire [ELEN*NLANES-1:0] mem_rdata,
    output reg mem_req,
    output reg mem_we,
    input wire mem_ack
);

    // Instruction fields
    wire [6:0] opcode = instruction[6:0];
    wire [4:0] vd = instruction[11:7];
    wire [4:0] vs1 = instruction[19:15];
    wire [4:0] vs2 = instruction[24:20];
    wire [5:0] funct6 = instruction[31:26];
    
    // Vector lanes
    genvar i;
    generate
        for (i = 0; i < NLANES; i++) begin : vector_lanes
            vector_lane #(
                .ELEN(ELEN),
                .LANE_ID(i)
            ) vlane (
                .clk(clk),
                .rst_n(rst_n),
                .opcode(opcode),
                .funct6(funct6),
                .operand_a(vrs1_data[(i+1)*ELEN-1:i*ELEN]),
                .operand_b(vrs2_data[(i+1)*ELEN-1:i*ELEN]),
                .result(vrd_data[(i+1)*ELEN-1:i*ELEN])
            );
        end
    endgenerate
    
    // Control logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inst_ready <= 1;
            vrf_we <= 0;
            mem_req <= 0;
        end else begin
            inst_ready <= 1; // Simplified - always ready
            vrf_we <= inst_valid; // Write result when instruction is valid
            
            // Memory operations (simplified)
            if (opcode == 7'b0000111) begin // Vector load
                mem_req <= inst_valid;
                mem_we <= 0;
                mem_addr <= {{{{27{{1'b0}}}}, vs1}};
            end else if (opcode == 7'b0100111) begin // Vector store
                mem_req <= inst_valid;
                mem_we <= 1;
                mem_addr <= {{{{27{{1'b0}}}}, vs1}};
                mem_wdata <= vrs2_data[ELEN*NLANES-1:0];
            end else begin
                mem_req <= 0;
            end
        end
    end

endmodule

//==============================================================================
// Vector Lane
//==============================================================================

module vector_lane #(
    parameter ELEN = 32,
    parameter LANE_ID = 0
) (
    input wire clk,
    input wire rst_n,
    
    input wire [6:0] opcode,
    input wire [5:0] funct6,
    input wire [ELEN-1:0] operand_a,
    input wire [ELEN-1:0] operand_b,
    output reg [ELEN-1:0] result
);

    // Vector arithmetic operations
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
        end else begin
            case ({{opcode, funct6}})
                // Standard vector operations
                {{7'b1010111, 6'b000000}}: result <= operand_a + operand_b;      // VADD
                {{7'b1010111, 6'b000010}}: result <= operand_a - operand_b;      // VSUB
                {{7'b1010111, 6'b100101}}: result <= operand_a * operand_b;      // VMUL
'''
        
        # Add custom instruction implementations
        for inst in self.custom_instructions:
            custom_opcode = inst.opcode.replace("0x", "")
            rtl_template += f'''
                13'h{custom_opcode}: begin // {inst.name.upper()}
                    // Custom instruction: {inst.description}
                    case ("{inst.name}")
                        "relu": result <= (operand_a[ELEN-1] == 0) ? operand_a : 0;
                        "sigmoid": result <= operand_a; // Simplified - would need LUT
                        "conv3x3": result <= operand_a + operand_b; // Simplified
                        default: result <= operand_a;
                    endcase
                end'''
        
        rtl_template += '''
                default: result <= operand_a; // Pass-through for unknown ops
            endcase
        end
    end

endmodule

//==============================================================================
// Vector Decode Unit
//==============================================================================

module vector_decode (
    input wire [31:0] instruction,
    output wire [4:0] vrd_addr,
    output wire [4:0] vrs1_addr,
    output wire [4:0] vrs2_addr
);

    assign vrd_addr = instruction[11:7];
    assign vrs1_addr = instruction[19:15];
    assign vrs2_addr = instruction[24:20];

endmodule
'''
        
        return rtl_template.strip()
    
    def _generate_intrinsics_header(self) -> str:
        """Generate C intrinsics header for custom instructions."""
        header = f'''
//==============================================================================
// Vector Processor Intrinsics
// Generated by AI Hardware Co-Design Playground
//==============================================================================

#ifndef VECTOR_INTRINSICS_H
#define VECTOR_INTRINSICS_H

#include <stdint.h>

// Vector types
typedef int{self.data_width}_t v{self.data_width};
typedef float vfloat32;

// Vector length
#define VLEN {self.vector_length}
#define NLANES {self.num_lanes}

// Standard vector operations
static inline v{self.data_width} vadd_v{self.data_width}(v{self.data_width} a, v{self.data_width} b) {{
    return __builtin_riscv_vadd(a, b);
}}

static inline v{self.data_width} vmul_v{self.data_width}(v{self.data_width} a, v{self.data_width} b) {{
    return __builtin_riscv_vmul(a, b);
}}

// Custom AI instructions
'''
        
        for inst in self.custom_instructions:
            header += f'''
static inline v{self.data_width} {inst.name}_v{self.data_width}('''
            
            # Generate parameter list based on operands
            params = []
            for i, operand in enumerate(inst.operands):
                if operand == "vector":
                    params.append(f"v{self.data_width} op{i}")
                elif operand == "scalar":
                    params.append(f"int{self.data_width}_t scalar{i}")
            
            header += ", ".join(params)
            header += f''') {{
    // {inst.description}
    // Latency: {inst.latency} cycles, Throughput: {inst.throughput} ops/cycle
    return __builtin_riscv_custom_{inst.name}({", ".join([f"op{i}" for i in range(len(params))])});
}}
'''
        
        header += '''
#endif // VECTOR_INTRINSICS_H
'''
        
        return header.strip()
    
    def _generate_assembly_templates(self) -> str:
        """Generate assembly templates for custom instructions."""
        asm_code = f'''
# Vector Processor Assembly Templates
# Generated by AI Hardware Co-Design Playground

.text
.global _start

# Vector configuration
vsetvli t0, zero, e{self.data_width}, m1  # Set vector length

'''
        
        for inst in self.custom_instructions:
            asm_code += f'''
# {inst.name.upper()} - {inst.description}
# Latency: {inst.latency} cycles, Throughput: {inst.throughput} ops/cycle
{inst.name}_template:
    # Load vector operands
    vle.v v1, (a0)      # Load first vector
    vle.v v2, (a1)      # Load second vector
    
    # Execute custom instruction
    .insn r {inst.opcode}, 0, 0, v3, v1, v2
    
    # Store result
    vse.v v3, (a2)      # Store result vector
    ret

'''
        
        return asm_code.strip()
    
    def __str__(self) -> str:
        """String representation of vector processor."""
        return f"VectorProcessor(V{self.vector_length}, L{self.num_lanes}, {self.data_width}b, {len(self.custom_instructions)} custom ops)"