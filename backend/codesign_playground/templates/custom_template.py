"""
Custom Template framework for creating user-defined hardware accelerators.

This module provides a flexible framework for users to define their own
hardware accelerator templates with custom operations and optimizations.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OperationSpec:
    """Specification for a custom operation."""
    name: str
    inputs: List[str]
    outputs: List[str]
    latency: int
    throughput: int
    resource_cost: Dict[str, int] = field(default_factory=dict)
    description: str = ""
    implementation: Optional[str] = None  # RTL or HLS code


@dataclass
class DatapathSpec:
    """Specification for custom datapath."""
    width: int
    type: str  # "integer", "fixed_point", "floating_point"
    signed: bool = True
    fractional_bits: int = 0  # For fixed-point


@dataclass
class MemorySpec:
    """Specification for memory hierarchy."""
    name: str
    size_kb: int
    type: str  # "sram", "bram", "dram", "cache"
    ports: int = 1
    bandwidth_gb_s: float = 1.0
    latency_cycles: int = 1


class HardwareTemplate(ABC):
    """Abstract base class for hardware templates."""
    
    @abstractmethod
    def generate_rtl(self, output_dir: str) -> str:
        """Generate RTL code for the template."""
        pass
    
    @abstractmethod
    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate hardware resource requirements."""
        pass
    
    @abstractmethod
    def estimate_performance(self, workload: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for given workload."""
        pass


class CustomTemplate(HardwareTemplate):
    """
    User-defined custom hardware accelerator template.
    
    Supports:
    - Custom operation definitions
    - Flexible datapath configurations
    - User-defined memory hierarchies
    - Performance modeling
    - RTL generation from specifications
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        config_file: Optional[str] = None
    ):
        """
        Initialize custom template.
        
        Args:
            name: Template name
            description: Template description
            config_file: Optional configuration file path
        """
        self.name = name
        self.description = description
        
        # Template specifications
        self.operations: List[OperationSpec] = []
        self.datapaths: List[DatapathSpec] = []
        self.memory_hierarchy: List[MemorySpec] = []
        
        # Configuration
        self.config = {
            "parallel_units": 1,
            "pipeline_stages": 1,
            "frequency_mhz": 200.0,
            "optimization_level": "balanced"  # "area", "performance", "power", "balanced"
        }
        
        # Performance model
        self.performance_model = {}
        
        # Load configuration if provided
        if config_file:
            self.load_from_file(config_file)
        
        logger.info(f"Initialized CustomTemplate: {name}")
    
    def add_operation(
        self,
        name: str,
        inputs: List[str],
        outputs: List[str],
        latency: int,
        throughput: int,
        description: str = "",
        resource_cost: Optional[Dict[str, int]] = None,
        implementation: Optional[str] = None
    ) -> None:
        """
        Add custom operation to template.
        
        Args:
            name: Operation name
            inputs: List of input types
            outputs: List of output types
            latency: Operation latency in cycles
            throughput: Operations per cycle
            description: Operation description
            resource_cost: Hardware resource costs
            implementation: RTL/HLS implementation code
        """
        operation = OperationSpec(
            name=name,
            inputs=inputs,
            outputs=outputs,
            latency=latency,
            throughput=throughput,
            resource_cost=resource_cost or {},
            description=description,
            implementation=implementation
        )
        
        self.operations.append(operation)
        logger.info(f"Added operation '{name}' to template {self.name}")
    
    def add_datapath(
        self,
        width: int,
        data_type: str,
        signed: bool = True,
        fractional_bits: int = 0
    ) -> None:
        """
        Add datapath specification.
        
        Args:
            width: Bit width
            data_type: Data type ("integer", "fixed_point", "floating_point")
            signed: Whether data is signed
            fractional_bits: Number of fractional bits (for fixed-point)
        """
        datapath = DatapathSpec(
            width=width,
            type=data_type,
            signed=signed,
            fractional_bits=fractional_bits
        )
        
        self.datapaths.append(datapath)
        logger.info(f"Added {width}-bit {data_type} datapath to template {self.name}")
    
    def add_memory(
        self,
        name: str,
        size_kb: int,
        memory_type: str,
        ports: int = 1,
        bandwidth_gb_s: float = 1.0,
        latency_cycles: int = 1
    ) -> None:
        """
        Add memory specification.
        
        Args:
            name: Memory name
            size_kb: Size in KB
            memory_type: Memory type ("sram", "bram", "dram", "cache")
            ports: Number of ports
            bandwidth_gb_s: Bandwidth in GB/s
            latency_cycles: Access latency in cycles
        """
        memory = MemorySpec(
            name=name,
            size_kb=size_kb,
            type=memory_type,
            ports=ports,
            bandwidth_gb_s=bandwidth_gb_s,
            latency_cycles=latency_cycles
        )
        
        self.memory_hierarchy.append(memory)
        logger.info(f"Added {size_kb}KB {memory_type} memory '{name}' to template {self.name}")
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration parameter."""
        self.config[key] = value
        logger.info(f"Set config {key} = {value} for template {self.name}")
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load template configuration from file.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Determine file format
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Load configuration
        if 'name' in config_data:
            self.name = config_data['name']
        if 'description' in config_data:
            self.description = config_data['description']
        if 'config' in config_data:
            self.config.update(config_data['config'])
        
        # Load operations
        if 'operations' in config_data:
            for op_data in config_data['operations']:
                self.add_operation(
                    name=op_data['name'],
                    inputs=op_data['inputs'],
                    outputs=op_data['outputs'],
                    latency=op_data['latency'],
                    throughput=op_data['throughput'],
                    description=op_data.get('description', ''),
                    resource_cost=op_data.get('resource_cost', {}),
                    implementation=op_data.get('implementation')
                )
        
        # Load datapaths
        if 'datapaths' in config_data:
            for dp_data in config_data['datapaths']:
                self.add_datapath(
                    width=dp_data['width'],
                    data_type=dp_data['type'],
                    signed=dp_data.get('signed', True),
                    fractional_bits=dp_data.get('fractional_bits', 0)
                )
        
        # Load memory hierarchy
        if 'memory_hierarchy' in config_data:
            for mem_data in config_data['memory_hierarchy']:
                self.add_memory(
                    name=mem_data['name'],
                    size_kb=mem_data['size_kb'],
                    memory_type=mem_data['type'],
                    ports=mem_data.get('ports', 1),
                    bandwidth_gb_s=mem_data.get('bandwidth_gb_s', 1.0),
                    latency_cycles=mem_data.get('latency_cycles', 1)
                )
        
        logger.info(f"Loaded template configuration from {config_file}")
    
    def save_to_file(self, output_file: str, format: str = "yaml") -> None:
        """
        Save template configuration to file.
        
        Args:
            output_file: Output file path
            format: File format ("yaml" or "json")
        """
        config_data = {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "operations": [
                {
                    "name": op.name,
                    "inputs": op.inputs,
                    "outputs": op.outputs,
                    "latency": op.latency,
                    "throughput": op.throughput,
                    "description": op.description,
                    "resource_cost": op.resource_cost,
                    "implementation": op.implementation
                }
                for op in self.operations
            ],
            "datapaths": [
                {
                    "width": dp.width,
                    "type": dp.type,
                    "signed": dp.signed,
                    "fractional_bits": dp.fractional_bits
                }
                for dp in self.datapaths
            ],
            "memory_hierarchy": [
                {
                    "name": mem.name,
                    "size_kb": mem.size_kb,
                    "type": mem.type,
                    "ports": mem.ports,
                    "bandwidth_gb_s": mem.bandwidth_gb_s,
                    "latency_cycles": mem.latency_cycles
                }
                for mem in self.memory_hierarchy
            ]
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved template configuration to {output_file}")
    
    def generate_rtl(self, output_dir: str = "./rtl") -> str:
        """Generate RTL code for custom template."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        rtl_file = output_path / f"{self.name.lower().replace(' ', '_')}_accelerator.sv"
        
        # Generate RTL based on specifications
        rtl_code = self._generate_custom_rtl()
        
        with open(rtl_file, 'w') as f:
            f.write(rtl_code)
        
        logger.info(f"Generated custom RTL: {rtl_file}")
        return str(rtl_file)
    
    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate hardware resource requirements."""
        total_resources = {
            "luts": 0,
            "ffs": 0,  
            "dsps": 0,
            "bram_kb": 0,
            "power_mw": 0
        }
        
        # Sum resources from operations
        for operation in self.operations:
            for resource, cost in operation.resource_cost.items():
                if resource in total_resources:
                    total_resources[resource] += cost * self.config["parallel_units"]
        
        # Add memory resources
        for memory in self.memory_hierarchy:
            if memory.type in ["sram", "bram"]:
                total_resources["bram_kb"] += memory.size_kb
            
            # Estimate memory controller resources
            total_resources["luts"] += memory.ports * 100
            total_resources["ffs"] += memory.ports * 50
        
        # Add baseline resources for control logic
        total_resources["luts"] += 500 * self.config["parallel_units"]
        total_resources["ffs"] += 300 * self.config["parallel_units"]
        
        # Estimate power
        total_resources["power_mw"] = (
            total_resources["luts"] * 0.01 +
            total_resources["dsps"] * 2.0 +
            total_resources["bram_kb"] * 0.1
        )
        
        return total_resources
    
    def estimate_performance(self, workload: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for given workload."""
        if not self.operations:
            logger.warning("No operations defined for performance estimation")
            return {}
        
        # Calculate theoretical peak performance
        total_throughput = sum(op.throughput for op in self.operations)
        peak_ops_per_cycle = total_throughput * self.config["parallel_units"]
        peak_ops_per_second = peak_ops_per_cycle * self.config["frequency_mhz"] * 1e6
        
        # Estimate actual performance with utilization factor
        utilization = self._estimate_utilization(workload)
        actual_ops_per_second = peak_ops_per_second * utilization
        
        # Calculate latency
        workload_ops = workload.get("operations", 1000)
        cycles_needed = workload_ops / (peak_ops_per_cycle * utilization)
        latency_ms = cycles_needed / (self.config["frequency_mhz"] * 1000)
        
        # Memory analysis
        memory_bound = self._check_memory_bound(workload)
        
        return {
            "peak_ops_per_second": peak_ops_per_second,
            "actual_ops_per_second": actual_ops_per_second,
            "utilization": utilization,
            "latency_ms": latency_ms,
            "throughput_fps": 1000 / latency_ms if latency_ms > 0 else 0,
            "memory_bound": memory_bound,
            "efficiency": utilization * 0.9,  # Account for overhead
        }
    
    def optimize_for_objective(self, objective: str) -> None:
        """
        Optimize template for specific objective.
        
        Args:
            objective: Optimization objective ("area", "performance", "power", "balanced")
        """
        if objective == "area":
            self.config["parallel_units"] = 1
            self.config["pipeline_stages"] = min(2, self.config.get("pipeline_stages", 1))
            self.config["optimization_level"] = "area"
        
        elif objective == "performance":
            # Increase parallelism
            max_parallel = min(16, len(self.operations) * 2)
            self.config["parallel_units"] = max_parallel
            self.config["pipeline_stages"] = max(4, self.config.get("pipeline_stages", 1))
            self.config["optimization_level"] = "performance"
        
        elif objective == "power":
            # Reduce frequency and parallelism
            self.config["frequency_mhz"] = min(100.0, self.config["frequency_mhz"])
            self.config["parallel_units"] = max(1, self.config["parallel_units"] // 2)
            self.config["optimization_level"] = "power"
        
        elif objective == "balanced":
            # Balanced configuration
            self.config["parallel_units"] = min(4, len(self.operations))
            self.config["pipeline_stages"] = 3
            self.config["optimization_level"] = "balanced"
        
        logger.info(f"Optimized template {self.name} for {objective}")
    
    def validate_template(self) -> List[str]:
        """
        Validate template configuration.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for operations
        if not self.operations:
            issues.append("No operations defined")
        
        # Check for datapaths
        if not self.datapaths:
            issues.append("No datapaths defined")
        
        # Validate operation resource costs
        for operation in self.operations:
            if operation.latency <= 0:
                issues.append(f"Operation '{operation.name}' has invalid latency: {operation.latency}")
            
            if operation.throughput <= 0:
                issues.append(f"Operation '{operation.name}' has invalid throughput: {operation.throughput}")
        
        # Check memory hierarchy
        total_memory = sum(mem.size_kb for mem in self.memory_hierarchy)
        if total_memory == 0:
            issues.append("No memory defined in hierarchy")
        
        # Check configuration
        if self.config["parallel_units"] <= 0:
            issues.append("Invalid parallel_units configuration")
        
        if self.config["frequency_mhz"] <= 0:
            issues.append("Invalid frequency_mhz configuration")
        
        if issues:
            logger.warning(f"Template validation found {len(issues)} issues")
        else:
            logger.info(f"Template {self.name} validation passed")
        
        return issues
    
    def _estimate_utilization(self, workload: Dict[str, Any]) -> float:
        """Estimate hardware utilization for workload."""
        # Base utilization
        base_util = 0.8
        
        # Adjust for parallel units
        parallel_efficiency = min(1.0, self.config["parallel_units"] / len(self.operations))
        
        # Adjust for memory characteristics
        memory_efficiency = 0.9  # Assume good memory efficiency
        
        # Adjust for workload characteristics
        workload_fit = workload.get("fit_factor", 1.0)
        
        return base_util * parallel_efficiency * memory_efficiency * workload_fit
    
    def _check_memory_bound(self, workload: Dict[str, Any]) -> float:
        """Check if workload is memory bound."""
        if not self.memory_hierarchy:
            return 0.0
        
        # Calculate memory bandwidth requirement
        data_movement = workload.get("data_movement_gb", 1.0)
        compute_time = workload.get("compute_time_s", 1.0)
        required_bandwidth = data_movement / compute_time
        
        # Calculate available bandwidth
        available_bandwidth = sum(mem.bandwidth_gb_s for mem in self.memory_hierarchy)
        
        # Memory bound ratio
        memory_bound_ratio = required_bandwidth / available_bandwidth if available_bandwidth > 0 else 1.0
        
        return min(1.0, memory_bound_ratio)
    
    def _generate_custom_rtl(self) -> str:
        """Generate RTL code based on template specifications."""
        # Get primary datapath
        primary_datapath = self.datapaths[0] if self.datapaths else DatapathSpec(32, "integer")
        
        rtl_template = f'''
//==============================================================================
// Custom Accelerator: {self.name}
// {self.description}
// Generated by AI Hardware Co-Design Playground
//==============================================================================

`timescale 1ns/1ps

module {self.name.lower().replace(" ", "_")}_accelerator #(
    parameter DATA_WIDTH = {primary_datapath.width},
    parameter PARALLEL_UNITS = {self.config["parallel_units"]},
    parameter PIPELINE_STAGES = {self.config.get("pipeline_stages", 1)}
) (
    input wire clk,
    input wire rst_n,
    
    // Control interface
    input wire start,
    input wire [31:0] config_data,
    output wire done,
    output wire ready,
    
    // Data interface
    input wire [DATA_WIDTH-1:0] data_in,
    input wire data_valid,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire data_ready,
    
    // Memory interface
    output wire [31:0] mem_addr,
    output wire [DATA_WIDTH-1:0] mem_wdata,
    input wire [DATA_WIDTH-1:0] mem_rdata,
    output wire mem_req,
    output wire mem_we,
    input wire mem_ack
);

    // State machine
    typedef enum logic [2:0] {{
        IDLE,
        CONFIG,
        COMPUTE,
        OUTPUT,
        DONE
    }} state_t;
    
    state_t current_state, next_state;
    
    // Processing units
    genvar i;
    generate
        for (i = 0; i < PARALLEL_UNITS; i++) begin : processing_units
            processing_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .UNIT_ID(i)
            ) pu_inst (
                .clk(clk),
                .rst_n(rst_n),
                .enable(current_state == COMPUTE),
                .data_in(data_in),
                .data_valid(data_valid),
                .data_out(data_out),
                .data_ready(data_ready)
            );
        end
    endgenerate
    
    // State machine logic
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
            IDLE: if (start) next_state = CONFIG;
            CONFIG: next_state = COMPUTE;
            COMPUTE: if (/* computation done */) next_state = OUTPUT;
            OUTPUT: if (data_ready) next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    // Output assignments
    assign done = (current_state == DONE);
    assign ready = (current_state == IDLE);
    assign mem_req = (current_state == COMPUTE);
    assign mem_we = 0; // Read-only for now

endmodule

//==============================================================================
// Processing Unit
//==============================================================================

module processing_unit #(
    parameter DATA_WIDTH = 32,
    parameter UNIT_ID = 0
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] data_in,
    input wire data_valid,
    output reg [DATA_WIDTH-1:0] data_out,
    output reg data_ready
);

    // Pipeline registers
    reg [DATA_WIDTH-1:0] pipeline_data [0:PIPELINE_STAGES-1];
    reg pipeline_valid [0:PIPELINE_STAGES-1];
    
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
            data_ready <= 0;
            for (i = 0; i < PIPELINE_STAGES; i++) begin
                pipeline_data[i] <= 0;
                pipeline_valid[i] <= 0;
            end
        end else if (enable) begin
            // Pipeline stage 0 - input
            pipeline_data[0] <= data_in;
            pipeline_valid[0] <= data_valid;
            
            // Pipeline stages 1 to N-1
            for (i = 1; i < PIPELINE_STAGES; i++) begin
                pipeline_data[i] <= pipeline_data[i-1];
                pipeline_valid[i] <= pipeline_valid[i-1];
            end
            
            // Custom operations implementation
'''
        
        # Add custom operation implementations
        for operation in self.operations:
            if operation.implementation:
                rtl_template += f'''
            // {operation.name}: {operation.description}
            {operation.implementation}
'''
            else:
                # Generate simple operation
                rtl_template += f'''
            // {operation.name}: {operation.description}
            // Latency: {operation.latency} cycles, Throughput: {operation.throughput} ops/cycle
            if (pipeline_valid[PIPELINE_STAGES-1]) begin
                data_out <= pipeline_data[PIPELINE_STAGES-1] + 1; // Placeholder operation
            end
'''
        
        rtl_template += '''
            
            // Output stage
            data_ready <= pipeline_valid[PIPELINE_STAGES-1];
        end else begin
            data_ready <= 0;
        end
    end

endmodule
'''
        
        return rtl_template.strip()
    
    def __str__(self) -> str:
        """String representation of custom template."""
        return f"CustomTemplate('{self.name}', {len(self.operations)} ops, {len(self.datapaths)} datapaths)"