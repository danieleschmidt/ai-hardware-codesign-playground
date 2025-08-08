"""
Command-line interface for AI Hardware Co-Design Playground.

This module provides the main CLI entry point for the codesign-playground tool,
supporting various commands for model analysis, hardware design, and workflow management.
"""

import typer
from typing import Optional, List
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from .core import (AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer, Workflow,
                  CycleAccurateSimulator, PowerAnalyzer, AreaEstimator, PerformanceOptimizer,
                  SimulationBackend)
from .templates import SystolicArray, VectorProcessor, TransformerAccelerator, CustomTemplate

app = typer.Typer(
    name="codesign-playground",
    help="AI Hardware Co-Design Playground - Interactive environment for neural network and hardware accelerator co-optimization",
    add_completion=False
)

console = Console()


@app.command()
def verify() -> None:
    """Verify installation and dependencies."""
    console.print("[bold green]AI Hardware Co-Design Playground[/bold green]")
    console.print("Verifying installation...")
    
    # Check core components
    try:
        designer = AcceleratorDesigner()
        console.print("✓ AcceleratorDesigner loaded")
        
        explorer = DesignSpaceExplorer()
        console.print("✓ DesignSpaceExplorer loaded")
        
        # Test basic functionality
        mock_model = {"type": "test"}
        profile = designer.profile_model(mock_model, (224, 224, 3))
        console.print(f"✓ Model profiling works ({profile.peak_gflops:.2f} GFLOPS)")
        
        accelerator = designer.design(compute_units=16)
        console.print(f"✓ Accelerator design works ({accelerator.compute_units} compute units)")
        
        console.print("[bold green]Installation verified successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Verification failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def profile_model(
    model_path: str = typer.Argument(..., help="Path to model file"),
    input_shape: str = typer.Option("224,224,3", help="Input shape as comma-separated values"),
    output: Optional[str] = typer.Option(None, help="Output file for profile results"),
) -> None:
    """Profile a neural network model to analyze computational requirements."""
    
    console.print(f"[bold]Profiling model:[/bold] {model_path}")
    
    try:
        # Parse input shape
        shape_dims = tuple(map(int, input_shape.split(',')))
        
        # Create designer and profile model
        designer = AcceleratorDesigner()
        mock_model = {"path": model_path, "type": "inference"}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing model...", total=None)
            profile = designer.profile_model(mock_model, shape_dims)
            progress.update(task, completed=True)
        
        # Display results
        table = Table(title="Model Profile")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Peak GFLOPS", f"{profile.peak_gflops:.2f}")
        table.add_row("Memory Bandwidth", f"{profile.bandwidth_gb_s:.2f} GB/s")
        table.add_row("Parameters", f"{profile.parameters:,}")
        table.add_row("Model Size", f"{profile.model_size_mb:.2f} MB")
        table.add_row("Compute Intensity", f"{profile.compute_intensity:.2f}")
        table.add_row("Layer Types", ", ".join(profile.layer_types))
        
        console.print(table)
        
        # Save results if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            console.print(f"Profile saved to: {output_path}")
            
    except Exception as e:
        console.print(f"[bold red]Error profiling model: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def design_accelerator(
    compute_units: int = typer.Option(64, help="Number of compute units"),
    dataflow: str = typer.Option("weight_stationary", help="Dataflow pattern"),
    memory: str = typer.Option("sram_64kb,dram", help="Memory hierarchy (comma-separated)"),
    frequency: float = typer.Option(200.0, help="Operating frequency in MHz"),
    precision: str = typer.Option("int8", help="Numerical precision"),
    output_dir: str = typer.Option("./output", help="Output directory for RTL"),
) -> None:
    """Design a hardware accelerator with specified parameters."""
    
    console.print("[bold]Designing Hardware Accelerator[/bold]")
    
    try:
        # Parse memory hierarchy
        memory_hierarchy = memory.split(',')
        
        # Create designer
        designer = AcceleratorDesigner()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Designing accelerator...", total=None)
            
            accelerator = designer.design(
                compute_units=compute_units,
                memory_hierarchy=memory_hierarchy,
                dataflow=dataflow,
                frequency_mhz=frequency,
                precision=precision
            )
            
            progress.update(task, completed=True)
        
        # Display accelerator specs
        table = Table(title="Accelerator Design")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Compute Units", str(accelerator.compute_units))
        table.add_row("Dataflow", accelerator.dataflow)
        table.add_row("Memory Hierarchy", " → ".join(accelerator.memory_hierarchy))
        table.add_row("Frequency", f"{accelerator.frequency_mhz:.1f} MHz")
        table.add_row("Precision", accelerator.precision)
        table.add_row("Data Width", f"{accelerator.data_width} bits")
        
        console.print(table)
        
        # Generate RTL
        rtl_path = Path(output_dir) / "accelerator.v"
        rtl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating RTL...", total=None)
            accelerator.generate_rtl(str(rtl_path))
            progress.update(task, completed=True)
        
        # Show performance estimates
        perf = accelerator.estimate_performance()
        
        perf_table = Table(title="Performance Estimates")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("Throughput", f"{perf['throughput_ops_s']/1e9:.2f} GOP/s")
        perf_table.add_row("Latency", f"{perf['latency_ms']:.2f} ms")
        perf_table.add_row("Power", f"{perf['power_w']:.2f} W")
        perf_table.add_row("Efficiency", f"{perf['efficiency_ops_w']/1e6:.2f} MOP/s/W")
        perf_table.add_row("Area", f"{perf['area_mm2']:.2f} mm²")
        
        console.print(perf_table)
        console.print(f"RTL generated: {rtl_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error designing accelerator: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def explore_design_space(
    model_path: str = typer.Argument(..., help="Path to model file"),
    input_shape: str = typer.Option("224,224,3", help="Input shape as comma-separated values"),
    objectives: str = typer.Option("latency,power,area", help="Optimization objectives"),
    num_samples: int = typer.Option(50, help="Number of design points to evaluate"),
    strategy: str = typer.Option("random", help="Exploration strategy"),
    output: str = typer.Option("./design_space_results.json", help="Output file for results"),
) -> None:
    """Explore design space for optimal hardware-software configurations."""
    
    console.print("[bold]Design Space Exploration[/bold]")
    
    try:
        # Parse parameters
        shape_dims = tuple(map(int, input_shape.split(',')))
        obj_list = objectives.split(',')
        
        # Define design space
        design_space = {
            "compute_units": [16, 32, 64, 128],
            "memory_hierarchy": [["sram_32kb", "dram"], ["sram_64kb", "dram"], ["sram_128kb", "dram"]],
            "dataflow": ["weight_stationary", "output_stationary"],
            "frequency_mhz": [100, 200, 400],
            "precision": ["int8", "fp16"],
        }
        
        # Create mock model
        mock_model = {"path": model_path, "type": "inference"}
        
        # Run exploration
        explorer = DesignSpaceExplorer()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Exploring {num_samples} design points...", total=None)
            
            results = explorer.explore(
                model=mock_model,
                design_space=design_space,
                objectives=obj_list,
                num_samples=num_samples,
                strategy=strategy
            )
            
            progress.update(task, completed=True)
        
        # Display results summary
        console.print(f"[bold green]Exploration completed![/bold green]")
        console.print(f"Evaluated {results.total_evaluations} design points in {results.exploration_time:.2f}s")
        console.print(f"Found {len(results.pareto_frontier)} Pareto-optimal designs")
        
        # Show best designs
        best_table = Table(title="Best Designs by Objective")
        best_table.add_column("Objective", style="cyan")
        best_table.add_column("Value", style="green")
        best_table.add_column("Configuration", style="yellow")
        
        for obj_name, design_point in results.best_designs.items():
            obj = obj_name.replace("best_", "")
            value = design_point.metrics[obj]
            config_str = f"CU:{design_point.config['compute_units']}, {design_point.config['dataflow']}"
            best_table.add_row(obj.title(), f"{value:.3f}", config_str)
        
        console.print(best_table)
        
        # Save results
        results.save(output)
        console.print(f"Results saved to: {output}")
        
        # Offer to generate Pareto plot
        if typer.confirm("Generate Pareto frontier plot?"):
            plot_path = output.replace('.json', '_pareto.html')
            explorer.plot_pareto(results, save_to=plot_path)
            console.print(f"Pareto plot saved to: {plot_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error in design space exploration: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def run_workflow(
    name: str = typer.Argument(..., help="Workflow name"),
    model_path: str = typer.Option(..., help="Path to model file"),
    input_shape: str = typer.Option("224,224,3", help="Input shape"),
    template: str = typer.Option("systolic_array", help="Hardware template"),
    size: str = typer.Option("16,16", help="Hardware dimensions"),
    output_dir: str = typer.Option("./workflows", help="Output directory"),
) -> None:
    """Run end-to-end workflow from model to RTL."""
    
    console.print(f"[bold]Running Workflow: {name}[/bold]")
    
    try:
        # Parse parameters
        shape_dims = tuple(map(int, input_shape.split(',')))
        hw_size = tuple(map(int, size.split(',')))
        
        # Create workflow
        workflow = Workflow(name, output_dir)
        
        # Execute workflow steps
        with Progress(console=console) as progress:
            main_task = progress.add_task("Workflow Progress", total=100)
            
            # Step 1: Import model
            progress.update(main_task, completed=10)
            workflow.import_model(
                model_path,
                {"input": shape_dims}
            )
            console.print("✓ Model imported and profiled")
            
            # Step 2: Map to hardware
            progress.update(main_task, completed=30)
            workflow.map_to_hardware(
                template=template,
                size=hw_size,
                precision="int8"
            )
            console.print("✓ Hardware mapping completed")
            
            # Step 3: Compile
            progress.update(main_task, completed=50)
            workflow.compile(
                optimizer="tvm",
                target="custom_accelerator",
                optimizations=["layer_fusion", "tensorization"]
            )
            console.print("✓ Compilation completed")
            
            # Step 4: Simulate
            progress.update(main_task, completed=70)
            metrics = workflow.simulate(
                testbench="mock_testbench",
                cycles_limit=1000000
            )
            console.print("✓ Simulation completed")
            
            # Step 5: Generate RTL
            progress.update(main_task, completed=90)
            workflow.generate_rtl(include_testbench=True)
            console.print("✓ RTL generated")
            
            progress.update(main_task, completed=100)
        
        # Display final results
        results_table = Table(title=f"Workflow Results: {name}")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Throughput", f"{metrics.images_per_second:.1f} img/s")
        results_table.add_row("Power", f"{metrics.average_power:.2f} W")
        results_table.add_row("Efficiency", f"{metrics.tops_per_watt:.3f} TOPS/W")
        results_table.add_row("Latency", f"{metrics.latency_ms:.2f} ms")
        results_table.add_row("Area", f"{metrics.area_mm2:.2f} mm²")
        
        console.print(results_table)
        
        # Show artifacts
        status = workflow.get_status()
        console.print(f"[bold green]Workflow completed successfully![/bold green]")
        console.print(f"Output directory: {workflow.output_dir}")
        console.print("Generated artifacts:")
        for name, path in status["artifacts"].items():
            console.print(f"  • {name}: {path}")
        
    except Exception as e:
        console.print(f"[bold red]Workflow failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: str = typer.Argument(..., help="Path to model file"),
    accelerator_config: str = typer.Option(..., help="Path to accelerator config JSON"),
    iterations: int = typer.Option(10, help="Number of benchmark iterations"),
) -> None:
    """Benchmark model performance on specified accelerator."""
    
    console.print("[bold]Running Performance Benchmark[/bold]")
    
    try:
        # Load accelerator configuration
        with open(accelerator_config, 'r') as f:
            config = json.load(f)
        
        # Create accelerator from config
        designer = AcceleratorDesigner()
        accelerator = designer.design(**config)
        
        # Mock model
        mock_model = {"path": model_path, "type": "benchmark"}
        
        # Create optimizer
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        # Run benchmark iterations
        results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task(f"Running {iterations} iterations...", total=iterations)
            
            for i in range(iterations):
                # Simulate benchmark run
                metrics = accelerator.estimate_performance()
                results.append(metrics)
                progress.advance(task)
                time.sleep(0.1)  # Simulate work
        
        # Calculate statistics
        throughputs = [r["throughput_ops_s"] for r in results]
        powers = [r["power_w"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        
        # Display benchmark results
        bench_table = Table(title="Benchmark Results")
        bench_table.add_column("Metric", style="cyan")
        bench_table.add_column("Mean", style="green")
        bench_table.add_column("Std Dev", style="yellow")
        bench_table.add_column("Min", style="blue")
        bench_table.add_column("Max", style="red")
        
        import statistics
        
        bench_table.add_row(
            "Throughput (GOP/s)",
            f"{statistics.mean(throughputs)/1e9:.2f}",
            f"{statistics.stdev(throughputs)/1e9:.2f}",
            f"{min(throughputs)/1e9:.2f}",
            f"{max(throughputs)/1e9:.2f}"
        )
        
        bench_table.add_row(
            "Power (W)",
            f"{statistics.mean(powers):.2f}",
            f"{statistics.stdev(powers):.2f}",
            f"{min(powers):.2f}",
            f"{max(powers):.2f}"
        )
        
        bench_table.add_row(
            "Latency (ms)",
            f"{statistics.mean(latencies):.2f}",
            f"{statistics.stdev(latencies):.2f}",
            f"{min(latencies):.2f}",
            f"{max(latencies):.2f}"
        )
        
        console.print(bench_table)
        console.print(f"[bold green]Benchmark completed: {iterations} iterations[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Benchmark failed: {e}[/bold red]")
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


@app.command()
def simulate_hardware(
    rtl_file: str = typer.Argument(..., help="Path to RTL file"),
    testbench: str = typer.Option("./testbench.sv", help="Path to testbench file"),
    backend: str = typer.Option("analytical", help="Simulation backend"),
    max_cycles: int = typer.Option(1000000, help="Maximum simulation cycles"),
    save_waveform: bool = typer.Option(False, help="Save waveform data"),
) -> None:
    """Run cycle-accurate hardware simulation."""
    
    console.print("[bold]Hardware Simulation[/bold]")
    
    try:
        # Create simulator
        sim_backend = SimulationBackend(backend.upper())
        simulator = CycleAccurateSimulator(sim_backend)
        
        # Mock input data
        class MockInputData:
            def __init__(self):
                self.size = 1000
        
        input_data = MockInputData()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running simulation...", total=None)
            
            metrics = simulator.run(
                rtl_file=rtl_file,
                testbench=testbench,
                input_data=input_data,
                max_cycles=max_cycles,
                save_waveform=save_waveform
            )
            
            progress.update(task, completed=True)
        
        # Display results
        sim_table = Table(title="Simulation Results")
        sim_table.add_column("Metric", style="cyan")
        sim_table.add_column("Value", style="green")
        
        sim_table.add_row("Operations/Second", f"{metrics.operations_per_second/1e9:.2f} GOP/s")
        sim_table.add_row("Frames/Second", f"{metrics.frames_per_second:.1f}")
        sim_table.add_row("Latency", f"{metrics.latency_ms:.2f} ms")
        sim_table.add_row("Compute Utilization", f"{metrics.compute_utilization:.1%}")
        sim_table.add_row("Memory Utilization", f"{metrics.memory_utilization:.1%}")
        sim_table.add_row("Cache Hit Rate", f"{metrics.cache_hit_rate:.1%}")
        sim_table.add_row("Accuracy", f"{metrics.accuracy:.3f}")
        
        console.print(sim_table)
        console.print(f"[bold green]Simulation completed: {metrics.latency_cycles:,} cycles[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Simulation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def analyze_power(
    design_file: str = typer.Argument(..., help="Path to design file"),
    technology: str = typer.Option("tsmc_28nm", help="Technology node"),
    frequency: float = typer.Option(200.0, help="Operating frequency (MHz)"),
    voltage: float = typer.Option(0.9, help="Supply voltage (V)"),
    temperature: float = typer.Option(25.0, help="Operating temperature (C)"),
    activity_file: Optional[str] = typer.Option(None, help="Activity file (VCD/SAIF)"),
) -> None:
    """Analyze power consumption of hardware design."""
    
    console.print("[bold]Power Analysis[/bold]")
    
    try:
        # Create power analyzer
        analyzer = PowerAnalyzer(technology)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing power...", total=None)
            
            power_report = analyzer.analyze(
                design_file=design_file,
                activity_file=activity_file,
                frequency_mhz=frequency,
                voltage=voltage,
                temperature=temperature
            )
            
            progress.update(task, completed=True)
        
        # Display power breakdown
        power_table = Table(title="Power Analysis Report")
        power_table.add_column("Component", style="cyan")
        power_table.add_column("Power (mW)", style="green")
        power_table.add_column("Percentage", style="yellow")
        
        total_power = power_report.total_power_mw()
        
        power_table.add_row("Dynamic", f"{power_report.dynamic_power_mw:.1f}", 
                           f"{power_report.dynamic_power_mw/total_power*100:.1f}%")
        power_table.add_row("Static", f"{power_report.static_power_mw:.1f}", 
                           f"{power_report.static_power_mw/total_power*100:.1f}%")
        power_table.add_row("Compute", f"{power_report.compute_power_mw:.1f}", 
                           f"{power_report.compute_power_mw/total_power*100:.1f}%")
        power_table.add_row("Memory", f"{power_report.memory_power_mw:.1f}", 
                           f"{power_report.memory_power_mw/total_power*100:.1f}%")
        power_table.add_row("I/O", f"{power_report.io_power_mw:.1f}", 
                           f"{power_report.io_power_mw/total_power*100:.1f}%")
        power_table.add_row("Clock", f"{power_report.clock_power_mw:.1f}", 
                           f"{power_report.clock_power_mw/total_power*100:.1f}%")
        power_table.add_row("[bold]Total[/bold]", f"[bold]{total_power:.1f}[/bold]", "[bold]100.0%[/bold]")
        
        console.print(power_table)
        
        # Temperature info
        temp_table = Table(title="Thermal Analysis")
        temp_table.add_column("Metric", style="cyan")
        temp_table.add_column("Value", style="green")
        
        temp_table.add_row("Max Temperature", f"{power_report.max_temperature_c:.1f}°C")
        temp_table.add_row("Average Temperature", f"{power_report.average_temperature_c:.1f}°C")
        
        console.print(temp_table)
        
        # Power optimization suggestions
        suggestions = analyzer.suggest_optimizations(power_report)
        if suggestions:
            console.print("\n[bold]Power Optimization Suggestions:[/bold]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"{i}. [bold]{suggestion['technique']}[/bold]: {suggestion['description']}")
                console.print(f"   Potential savings: {suggestion['power_savings_mw']:.1f} mW (Effort: {suggestion['effort']})")
        
    except Exception as e:
        console.print(f"[bold red]Power analysis failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def estimate_area(
    design_file: str = typer.Argument(..., help="Path to design file"),
    technology: str = typer.Option("sky130", help="Technology node"),
    target_frequency: float = typer.Option(200.0, help="Target frequency (MHz)"),
    output_floorplan: Optional[str] = typer.Option(None, help="Output floorplan image"),
) -> None:
    """Estimate chip area for hardware design."""
    
    console.print("[bold]Area Estimation[/bold]")
    
    try:
        # Create area estimator
        estimator = AreaEstimator(technology)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Estimating area...", total=None)
            
            area_report = estimator.estimate(
                design_file=design_file,
                target_frequency=target_frequency
            )
            
            progress.update(task, completed=True)
        
        # Display area breakdown
        area_table = Table(title="Area Estimation Report")
        area_table.add_column("Component", style="cyan")
        area_table.add_column("Area (mm²)", style="green")
        area_table.add_column("Percentage", style="yellow")
        
        total_area = area_report.total_area_mm2
        
        area_table.add_row("Logic", f"{area_report.logic_area_mm2:.3f}", 
                          f"{area_report.logic_area_mm2/total_area*100:.1f}%")
        area_table.add_row("Memory", f"{area_report.memory_area_mm2:.3f}", 
                          f"{area_report.memory_area_mm2/total_area*100:.1f}%")
        area_table.add_row("I/O", f"{area_report.io_area_mm2:.3f}", 
                          f"{area_report.io_area_mm2/total_area*100:.1f}%")
        area_table.add_row("[bold]Total[/bold]", f"[bold]{total_area:.3f}[/bold]", "[bold]100.0%[/bold]")
        
        console.print(area_table)
        
        # FPGA resource utilization
        if area_report.luts > 0:
            fpga_table = Table(title="FPGA Resource Utilization")
            fpga_table.add_column("Resource", style="cyan")
            fpga_table.add_column("Count", style="green")
            
            fpga_table.add_row("LUTs", f"{area_report.luts:,}")
            fpga_table.add_row("Flip-Flops", f"{area_report.ffs:,}")
            fpga_table.add_row("DSPs", f"{area_report.dsps:,}")
            fpga_table.add_row("BRAMs", f"{area_report.brams:,}")
            fpga_table.add_row("Utilization", f"{area_report.utilization_percent:.1f}%")
            
            console.print(fpga_table)
        
        # Generate floorplan if requested
        if output_floorplan:
            estimator.visualize_floorplan(area_report, output_floorplan)
            console.print(f"Floorplan saved to: {output_floorplan}")
        
    except Exception as e:
        console.print(f"[bold red]Area estimation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def create_template(
    template_type: str = typer.Argument(..., help="Template type (systolic_array, vector_processor, transformer)"),
    name: str = typer.Option("my_accelerator", help="Accelerator name"),
    output_dir: str = typer.Option("./templates", help="Output directory"),
    **kwargs
) -> None:
    """Create hardware accelerator from template."""
    
    console.print(f"[bold]Creating {template_type} template[/bold]")
    
    try:
        template = None
        
        if template_type == "systolic_array":
            rows = typer.prompt("Number of rows", type=int, default=16)
            cols = typer.prompt("Number of columns", type=int, default=16)
            data_width = typer.prompt("Data width (bits)", type=int, default=8)
            dataflow = typer.prompt("Dataflow pattern", default="weight_stationary")
            
            template = SystolicArray(
                rows=rows,
                cols=cols,
                data_width=data_width,
                dataflow=dataflow
            )
            
        elif template_type == "vector_processor":
            vector_length = typer.prompt("Vector length", type=int, default=512)
            num_lanes = typer.prompt("Number of lanes", type=int, default=8)
            data_width = typer.prompt("Data width (bits)", type=int, default=32)
            
            template = VectorProcessor(
                vector_length=vector_length,
                num_lanes=num_lanes,
                data_width=data_width
            )
            
        elif template_type == "transformer":
            seq_length = typer.prompt("Max sequence length", type=int, default=2048)
            embedding_dim = typer.prompt("Embedding dimension", type=int, default=768)
            num_heads = typer.prompt("Number of heads", type=int, default=12)
            precision = typer.prompt("Precision", default="fp16")
            
            template = TransformerAccelerator(
                max_sequence_length=seq_length,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                precision=precision
            )
            
        elif template_type == "custom":
            template = CustomTemplate(
                name=name,
                description=f"Custom {name} accelerator"
            )
            
            # Interactive configuration
            add_ops = typer.confirm("Add custom operations?")
            if add_ops:
                while True:
                    op_name = typer.prompt("Operation name")
                    latency = typer.prompt("Latency (cycles)", type=int)
                    throughput = typer.prompt("Throughput (ops/cycle)", type=int)
                    description = typer.prompt("Description", default="")
                    
                    template.add_operation(
                        name=op_name,
                        inputs=["vector", "vector"],
                        outputs=["vector"],
                        latency=latency,
                        throughput=throughput,
                        description=description
                    )
                    
                    if not typer.confirm("Add another operation?"):
                        break
        
        else:
            console.print(f"[bold red]Unknown template type: {template_type}[/bold red]")
            raise typer.Exit(1)
        
        # Generate RTL
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating template...", total=None)
            
            rtl_path = template.generate_rtl(output_dir)
            
            progress.update(task, completed=True)
        
        console.print(f"[bold green]Template created successfully![/bold green]")
        console.print(f"RTL generated: {rtl_path}")
        
        # Show template info
        console.print(f"Template: {template}")
        
        # Generate additional files for some templates
        if hasattr(template, 'generate_compiler_support'):
            compiler_files = template.generate_compiler_support(f"{output_dir}/compiler")
            console.print("Compiler support generated:")
            for file_type, path in compiler_files.items():
                console.print(f"  • {file_type}: {path}")
        
        if hasattr(template, 'estimate_resources'):
            resources = template.estimate_resources()
            console.print(f"Estimated resources: {resources}")
        
    except Exception as e:
        console.print(f"[bold red]Template creation failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()