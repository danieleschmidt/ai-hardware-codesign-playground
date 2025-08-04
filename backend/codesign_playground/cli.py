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

from .core import AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer, Workflow

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


if __name__ == "__main__":
    main()