"""
Background worker for AI Hardware Co-Design Playground.

This module provides Celery-based background task processing for compute-intensive
operations like design space exploration and model optimization.
"""

# Optional dependency with fallback
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    Celery = None
    CELERY_AVAILABLE = False
from typing import Dict, List, Any, Optional
import json
import time
from pathlib import Path

from .core import AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer, Workflow

# Celery configuration
if CELERY_AVAILABLE:
    celery_app = Celery(
        "codesign_playground_worker",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0",
        include=["codesign_playground.worker"]
    )
else:
    celery_app = None

# Celery settings
if CELERY_AVAILABLE and celery_app:
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


@celery_app.task(bind=True)
def profile_model_task(self, model_path: str, input_shape: List[int], framework: str = "auto"):
    """
    Background task for model profiling.
    
    Args:
        model_path: Path to model file
        input_shape: Input tensor shape
        framework: ML framework
        
    Returns:
        Model profile dictionary
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Initializing profiler"})
        
        designer = AcceleratorDesigner()
        mock_model = {"path": model_path, "framework": framework}
        
        self.update_state(state="PROGRESS", meta={"progress": 0.5, "status": "Analyzing model"})
        
        profile = designer.profile_model(mock_model, tuple(input_shape))
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "Profiling completed"})
        
        return {
            "status": "success",
            "profile": profile.to_dict(),
            "model_path": model_path,
            "input_shape": input_shape,
            "framework": framework
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Model profiling failed"}
        )
        raise


@celery_app.task(bind=True)
def design_accelerator_task(
    self,
    compute_units: int = 64,
    memory_hierarchy: List[str] = None,
    dataflow: str = "weight_stationary",
    frequency_mhz: float = 200.0,
    precision: str = "int8",
    power_budget_w: float = 5.0
):
    """
    Background task for accelerator design.
    
    Args:
        compute_units: Number of compute units
        memory_hierarchy: Memory system configuration
        dataflow: Data movement pattern
        frequency_mhz: Operating frequency
        precision: Numerical precision
        power_budget_w: Power budget
        
    Returns:
        Accelerator design dictionary
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Initializing designer"})
        
        if memory_hierarchy is None:
            memory_hierarchy = ["sram_64kb", "dram"]
        
        designer = AcceleratorDesigner()
        
        self.update_state(state="PROGRESS", meta={"progress": 0.5, "status": "Designing accelerator"})
        
        accelerator = designer.design(
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dataflow=dataflow,
            frequency_mhz=frequency_mhz,
            precision=precision,
            power_budget_w=power_budget_w
        )
        
        self.update_state(state="PROGRESS", meta={"progress": 0.8, "status": "Estimating performance"})
        
        performance = accelerator.estimate_performance()
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "Design completed"})
        
        return {
            "status": "success",
            "accelerator": accelerator.to_dict(),
            "performance": performance
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Accelerator design failed"}
        )
        raise


@celery_app.task(bind=True)
def co_optimize_task(
    self,
    model_path: str,
    accelerator_config: Dict[str, Any],
    target_fps: float = 30.0,
    power_budget: float = 5.0,
    iterations: int = 10,
    strategy: str = "balanced"
):
    """
    Background task for model-hardware co-optimization.
    
    Args:
        model_path: Path to model file
        accelerator_config: Accelerator configuration
        target_fps: Target inference rate
        power_budget: Power budget
        iterations: Number of optimization iterations
        strategy: Optimization strategy
        
    Returns:
        Optimization results dictionary
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Setting up optimization"})
        
        # Create accelerator from config
        designer = AcceleratorDesigner()
        accelerator = designer.design(**accelerator_config)
        
        # Mock model
        mock_model = {"path": model_path, "type": "optimization"}
        
        # Create optimizer
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        self.update_state(state="PROGRESS", meta={"progress": 0.3, "status": "Starting co-optimization"})
        
        # Run optimization with progress updates
        result = optimizer.co_optimize(
            target_fps=target_fps,
            power_budget=power_budget,
            iterations=iterations,
            optimization_strategy=strategy
        )
        
        # Update progress during optimization
        for i in range(iterations):
            progress = 0.3 + (0.6 * (i + 1) / iterations)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": f"Optimization iteration {i+1}/{iterations}"
                }
            )
            time.sleep(0.1)  # Simulate work
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "Co-optimization completed"})
        
        return {
            "status": "success",
            "result": result.to_dict(),
            "model_path": model_path,
            "target_fps": target_fps,
            "power_budget": power_budget,
            "iterations": iterations,
            "strategy": strategy
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Co-optimization failed"}
        )
        raise


@celery_app.task(bind=True)
def explore_design_space_task(
    self,
    model_path: str,
    input_shape: List[int],
    design_space: Dict[str, List[Any]],
    objectives: List[str] = None,
    num_samples: int = 100,
    strategy: str = "random",
    constraints: Optional[Dict[str, float]] = None
):
    """
    Background task for design space exploration.
    
    Args:
        model_path: Path to model file
        input_shape: Input tensor shape
        design_space: Design space parameters
        objectives: Optimization objectives
        num_samples: Number of design points
        strategy: Exploration strategy
        constraints: Design constraints
        
    Returns:
        Design space exploration results
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Initializing exploration"})
        
        if objectives is None:
            objectives = ["latency", "power", "area"]
        
        # Create explorer
        explorer = DesignSpaceExplorer()
        
        # Mock model
        mock_model = {"path": model_path, "input_shape": input_shape}
        
        self.update_state(state="PROGRESS", meta={"progress": 0.2, "status": f"Exploring {num_samples} design points"})
        
        # Run exploration with progress tracking
        results = explorer.explore(
            model=mock_model,
            design_space=design_space,
            objectives=objectives,
            num_samples=num_samples,
            strategy=strategy,
            constraints=constraints
        )
        
        # Simulate progress updates during exploration
        for i in range(10):
            progress = 0.2 + (0.7 * (i + 1) / 10)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": f"Evaluating design points ({i*10}%)"
                }
            )
            time.sleep(0.1)
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "Exploration completed"})
        
        return {
            "status": "success",
            "results": results.to_dict(),
            "model_path": model_path,
            "input_shape": input_shape,
            "num_samples": num_samples,
            "strategy": strategy,
            "pareto_points": len(results.pareto_frontier),
            "exploration_time": results.exploration_time
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Design space exploration failed"}
        )
        raise


@celery_app.task(bind=True)
def run_workflow_task(
    self,
    workflow_name: str,
    model_path: str,
    input_shapes: Dict[str, List[int]],
    hardware_template: str = "systolic_array",
    hardware_size: List[int] = None,
    precision: str = "int8",
    optimizer: str = "tvm",
    optimizations: List[str] = None,
    output_dir: Optional[str] = None
):
    """
    Background task for complete workflow execution.
    
    Args:
        workflow_name: Workflow identifier
        model_path: Path to model file
        input_shapes: Input tensor shapes
        hardware_template: Hardware template
        hardware_size: Hardware dimensions
        precision: Numerical precision
        optimizer: Compiler framework
        optimizations: Optimization passes
        output_dir: Output directory
        
    Returns:
        Workflow execution results
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Initializing workflow"})
        
        if hardware_size is None:
            hardware_size = [16, 16]
        if optimizations is None:
            optimizations = ["layer_fusion", "tensorization"]
        
        # Create workflow
        workflow = Workflow(workflow_name, output_dir)
        
        # Step 1: Import model
        self.update_state(state="PROGRESS", meta={"progress": 0.2, "status": "Importing model"})
        workflow.import_model(
            model_path,
            {k: tuple(v) for k, v in input_shapes.items()}
        )
        
        # Step 2: Map to hardware
        self.update_state(state="PROGRESS", meta={"progress": 0.4, "status": "Mapping to hardware"})
        workflow.map_to_hardware(
            template=hardware_template,
            size=tuple(hardware_size),
            precision=precision
        )
        
        # Step 3: Compile
        self.update_state(state="PROGRESS", meta={"progress": 0.6, "status": "Compiling model"})
        workflow.compile(
            optimizer=optimizer,
            target="custom_accelerator",
            optimizations=optimizations
        )
        
        # Step 4: Simulate
        self.update_state(state="PROGRESS", meta={"progress": 0.8, "status": "Running simulation"})
        metrics = workflow.simulate("mock_testbench")
        
        # Step 5: Generate RTL
        self.update_state(state="PROGRESS", meta={"progress": 0.9, "status": "Generating RTL"})
        workflow.generate_rtl(include_testbench=True)
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "Workflow completed"})
        
        return {
            "status": "success",
            "workflow_name": workflow_name,
            "metrics": metrics.to_dict(),
            "workflow_status": workflow.get_status(),
            "output_directory": str(workflow.output_dir)
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Workflow execution failed"}
        )
        raise


@celery_app.task(bind=True)
def generate_rtl_task(
    self,
    accelerator_config: Dict[str, Any],
    output_path: str,
    include_testbench: bool = True
):
    """
    Background task for RTL generation.
    
    Args:
        accelerator_config: Accelerator configuration
        output_path: Output file path
        include_testbench: Whether to include testbench
        
    Returns:
        RTL generation results
    """
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "status": "Setting up RTL generation"})
        
        # Create accelerator from config
        designer = AcceleratorDesigner()
        accelerator = designer.design(**accelerator_config)
        
        self.update_state(state="PROGRESS", meta={"progress": 0.5, "status": "Generating RTL code"})
        
        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate RTL
        accelerator.generate_rtl(str(output_file))
        
        if include_testbench:
            self.update_state(state="PROGRESS", meta={"progress": 0.8, "status": "Generating testbench"})
            # Testbench would be generated here
        
        self.update_state(state="PROGRESS", meta={"progress": 1.0, "status": "RTL generation completed"})
        
        return {
            "status": "success",
            "output_path": str(output_file),
            "accelerator_config": accelerator_config,
            "include_testbench": include_testbench,
            "rtl_size_bytes": output_file.stat().st_size if output_file.exists() else 0
        }
        
    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "RTL generation failed"}
        )
        raise


# Task monitoring and management
@celery_app.task
def health_check():
    """Health check task for worker monitoring."""
    return {
        "status": "healthy",
        "worker": "codesign_playground_worker",
        "timestamp": time.time()
    }


@celery_app.task
def cleanup_old_files(max_age_hours: int = 24):
    """Cleanup old temporary files and results."""
    import tempfile
    import shutil
    from datetime import datetime, timedelta
    
    try:
        temp_dir = Path(tempfile.gettempdir())
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        cleaned_files = 0
        for file_path in temp_dir.glob("codesign_*"):
            if file_path.stat().st_mtime < cutoff_time.timestamp():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                cleaned_files += 1
        
        return {
            "status": "success",
            "cleaned_files": cleaned_files,
            "max_age_hours": max_age_hours
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    """Main worker entry point."""
    celery_app.start()


if __name__ == "__main__":
    main()