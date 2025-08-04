"""
FastAPI server for AI Hardware Co-Design Playground.

This module provides the web API interface for the codesign playground,
enabling remote access to design tools and workflow orchestration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
from pathlib import Path
import tempfile
import uuid
from datetime import datetime

from .core import AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer, Workflow
from .core.accelerator import ModelProfile, Accelerator
from .core.optimizer import OptimizationResult
from .core.explorer import DesignSpaceResult
from .core.workflow import WorkflowMetrics

# FastAPI app configuration
app = FastAPI(
    title="AI Hardware Co-Design Playground API",
    description="Interactive environment for co-optimizing neural networks and hardware accelerators",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
active_workflows: Dict[str, Workflow] = {}
job_results: Dict[str, Any] = {}


# Pydantic models for API requests/responses
class ModelProfileRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    framework: str = Field("auto", description="ML framework")


class AcceleratorDesignRequest(BaseModel):
    compute_units: int = Field(64, description="Number of compute units")
    memory_hierarchy: List[str] = Field(["sram_64kb", "dram"], description="Memory hierarchy")
    dataflow: str = Field("weight_stationary", description="Dataflow pattern")
    frequency_mhz: float = Field(200.0, description="Operating frequency")
    precision: str = Field("int8", description="Numerical precision")
    power_budget_w: float = Field(5.0, description="Power budget in watts")


class OptimizationRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file")
    accelerator_config: AcceleratorDesignRequest = Field(..., description="Accelerator configuration")
    target_fps: float = Field(30.0, description="Target inference rate")
    power_budget: float = Field(5.0, description="Power budget in watts")
    iterations: int = Field(10, description="Optimization iterations")
    strategy: str = Field("balanced", description="Optimization strategy")


class DesignSpaceRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    design_space: Dict[str, List[Any]] = Field(..., description="Design space parameters")
    objectives: List[str] = Field(["latency", "power", "area"], description="Optimization objectives")
    num_samples: int = Field(100, description="Number of design points")
    strategy: str = Field("random", description="Exploration strategy")
    constraints: Optional[Dict[str, float]] = Field(None, description="Design constraints")


class WorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow identifier")
    model_path: str = Field(..., description="Path to model file")
    input_shapes: Dict[str, List[int]] = Field(..., description="Input tensor shapes")
    hardware_template: str = Field("systolic_array", description="Hardware template")
    hardware_size: List[int] = Field([16, 16], description="Hardware dimensions")
    precision: str = Field("int8", description="Numerical precision")
    optimizer: str = Field("tvm", description="Compiler framework")
    optimizations: List[str] = Field(["layer_fusion", "tensorization"], description="Optimization passes")


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Any] = None
    created_at: datetime
    updated_at: datetime


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Model profiling endpoints
@app.post("/api/v1/model/profile", response_model=Dict[str, Any])
async def profile_model(request: ModelProfileRequest):
    """Profile a neural network model to analyze computational requirements."""
    try:
        designer = AcceleratorDesigner()
        
        # Mock model for profiling
        mock_model = {"path": request.model_path, "framework": request.framework}
        
        profile = designer.profile_model(mock_model, tuple(request.input_shape))
        
        return {
            "status": "success",
            "profile": profile.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model profiling failed: {str(e)}"
        )


# Accelerator design endpoints
@app.post("/api/v1/accelerator/design", response_model=Dict[str, Any])
async def design_accelerator(request: AcceleratorDesignRequest):
    """Design a hardware accelerator with specified parameters."""
    try:
        designer = AcceleratorDesigner()
        
        accelerator = designer.design(
            compute_units=request.compute_units,
            memory_hierarchy=request.memory_hierarchy,
            dataflow=request.dataflow,
            frequency_mhz=request.frequency_mhz,
            precision=request.precision,
            power_budget_w=request.power_budget_w
        )
        
        # Generate performance estimates
        performance = accelerator.estimate_performance()
        
        return {
            "status": "success",
            "accelerator": accelerator.to_dict(),
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Accelerator design failed: {str(e)}"
        )


@app.post("/api/v1/accelerator/rtl", response_model=Dict[str, Any])
async def generate_rtl(request: AcceleratorDesignRequest):
    """Generate RTL code for accelerator design."""
    try:
        designer = AcceleratorDesigner()
        
        accelerator = designer.design(
            compute_units=request.compute_units,
            memory_hierarchy=request.memory_hierarchy,
            dataflow=request.dataflow,
            frequency_mhz=request.frequency_mhz,
            precision=request.precision,
            power_budget_w=request.power_budget_w
        )
        
        # Generate RTL to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            accelerator.generate_rtl(f.name)
            rtl_path = f.name
        
        return {
            "status": "success",
            "rtl_file": rtl_path,
            "accelerator": accelerator.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RTL generation failed: {str(e)}"
        )


# Optimization endpoints
@app.post("/api/v1/optimization/co-optimize")
async def start_co_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start model-hardware co-optimization job."""
    job_id = str(uuid.uuid4())
    
    # Add background task
    background_tasks.add_task(
        run_co_optimization,
        job_id,
        request
    )
    
    return {
        "status": "started",
        "job_id": job_id,
        "message": "Co-optimization job started",
        "timestamp": datetime.now().isoformat()
    }


async def run_co_optimization(job_id: str, request: OptimizationRequest):
    """Background task for co-optimization."""
    try:
        # Update job status
        job_results[job_id] = JobStatus(
            job_id=job_id,
            status="running",
            progress=0.0,
            message="Starting co-optimization",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Create accelerator from request
        designer = AcceleratorDesigner()
        accelerator = designer.design(
            compute_units=request.accelerator_config.compute_units,
            memory_hierarchy=request.accelerator_config.memory_hierarchy,
            dataflow=request.accelerator_config.dataflow,
            frequency_mhz=request.accelerator_config.frequency_mhz,
            precision=request.accelerator_config.precision,
            power_budget_w=request.accelerator_config.power_budget_w
        )
        
        # Mock model
        mock_model = {"path": request.model_path, "type": "optimization"}
        
        # Create optimizer
        optimizer = ModelOptimizer(mock_model, accelerator)
        
        # Update progress
        job_results[job_id].status = "optimizing"
        job_results[job_id].progress = 0.3
        job_results[job_id].message = "Running co-optimization"
        job_results[job_id].updated_at = datetime.now()
        
        # Run optimization
        result = optimizer.co_optimize(
            target_fps=request.target_fps,
            power_budget=request.power_budget,
            iterations=request.iterations,
            optimization_strategy=request.strategy
        )
        
        # Complete job
        job_results[job_id].status = "completed"
        job_results[job_id].progress = 1.0
        job_results[job_id].message = "Co-optimization completed"
        job_results[job_id].result = result.to_dict()
        job_results[job_id].updated_at = datetime.now()
        
    except Exception as e:
        job_results[job_id].status = "failed"
        job_results[job_id].message = f"Co-optimization failed: {str(e)}"
        job_results[job_id].updated_at = datetime.now()


# Design space exploration endpoints
@app.post("/api/v1/exploration/design-space")
async def start_design_space_exploration(request: DesignSpaceRequest, background_tasks: BackgroundTasks):
    """Start design space exploration job."""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        run_design_space_exploration,
        job_id,
        request
    )
    
    return {
        "status": "started",
        "job_id": job_id,
        "message": "Design space exploration started",
        "timestamp": datetime.now().isoformat()
    }


async def run_design_space_exploration(job_id: str, request: DesignSpaceRequest):
    """Background task for design space exploration."""
    try:
        job_results[job_id] = JobStatus(
            job_id=job_id,
            status="running",
            progress=0.0,
            message="Starting design space exploration",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Create explorer
        explorer = DesignSpaceExplorer()
        
        # Mock model
        mock_model = {"path": request.model_path, "input_shape": request.input_shape}
        
        # Update progress
        job_results[job_id].progress = 0.2
        job_results[job_id].message = "Exploring design space"
        job_results[job_id].updated_at = datetime.now()
        
        # Run exploration
        results = explorer.explore(
            model=mock_model,
            design_space=request.design_space,
            objectives=request.objectives,
            num_samples=request.num_samples,
            strategy=request.strategy,
            constraints=request.constraints
        )
        
        # Complete job
        job_results[job_id].status = "completed"
        job_results[job_id].progress = 1.0
        job_results[job_id].message = f"Explored {results.total_evaluations} design points"
        job_results[job_id].result = results.to_dict()
        job_results[job_id].updated_at = datetime.now()
        
    except Exception as e:
        job_results[job_id].status = "failed"
        job_results[job_id].message = f"Design space exploration failed: {str(e)}"
        job_results[job_id].updated_at = datetime.now()


# Workflow endpoints
@app.post("/api/v1/workflow/create")
async def create_workflow(request: WorkflowRequest):
    """Create a new end-to-end workflow."""
    try:
        workflow = Workflow(request.name)
        active_workflows[request.name] = workflow
        
        return {
            "status": "created",
            "workflow_id": request.name,
            "message": f"Workflow '{request.name}' created",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow creation failed: {str(e)}"
        )


@app.post("/api/v1/workflow/{workflow_id}/run")
async def run_workflow(workflow_id: str, request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Run complete workflow from model to RTL."""
    if workflow_id not in active_workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )
    
    background_tasks.add_task(
        execute_workflow,
        workflow_id,
        request
    )
    
    return {
        "status": "started",
        "workflow_id": workflow_id,
        "message": "Workflow execution started",
        "timestamp": datetime.now().isoformat()
    }


async def execute_workflow(workflow_id: str, request: WorkflowRequest):
    """Background task for workflow execution."""
    try:
        workflow = active_workflows[workflow_id]
        
        # Step 1: Import model
        workflow.import_model(
            request.model_path,
            {k: tuple(v) for k, v in request.input_shapes.items()}
        )
        
        # Step 2: Map to hardware
        workflow.map_to_hardware(
            template=request.hardware_template,
            size=tuple(request.hardware_size),
            precision=request.precision
        )
        
        # Step 3: Compile
        workflow.compile(
            optimizer=request.optimizer,
            target="custom_accelerator",
            optimizations=request.optimizations
        )
        
        # Step 4: Simulate
        metrics = workflow.simulate("mock_testbench")
        
        # Step 5: Generate RTL
        workflow.generate_rtl(include_testbench=True)
        
    except Exception as e:
        # Handle workflow errors
        workflow.state.stage = "failed"
        workflow.state.add_message(f"Workflow failed: {str(e)}")


@app.get("/api/v1/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow execution status."""
    if workflow_id not in active_workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )
    
    workflow = active_workflows[workflow_id]
    return workflow.get_status()


# Job status endpoints
@app.get("/api/v1/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of background job."""
    if job_id not in job_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found"
        )
    
    job = job_results[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat()
    }


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all jobs and their status."""
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat()
            }
            for job in job_results.values()
        ]
    }


# File download endpoints
@app.get("/api/v1/download/{file_path:path}")
async def download_file(file_path: str):
    """Download generated files (RTL, reports, etc.)."""
    file_full_path = Path(file_path)
    
    if not file_full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_path}' not found"
        )
    
    return FileResponse(
        path=file_full_path,
        filename=file_full_path.name,
        media_type='application/octet-stream'
    )


# System information endpoints
@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information and capabilities."""
    return {
        "version": "0.1.0",
        "supported_frameworks": ["pytorch", "tensorflow", "onnx"],
        "hardware_templates": ["systolic_array", "vector_processor", "transformer_accelerator"],
        "dataflow_options": ["weight_stationary", "output_stationary", "row_stationary"],
        "precision_options": ["int8", "fp16", "fp32"],
        "optimization_strategies": ["performance", "power", "balanced"],
        "exploration_strategies": ["random", "grid", "evolutionary"],
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Main server entry point."""
    import uvicorn
    uvicorn.run(
        "codesign_playground.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()