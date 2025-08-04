"""
FastAPI server for AI Hardware Co-Design Playground.

This module provides the web API interface for the codesign playground,
enabling remote access to design tools and workflow orchestration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timedelta
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from .core import AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer, Workflow
from .core.accelerator import ModelProfile, Accelerator
from .core.optimizer import OptimizationResult
from .core.explorer import DesignSpaceResult
from .core.workflow import WorkflowMetrics
from .utils.monitoring import get_system_monitor, record_metric, get_health_status, monitor_function
from .utils.validation import SecurityValidator, validate_inputs
from .utils.logging import get_logger
from .utils.exceptions import ValidationError, SecurityError

logger = get_logger(__name__)

# Rate limiting configuration
rate_limit_store: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"requests": [], "blocked_until": None})
MAX_REQUESTS_PER_MINUTE = 60
MAX_REQUESTS_PER_HOUR = 1000
BLOCK_DURATION_MINUTES = 15

# Security validator
security_validator = SecurityValidator()
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    monitor = get_system_monitor()
    logger.info("FastAPI server starting up")
    record_metric("server_startup", 1, "counter")
    yield
    # Shutdown
    logger.info("FastAPI server shutting down")
    record_metric("server_shutdown", 1, "counter")

# FastAPI app configuration with enhanced security
app = FastAPI(
    title="AI Hardware Co-Design Playground API",
    description="Interactive environment for co-optimizing neural networks and hardware accelerators",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# CORS middleware with stricter configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=600,
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware to prevent abuse."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests
    cutoff_time = current_time - 3600  # 1 hour ago
    rate_limit_store[client_ip]["requests"] = [
        req_time for req_time in rate_limit_store[client_ip]["requests"] 
        if req_time > cutoff_time
    ]
    
    # Check if client is blocked
    if (rate_limit_store[client_ip]["blocked_until"] and 
        current_time < rate_limit_store[client_ip]["blocked_until"]):
        record_metric("rate_limit_blocked", 1, "counter", {"client_ip": client_ip})
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": "900"}  # 15 minutes
        )
    
    # Check rate limits
    recent_requests = rate_limit_store[client_ip]["requests"]
    minute_ago = current_time - 60
    requests_last_minute = len([req for req in recent_requests if req > minute_ago])
    requests_last_hour = len(recent_requests)
    
    if requests_last_minute > MAX_REQUESTS_PER_MINUTE or requests_last_hour > MAX_REQUESTS_PER_HOUR:
        # Block client
        rate_limit_store[client_ip]["blocked_until"] = current_time + (BLOCK_DURATION_MINUTES * 60)
        record_metric("rate_limit_exceeded", 1, "counter", {"client_ip": client_ip})
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Client blocked."},
            headers={"Retry-After": str(BLOCK_DURATION_MINUTES * 60)}
        )
    
    # Record request
    rate_limit_store[client_ip]["requests"].append(current_time)
    
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit-Minute"] = str(MAX_REQUESTS_PER_MINUTE)
    response.headers["X-RateLimit-Limit-Hour"] = str(MAX_REQUESTS_PER_HOUR)
    response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, MAX_REQUESTS_PER_MINUTE - requests_last_minute))
    response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, MAX_REQUESTS_PER_HOUR - requests_last_hour))
    
    record_metric("api_request", 1, "counter", {
        "method": request.method,
        "endpoint": str(request.url.path),
        "status_code": str(response.status_code)
    })
    
    return response

# Global state management with size limits
MAX_ACTIVE_WORKFLOWS = 100
MAX_JOB_RESULTS = 1000

active_workflows: Dict[str, Workflow] = {}
job_results: Dict[str, Any] = {}

# State cleanup functions
def cleanup_old_workflows():
    """Clean up old workflows to prevent memory leaks."""
    if len(active_workflows) > MAX_ACTIVE_WORKFLOWS:
        # Remove oldest workflows
        sorted_workflows = sorted(
            active_workflows.items(),
            key=lambda x: x[1].state.start_time
        )
        for name, _ in sorted_workflows[:len(active_workflows) - MAX_ACTIVE_WORKFLOWS]:
            del active_workflows[name]
            record_metric("workflow_cleanup", 1, "counter")

def cleanup_old_jobs():
    """Clean up old job results to prevent memory leaks."""
    if len(job_results) > MAX_JOB_RESULTS:
        # Remove oldest jobs
        sorted_jobs = sorted(
            job_results.items(),
            key=lambda x: x[1].created_at
        )
        for job_id, _ in sorted_jobs[:len(job_results) - MAX_JOB_RESULTS]:
            del job_results[job_id]
            record_metric("job_cleanup", 1, "counter")


# Pydantic models for API requests/responses
class ModelProfileRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file", max_length=500)
    input_shape: List[int] = Field(..., description="Input tensor shape")
    framework: str = Field("auto", description="ML framework")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not security_validator.validate_file_path(v):
            raise ValueError('Invalid model path')
        return v
    
    @validator('input_shape')
    def validate_input_shape(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Input shape cannot be empty')
        if len(v) > 10:  # Reasonable limit
            raise ValueError('Input shape too complex')
        if any(dim <= 0 or dim > 10000 for dim in v):
            raise ValueError('Invalid input shape dimensions')
        return v
    
    @validator('framework')
    def validate_framework(cls, v):
        valid_frameworks = ['auto', 'pytorch', 'tensorflow', 'onnx']
        if v not in valid_frameworks:
            raise ValueError(f'Framework must be one of: {valid_frameworks}')
        return v


class AcceleratorDesignRequest(BaseModel):
    compute_units: int = Field(64, description="Number of compute units", ge=1, le=10000)
    memory_hierarchy: List[str] = Field(["sram_64kb", "dram"], description="Memory hierarchy")
    dataflow: str = Field("weight_stationary", description="Dataflow pattern")
    frequency_mhz: float = Field(200.0, description="Operating frequency", ge=1.0, le=2000.0)
    precision: str = Field("int8", description="Numerical precision")
    power_budget_w: float = Field(5.0, description="Power budget in watts", ge=0.1, le=1000.0)
    
    @validator('memory_hierarchy')
    def validate_memory_hierarchy(cls, v):
        valid_memories = ['sram_8kb', 'sram_16kb', 'sram_32kb', 'sram_64kb', 'sram_128kb', 'dram', 'hbm']
        if not all(mem in valid_memories for mem in v):
            raise ValueError(f'Invalid memory type. Valid options: {valid_memories}')
        if len(v) > 5:  # Reasonable hierarchy depth
            raise ValueError('Memory hierarchy too complex')
        return v
    
    @validator('dataflow')
    def validate_dataflow(cls, v):
        valid_dataflows = ['weight_stationary', 'output_stationary', 'row_stationary']
        if v not in valid_dataflows:
            raise ValueError(f'Dataflow must be one of: {valid_dataflows}')
        return v
    
    @validator('precision')
    def validate_precision(cls, v):
        valid_precisions = ['int8', 'int16', 'fp16', 'fp32']
        if v not in valid_precisions:
            raise ValueError(f'Precision must be one of: {valid_precisions}')
        return v


class OptimizationRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file", max_length=500)
    accelerator_config: AcceleratorDesignRequest = Field(..., description="Accelerator configuration")
    target_fps: float = Field(30.0, description="Target inference rate", ge=0.1, le=1000.0)
    power_budget: float = Field(5.0, description="Power budget in watts", ge=0.1, le=1000.0)
    iterations: int = Field(10, description="Optimization iterations", ge=1, le=100)
    strategy: str = Field("balanced", description="Optimization strategy")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not security_validator.validate_file_path(v):
            raise ValueError('Invalid model path')
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['performance', 'power', 'balanced']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v


class DesignSpaceRequest(BaseModel):
    model_path: str = Field(..., description="Path to model file", max_length=500)
    input_shape: List[int] = Field(..., description="Input tensor shape")
    design_space: Dict[str, List[Any]] = Field(..., description="Design space parameters")
    objectives: List[str] = Field(["latency", "power", "area"], description="Optimization objectives")
    num_samples: int = Field(100, description="Number of design points", ge=1, le=1000)
    strategy: str = Field("random", description="Exploration strategy")
    constraints: Optional[Dict[str, float]] = Field(None, description="Design constraints")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not security_validator.validate_file_path(v):
            raise ValueError('Invalid model path')
        return v
    
    @validator('input_shape')
    def validate_input_shape(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Input shape cannot be empty')
        if len(v) > 10:
            raise ValueError('Input shape too complex')
        if any(dim <= 0 or dim > 10000 for dim in v):
            raise ValueError('Invalid input shape dimensions')
        return v
    
    @validator('design_space')
    def validate_design_space(cls, v):
        if not v:
            raise ValueError('Design space cannot be empty')
        if len(v) > 20:  # Reasonable parameter limit
            raise ValueError('Design space too complex')
        return v
    
    @validator('objectives')
    def validate_objectives(cls, v):
        valid_objectives = ['latency', 'power', 'area', 'throughput', 'efficiency']
        if not all(obj in valid_objectives for obj in v):
            raise ValueError(f'Invalid objective. Valid options: {valid_objectives}')
        if len(v) > 5:
            raise ValueError('Too many objectives')
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['random', 'grid', 'evolutionary']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v


class WorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow identifier", max_length=100)
    model_path: str = Field(..., description="Path to model file", max_length=500)
    input_shapes: Dict[str, List[int]] = Field(..., description="Input tensor shapes")
    hardware_template: str = Field("systolic_array", description="Hardware template")
    hardware_size: List[int] = Field([16, 16], description="Hardware dimensions")
    precision: str = Field("int8", description="Numerical precision")
    optimizer: str = Field("tvm", description="Compiler framework")
    optimizations: List[str] = Field(["layer_fusion", "tensorization"], description="Optimization passes")
    
    @validator('name')
    def validate_name(cls, v):
        if not security_validator.validate_identifier(v):
            raise ValueError('Invalid workflow name')
        return v
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not security_validator.validate_file_path(v):
            raise ValueError('Invalid model path')
        return v
    
    @validator('input_shapes')
    def validate_input_shapes(cls, v):
        if not v:
            raise ValueError('Input shapes cannot be empty')
        if len(v) > 10:  # Multiple inputs limit
            raise ValueError('Too many input shapes')
        for name, shape in v.items():
            if not security_validator.validate_identifier(name):
                raise ValueError(f'Invalid input name: {name}')
            if not shape or len(shape) == 0:
                raise ValueError(f'Invalid shape for input {name}')
            if any(dim <= 0 or dim > 10000 for dim in shape):
                raise ValueError(f'Invalid dimensions for input {name}')
        return v
    
    @validator('hardware_template')
    def validate_hardware_template(cls, v):
        valid_templates = ['systolic_array', 'vector_processor', 'transformer_accelerator']
        if v not in valid_templates:
            raise ValueError(f'Hardware template must be one of: {valid_templates}')
        return v
    
    @validator('hardware_size')
    def validate_hardware_size(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Hardware size cannot be empty')
        if len(v) > 3:  # 3D max
            raise ValueError('Hardware size too complex')
        if any(dim <= 0 or dim > 1000 for dim in v):
            raise ValueError('Invalid hardware dimensions')
        return v
    
    @validator('precision')
    def validate_precision(cls, v):
        valid_precisions = ['int8', 'int16', 'fp16', 'fp32']
        if v not in valid_precisions:
            raise ValueError(f'Precision must be one of: {valid_precisions}')
        return v
    
    @validator('optimizer')
    def validate_optimizer(cls, v):
        valid_optimizers = ['tvm', 'mlir', 'custom']
        if v not in valid_optimizers:
            raise ValueError(f'Optimizer must be one of: {valid_optimizers}')
        return v
    
    @validator('optimizations')
    def validate_optimizations(cls, v):
        valid_opts = ['layer_fusion', 'tensorization', 'quantization', 'pruning', 'operator_fusion']
        if not all(opt in valid_opts for opt in v):
            raise ValueError(f'Invalid optimization. Valid options: {valid_opts}')
        if len(v) > 10:  # Reasonable limit
            raise ValueError('Too many optimizations')
        return v


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Any] = None
    created_at: datetime
    updated_at: datetime


# Enhanced health check endpoint
@app.get("/health")
@monitor_function("health_check")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = get_health_status()
        system_monitor = get_system_monitor()
        monitoring_summary = system_monitor.get_monitoring_summary()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_health": health_status,
            "system_metrics": {
                "cpu_percent": monitoring_summary["system_metrics"]["cpu_percent"],
                "memory_percent": monitoring_summary["system_metrics"]["memory_percent"],
                "uptime_seconds": monitoring_summary["system_metrics"]["uptime_seconds"]
            },
            "monitoring_active": monitoring_summary["monitoring"]["monitoring_active"]
        }
    except Exception as e:
        logger.error("Health check failed", exception=e)
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": "Health check partially failed"
        }


# Model profiling endpoints
@app.post("/api/v1/model/profile", response_model=Dict[str, Any])
@monitor_function("api_model_profile")
async def profile_model(request: ModelProfileRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Profile a neural network model to analyze computational requirements."""
    try:
        # Enhanced security validation
        if not security_validator.validate_request_size(request.dict()):
            record_metric("api_security_violation", 1, "counter", {"type": "request_size"})
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        record_metric("model_profile_request", 1, "counter", {"framework": request.framework})
        
        designer = AcceleratorDesigner()
        
        # Mock model for profiling with validation
        mock_model = {
            "path": request.model_path, 
            "framework": request.framework,
            "validated": True
        }
        
        profile = designer.profile_model(mock_model, tuple(request.input_shape))
        
        record_metric("model_profile_success", 1, "counter")
        
        return {
            "status": "success",
            "profile": profile.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }
        
    except ValidationError as e:
        record_metric("model_profile_validation_error", 1, "counter")
        logger.warning(f"Model profiling validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except SecurityError as e:
        record_metric("model_profile_security_error", 1, "counter")
        logger.error(f"Model profiling security error: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security validation failed"
        )
    except Exception as e:
        record_metric("model_profile_error", 1, "counter")
        logger.error(f"Model profiling failed: {e}", exception=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model profiling failed. Please try again."
        )


# Accelerator design endpoints
@app.post("/api/v1/accelerator/design", response_model=Dict[str, Any])
@monitor_function("api_accelerator_design")
async def design_accelerator(request: AcceleratorDesignRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Design a hardware accelerator with specified parameters."""
    try:
        # Enhanced validation
        if not security_validator.validate_request_size(request.dict()):
            record_metric("api_security_violation", 1, "counter", {"type": "request_size"})
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        record_metric("accelerator_design_request", 1, "counter", {
            "dataflow": request.dataflow,
            "precision": request.precision
        })
        
        designer = AcceleratorDesigner()
        
        accelerator = designer.design(
            compute_units=request.compute_units,
            memory_hierarchy=request.memory_hierarchy,
            dataflow=request.dataflow,
            frequency_mhz=request.frequency_mhz,
            precision=request.precision,
            power_budget_w=request.power_budget_w
        )
        
        # Generate performance estimates with error handling
        try:
            performance = accelerator.estimate_performance()
        except Exception as perf_error:
            logger.warning(f"Performance estimation failed: {perf_error}")
            performance = {"error": "Performance estimation unavailable"}
        
        record_metric("accelerator_design_success", 1, "counter")
        record_metric("accelerator_compute_units", request.compute_units, "gauge")
        record_metric("accelerator_power_budget", request.power_budget_w, "gauge")
        
        return {
            "status": "success",
            "accelerator": accelerator.to_dict(),
            "performance": performance,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }
        
    except ValidationError as e:
        record_metric("accelerator_design_validation_error", 1, "counter")
        logger.warning(f"Accelerator design validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except SecurityError as e:
        record_metric("accelerator_design_security_error", 1, "counter")
        logger.error(f"Accelerator design security error: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security validation failed"
        )
    except Exception as e:
        record_metric("accelerator_design_error", 1, "counter")
        logger.error(f"Accelerator design failed: {e}", exception=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Accelerator design failed. Please try again."
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
@monitor_function("api_file_download")
async def download_file(file_path: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Download generated files (RTL, reports, etc.) with security validation."""
    try:
        # Security validation for file path
        if not security_validator.validate_file_path(file_path):
            record_metric("file_download_security_violation", 1, "counter")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Invalid file path"
            )
        
        file_full_path = Path(file_path).resolve()
        
        # Additional security: ensure file is within allowed directories
        allowed_dirs = [Path("/tmp"), Path("./workflows"), Path("./rtl")]
        if not any(str(file_full_path).startswith(str(allowed_dir.resolve())) for allowed_dir in allowed_dirs):
            record_metric("file_download_path_violation", 1, "counter")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: File outside allowed directories"
            )
        
        if not file_full_path.exists():
            record_metric("file_download_not_found", 1, "counter")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found"
            )
        
        # Check file size limit (100MB)
        if file_full_path.stat().st_size > 100 * 1024 * 1024:
            record_metric("file_download_too_large", 1, "counter")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large for download"
            )
        
        record_metric("file_download_success", 1, "counter", {"file_type": file_full_path.suffix})
        
        return FileResponse(
            path=file_full_path,
            filename=file_full_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        record_metric("file_download_error", 1, "counter")
        logger.error(f"File download failed: {e}", exception=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File download failed"
        )


# System information endpoints
@app.get("/api/v1/system/info")
@monitor_function("api_system_info")
async def get_system_info():
    """Get system information and capabilities."""
    try:
        system_monitor = get_system_monitor()
        health_status = get_health_status()
        
        return {
            "version": "0.1.0",
            "status": health_status["overall_status"],
            "supported_frameworks": ["pytorch", "tensorflow", "onnx"],
            "hardware_templates": ["systolic_array", "vector_processor", "transformer_accelerator"],
            "dataflow_options": ["weight_stationary", "output_stationary", "row_stationary"],
            "precision_options": ["int8", "int16", "fp16", "fp32"],
            "optimization_strategies": ["performance", "power", "balanced"],
            "exploration_strategies": ["random", "grid", "evolutionary"],
            "limits": {
                "max_compute_units": 10000,
                "max_frequency_mhz": 2000.0,
                "max_power_budget_w": 1000.0,
                "max_design_samples": 1000,
                "max_optimization_iterations": 100,
                "rate_limits": {
                    "requests_per_minute": MAX_REQUESTS_PER_MINUTE,
                    "requests_per_hour": MAX_REQUESTS_PER_HOUR
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System info failed: {e}", exception=e)
        return {
            "version": "0.1.0",
            "status": "unknown",
            "error": "Partial system info unavailable",
            "timestamp": datetime.now().isoformat()
        }

# Enhanced monitoring endpoint
@app.get("/api/v1/system/metrics")
@monitor_function("api_system_metrics")
async def get_system_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get detailed system metrics (admin only)."""
    try:
        system_monitor = get_system_monitor()
        monitoring_data = system_monitor.export_monitoring_data("json", include_history=False)
        
        record_metric("system_metrics_request", 1, "counter")
        
        return {
            "status": "success",
            "metrics": json.loads(monitoring_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System metrics failed: {e}", exception=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System metrics unavailable"
        )


def main():
    """Main server entry point with enhanced configuration."""
    import uvicorn
    
    # Configure server with security settings
    config = uvicorn.Config(
        "codesign_playground.server:app",
        host="127.0.0.1",  # More secure default
        port=8000,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=True,
        server_header=False,  # Hide server header
        date_header=False,    # Hide date header
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting AI Hardware Co-Design Playground API server")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exception=e)
        raise


if __name__ == "__main__":
    main()