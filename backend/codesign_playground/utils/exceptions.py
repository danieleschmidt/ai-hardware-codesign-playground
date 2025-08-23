"""
Custom exceptions for AI Hardware Co-Design Playground.

This module defines the exception hierarchy for the platform,
providing specific error types for different components.
"""

from typing import Optional, Dict, Any


class CodesignError(Exception):
    """Base exception for all codesign platform errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(CodesignError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.constraints = constraints or {}
        
        self.details.update({
            "field": field,
            "value": str(value) if value is not None else None,
            "constraints": constraints,
        })


class ModelError(CodesignError):
    """Raised when model-related operations fail."""
    
    def __init__(
        self, 
        message: str, 
        model_path: Optional[str] = None,
        framework: Optional[str] = None,
        operation: Optional[str] = None
    ):
        super().__init__(message, "MODEL_ERROR")
        self.model_path = model_path
        self.framework = framework
        self.operation = operation
        
        self.details.update({
            "model_path": model_path,
            "framework": framework,
            "operation": operation,
        })


class HardwareError(CodesignError):
    """Raised when hardware design or simulation fails."""
    
    def __init__(
        self, 
        message: str, 
        accelerator_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None
    ):
        super().__init__(message, "HARDWARE_ERROR")
        self.accelerator_type = accelerator_type
        self.config = config
        self.stage = stage
        
        self.details.update({
            "accelerator_type": accelerator_type,
            "config": config,
            "stage": stage,
        })


class PerformanceError(CodesignError):
    """Performance optimization related errors."""
    pass


class ScalingError(CodesignError):
    """Auto-scaling related errors."""
    pass


class OptimizationError(CodesignError):
    """Raised when optimization processes fail."""
    
    def __init__(
        self, 
        message: str, 
        strategy: Optional[str] = None,
        iteration: Optional[int] = None,
        objective: Optional[str] = None
    ):
        super().__init__(message, "OPTIMIZATION_ERROR")
        self.strategy = strategy
        self.iteration = iteration
        self.objective = objective
        
        self.details.update({
            "strategy": strategy,
            "iteration": iteration,
            "objective": objective,
        })


class WorkflowError(CodesignError):
    """Raised when workflow execution fails."""
    
    def __init__(
        self, 
        message: str, 
        workflow_id: Optional[str] = None,
        stage: Optional[str] = None,
        step: Optional[str] = None
    ):
        super().__init__(message, "WORKFLOW_ERROR")
        self.workflow_id = workflow_id
        self.stage = stage
        self.step = step
        
        self.details.update({
            "workflow_id": workflow_id,
            "stage": stage,
            "step": step,
        })


class SecurityError(CodesignError):
    """Raised when security violations are detected."""
    
    def __init__(
        self, 
        message: str, 
        violation_type: Optional[str] = None,
        resource: Optional[str] = None
    ):
        super().__init__(message, "SECURITY_ERROR")
        self.violation_type = violation_type
        self.resource = resource
        
        self.details.update({
            "violation_type": violation_type,
            "resource": resource,
        })


class ResourceError(CodesignError):
    """Raised when resource constraints are violated."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        limit: Optional[float] = None,
        usage: Optional[float] = None
    ):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.limit = limit
        self.usage = usage
        
        self.details.update({
            "resource_type": resource_type,
            "limit": limit,
            "usage": usage,
        })


class ConfigurationError(CodesignError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.config_file = config_file
        
        self.details.update({
            "config_key": config_key,
            "config_file": config_file,
        })


class TimeoutError(CodesignError):
    """Raised when operations exceed time limits."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        
        self.details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
        })


class ExternalServiceError(CodesignError):
    """Raised when external services fail."""
    
    def __init__(
        self, 
        message: str, 
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[str] = None
    ):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR")
        self.service = service
        self.status_code = status_code
        self.response = response
        
        self.details.update({
            "service": service,
            "status_code": status_code,
            "response": response,
        })


class MonitoringError(CodesignError):
    """Raised when monitoring system fails."""
    
    def __init__(
        self, 
        message: str, 
        metric_name: Optional[str] = None,
        component: Optional[str] = None
    ):
        super().__init__(message, "MONITORING_ERROR")
        self.metric_name = metric_name
        self.component = component
        
        self.details.update({
            "metric_name": metric_name,
            "component": component,
        })


class SystemHealthError(CodesignError):
    """Raised when system health checks fail."""
    
    def __init__(
        self, 
        message: str, 
        subsystem: Optional[str] = None,
        severity: str = "warning"
    ):
        super().__init__(message, "SYSTEM_HEALTH_ERROR")
        self.subsystem = subsystem
        self.severity = severity
        
        self.details.update({
            "subsystem": subsystem,
            "severity": severity,
        })


class HardwareModelingError(CodesignError):
    """Raised when hardware modeling operations fail."""
    pass