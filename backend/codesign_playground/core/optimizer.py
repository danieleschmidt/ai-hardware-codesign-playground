"""
Model optimization and co-design functionality.

This module provides optimization algorithms for jointly optimizing neural networks
and their corresponding hardware accelerators.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import random
import math
import time
import threading
from contextlib import contextmanager

from .accelerator import Accelerator, ModelProfile
from ..utils.monitoring import record_metric, monitor_function
from ..utils.validation import validate_inputs, validate_model, SecurityValidator
from ..utils.exceptions import OptimizationError, ValidationError
from ..utils.circuit_breaker import AdvancedCircuitBreaker, circuit_breaker
from ..utils.resilience import RetryConfig
from ..utils.health_monitoring import get_health_monitor
from ..utils.logging import get_logger
from ..utils.compliance import record_processing, DataCategory


@dataclass
class OptimizationResult:
    """Results from model-hardware co-optimization."""
    
    optimized_model: Any  # Placeholder for model object
    optimized_accelerator: Accelerator
    metrics: Dict[str, float]
    iterations: int
    convergence_history: List[Dict[str, float]]
    optimization_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "metrics": self.metrics,
            "iterations": self.iterations,
            "convergence_history": self.convergence_history,
            "optimization_time": self.optimization_time,
            "accelerator": self.optimized_accelerator.to_dict(),
        }


class ModelOptimizer:
    """Co-optimization of neural networks and hardware accelerators with enhanced robustness."""
    
    def __init__(self, model: Any, accelerator: Accelerator, user_id: Optional[str] = None):
        """
        Initialize the model-hardware co-optimizer.
        
        Args:
            model: Neural network model to optimize
            accelerator: Hardware accelerator to co-optimize
            user_id: User identifier for compliance tracking
        """
        self.model = model
        self.accelerator = accelerator
        self.user_id = user_id
        self.optimization_history = []
        
        # Robustness components
        self.logger = get_logger(__name__)
        self.health_monitor = get_health_monitor()
        self.circuit_breaker = AdvancedCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=OptimizationError
        )
        self._optimization_lock = threading.RLock()
        self._metrics_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Security validation
        self.security_validator = SecurityValidator()
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_optimization_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Record data processing for compliance
        if self.user_id:
            record_processing(
                user_id=self.user_id,
                data_category=DataCategory.MODEL_ARTIFACTS,
                purpose="model_optimization",
                legal_basis="legitimate_interests"
            )
        
        self.health_monitor.update_status("healthy", {"component": "initialized"})
        self.logger.info(f"ModelOptimizer initialized for user {user_id}")
        
    @monitor_function("model_co_optimization")
    @validate_inputs
    @circuit_breaker
    # @resilient_operation(RetryConfig(max_attempts=3, base_delay=1.0))
    def co_optimize(
        self,
        target_fps: float,
        power_budget: float,
        iterations: int = 10,
        optimization_strategy: str = "balanced",
        enable_caching: bool = True,
        timeout_seconds: int = 300
    ) -> OptimizationResult:
        """
        Jointly optimize model and hardware for target metrics.
        
        Args:
            target_fps: Target inference rate (frames per second)
            power_budget: Maximum power consumption (watts)
            iterations: Number of optimization iterations
            optimization_strategy: Strategy ("performance", "power", "balanced")
            
        Returns:
            OptimizationResult with optimized model and accelerator
        """
        # Comprehensive input validation with security checks
        self._validate_optimization_inputs(target_fps, power_budget, iterations, optimization_strategy, timeout_seconds)
        
        # Check cache first
        cache_key = self._generate_cache_key(target_fps, power_budget, iterations, optimization_strategy)
        if enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.optimization_stats["cache_hits"] += 1
                self.logger.info(f"Returning cached optimization result for key {cache_key}")
                return cached_result
            else:
                self.optimization_stats["cache_misses"] += 1
        
        # Record optimization start with comprehensive tracking
        self.optimization_stats["total_optimizations"] += 1
        optimization_id = f"opt_{int(time.time())}_{random.randint(1000, 9999)}"
        
        try:
            record_metric("co_optimization_started", 1, "counter", {
                "strategy": optimization_strategy,
                "optimization_id": optimization_id,
                "user_id": self.user_id or "anonymous"
            })
        except Exception as e:
            self.logger.warning(f"Failed to record start metric: {e}")
        
        # Thread-safe optimization with timeout
        with self._optimization_lock:
            start_time = time.time()
            
            convergence_history = []
            best_design = None
            best_score = float('-inf')
            
            # Health check before starting
            if not self.health_monitor.is_healthy():
                raise OptimizationError("System health check failed before optimization")
        
            try:
                # Optimization loop with timeout and health monitoring
                for iteration in range(iterations):
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        self.logger.warning(f"Optimization timeout after {timeout_seconds}s")
                        break
                    
                    # Health check during optimization
                    if iteration % 5 == 0:  # Check every 5 iterations
                        if not self.health_monitor.is_healthy():
                            self.logger.warning("Health check failed during optimization")
                            break
                    
                    try:
                        iteration_start = time.time()
                        
                        # Model optimization step with resilience
                        current_model = self._optimize_model_step_resilient(target_fps, power_budget, iteration)
                        
                        # Hardware optimization step with resilience
                        current_accelerator = self._optimize_hardware_step_resilient(target_fps, power_budget, iteration)
                        
                        # Evaluate combined design with error handling
                        metrics = self._evaluate_design_resilient(current_model, current_accelerator)
                        score = self._compute_objective_score(metrics, optimization_strategy)
                        
                        iteration_time = time.time() - iteration_start
                        
                        convergence_history.append({
                            "iteration": iteration,
                            "score": score,
                            "fps": metrics["fps"],
                            "power": metrics["power"],
                            "accuracy": metrics["accuracy"],
                            "iteration_time": iteration_time,
                            "timestamp": time.time()
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_design = (current_model, current_accelerator, metrics)
                            self.logger.debug(f"New best design found at iteration {iteration}: score={score:.4f}")
                        
                        # Update health status
                        self.health_monitor.update_status("healthy", {
                            "current_iteration": iteration,
                            "best_score": best_score,
                            "iteration_time": iteration_time
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Optimization iteration {iteration} failed: {e}")
                        self.health_monitor.update_status("degraded", {
                            "failed_iteration": iteration,
                            "error": str(e)
                        })
                        
                        # Record iteration failure but continue
                        try:
                            record_metric("optimization_iteration_failure", 1, "counter", {
                                "iteration": iteration,
                                "optimization_id": optimization_id
                            })
                        except:
                            pass
                        
                        # Continue with next iteration instead of failing completely
                        continue
                        
            except Exception as e:
                self.logger.error(f"Critical optimization failure: {e}")
                self.health_monitor.update_status("unhealthy", {"critical_error": str(e)})
                self.optimization_stats["failed_optimizations"] += 1
                raise OptimizationError(f"Co-optimization failed: {e}")
        
            optimization_time = time.time() - start_time
            
            # Validate results
            if best_design is None:
                self.optimization_stats["failed_optimizations"] += 1
                self.health_monitor.update_status("unhealthy", {"no_valid_design": True})
                raise OptimizationError("No valid design found during optimization")
            
            best_model, best_accelerator, best_metrics = best_design
            
            # Create optimization result
            result = OptimizationResult(
                optimized_model=best_model,
                optimized_accelerator=best_accelerator,
                metrics=best_metrics,
                iterations=len(convergence_history),  # Actual iterations completed
                convergence_history=convergence_history,
                optimization_time=optimization_time,
            )
            
            # Cache successful result
            if enable_caching:
                self._cache_result(cache_key, result)
            
            # Update statistics
            self.optimization_stats["successful_optimizations"] += 1
            self.optimization_stats["average_optimization_time"] = (
                (self.optimization_stats["average_optimization_time"] * 
                 (self.optimization_stats["successful_optimizations"] - 1) + optimization_time) /
                self.optimization_stats["successful_optimizations"]
            )
            
            # Record comprehensive completion metrics
            try:
                record_metric("co_optimization_completed", 1, "counter", {
                    "strategy": optimization_strategy,
                    "optimization_id": optimization_id,
                    "success": True
                })
                record_metric("co_optimization_time", optimization_time, "timer")
                record_metric("co_optimization_best_score", best_score, "gauge")
                record_metric("co_optimization_iterations_completed", len(convergence_history), "gauge")
            except Exception as e:
                self.logger.warning(f"Failed to record completion metrics: {e}")
            
            # Final health status update
            self.health_monitor.update_status("healthy", {
                "optimization_completed": True,
                "final_score": best_score,
                "total_time": optimization_time
            })
            
            self.logger.info(
                f"Optimization completed successfully",
                extra={
                    "optimization_id": optimization_id,
                    "strategy": optimization_strategy,
                    "final_score": best_score,
                    "optimization_time": optimization_time,
                    "iterations_completed": len(convergence_history),
                    "user_id": self.user_id
                }
            )
            
            return result
    
    def _validate_optimization_inputs(self, target_fps: float, power_budget: float, 
                                    iterations: int, strategy: str, timeout: int) -> None:
        """Comprehensive input validation with security checks."""
        # Basic type and value validation
        if not isinstance(target_fps, (int, float)) or target_fps <= 0:
            raise ValidationError("target_fps must be a positive number")
        if not isinstance(power_budget, (int, float)) or power_budget <= 0:
            raise ValidationError("power_budget must be a positive number")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValidationError("iterations must be a positive integer")
        if strategy not in ["performance", "power", "balanced"]:
            raise ValidationError("optimization_strategy must be one of: performance, power, balanced")
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValidationError("timeout_seconds must be a positive integer")
        
        # Security validation with strict bounds
        if not self.security_validator.validate_numeric_input(target_fps, "target_fps", min_value=0.1, max_value=1000):
            raise SecurityError("target_fps outside safe bounds")
        if not self.security_validator.validate_numeric_input(power_budget, "power_budget", min_value=0.1, max_value=100):
            raise SecurityError("power_budget outside safe bounds")
        if iterations > 1000:
            raise SecurityError("iterations count too high")
        if timeout > 3600:  # 1 hour max
            raise SecurityError("timeout too high")
    
    def _generate_cache_key(self, target_fps: float, power_budget: float, 
                          iterations: int, strategy: str) -> str:
        """Generate cache key for optimization parameters."""
        import hashlib
        params = f"{target_fps}_{power_budget}_{iterations}_{strategy}_{hash(str(self.model))}_{hash(str(self.accelerator))}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[OptimizationResult]:
        """Get cached optimization result if valid."""
        if cache_key not in self._metrics_cache:
            return None
        
        cached_entry = self._metrics_cache[cache_key]
        current_time = time.time()
        
        # Check if cache entry is still valid
        if current_time - cached_entry["timestamp"] > self._cache_ttl:
            del self._metrics_cache[cache_key]
            return None
        
        return cached_entry["result"]
    
    def _cache_result(self, cache_key: str, result: OptimizationResult) -> None:
        """Cache optimization result."""
        self._metrics_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Limit cache size
        if len(self._metrics_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self._metrics_cache.keys(), 
                           key=lambda k: self._metrics_cache[k]["timestamp"])
            del self._metrics_cache[oldest_key]
    
    def _optimize_model_step_resilient(self, target_fps: float, power_budget: float, iteration: int) -> Any:
        """Model optimization step with enhanced error handling."""
        try:
            return self._optimize_model_step(target_fps, power_budget)
        except Exception as e:
            self.logger.warning(f"Model optimization step {iteration} failed: {e}")
            # Return current model as fallback
            return self.model
    
    def _optimize_hardware_step_resilient(self, target_fps: float, power_budget: float, iteration: int) -> Accelerator:
        """Hardware optimization step with enhanced error handling."""
        try:
            return self._optimize_hardware_step(target_fps, power_budget)
        except Exception as e:
            self.logger.warning(f"Hardware optimization step {iteration} failed: {e}")
            # Return current accelerator as fallback
            return self.accelerator
    
    def _evaluate_design_resilient(self, model: Any, accelerator: Accelerator) -> Dict[str, float]:
        """Design evaluation with enhanced error handling."""
        try:
            return self._evaluate_design(model, accelerator)
        except Exception as e:
            self.logger.warning(f"Design evaluation failed: {e}")
            # Return conservative default metrics
            return {
                "fps": 1.0,
                "power": 10.0,
                "accuracy": 0.5,
                "latency_ms": 1000.0,
                "efficiency": 0.1,
                "area_mm2": 50.0,
            }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            **self.optimization_stats,
            "cache_size": len(self._metrics_cache),
            "health_status": self.health_monitor.get_status(),
            "circuit_breaker_status": {
                "state": self.circuit_breaker._state.name,
                "failure_count": self.circuit_breaker._failure_count,
                "last_failure_time": self.circuit_breaker._last_failure_time
            }
        }
    
    def clear_cache(self) -> None:
        """Clear optimization cache."""
        self._metrics_cache.clear()
        self.optimization_stats["cache_hits"] = 0
        self.optimization_stats["cache_misses"] = 0
        self.logger.info("Optimization cache cleared")
    
    @contextmanager
    def optimization_context(self, context_name: str):
        """Context manager for optimization operations."""
        self.logger.debug(f"Entering optimization context: {context_name}")
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in optimization context {context_name}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.logger.debug(f"Exiting optimization context: {context_name} (duration: {duration:.2f}s)")
    
    def apply_hardware_constraints(self, model: Any, constraints: Dict[str, Any]) -> Any:
        """
        Apply hardware constraints to model optimization.
        
        Args:
            model: Neural network model
            constraints: Hardware constraints (memory, compute, etc.)
            
        Returns:
            Model adapted for hardware constraints
        """
        # Mock model adaptation
        adapted_model = model  # In reality, would apply quantization, pruning, etc.
        
        # Apply quantization if needed
        if constraints.get("precision") == "int8":
            adapted_model = self._apply_quantization(adapted_model, 8)
        elif constraints.get("precision") == "fp16":
            adapted_model = self._apply_quantization(adapted_model, 16)
        
        # Apply pruning if memory constrained
        memory_limit = constraints.get("memory_limit_mb", float('inf'))
        if hasattr(adapted_model, 'size_mb') and adapted_model.size_mb > memory_limit:
            adapted_model = self._apply_pruning(adapted_model, target_size=memory_limit)
        
        return adapted_model
    
    def _optimize_model_step(self, target_fps: float, power_budget: float) -> Any:
        """Single model optimization step."""
        # Simplified model optimization
        # In practice: quantization, pruning, architecture search, etc.
        optimized_model = self.model
        
        # Mock model modifications based on constraints
        if power_budget < 3.0:
            # Aggressive optimization for low power
            optimized_model = self._apply_quantization(optimized_model, 8)
            optimized_model = self._apply_pruning(optimized_model, sparsity=0.5)
        elif target_fps > 60:
            # Optimize for high performance
            optimized_model = self._apply_quantization(optimized_model, 16)
            optimized_model = self._optimize_operations(optimized_model)
        
        return optimized_model
    
    def _optimize_hardware_step(self, target_fps: float, power_budget: float) -> Accelerator:
        """Single hardware optimization step."""
        # Create modified accelerator based on targets
        new_accelerator = Accelerator(
            compute_units=self.accelerator.compute_units,
            memory_hierarchy=self.accelerator.memory_hierarchy.copy(),
            dataflow=self.accelerator.dataflow,
            frequency_mhz=self.accelerator.frequency_mhz,
            data_width=self.accelerator.data_width,
            precision=self.accelerator.precision,
            power_budget_w=power_budget,
        )
        
        # Adjust compute units based on performance target
        if target_fps > 30:
            scale_factor = min(2.0, target_fps / 30)
            new_accelerator.compute_units = int(new_accelerator.compute_units * scale_factor)
        
        # Adjust frequency based on power budget
        if power_budget < 3.0:
            new_accelerator.frequency_mhz = min(100, new_accelerator.frequency_mhz)
        elif power_budget > 8.0:
            new_accelerator.frequency_mhz = min(400, new_accelerator.frequency_mhz * 1.5)
        
        # Update performance estimates
        new_accelerator.estimate_performance()
        
        return new_accelerator
    
    def _evaluate_design(self, model: Any, accelerator: Accelerator) -> Dict[str, float]:
        """Evaluate combined model-hardware design."""
        # Mock evaluation - in practice would run actual simulation
        perf = accelerator.performance_model or accelerator.estimate_performance()
        
        # Simulate performance based on model and hardware characteristics
        base_fps = perf["throughput_ops_s"] / 1e6  # Convert to rough FPS estimate
        power = perf["power_w"]
        
        # Model complexity affects performance
        model_complexity = getattr(model, 'complexity', 1.0)
        actual_fps = base_fps / model_complexity
        
        # Simulate accuracy (would be measured in practice)
        base_accuracy = 0.95
        if hasattr(model, 'quantization_bits') and model.quantization_bits < 16:
            accuracy_penalty = (16 - model.quantization_bits) * 0.01
            accuracy = base_accuracy - accuracy_penalty
        else:
            accuracy = base_accuracy
        
        return {
            "fps": actual_fps,
            "power": power,
            "accuracy": accuracy,
            "latency_ms": perf["latency_ms"],
            "efficiency": actual_fps / power if power > 0 else 0,
            "area_mm2": perf["area_mm2"],
        }
    
    def _compute_objective_score(
        self, 
        metrics: Dict[str, float], 
        strategy: str
    ) -> float:
        """Compute optimization objective score."""
        fps = metrics["fps"]
        power = metrics["power"]
        accuracy = metrics["accuracy"]
        
        if strategy == "performance":
            # Maximize FPS while maintaining accuracy
            return fps * accuracy
        elif strategy == "power":
            # Maximize efficiency (FPS/power) while maintaining accuracy
            efficiency = fps / power if power > 0 else 0
            return efficiency * accuracy
        else:  # balanced
            # Balanced objective
            normalized_fps = fps / 60.0  # Normalize to 60 FPS
            normalized_power = 5.0 / power if power > 0 else 0  # Normalize to 5W budget
            return (normalized_fps + normalized_power + accuracy) / 3.0
    
    def _apply_quantization(self, model: Any, bits: int) -> Any:
        """Apply quantization to model."""
        # Mock quantization
        class QuantizedModel:
            def __init__(self, original_model, bits):
                self.original_model = original_model
                self.quantization_bits = bits
                self.complexity = getattr(original_model, 'complexity', 1.0) * (16 / bits)
        
        return QuantizedModel(model, bits)
    
    def _apply_pruning(self, model: Any, sparsity: float = 0.5, target_size: Optional[float] = None) -> Any:
        """Apply pruning to model."""
        # Mock pruning
        class PrunedModel:
            def __init__(self, original_model, sparsity):
                self.original_model = original_model
                self.sparsity = sparsity
                self.complexity = getattr(original_model, 'complexity', 1.0) * (1 - sparsity * 0.5)
                if hasattr(original_model, 'size_mb'):
                    self.size_mb = original_model.size_mb * (1 - sparsity)
        
        return PrunedModel(model, sparsity)
    
    def _optimize_operations(self, model: Any) -> Any:
        """Apply operation-level optimizations."""
        # Mock operation optimization (fusion, etc.)
        class OptimizedModel:
            def __init__(self, original_model):
                self.original_model = original_model
                self.complexity = getattr(original_model, 'complexity', 1.0) * 0.8  # 20% improvement
        
        return OptimizedModel(model)


class HardwareAwareTraining:
    """Hardware-aware neural network training."""
    
    def __init__(self, model: Any, hardware: Dict[str, float]):
        """
        Initialize hardware-aware training.
        
        Args:
            model: Neural network model
            hardware: Hardware constraints and characteristics
        """
        self.model = model
        self.hardware_constraints = hardware
        self.training_history = []
    
    def compile(
        self,
        optimizer: str = "adam",
        loss: Union[str, List[str]] = "accuracy",
        loss_weights: Optional[List[float]] = None
    ) -> None:
        """
        Compile model with hardware-aware loss functions.
        
        Args:
            optimizer: Optimization algorithm
            loss: Loss function(s) to use
            loss_weights: Weights for multiple loss functions
        """
        self.optimizer = optimizer
        self.loss_functions = loss if isinstance(loss, list) else [loss]
        self.loss_weights = loss_weights or [1.0] * len(self.loss_functions)
    
    def fit(
        self,
        train_data: Any,
        epochs: int = 50,
        callbacks: Optional[List[str]] = None
    ) -> Any:
        """
        Train model with hardware awareness.
        
        Args:
            train_data: Training dataset
            epochs: Number of training epochs
            callbacks: Training callbacks
            
        Returns:
            Trained model
        """
        callbacks = callbacks or []
        
        # Mock training process
        trained_model = self.model
        
        for epoch in range(epochs):
            # Simulate training metrics
            accuracy = 0.5 + (epoch / epochs) * 0.4 + random.gauss(0, 0.02)
            hardware_efficiency = self._compute_hardware_efficiency(trained_model)
            
            # Apply hardware-aware callbacks
            if "layer_pruning" in callbacks and epoch % 10 == 0:
                trained_model = self._apply_structured_pruning(trained_model)
            
            if "quantization_aware" in callbacks and epoch > epochs // 2:
                trained_model = self._apply_quantization_aware_training(trained_model)
            
            self.training_history.append({
                "epoch": epoch,
                "accuracy": accuracy,
                "hardware_efficiency": hardware_efficiency,
            })
        
        return trained_model
    
    def _compute_hardware_efficiency(self, model: Any) -> float:
        """Compute hardware efficiency metric."""
        # Mock hardware efficiency calculation
        compute_roof = self.hardware_constraints.get("compute_roof", 100)
        memory_bandwidth = self.hardware_constraints.get("memory_bandwidth", 25.6)
        
        # Simple efficiency model
        model_ops = getattr(model, 'operations', 50)  # GOPS
        efficiency = min(1.0, compute_roof / model_ops) * 0.8 + random.gauss(0, 0.05)
        return max(0.0, min(1.0, efficiency))
    
    def _apply_structured_pruning(self, model: Any) -> Any:
        """Apply structured pruning during training."""
        # Mock structured pruning
        class StructurallyPrunedModel:
            def __init__(self, original_model):
                self.original_model = original_model
                self.operations = getattr(original_model, 'operations', 50) * 0.9
        
        return StructurallyPrunedModel(model)
    
    def _apply_quantization_aware_training(self, model: Any) -> Any:
        """Apply quantization-aware training."""
        # Mock QAT
        class QATModel:
            def __init__(self, original_model):
                self.original_model = original_model
                self.quantization_ready = True
        
        return QATModel(model)