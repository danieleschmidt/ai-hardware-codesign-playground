"""
Performance optimization and monitoring for AI Hardware Co-Design Playground.

This module provides performance profiling, optimization hints, and
adaptive scaling mechanisms for the platform.
"""

import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..utils.logging import get_logger, get_performance_logger
from ..utils.monitoring import get_system_monitor, record_metric, monitor_function
from .cache import cached, get_thread_pool

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_used_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (operations per second)."""
        return 1.0 / self.duration if self.duration > 0 else 0.0


class PerformanceProfiler:
    """System performance profiler and monitor."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance profiler.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self._metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
        self._active_operations = {}
        
        # System monitoring
        self._system_metrics = deque(maxlen=window_size)
        self._monitor_thread = None
        self._monitoring = False
        
        logger.info("Initialized PerformanceProfiler", window_size=window_size)
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        Start system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Started system monitoring", interval=interval)
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped system monitoring")
    
    def start_operation(self, operation_name: str, **metadata) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{time.time()}_{id(threading.current_thread())}"
        
        start_data = {
            "operation_name": operation_name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": psutil.cpu_percent(),
            "metadata": metadata
        }
        
        with self._lock:
            self._active_operations[operation_id] = start_data
        
        # Record operation start in monitoring system
        record_metric(f"operation_{operation_name}_started", 1, "counter")
        
        logger.debug(
            "Started operation tracking",
            operation_name=operation_name,
            operation_id=operation_id,
            **metadata
        )
        
        return operation_id
    
    def end_operation(
        self, 
        operation_id: str, 
        success: bool = True, 
        error_message: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        End tracking an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            error_message: Error message if failed
            
        Returns:
            Performance metrics for the operation
        """
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning("Unknown operation ID", operation_id=operation_id)
                return None
            
            start_data = self._active_operations.pop(operation_id)
        
        # Calculate metrics
        duration = end_time - start_data["start_time"]
        memory_used = end_memory - start_data["start_memory"]
        avg_cpu = (start_data["start_cpu"] + end_cpu) / 2.0
        
        metrics = PerformanceMetrics(
            operation_name=start_data["operation_name"],
            start_time=start_data["start_time"],
            end_time=end_time,
            duration=duration,
            memory_used_mb=memory_used,
            cpu_percent=avg_cpu,
            success=success,
            error_message=error_message,
            metadata=start_data["metadata"]
        )
        
        # Store metrics
        with self._lock:
            self._metrics_history[start_data["operation_name"]].append(metrics)
        
        # Record operation completion in monitoring system
        status = "success" if success else "error"
        record_metric(f"operation_{start_data['operation_name']}_completed", 1, "counter", {"status": status})
        record_metric(f"operation_{start_data['operation_name']}_duration", duration, "timer")
        record_metric(f"operation_{start_data['operation_name']}_memory", memory_used, "gauge")
        
        logger.info(
            "Completed operation tracking",
            operation_name=start_data["operation_name"],
            duration=duration,
            memory_used_mb=memory_used,
            success=success
        )
        
        return metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if operation_name not in self._metrics_history:
                return {"error": "No data for operation"}
            
            metrics_list = list(self._metrics_history[operation_name])
        
        if not metrics_list:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        durations = [m.duration for m in metrics_list]
        memory_usage = [m.memory_used_mb for m in metrics_list]
        cpu_usage = [m.cpu_percent for m in metrics_list]
        success_count = sum(1 for m in metrics_list if m.success)
        
        return {
            "operation_name": operation_name,
            "total_calls": len(metrics_list),
            "success_rate": success_count / len(metrics_list),
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0.0
            },
            "memory_stats": {
                "mean_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage),
                "min_mb": min(memory_usage)
            },
            "cpu_stats": {
                "mean_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage)
            },
            "throughput_ops_per_sec": 1.0 / statistics.mean(durations) if durations else 0.0
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        with self._lock:
            if not self._system_metrics:
                return {"error": "No system metrics available"}
            
            recent_metrics = list(self._system_metrics)[-10:]  # Last 10 readings
        
        cpu_values = [m["cpu_percent"] for m in recent_metrics]
        memory_values = [m["memory_percent"] for m in recent_metrics]
        
        return {
            "current_cpu_percent": psutil.cpu_percent(),
            "current_memory_percent": psutil.virtual_memory().percent,
            "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else 0.0,
            "avg_memory_percent": statistics.mean(memory_values) if memory_values else 0.0,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "active_operations": len(self._active_operations)
        }
    
    def get_performance_recommendations(self) -> List[Dict[str, str]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        system_stats = self.get_system_stats()
        
        # CPU recommendations
        if system_stats.get("avg_cpu_percent", 0) > 80:
            recommendations.append({
                "type": "cpu",
                "severity": "high",
                "message": "High CPU usage detected. Consider reducing concurrent operations or optimizing algorithms.",
                "suggestion": "Enable caching or reduce parallel processing threads"
            })
        
        # Memory recommendations
        if system_stats.get("avg_memory_percent", 0) > 85:
            recommendations.append({
                "type": "memory",
                "severity": "high", 
                "message": "High memory usage detected. Consider clearing caches or reducing batch sizes.",
                "suggestion": "Clear unused caches or implement memory-efficient processing"
            })
        
        # Operation-specific recommendations
        with self._lock:
            for operation_name in self._metrics_history:
                stats = self.get_operation_stats(operation_name)
                
                if stats.get("success_rate", 1.0) < 0.95:
                    recommendations.append({
                        "type": "reliability",
                        "severity": "medium",
                        "message": f"Operation '{operation_name}' has low success rate ({stats['success_rate']:.1%})",
                        "suggestion": "Review error handling and input validation"
                    })
                
                if stats.get("duration_stats", {}).get("mean", 0) > 30.0:
                    recommendations.append({
                        "type": "performance",
                        "severity": "medium",
                        "message": f"Operation '{operation_name}' is slow (avg: {stats['duration_stats']['mean']:.1f}s)",
                        "suggestion": "Consider caching, optimization, or parallel processing"
                    })
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _monitor_system(self, interval: float) -> None:
        """Background system monitoring thread."""
        while self._monitoring:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                
                with self._lock:
                    self._system_metrics.append(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error("System monitoring error", exception=e)
                time.sleep(interval)


class AdaptiveOptimizer:
    """Adaptive optimization system that learns from performance patterns."""
    
    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize adaptive optimizer.
        
        Args:
            profiler: Performance profiler instance
        """
        self.profiler = profiler
        self._optimization_rules = {}
        self._adaptation_history = []
        
        # Load default optimization rules
        self._load_default_rules()
        
        logger.info("Initialized AdaptiveOptimizer")
    
    def optimize_operation(
        self, 
        operation_name: str, 
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with adaptive optimizations.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Operation result
        """
        # Get optimization strategy
        strategy = self._get_optimization_strategy(operation_name)
        
        # Apply optimizations based on strategy
        if strategy.get("use_caching", False):
            # Apply caching if beneficial
            cached_func = self._apply_caching(operation_func, operation_name)
            operation_func = cached_func
        
        if strategy.get("use_parallelization", False):
            # Apply parallelization if beneficial
            return self._execute_parallel(operation_name, operation_func, *args, **kwargs)
        
        # Execute with profiling
        op_id = self.profiler.start_operation(operation_name, strategy=strategy)
        
        try:
            result = operation_func(*args, **kwargs)
            self.profiler.end_operation(op_id, success=True)
            
            # Learn from this execution
            self._update_optimization_strategy(operation_name, True)
            
            return result
            
        except Exception as e:
            self.profiler.end_operation(op_id, success=False, error_message=str(e))
            self._update_optimization_strategy(operation_name, False)
            raise
    
    def _get_optimization_strategy(self, operation_name: str) -> Dict[str, Any]:
        """Get optimization strategy for operation."""
        if operation_name not in self._optimization_rules:
            self._optimization_rules[operation_name] = {
                "use_caching": False,
                "use_parallelization": False,
                "batch_size": 1,
                "confidence": 0.0
            }
        
        return self._optimization_rules[operation_name].copy()
    
    def _update_optimization_strategy(self, operation_name: str, success: bool) -> None:
        """Update optimization strategy based on results."""
        stats = self.profiler.get_operation_stats(operation_name)
        
        if "error" in stats:
            return
        
        strategy = self._optimization_rules[operation_name]
        
        # Increase confidence if successful
        if success:
            strategy["confidence"] = min(1.0, strategy["confidence"] + 0.1)
        else:
            strategy["confidence"] = max(0.0, strategy["confidence"] - 0.2)
        
        # Adapt based on performance patterns
        avg_duration = stats["duration_stats"]["mean"]
        success_rate = stats["success_rate"]
        
        # Enable caching for slow, successful operations
        if avg_duration > 5.0 and success_rate > 0.9 and not strategy["use_caching"]:
            strategy["use_caching"] = True
            logger.info(
                "Enabled caching for operation",
                operation_name=operation_name,
                avg_duration=avg_duration
            )
        
        # Enable parallelization for CPU-intensive operations
        cpu_usage = stats.get("cpu_stats", {}).get("mean_percent", 0)
        if cpu_usage > 50 and avg_duration > 10.0 and not strategy["use_parallelization"]:
            strategy["use_parallelization"] = True
            logger.info(
                "Enabled parallelization for operation",
                operation_name=operation_name,
                cpu_usage=cpu_usage
            )
        
        self._adaptation_history.append({
            "timestamp": time.time(),
            "operation_name": operation_name,
            "strategy": strategy.copy(),
            "stats": stats
        })
    
    def _apply_caching(self, operation_func: Callable, operation_name: str) -> Callable:
        """Apply caching to operation function."""
        # Use intelligent cache type selection
        if "model" in operation_name.lower():
            cache_type = "model"
        elif "accelerator" in operation_name.lower():
            cache_type = "accelerator"
        elif "exploration" in operation_name.lower():
            cache_type = "exploration"
        elif "optimization" in operation_name.lower():
            cache_type = "optimization"
        else:
            cache_type = "default"
        
        return cached(cache_type=cache_type, ttl=3600.0)(operation_func)
    
    def _execute_parallel(
        self, 
        operation_name: str, 
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with parallelization."""
        # Simple parallel execution for batch operations
        thread_pool = get_thread_pool()
        
        # Check if args contain batch data
        batch_data = None
        for arg in args:
            if isinstance(arg, (list, tuple)) and len(arg) > 1:
                batch_data = arg
                break
        
        if batch_data and len(batch_data) > 2:
            # Split batch and execute in parallel
            batch_size = max(1, len(batch_data) // 4)  # Use 4 threads
            batches = [
                batch_data[i:i + batch_size] 
                for i in range(0, len(batch_data), batch_size)
            ]
            
            futures = []
            for batch in batches:
                # Replace batch data in args
                parallel_args = list(args)
                for i, arg in enumerate(parallel_args):
                    if arg is batch_data:
                        parallel_args[i] = batch
                        break
                
                future = thread_pool.submit(operation_func, *parallel_args, **kwargs)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, (list, tuple)):
                        results.extend(result)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        "Parallel execution error",
                        operation_name=operation_name,
                        exception=e
                    )
                    raise
            
            return results
        
        # Fall back to sequential execution
        return operation_func(*args, **kwargs)
    
    def _load_default_rules(self) -> None:
        """Load default optimization rules."""
        default_rules = {
            "model_profiling": {
                "use_caching": True,
                "use_parallelization": False,
                "batch_size": 1,
                "confidence": 0.8
            },
            "accelerator_design": {
                "use_caching": True,
                "use_parallelization": False,
                "batch_size": 1,
                "confidence": 0.8
            },
            "design_space_exploration": {
                "use_caching": True,
                "use_parallelization": True,
                "batch_size": 4,
                "confidence": 0.9
            },
            "optimization": {
                "use_caching": False,  # Optimization results vary
                "use_parallelization": False,
                "batch_size": 1,
                "confidence": 0.5
            }
        }
        
        self._optimization_rules.update(default_rules)
        logger.info("Loaded default optimization rules")


# Global performance system
_profiler = None
_optimizer = None


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _profiler
    
    if _profiler is None:
        _profiler = PerformanceProfiler()
        _profiler.start_monitoring()
    
    return _profiler


def get_optimizer() -> AdaptiveOptimizer:
    """Get global adaptive optimizer."""
    global _optimizer
    
    if _optimizer is None:
        profiler = get_profiler()
        _optimizer = AdaptiveOptimizer(profiler)
    
    return _optimizer


def profile_operation(operation_name: str):
    """
    Decorator for profiling operations.
    
    Args:
        operation_name: Name of the operation to profile
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            op_id = profiler.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                profiler.end_operation(op_id, success=True)
                return result
            except Exception as e:
                profiler.end_operation(op_id, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator


def optimize_operation(operation_name: str):
    """
    Decorator for adaptive operation optimization.
    
    Args:
        operation_name: Name of the operation to optimize
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            return optimizer.optimize_operation(operation_name, func, *args, **kwargs)
        
        return wrapper
    return decorator