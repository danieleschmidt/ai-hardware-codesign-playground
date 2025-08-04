"""
Auto-scaling and load balancing for AI Hardware Co-Design Playground.

This module provides intelligent scaling mechanisms, load balancing,
and resource management for high-performance operation.
"""

import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue

from ..utils.logging import get_logger, get_performance_logger
from .performance import get_profiler, PerformanceProfiler

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)


class ScalingMode(Enum):
    """Scaling modes for the system."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    queue_size: int
    active_tasks: int
    response_time_ms: float
    error_rate: float
    throughput_ops_per_sec: float


@dataclass 
class ScalingAction:
    """Scaling action to be taken."""
    
    action_type: str  # "scale_up", "scale_down", "no_action"
    target_workers: int
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class LoadBalancer:
    """Intelligent load balancer for distributing tasks."""
    
    def __init__(self, initial_workers: int = 4):
        """
        Initialize load balancer.
        
        Args:
            initial_workers: Initial number of worker threads
        """
        self.initial_workers = initial_workers
        self.current_workers = initial_workers
        
        # Worker management
        self._thread_pool = ThreadPoolExecutor(max_workers=initial_workers)
        self._process_pool = None  # Created on demand
        
        # Task queues
        self._task_queue = queue.PriorityQueue()
        self._results = {}
        
        # Load balancing state
        self._worker_stats = {}
        self._lock = threading.Lock()
        
        # Metrics collection
        self._metrics_history = deque(maxlen=100)
        self._active_tasks = 0
        
        logger.info("Initialized LoadBalancer", initial_workers=initial_workers)
    
    def submit_task(
        self,
        task_func: Callable,
        priority: int = 5,
        use_process: bool = False,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Submit task for execution.
        
        Args:
            task_func: Function to execute
            priority: Task priority (lower = higher priority)
            use_process: Whether to use process pool
            timeout: Task timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        task_id = f"task_{time.time()}_{id(task_func)}"
        
        task_data = {
            "task_id": task_id,
            "task_func": task_func,
            "args": args,
            "kwargs": kwargs,
            "use_process": use_process,
            "timeout": timeout,
            "submitted_at": time.time(),
            "priority": priority
        }
        
        # Add to queue
        self._task_queue.put((priority, time.time(), task_data))
        
        # Execute task
        if use_process:
            future = self._submit_to_process_pool(task_data)
        else:
            future = self._submit_to_thread_pool(task_data)
        
        self._results[task_id] = {
            "future": future,
            "submitted_at": time.time(),
            "status": "submitted"
        }
        
        with self._lock:
            self._active_tasks += 1
        
        logger.debug(
            "Submitted task",
            task_id=task_id,
            priority=priority,
            use_process=use_process
        )
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result.
        
        Args:
            task_id: Task ID from submit_task
            timeout: Wait timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            KeyError: If task ID not found
            TimeoutError: If timeout exceeded
        """
        if task_id not in self._results:
            raise KeyError(f"Task {task_id} not found")
        
        result_data = self._results[task_id]
        future = result_data["future"]
        
        try:
            result = future.result(timeout=timeout)
            
            with self._lock:
                self._active_tasks -= 1
            
            # Update result status
            result_data["status"] = "completed"
            result_data["completed_at"] = time.time()
            result_data["result"] = result
            
            # Record metrics
            duration = result_data["completed_at"] - result_data["submitted_at"]
            self._record_task_completion(task_id, duration, success=True)
            
            return result
            
        except Exception as e:
            with self._lock:
                self._active_tasks -= 1
            
            result_data["status"] = "failed"
            result_data["error"] = str(e)
            
            # Record failure
            duration = time.time() - result_data["submitted_at"]
            self._record_task_completion(task_id, duration, success=False)
            
            raise
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            queue_size = self._task_queue.qsize()
            active_tasks = self._active_tasks
        
        # Calculate recent metrics
        recent_metrics = list(self._metrics_history)[-10:]  # Last 10
        
        if recent_metrics:
            avg_response_time = statistics.mean([m["duration"] for m in recent_metrics])
            error_rate = sum(1 for m in recent_metrics if not m["success"]) / len(recent_metrics)
            throughput = len(recent_metrics) / 60.0  # Last minute approximation
        else:
            avg_response_time = 0.0
            error_rate = 0.0
            throughput = 0.0
        
        return {
            "current_workers": self.current_workers,
            "queue_size": queue_size,
            "active_tasks": active_tasks,
            "avg_response_time_ms": avg_response_time * 1000,
            "error_rate": error_rate,
            "throughput_ops_per_sec": throughput,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    
    def _submit_to_thread_pool(self, task_data: Dict[str, Any]) -> Any:
        """Submit task to thread pool."""
        return self._thread_pool.submit(
            self._execute_task,
            task_data["task_func"],
            task_data["args"],
            task_data["kwargs"]
        )
    
    def _submit_to_process_pool(self, task_data: Dict[str, Any]) -> Any:
        """Submit task to process pool."""
        if self._process_pool is None:
            # Create process pool on demand
            max_workers = min(8, psutil.cpu_count() or 1)
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
            logger.info("Created process pool", max_workers=max_workers)
        
        return self._process_pool.submit(
            self._execute_task,
            task_data["task_func"],
            task_data["args"],
            task_data["kwargs"]
        )
    
    def _execute_task(self, task_func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute task with error handling."""
        try:
            return task_func(*args, **kwargs)
        except Exception as e:
            logger.error("Task execution failed", exception=e)
            raise
    
    def _record_task_completion(self, task_id: str, duration: float, success: bool) -> None:
        """Record task completion metrics."""
        metrics = {
            "task_id": task_id,
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        }
        
        self._metrics_history.append(metrics)
        
        perf_logger.record_metric(
            "task_completion_time",
            duration,
            task_id=task_id,
            success=success
        )


class AutoScaler:
    """Automatic scaling system based on performance metrics."""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        mode: ScalingMode = ScalingMode.BALANCED,
        min_workers: int = 2,
        max_workers: int = 16,
        check_interval: float = 30.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            load_balancer: Load balancer to scale
            mode: Scaling mode
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            check_interval: Scaling check interval in seconds
        """
        self.load_balancer = load_balancer
        self.mode = mode
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.check_interval = check_interval
        
        # Scaling state
        self._scaling_enabled = False
        self._scaling_thread = None
        self._metrics_history = deque(maxlen=20)
        self._last_scaling_action = None
        self._scaling_lock = threading.Lock()
        
        # Scaling thresholds based on mode
        self._thresholds = self._get_scaling_thresholds(mode)
        
        logger.info(
            "Initialized AutoScaler",
            mode=mode.value,
            min_workers=min_workers,
            max_workers=max_workers,
            check_interval=check_interval
        )
    
    def start_scaling(self) -> None:
        """Start automatic scaling."""
        if self._scaling_enabled:
            return
        
        self._scaling_enabled = True
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self._scaling_thread.start()
        
        logger.info("Started automatic scaling")
    
    def stop_scaling(self) -> None:
        """Stop automatic scaling."""
        self._scaling_enabled = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
        
        logger.info("Stopped automatic scaling")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self._scaling_lock:
            recent_metrics = list(self._metrics_history)[-5:]  # Last 5 checks
        
        if recent_metrics:
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            avg_queue_size = statistics.mean([m.queue_size for m in recent_metrics])
            avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
        else:
            avg_cpu = avg_memory = avg_queue_size = avg_response_time = 0.0
        
        return {
            "scaling_enabled": self._scaling_enabled,
            "current_workers": self.load_balancer.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "mode": self.mode.value,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_queue_size": avg_queue_size,
            "avg_response_time_ms": avg_response_time,
            "last_scaling_action": self._last_scaling_action.__dict__ if self._last_scaling_action else None,
            "thresholds": self._thresholds
        }
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self._scaling_enabled:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()
                
                with self._scaling_lock:
                    self._metrics_history.append(metrics)
                
                # Make scaling decision
                action = self._make_scaling_decision(metrics)
                
                if action.action_type != "no_action":
                    self._execute_scaling_action(action)
                    self._last_scaling_action = action
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("Scaling loop error", exception=e)
                time.sleep(self.check_interval)
    
    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect current scaling metrics."""
        load_stats = self.load_balancer.get_load_stats()
        
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_percent=load_stats["cpu_percent"],
            memory_percent=load_stats["memory_percent"],
            queue_size=load_stats["queue_size"],
            active_tasks=load_stats["active_tasks"],
            response_time_ms=load_stats["avg_response_time_ms"],
            error_rate=load_stats["error_rate"],
            throughput_ops_per_sec=load_stats["throughput_ops_per_sec"]
        )
    
    def _make_scaling_decision(self, current_metrics: ScalingMetrics) -> ScalingAction:
        """Make scaling decision based on metrics."""
        current_workers = self.load_balancer.current_workers
        
        # Check if we need to scale up
        scale_up_score = self._calculate_scale_up_score(current_metrics)
        scale_down_score = self._calculate_scale_down_score(current_metrics)
        
        # Prevent rapid scaling changes
        if (self._last_scaling_action and 
            time.time() - self._last_scaling_action.timestamp < 60.0):
            return ScalingAction(
                action_type="no_action",
                target_workers=current_workers,
                reason="Recent scaling action - cooling down",
                confidence=1.0
            )
        
        # Make decision
        if scale_up_score > self._thresholds["scale_up_threshold"]:
            target_workers = min(self.max_workers, current_workers + 1)
            if target_workers > current_workers:
                return ScalingAction(
                    action_type="scale_up",
                    target_workers=target_workers,
                    reason=f"High load detected (score: {scale_up_score:.2f})",
                    confidence=scale_up_score
                )
        
        elif scale_down_score > self._thresholds["scale_down_threshold"]:
            target_workers = max(self.min_workers, current_workers - 1)
            if target_workers < current_workers:
                return ScalingAction(
                    action_type="scale_down", 
                    target_workers=target_workers,
                    reason=f"Low load detected (score: {scale_down_score:.2f})",
                    confidence=scale_down_score
                )
        
        return ScalingAction(
            action_type="no_action",
            target_workers=current_workers,
            reason="Load within acceptable range",
            confidence=0.5
        )
    
    def _calculate_scale_up_score(self, metrics: ScalingMetrics) -> float:
        """Calculate score for scaling up."""
        score = 0.0
        
        # CPU pressure
        if metrics.cpu_percent > self._thresholds["cpu_high"]:
            score += (metrics.cpu_percent - self._thresholds["cpu_high"]) / 20.0
        
        # Memory pressure
        if metrics.memory_percent > self._thresholds["memory_high"]:
            score += (metrics.memory_percent - self._thresholds["memory_high"]) / 20.0
        
        # Queue size
        if metrics.queue_size > self._thresholds["queue_high"]:
            score += (metrics.queue_size - self._thresholds["queue_high"]) / 10.0
        
        # Response time
        if metrics.response_time_ms > self._thresholds["response_time_high"]:
            score += (metrics.response_time_ms - self._thresholds["response_time_high"]) / 1000.0
        
        # Error rate
        if metrics.error_rate > self._thresholds["error_rate_high"]:
            score += metrics.error_rate * 2.0
        
        return min(1.0, score)
    
    def _calculate_scale_down_score(self, metrics: ScalingMetrics) -> float:
        """Calculate score for scaling down."""
        score = 0.0
        
        # Low resource usage
        if metrics.cpu_percent < self._thresholds["cpu_low"]:
            score += (self._thresholds["cpu_low"] - metrics.cpu_percent) / 30.0
        
        if metrics.memory_percent < self._thresholds["memory_low"]:
            score += (self._thresholds["memory_low"] - metrics.memory_percent) / 30.0
        
        # Low activity
        if metrics.queue_size < self._thresholds["queue_low"]:
            score += 0.3
        
        if metrics.active_tasks < self.load_balancer.current_workers * 0.5:
            score += 0.2
        
        # Good performance metrics
        if metrics.response_time_ms < self._thresholds["response_time_low"]:
            score += 0.2
        
        if metrics.error_rate < self._thresholds["error_rate_low"]:
            score += 0.1
        
        return min(1.0, score)
    
    def _execute_scaling_action(self, action: ScalingAction) -> None:
        """Execute scaling action."""
        current_workers = self.load_balancer.current_workers
        target_workers = action.target_workers
        
        if action.action_type == "scale_up":
            # Increase thread pool size
            new_pool = ThreadPoolExecutor(max_workers=target_workers)
            old_pool = self.load_balancer._thread_pool
            self.load_balancer._thread_pool = new_pool
            self.load_balancer.current_workers = target_workers
            
            # Shutdown old pool gracefully
            old_pool.shutdown(wait=False)
            
            logger.info(
                "Scaled up workers",
                from_workers=current_workers,
                to_workers=target_workers,
                reason=action.reason
            )
        
        elif action.action_type == "scale_down":
            # Decrease thread pool size
            new_pool = ThreadPoolExecutor(max_workers=target_workers)
            old_pool = self.load_balancer._thread_pool
            self.load_balancer._thread_pool = new_pool
            self.load_balancer.current_workers = target_workers
            
            # Shutdown old pool gracefully
            old_pool.shutdown(wait=False)
            
            logger.info(
                "Scaled down workers",
                from_workers=current_workers,
                to_workers=target_workers,
                reason=action.reason
            )
        
        # Record scaling action
        perf_logger.record_metric(
            "scaling_action",
            target_workers - current_workers,
            action_type=action.action_type,
            confidence=action.confidence,
            reason=action.reason
        )
    
    def _get_scaling_thresholds(self, mode: ScalingMode) -> Dict[str, float]:
        """Get scaling thresholds based on mode."""
        if mode == ScalingMode.CONSERVATIVE:
            return {
                "cpu_high": 85.0,
                "cpu_low": 30.0,
                "memory_high": 85.0,
                "memory_low": 40.0,
                "queue_high": 20,
                "queue_low": 2,
                "response_time_high": 5000.0,  # 5 seconds
                "response_time_low": 500.0,    # 0.5 seconds
                "error_rate_high": 0.1,        # 10%
                "error_rate_low": 0.01,        # 1%
                "scale_up_threshold": 0.7,
                "scale_down_threshold": 0.6
            }
        elif mode == ScalingMode.AGGRESSIVE:
            return {
                "cpu_high": 70.0,
                "cpu_low": 40.0,
                "memory_high": 75.0,
                "memory_low": 50.0,
                "queue_high": 10,
                "queue_low": 1,
                "response_time_high": 2000.0,  # 2 seconds
                "response_time_low": 200.0,    # 0.2 seconds
                "error_rate_high": 0.05,       # 5%
                "error_rate_low": 0.005,       # 0.5%
                "scale_up_threshold": 0.4,
                "scale_down_threshold": 0.5
            }
        else:  # BALANCED
            return {
                "cpu_high": 75.0,
                "cpu_low": 35.0,
                "memory_high": 80.0,
                "memory_low": 45.0,
                "queue_high": 15,
                "queue_low": 1,
                "response_time_high": 3000.0,  # 3 seconds
                "response_time_low": 300.0,    # 0.3 seconds
                "error_rate_high": 0.08,       # 8%
                "error_rate_low": 0.01,        # 1%
                "scale_up_threshold": 0.6,
                "scale_down_threshold": 0.5
            }


# Global scaling system
_load_balancer = None
_auto_scaler = None


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer."""
    global _load_balancer
    
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    
    return _load_balancer


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler."""
    global _auto_scaler
    
    if _auto_scaler is None:
        load_balancer = get_load_balancer()
        _auto_scaler = AutoScaler(load_balancer)
        _auto_scaler.start_scaling()
    
    return _auto_scaler


def distributed_execute(
    tasks: List[Tuple[Callable, tuple, dict]],
    max_workers: Optional[int] = None,
    use_processes: bool = False
) -> List[Any]:
    """
    Execute tasks in a distributed manner.
    
    Args:
        tasks: List of (function, args, kwargs) tuples
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        
    Returns:
        List of results in the same order as tasks
    """
    load_balancer = get_load_balancer()
    
    # Submit all tasks
    task_ids = []
    for task_func, args, kwargs in tasks:
        task_id = load_balancer.submit_task(
            task_func,
            use_process=use_processes,
            *args,
            **kwargs
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        try:
            result = load_balancer.get_result(task_id, timeout=300.0)  # 5 minute timeout
            results.append(result)
        except Exception as e:
            logger.error("Distributed task failed", task_id=task_id, exception=e)
            results.append(None)
    
    return results


def get_scaling_stats() -> Dict[str, Any]:
    """Get comprehensive scaling statistics."""
    load_balancer = get_load_balancer()
    auto_scaler = get_auto_scaler()
    
    return {
        "load_balancer": load_balancer.get_load_stats(),
        "auto_scaler": auto_scaler.get_scaling_stats(),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "current_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    }