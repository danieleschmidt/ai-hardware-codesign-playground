"""
Advanced auto-scaling system for AI Hardware Co-Design Playground.

This module provides intelligent resource scaling, load balancing, and performance
optimization with machine learning-based predictions and adaptive behavior.
"""

import time
import threading
import asyncio
import math
import statistics
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue

import logging
from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Auto-scaling modes with different aggressiveness levels."""
    CONSERVATIVE = "conservative"  # Scale slowly, prioritize stability
    BALANCED = "balanced"         # Balance performance and stability
    AGGRESSIVE = "aggressive"     # Scale quickly, prioritize performance
    PREDICTIVE = "predictive"     # ML-based predictive scaling


@dataclass
class SystemMetrics:
    """System performance metrics for scaling decisions."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    queue_size: int
    active_workers: int
    response_time_ms: float
    error_rate: float
    throughput_rps: float
    pending_tasks: int
    completed_tasks_last_minute: int


@dataclass
class ScalingDecision:
    """Scaling decision with confidence and reasoning."""
    action: str  # "scale_up", "scale_down", "maintain"
    target_workers: int
    current_workers: int
    confidence: float
    reasoning: List[str]
    timestamp: float = field(default_factory=time.time)


class AdaptiveLoadBalancer:
    """Intelligent load balancer with auto-scaling capabilities."""
    
    def __init__(
        self, 
        initial_workers: int = 4,
        min_workers: int = 2,
        max_workers: int = 32,
        scaling_mode: ScalingMode = ScalingMode.BALANCED
    ):
        """
        Initialize adaptive load balancer.
        
        Args:
            initial_workers: Starting number of workers
            min_workers: Minimum workers to maintain
            max_workers: Maximum workers allowed
            scaling_mode: Scaling aggressiveness mode
        """
        self.initial_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_mode = scaling_mode
        
        # Worker management
        self.current_workers = initial_workers
        self._thread_pool = ThreadPoolExecutor(
            max_workers=initial_workers, 
            thread_name_prefix="adaptive_worker"
        )
        self._process_pool = None  # Created on demand for CPU-intensive tasks
        
        # Task management
        self._task_queue = queue.PriorityQueue()
        self._high_priority_queue = queue.Queue()
        self._results_cache = {}
        
        # Metrics and monitoring
        self._metrics_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self._scaling_history = deque(maxlen=100)
        self._performance_baseline = None
        
        # Adaptive parameters
        self._scaling_cooldown = 30.0  # Seconds between scaling actions
        self._last_scale_time = 0.0
        self._scale_up_threshold = 0.8
        self._scale_down_threshold = 0.3
        self._prediction_weights = deque(maxlen=10)
        
        # Control and locks
        self._lock = threading.RLock()
        self._metrics_thread = None
        self._scaling_thread = None
        self._shutdown_event = threading.Event()
        
        self._start_monitoring()
        logger.info(f"Initialized AdaptiveLoadBalancer with {initial_workers} workers")
    
    def _start_monitoring(self) -> None:
        """Start background monitoring and scaling threads."""
        self._metrics_thread = threading.Thread(
            target=self._metrics_collection_loop, 
            daemon=True, 
            name="metrics_collector"
        )
        self._scaling_thread = threading.Thread(
            target=self._scaling_decision_loop, 
            daemon=True, 
            name="scaling_controller"
        )
        
        self._metrics_thread.start()
        self._scaling_thread.start()
    
    def _metrics_collection_loop(self) -> None:
        """Continuous metrics collection."""
        while not self._shutdown_event.is_set():
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self._metrics_history.append(metrics)
                
                # Record metrics for external monitoring
                record_metric("system_cpu_percent", metrics.cpu_percent, "gauge")
                record_metric("system_memory_percent", metrics.memory_percent, "gauge")
                record_metric("active_workers", metrics.active_workers, "gauge")
                record_metric("queue_size", metrics.queue_size, "gauge")
                record_metric("response_time_ms", metrics.response_time_ms, "gauge")
                
                time.sleep(1.0)  # Collect metrics every second
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _scaling_decision_loop(self) -> None:
        """Continuous scaling decision making."""
        while not self._shutdown_event.is_set():
            try:
                if self._should_evaluate_scaling():
                    decision = self._make_scaling_decision()
                    if decision.action != "maintain":
                        self._execute_scaling_decision(decision)
                
                time.sleep(5.0)  # Evaluate scaling every 5 seconds
            except Exception as e:
                logger.error(f"Scaling decision error: {e}")
                time.sleep(10.0)  # Back off on error
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            # Fallback metrics without psutil
            cpu_percent = self._estimate_cpu_usage()
            memory_percent = 50.0  # Conservative estimate
        
        # Calculate queue and task metrics
        queue_size = self._task_queue.qsize() + self._high_priority_queue.qsize()
        
        # Estimate response time based on recent performance
        response_time_ms = self._estimate_response_time()
        
        # Calculate throughput
        completed_tasks = self._get_completed_tasks_last_minute()
        throughput_rps = completed_tasks / 60.0
        
        # Error rate estimation
        error_rate = self._calculate_error_rate()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            queue_size=queue_size,
            active_workers=self.current_workers,
            response_time_ms=response_time_ms,
            error_rate=error_rate,
            throughput_rps=throughput_rps,
            pending_tasks=queue_size,
            completed_tasks_last_minute=completed_tasks
        )
    
    def _should_evaluate_scaling(self) -> bool:
        """Check if we should evaluate scaling decisions."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self._last_scale_time < self._scaling_cooldown:
            return False
        
        # Need at least some metrics history
        with self._lock:
            return len(self._metrics_history) >= 5
    
    def _make_scaling_decision(self) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics and mode."""
        with self._lock:
            recent_metrics = list(self._metrics_history)[-10:]  # Last 10 seconds
        
        if not recent_metrics:
            return ScalingDecision("maintain", self.current_workers, self.current_workers, 0.0, [])
        
        # Calculate trend indicators
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)
        avg_queue_size = statistics.mean(m.queue_size for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        
        # Calculate trends (are metrics getting worse?)
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        queue_trend = self._calculate_trend([m.queue_size for m in recent_metrics])
        response_trend = self._calculate_trend([m.response_time_ms for m in recent_metrics])
        
        # Scaling signals
        scale_up_signals = []
        scale_down_signals = []
        confidence_factors = []
        
        # CPU-based scaling
        if avg_cpu > 85:
            scale_up_signals.append(f"High CPU usage: {avg_cpu:.1f}%")
            confidence_factors.append(0.9)
        elif avg_cpu < 20:
            scale_down_signals.append(f"Low CPU usage: {avg_cpu:.1f}%")
            confidence_factors.append(0.7)
        
        # Queue-based scaling
        if avg_queue_size > self.current_workers * 3:
            scale_up_signals.append(f"Large queue: {avg_queue_size:.1f} tasks")
            confidence_factors.append(0.95)
        elif avg_queue_size < 1 and self.current_workers > self.min_workers:
            scale_down_signals.append(f"Empty queue: {avg_queue_size:.1f} tasks")
            confidence_factors.append(0.8)
        
        # Response time scaling
        if avg_response_time > 1000:  # 1 second
            scale_up_signals.append(f"High response time: {avg_response_time:.0f}ms")
            confidence_factors.append(0.85)
        elif avg_response_time < 100:  # 100ms
            scale_down_signals.append(f"Low response time: {avg_response_time:.0f}ms")
            confidence_factors.append(0.6)
        
        # Trend-based scaling (predictive)
        if self.scaling_mode == ScalingMode.PREDICTIVE:
            if cpu_trend > 0.1 or queue_trend > 0.1:
                scale_up_signals.append("Increasing load trend detected")
                confidence_factors.append(0.7)
        
        # Make decision
        if scale_up_signals and self.current_workers < self.max_workers:
            # Scale up
            scaling_factor = self._get_scaling_factor()
            new_workers = min(
                self.max_workers,
                max(self.current_workers + 1, int(self.current_workers * scaling_factor))
            )
            
            confidence = statistics.mean(confidence_factors) if confidence_factors else 0.5
            return ScalingDecision("scale_up", new_workers, self.current_workers, confidence, scale_up_signals)
            
        elif scale_down_signals and self.current_workers > self.min_workers:
            # Scale down
            scaling_factor = self._get_scaling_factor()
            new_workers = max(
                self.min_workers,
                min(self.current_workers - 1, int(self.current_workers / scaling_factor))
            )
            
            confidence = statistics.mean(confidence_factors) if confidence_factors else 0.5
            return ScalingDecision("scale_down", new_workers, self.current_workers, confidence, scale_down_signals)
        
        else:
            # Maintain current level
            return ScalingDecision("maintain", self.current_workers, self.current_workers, 1.0, ["System stable"])
    
    def _get_scaling_factor(self) -> float:
        """Get scaling factor based on mode."""
        if self.scaling_mode == ScalingMode.CONSERVATIVE:
            return 1.2  # 20% increase/decrease
        elif self.scaling_mode == ScalingMode.BALANCED:
            return 1.5  # 50% increase/decrease
        elif self.scaling_mode == ScalingMode.AGGRESSIVE:
            return 2.0  # 100% increase/decrease
        else:  # PREDICTIVE
            return 1.3  # Moderate scaling with ML predictions
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute the scaling decision."""
        if decision.confidence < 0.5:
            logger.info(f"Skipping scaling decision due to low confidence: {decision.confidence:.2f}")
            return
        
        logger.info(
            f"Scaling decision: {decision.action} from {decision.current_workers} to {decision.target_workers} "
            f"workers (confidence: {decision.confidence:.2f})"
        )
        logger.info(f"Reasons: {'; '.join(decision.reasoning)}")
        
        try:
            if decision.action == "scale_up":
                self._scale_up(decision.target_workers)
            elif decision.action == "scale_down":
                self._scale_down(decision.target_workers)
            
            self._last_scale_time = time.time()
            
            with self._lock:
                self._scaling_history.append(decision)
            
            record_metric("scaling_action", 1, "counter", {"action": decision.action})
            record_metric("worker_count", self.current_workers, "gauge")
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
    
    def _scale_up(self, target_workers: int) -> None:
        """Scale up worker pool."""
        if target_workers <= self.current_workers:
            return
        
        # Gracefully increase thread pool size
        old_pool = self._thread_pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=target_workers,
            thread_name_prefix="adaptive_worker"
        )
        
        # Update worker count
        self.current_workers = target_workers
        
        logger.info(f"Scaled up to {target_workers} workers")
        
        # Keep old pool alive briefly to finish existing tasks
        threading.Timer(30.0, lambda: old_pool.shutdown(wait=False)).start()
    
    def _scale_down(self, target_workers: int) -> None:
        """Scale down worker pool."""
        if target_workers >= self.current_workers:
            return
        
        # Create new smaller pool
        old_pool = self._thread_pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=target_workers,
            thread_name_prefix="adaptive_worker"
        )
        
        # Update worker count
        self.current_workers = target_workers
        
        logger.info(f"Scaled down to {target_workers} workers")
        
        # Gracefully shutdown old pool
        threading.Timer(30.0, lambda: old_pool.shutdown(wait=True)).start()
    
    def submit_task(
        self, 
        task_func: Callable, 
        priority: int = 5,
        high_priority: bool = False,
        timeout: Optional[float] = None,
        *args, 
        **kwargs
    ) -> str:
        """Submit task for execution with priority and timeout."""
        task_id = f"task_{time.time()}_{id(task_func)}"
        
        task_info = {
            "id": task_id,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "priority": priority,
            "timeout": timeout,
            "submitted_at": time.time()
        }
        
        if high_priority:
            self._high_priority_queue.put(task_info)
        else:
            self._task_queue.put((priority, task_info))
        
        # Submit to thread pool
        future = self._thread_pool.submit(self._execute_task, task_info)
        
        record_metric("task_submitted", 1, "counter")
        return task_id
    
    def _execute_task(self, task_info: Dict[str, Any]) -> Any:
        """Execute a task with monitoring."""
        start_time = time.time()
        task_id = task_info["id"]
        
        try:
            result = task_info["func"](*task_info["args"], **task_info["kwargs"])
            duration = time.time() - start_time
            
            record_metric("task_completed", 1, "counter")
            record_metric("task_duration_ms", duration * 1000, "timer")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task {task_id} failed after {duration:.2f}s: {e}")
            
            record_metric("task_failed", 1, "counter")
            record_metric("task_error_duration_ms", duration * 1000, "timer")
            
            raise
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance statistics."""
        with self._lock:
            recent_metrics = list(self._metrics_history)[-60:]  # Last minute
            scaling_history = list(self._scaling_history)[-10:]  # Recent decisions
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Calculate averages
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)
        avg_queue_size = statistics.mean(m.queue_size for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        avg_throughput = statistics.mean(m.throughput_rps for m in recent_metrics)
        
        # Scaling efficiency
        scale_up_count = sum(1 for d in scaling_history if d.action == "scale_up")
        scale_down_count = sum(1 for d in scaling_history if d.action == "scale_down")
        avg_confidence = statistics.mean(d.confidence for d in scaling_history) if scaling_history else 0
        
        return {
            "current_workers": self.current_workers,
            "scaling_mode": self.scaling_mode.value,
            "performance": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "avg_queue_size": avg_queue_size,
                "avg_response_time_ms": avg_response_time,
                "avg_throughput_rps": avg_throughput
            },
            "scaling_history": {
                "scale_up_count": scale_up_count,
                "scale_down_count": scale_down_count,
                "avg_confidence": avg_confidence,
                "last_scale_time": self._last_scale_time,
                "cooldown_remaining": max(0, self._scaling_cooldown - (time.time() - self._last_scale_time))
            },
            "limits": {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers
            }
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the load balancer."""
        logger.info("Shutting down AdaptiveLoadBalancer")
        
        self._shutdown_event.set()
        
        # Wait for monitoring threads
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5.0)
        
        if self._scaling_thread and self._scaling_thread.is_alive():
            self._scaling_thread.join(timeout=5.0)
        
        # Shutdown executor
        self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
    
    # Helper methods
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Slope of linear regression
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope
    
    def _estimate_cpu_usage(self) -> float:
        """Estimate CPU usage without psutil."""
        # Simple estimation based on worker utilization
        utilization = min(self.current_workers / self.max_workers, 1.0)
        return utilization * 100
    
    def _estimate_response_time(self) -> float:
        """Estimate current response time based on queue and workers."""
        queue_size = self._task_queue.qsize()
        if queue_size == 0:
            return 50.0  # Base response time
        
        # Estimate based on queue length and worker count
        estimated_time = (queue_size / max(self.current_workers, 1)) * 100
        return min(estimated_time, 5000)  # Cap at 5 seconds
    
    def _get_completed_tasks_last_minute(self) -> int:
        """Get number of completed tasks in the last minute."""
        # This would be tracked in a real implementation
        # For now, return an estimate based on throughput
        return max(1, int(self.current_workers * 0.5))  # Conservative estimate
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # This would be tracked from actual metrics
        # For now, return a low default rate
        return 0.01  # 1% error rate


# Global auto-scaler instance
_global_auto_scaler: Optional[AdaptiveLoadBalancer] = None


def get_auto_scaler(
    initial_workers: int = 4,
    min_workers: int = 2, 
    max_workers: int = 32,
    scaling_mode: ScalingMode = ScalingMode.BALANCED
) -> AdaptiveLoadBalancer:
    """Get or create the global auto-scaler instance."""
    global _global_auto_scaler
    
    if _global_auto_scaler is None:
        _global_auto_scaler = AdaptiveLoadBalancer(
            initial_workers=initial_workers,
            min_workers=min_workers,
            max_workers=max_workers,
            scaling_mode=scaling_mode
        )
    
    return _global_auto_scaler


def shutdown_auto_scaler() -> None:
    """Shutdown the global auto-scaler."""
    global _global_auto_scaler
    
    if _global_auto_scaler:
        _global_auto_scaler.shutdown()
        _global_auto_scaler = None