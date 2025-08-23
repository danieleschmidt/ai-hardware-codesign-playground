"""
Advanced Performance Optimization Engine for AI Hardware Co-Design.

This module implements Generation 3 performance optimization techniques including
adaptive caching, concurrent processing, intelligent resource management, and
auto-scaling for maximum throughput and efficiency.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Fallback psutil implementation
    class FallbackPsutil:
        @staticmethod
        def cpu_count(logical=True):
            return 4 if logical else 2
        
        @staticmethod
        def cpu_percent(interval=None):
            return 50.0  # Mock 50% CPU usage
        
        @staticmethod
        def virtual_memory():
            return type('memory', (), {
                'total': 8 * 1024**3,  # 8GB
                'percent': 60.0
            })()
    
    psutil = FallbackPsutil()
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from enum import Enum

from ..utils.logging import get_logger
from ..utils.monitoring import record_metric
from ..utils.exceptions import PerformanceError

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    THROUGHPUT_MAXIMIZATION = "throughput_max"
    LATENCY_MINIMIZATION = "latency_min"  
    ENERGY_EFFICIENCY = "energy_efficient"
    BALANCED_PERFORMANCE = "balanced"
    ADAPTIVE_SCALING = "adaptive"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics collection."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0  # operations per second
    latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_tasks: int = 0
    queue_depth: int = 0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for monitoring."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "throughput": self.throughput,
            "latency_ms": self.latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "concurrent_tasks": self.concurrent_tasks,
            "queue_depth": self.queue_depth,
            "error_rate": self.error_rate
        }


@dataclass
class ResourcePool:
    """Dynamic resource pool for optimal allocation."""
    
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count(logical=False) or 4)
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    max_workers: int = field(default=None)
    thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
    process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
    
    def __post_init__(self):
        """Initialize resource pools."""
        if self.max_workers is None:
            self.max_workers = min(32, (self.cpu_cores * 4) + 4)
        
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="CoDesign-Thread"
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cpu_cores,
            mp_context=multiprocessing.get_context('spawn')
        )
    
    def shutdown(self):
        """Clean shutdown of resource pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class AdaptiveCache:
    """Intelligent adaptive caching system with performance optimization."""
    
    def __init__(self, initial_size: int = 1000, max_size: int = 10000):
        """
        Initialize adaptive cache.
        
        Args:
            initial_size: Initial cache size
            max_size: Maximum cache size before eviction
        """
        self.max_size = max_size
        self.current_size = 0
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.size_estimates = {}
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive parameters
        self.hit_rate_threshold = 0.8
        self.resize_factor = 1.2
        self.cleanup_interval = 100  # operations
        self.operation_count = 0
        
        self._lock = threading.RLock()
        
        logger.info(f"Initialized adaptive cache with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive behavior."""
        with self._lock:
            self.operation_count += 1
            
            if key in self.cache:
                self.hits += 1
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                
                # Adaptive hit rate tracking
                if self.operation_count % self.cleanup_interval == 0:
                    self._adaptive_optimization()
                
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, estimated_size: int = 1) -> bool:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            # Check if we need to evict
            if key not in self.cache and self.current_size >= self.max_size:
                if not self._intelligent_eviction():
                    logger.warning("Cache eviction failed, skipping put operation")
                    return False
            
            # Update cache
            if key not in self.cache:
                self.current_size += estimated_size
            
            self.cache[key] = value
            self.size_estimates[key] = estimated_size
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
            
            return True
    
    def _intelligent_eviction(self) -> bool:
        """Intelligent cache eviction using LFU + LRU hybrid approach."""
        if not self.cache:
            return True
        
        current_time = time.time()
        eviction_candidates = []
        
        # Score items based on access frequency and recency
        for key in self.cache.keys():
            access_count = self.access_counts[key]
            last_access = self.access_times[key]
            time_since_access = current_time - last_access
            
            # Composite score (lower is worse)
            score = access_count * 0.7 - time_since_access * 0.3
            eviction_candidates.append((score, key))
        
        # Sort by score and evict worst performers
        eviction_candidates.sort()
        evict_count = max(1, len(eviction_candidates) // 10)  # Evict 10%
        
        for _, key in eviction_candidates[:evict_count]:
            self._evict_key(key)
            self.evictions += 1
        
        return True
    
    def _evict_key(self, key: str):
        """Remove key from cache and update metrics."""
        if key in self.cache:
            del self.cache[key]
            self.current_size -= self.size_estimates.pop(key, 1)
            self.access_counts.pop(key, None)
            self.access_times.pop(key, None)
    
    def _adaptive_optimization(self):
        """Perform adaptive cache optimization."""
        hit_rate = self.get_hit_rate()
        
        # Adjust cache size based on performance
        if hit_rate < self.hit_rate_threshold and self.max_size < 50000:
            old_size = self.max_size
            self.max_size = int(self.max_size * self.resize_factor)
            logger.info(f"Cache resize: {old_size} -> {self.max_size} (hit_rate={hit_rate:.3f})")
        
        # Cleanup stale entries
        self._cleanup_stale_entries()
    
    def _cleanup_stale_entries(self):
        """Remove very old entries to prevent memory leaks."""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour
        
        stale_keys = [
            key for key, last_access in self.access_times.items()
            if current_time - last_access > stale_threshold
        ]
        
        for key in stale_keys:
            self._evict_key(key)
    
    def get_hit_rate(self) -> float:
        """Calculate current hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "current_size_estimate": self.current_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "evictions": self.evictions,
            "operations": self.operation_count
        }
    
    def clear(self):
        """Clear cache and reset statistics."""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.size_estimates.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.operation_count = 0


class PerformanceOrchestrator:
    """Orchestrates performance optimization across all system components."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED_PERFORMANCE):
        """
        Initialize performance orchestrator.
        
        Args:
            strategy: Performance optimization strategy to employ
        """
        self.strategy = strategy
        self.resource_pool = ResourcePool()
        self.adaptive_cache = AdaptiveCache()
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics()
        
        # Optimization controls
        self.optimization_enabled = True
        self.monitoring_enabled = True
        self.auto_scaling_enabled = True
        
        # Performance thresholds
        self.cpu_threshold = 0.85
        self.memory_threshold = 0.85
        self.latency_threshold_ms = 1000.0
        self.throughput_target = 100.0  # ops/sec
        
        # Background tasks
        self._monitoring_task = None
        self._optimization_task = None
        self._shutdown_event = threading.Event()
        
        # Start background optimization
        self._start_background_tasks()
        
        logger.info(f"Performance orchestrator initialized with {strategy.value} strategy")
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks."""
        if self.monitoring_enabled:
            self._monitoring_task = threading.Thread(
                target=self._monitoring_loop,
                name="PerformanceMonitor",
                daemon=True
            )
            self._monitoring_task.start()
        
        if self.optimization_enabled:
            self._optimization_task = threading.Thread(
                target=self._optimization_loop,
                name="PerformanceOptimizer", 
                daemon=True
            )
            self._optimization_task.start()
    
    def _monitoring_loop(self):
        """Background performance monitoring loop."""
        logger.info("Performance monitoring started")
        
        while not self._shutdown_event.is_set():
            try:
                self._collect_metrics()
                time.sleep(1.0)  # Monitor every second
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(5.0)  # Backoff on error
    
    def _optimization_loop(self):
        """Background optimization loop."""
        logger.info("Performance optimization started")
        
        while not self._shutdown_event.is_set():
            try:
                self._perform_optimization()
                time.sleep(10.0)  # Optimize every 10 seconds
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                time.sleep(30.0)  # Backoff on error
    
    def _collect_metrics(self):
        """Collect current system performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Cache metrics
            cache_stats = self.adaptive_cache.get_stats()
            
            # Create metrics snapshot
            self.current_metrics = PerformanceMetrics(
                cpu_usage=cpu_percent / 100.0,
                memory_usage=memory_percent / 100.0,
                cache_hit_rate=cache_stats["hit_rate"],
                concurrent_tasks=threading.active_count()
            )
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
            # Record to monitoring system
            record_metric("cpu_usage", cpu_percent, "gauge")
            record_metric("memory_usage", memory_percent, "gauge")
            record_metric("cache_hit_rate", cache_stats["hit_rate"], "gauge")
            record_metric("concurrent_tasks", threading.active_count(), "gauge")
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def _perform_optimization(self):
        """Perform adaptive performance optimization."""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Strategy-specific optimizations
        if self.strategy == OptimizationStrategy.THROUGHPUT_MAXIMIZATION:
            self._optimize_for_throughput(avg_cpu, avg_memory)
        elif self.strategy == OptimizationStrategy.LATENCY_MINIMIZATION:
            self._optimize_for_latency(avg_cpu, avg_memory)
        elif self.strategy == OptimizationStrategy.ENERGY_EFFICIENCY:
            self._optimize_for_energy(avg_cpu, avg_memory)
        elif self.strategy == OptimizationStrategy.BALANCED_PERFORMANCE:
            self._optimize_balanced(avg_cpu, avg_memory)
        elif self.strategy == OptimizationStrategy.ADAPTIVE_SCALING:
            self._optimize_adaptive(avg_cpu, avg_memory)
        
        # Trigger garbage collection if memory usage is high
        if avg_memory > self.memory_threshold:
            gc.collect()
            logger.info(f"Triggered garbage collection (memory usage: {avg_memory:.1%})")
    
    def _optimize_for_throughput(self, cpu_usage: float, memory_usage: float):
        """Optimize system for maximum throughput."""
        # Increase parallelism if resources available
        if cpu_usage < 0.7 and memory_usage < 0.8:
            current_workers = self.resource_pool.max_workers
            new_workers = min(current_workers + 2, 64)
            if new_workers != current_workers:
                self._resize_thread_pool(new_workers)
                logger.info(f"Throughput optimization: increased workers {current_workers} -> {new_workers}")
        
        # Optimize cache for throughput
        cache_hit_rate = self.adaptive_cache.get_hit_rate()
        if cache_hit_rate < 0.8:
            self.adaptive_cache.max_size = min(self.adaptive_cache.max_size * 1.1, 20000)
    
    def _optimize_for_latency(self, cpu_usage: float, memory_usage: float):
        """Optimize system for minimum latency."""
        # Reduce parallelism to minimize context switching
        if cpu_usage > 0.8:
            current_workers = self.resource_pool.max_workers
            new_workers = max(current_workers - 1, 2)
            if new_workers != current_workers:
                self._resize_thread_pool(new_workers)
                logger.info(f"Latency optimization: reduced workers {current_workers} -> {new_workers}")
        
        # Prioritize cache access speed
        if memory_usage > 0.8:
            self.adaptive_cache.max_size = max(self.adaptive_cache.max_size * 0.9, 500)
    
    def _optimize_for_energy(self, cpu_usage: float, memory_usage: float):
        """Optimize system for energy efficiency."""
        # Reduce unnecessary parallelism
        if cpu_usage < 0.5:
            current_workers = self.resource_pool.max_workers
            new_workers = max(current_workers - 1, psutil.cpu_count(logical=False))
            if new_workers != current_workers:
                self._resize_thread_pool(new_workers)
        
        # Minimize cache size to reduce memory power
        cache_stats = self.adaptive_cache.get_stats()
        if cache_stats["hit_rate"] > 0.9 and cache_stats["size"] > 1000:
            self.adaptive_cache.max_size = max(self.adaptive_cache.max_size * 0.95, 1000)
    
    def _optimize_balanced(self, cpu_usage: float, memory_usage: float):
        """Balanced optimization strategy."""
        target_cpu = 0.7
        target_memory = 0.7
        
        # Adjust worker count based on resource utilization
        if cpu_usage < target_cpu - 0.1 and memory_usage < target_memory:
            # Increase workers if underutilized
            current_workers = self.resource_pool.max_workers
            new_workers = min(current_workers + 1, 32)
            if new_workers != current_workers:
                self._resize_thread_pool(new_workers)
        elif cpu_usage > target_cpu + 0.1 or memory_usage > target_memory + 0.1:
            # Decrease workers if overutilized
            current_workers = self.resource_pool.max_workers
            new_workers = max(current_workers - 1, 2)
            if new_workers != current_workers:
                self._resize_thread_pool(new_workers)
    
    def _optimize_adaptive(self, cpu_usage: float, memory_usage: float):
        """Adaptive optimization based on current conditions."""
        # Dynamically switch strategies based on system state
        if memory_usage > 0.9:
            self._optimize_for_energy(cpu_usage, memory_usage)
        elif cpu_usage > 0.9:
            self._optimize_for_latency(cpu_usage, memory_usage)
        elif cpu_usage < 0.3 and memory_usage < 0.5:
            self._optimize_for_throughput(cpu_usage, memory_usage)
        else:
            self._optimize_balanced(cpu_usage, memory_usage)
    
    def _resize_thread_pool(self, new_size: int):
        """Dynamically resize thread pool."""
        try:
            # Note: ThreadPoolExecutor doesn't support dynamic resizing
            # This is a simplified version - in practice, you'd need a custom pool
            self.resource_pool.max_workers = new_size
            record_metric("thread_pool_size", new_size, "gauge")
        except Exception as e:
            logger.error(f"Failed to resize thread pool: {e}")
    
    async def execute_optimized(
        self,
        func: Callable,
        *args,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        estimated_duration: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Execute function with full performance optimization.
        
        Args:
            func: Function to execute
            *args: Function arguments
            use_cache: Whether to use caching
            cache_key: Custom cache key
            estimated_duration: Estimated execution time for scheduling
            **kwargs: Function keyword arguments
            
        Returns:
            Function result with performance optimization
        """
        start_time = time.time()
        
        # Generate cache key if not provided
        if use_cache and cache_key is None:
            cache_key = self._generate_cache_key(func, args, kwargs)
        
        # Check cache first
        if use_cache and cache_key:
            cached_result = self.adaptive_cache.get(cache_key)
            if cached_result is not None:
                record_metric("cache_hit", 1, "counter")
                return cached_result
        
        # Choose execution strategy based on estimated duration
        try:
            if estimated_duration < 0.1:  # Fast operations - execute directly
                result = func(*args, **kwargs)
            elif estimated_duration < 5.0:  # Medium operations - thread pool
                future = self.resource_pool.thread_pool.submit(func, *args, **kwargs)
                result = await asyncio.wrap_future(future)
            else:  # Long operations - process pool
                future = self.resource_pool.process_pool.submit(func, *args, **kwargs)
                result = await asyncio.wrap_future(future)
            
            # Cache successful result
            if use_cache and cache_key and result is not None:
                self.adaptive_cache.put(cache_key, result)
                record_metric("cache_miss", 1, "counter")
            
            # Record performance metrics
            execution_time = time.time() - start_time
            record_metric("function_execution_time", execution_time * 1000, "histogram")
            record_metric("function_executions", 1, "counter")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            record_metric("function_errors", 1, "counter")
            record_metric("function_error_time", execution_time * 1000, "histogram")
            raise PerformanceError(f"Optimized execution failed: {e}") from e
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function and arguments."""
        import hashlib
        
        # Create deterministic hash from function and arguments
        key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def batch_execute_optimized(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        max_concurrency: Optional[int] = None
    ) -> List[Any]:
        """
        Execute batch of tasks with optimal concurrency.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            max_concurrency: Maximum concurrent executions
            
        Returns:
            List of results in same order as input tasks
        """
        if not tasks:
            return []
        
        if max_concurrency is None:
            max_concurrency = self.resource_pool.max_workers
        
        # Execute tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = []
            for func, args, kwargs in tasks:
                future = executor.submit(func, *args, **kwargs)
                futures.append(future)
            
            # Collect results maintaining order
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch task failed: {e}")
                    results.append(None)
            
            return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last minute
        
        # Calculate statistics
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        cache_stats = self.adaptive_cache.get_stats()
        
        return {
            "strategy": self.strategy.value,
            "system_performance": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "current_threads": threading.active_count(),
                "max_workers": self.resource_pool.max_workers
            },
            "cache_performance": cache_stats,
            "optimization_status": {
                "monitoring_enabled": self.monitoring_enabled,
                "optimization_enabled": self.optimization_enabled,
                "auto_scaling_enabled": self.auto_scaling_enabled
            },
            "resource_utilization": {
                "cpu_cores": self.resource_pool.cpu_cores,
                "memory_gb": self.resource_pool.memory_gb,
                "cpu_utilization": avg_cpu,
                "memory_utilization": avg_memory
            }
        }
    
    def shutdown(self):
        """Graceful shutdown of performance orchestrator."""
        logger.info("Shutting down performance orchestrator")
        
        self._shutdown_event.set()
        
        # Wait for background threads
        if self._monitoring_task and self._monitoring_task.is_alive():
            self._monitoring_task.join(timeout=5.0)
        
        if self._optimization_task and self._optimization_task.is_alive():
            self._optimization_task.join(timeout=5.0)
        
        # Shutdown resource pools
        self.resource_pool.shutdown()
        
        # Clear cache
        self.adaptive_cache.clear()
        
        logger.info("Performance orchestrator shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global performance orchestrator instance
_performance_orchestrator: Optional[PerformanceOrchestrator] = None


def get_performance_orchestrator(
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED_PERFORMANCE
) -> PerformanceOrchestrator:
    """Get global performance orchestrator instance."""
    global _performance_orchestrator
    
    if _performance_orchestrator is None:
        _performance_orchestrator = PerformanceOrchestrator(strategy)
    
    return _performance_orchestrator


def optimized_execution(
    cache_enabled: bool = True,
    estimated_duration: float = 1.0,
    cache_key: Optional[str] = None
):
    """
    Decorator for automatic performance optimization of functions.
    
    Args:
        cache_enabled: Enable caching for this function
        estimated_duration: Estimated execution time in seconds
        cache_key: Custom cache key (auto-generated if None)
        
    Returns:
        Decorated function with performance optimization
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            orchestrator = get_performance_orchestrator()
            return await orchestrator.execute_optimized(
                func, *args,
                use_cache=cache_enabled,
                cache_key=cache_key,
                estimated_duration=estimated_duration,
                **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            orchestrator = get_performance_orchestrator()
            return asyncio.run(orchestrator.execute_optimized(
                func, *args,
                use_cache=cache_enabled,
                cache_key=cache_key,
                estimated_duration=estimated_duration,
                **kwargs
            ))
        
        # Return async wrapper if function is async, sync wrapper otherwise
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator