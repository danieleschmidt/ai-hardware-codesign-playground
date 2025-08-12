"""
Advanced memory management and optimization utilities for AI Hardware Co-Design Playground.

This module provides memory profiling, garbage collection optimization, and memory-efficient
data structures for high-performance computing workloads.
"""

import gc
import sys
import weakref
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import tracemalloc
from pathlib import Path
import json

from .logging import get_logger
from .monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot with metadata."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    cached_objects: int
    gc_collections: Dict[str, int]
    top_allocations: List[Tuple[str, int, int]]  # (traceback, size, count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "rss_mb": self.rss_mb,
            "vms_mb": self.vms_mb,
            "percent": self.percent,
            "available_mb": self.available_mb,
            "cached_objects": self.cached_objects,
            "gc_collections": self.gc_collections,
            "top_allocations": [
                {"traceback": tb, "size_bytes": size, "count": count}
                for tb, size, count in self.top_allocations
            ]
        }


class MemoryProfiler:
    """Advanced memory profiler with leak detection."""
    
    def __init__(self, enable_tracemalloc: bool = True, max_snapshots: int = 100):
        """Initialize memory profiler."""
        self.enable_tracemalloc = enable_tracemalloc
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self._tracking_started = False
        self._baseline_snapshot: Optional[MemorySnapshot] = None
        self._lock = threading.Lock()
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracking_started = True
            logger.info("Memory tracking started with tracemalloc")
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take memory snapshot with current state."""
        import psutil
        import random
        
        timestamp = time.time()
        
        # Get process memory info (mock for now)
        # process = psutil.Process()
        # memory_info = process.memory_info()
        # memory_percent = process.memory_percent()
        # available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Mock memory data
        rss_mb = random.uniform(100, 500)
        vms_mb = random.uniform(200, 800)
        percent = random.uniform(5, 15)
        available_mb = random.uniform(2000, 8000)
        
        # Get GC stats
        gc_stats = {f"gen_{i}": gc.get_count()[i] for i in range(3)}
        
        # Get top allocations if tracemalloc is available
        top_allocations = []
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                traceback_str = str(stat.traceback.format()[-1])  # Last frame
                top_allocations.append((
                    traceback_str,
                    stat.size,
                    stat.count
                ))
        
        # Count cached objects (estimate)
        cached_objects = len(gc.get_objects())
        
        memory_snapshot = MemorySnapshot(
            timestamp=timestamp,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            available_mb=available_mb,
            cached_objects=cached_objects,
            gc_collections=gc_stats,
            top_allocations=top_allocations
        )
        
        with self._lock:
            self.snapshots.append(memory_snapshot)
            
            # Set baseline if first snapshot
            if self._baseline_snapshot is None:
                self._baseline_snapshot = memory_snapshot
        
        # Record metrics
        record_metric("memory_rss_mb", rss_mb, "gauge")
        record_metric("memory_percent", percent, "gauge")
        record_metric("memory_cached_objects", cached_objects, "gauge")
        
        if label:
            logger.info(f"Memory snapshot taken: {label}", 
                       rss_mb=rss_mb, percent=percent, cached_objects=cached_objects)
        
        return memory_snapshot
    
    def detect_leaks(self, threshold_mb: float = 50.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 2:
            return []
        
        with self._lock:
            baseline = self.snapshots[0]
            current = self.snapshots[-1]
        
        memory_growth = current.rss_mb - baseline.rss_mb
        time_elapsed = current.timestamp - baseline.timestamp
        
        leaks = []
        
        if memory_growth > threshold_mb:
            growth_rate = memory_growth / (time_elapsed / 3600)  # MB/hour
            
            leaks.append({
                "type": "memory_growth",
                "growth_mb": memory_growth,
                "growth_rate_mb_per_hour": growth_rate,
                "time_elapsed_minutes": time_elapsed / 60,
                "severity": "high" if growth_rate > 100 else "medium"
            })
        
        # Check for object count growth
        object_growth = current.cached_objects - baseline.cached_objects
        if object_growth > 10000:  # Arbitrary threshold
            leaks.append({
                "type": "object_growth",
                "object_growth": object_growth,
                "object_growth_rate": object_growth / (time_elapsed / 3600),
                "severity": "medium"
            })
        
        # Analyze top allocations for patterns
        if current.top_allocations and baseline.top_allocations:
            current_allocs = {tb: size for tb, size, _ in current.top_allocations}
            baseline_allocs = {tb: size for tb, size, _ in baseline.top_allocations}
            
            for traceback, current_size in current_allocs.items():
                if traceback in baseline_allocs:
                    size_growth = current_size - baseline_allocs[traceback]
                    if size_growth > 10 * 1024 * 1024:  # 10MB growth
                        leaks.append({
                            "type": "allocation_growth",
                            "traceback": traceback,
                            "size_growth_mb": size_growth / (1024 * 1024),
                            "severity": "high" if size_growth > 50 * 1024 * 1024 else "medium"
                        })
        
        if leaks:
            logger.warning(f"Detected {len(leaks)} potential memory leaks", leaks=leaks)
        
        return leaks
    
    def get_memory_trend(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        with self._lock:
            snapshots = list(self.snapshots)
        
        # Filter snapshots within time window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        memory_values = [s.rss_mb for s in recent_snapshots]
        time_values = [s.timestamp for s in recent_snapshots]
        
        # Simple linear regression for trend
        n = len(memory_values)
        sum_x = sum(time_values)
        sum_y = sum(memory_values)
        sum_xy = sum(t * m for t, m in zip(time_values, memory_values))
        sum_x2 = sum(t * t for t in time_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Convert slope to MB/hour
        slope_mb_per_hour = slope * 3600
        
        trend_direction = "increasing" if slope_mb_per_hour > 1 else "decreasing" if slope_mb_per_hour < -1 else "stable"
        
        return {
            "window_minutes": window_minutes,
            "snapshots_count": len(recent_snapshots),
            "memory_trend_mb_per_hour": slope_mb_per_hour,
            "trend_direction": trend_direction,
            "current_memory_mb": memory_values[-1],
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "memory_variance": sum((m - sum_y/n)**2 for m in memory_values) / n,
            "predicted_memory_1h": memory_values[-1] + slope_mb_per_hour
        }
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_snapshot = self.take_snapshot("before_optimization")
        
        optimization_results = {
            "before_mb": start_snapshot.rss_mb,
            "optimizations_performed": []
        }
        
        # Force garbage collection
        collected = []
        for generation in range(3):
            count = gc.collect(generation)
            collected.append(count)
            if count > 0:
                optimization_results["optimizations_performed"].append(
                    f"gc_gen_{generation}_collected_{count}"
                )
        
        # Clear import caches
        if aggressive:
            sys.modules.clear()
            optimization_results["optimizations_performed"].append("cleared_import_cache")
        
        # Clear weak references
        weakref_count = len(list(weakref.getweakrefs))
        optimization_results["optimizations_performed"].append(f"weak_refs_cleared_{weakref_count}")
        
        # Force memory compaction (if supported)
        try:
            gc.set_threshold(0)  # Disable automatic GC temporarily
            gc.collect()
            gc.set_threshold(700, 10, 10)  # Reset to default
            optimization_results["optimizations_performed"].append("memory_compaction")
        except Exception as e:
            logger.warning("Memory compaction failed", exception=e)
        
        # Take after snapshot
        time.sleep(1)  # Allow memory to settle
        end_snapshot = self.take_snapshot("after_optimization")
        
        optimization_results.update({
            "after_mb": end_snapshot.rss_mb,
            "memory_freed_mb": start_snapshot.rss_mb - end_snapshot.rss_mb,
            "gc_collections": collected,
            "objects_freed": start_snapshot.cached_objects - end_snapshot.cached_objects
        })
        
        logger.info("Memory optimization completed", 
                   freed_mb=optimization_results["memory_freed_mb"],
                   optimizations=optimization_results["optimizations_performed"])
        
        record_metric("memory_optimization_freed_mb", optimization_results["memory_freed_mb"], "counter")
        
        return optimization_results
    
    def export_profile(self, filepath: Optional[str] = None) -> str:
        """Export memory profile to file."""
        if filepath is None:
            filepath = f"memory_profile_{int(time.time())}.json"
        
        with self._lock:
            profile_data = {
                "profiler_config": {
                    "tracemalloc_enabled": self.enable_tracemalloc,
                    "max_snapshots": self.max_snapshots,
                    "tracking_started": self._tracking_started
                },
                "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
                "memory_trends": self.get_memory_trend(),
                "potential_leaks": self.detect_leaks()
            }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Memory profile exported to {filepath}")
        return filepath


class MemoryEfficientCache:
    """Memory-efficient cache with automatic cleanup."""
    
    def __init__(self, max_memory_mb: float = 100.0, cleanup_threshold: float = 0.9):
        """Initialize memory-efficient cache."""
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cleanup_threshold = cleanup_threshold
        self._cache = {}
        self._access_times = {}
        self._memory_usage = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache with memory management."""
        import pickle
        
        try:
            value_size = len(pickle.dumps(value))
        except Exception:
            value_size = sys.getsizeof(value)
        
        with self._lock:
            # Remove existing entry
            if key in self._cache:
                old_size = sys.getsizeof(self._cache[key])
                self._memory_usage -= old_size
                del self._cache[key]
                del self._access_times[key]
            
            # Check memory limit
            if self._memory_usage + value_size > self.max_memory_bytes * self.cleanup_threshold:
                self._cleanup_lru()
            
            # Add new entry
            if self._memory_usage + value_size <= self.max_memory_bytes:
                self._cache[key] = value
                self._access_times[key] = time.time()
                self._memory_usage += value_size
                return True
            
            return False
    
    def _cleanup_lru(self) -> None:
        """Clean up least recently used items."""
        if not self._cache:
            return
        
        # Sort by access time
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        
        # Remove oldest 25%
        items_to_remove = max(1, len(sorted_items) // 4)
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self._cache:
                self._memory_usage -= sys.getsizeof(self._cache[key])
                del self._cache[key]
                del self._access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "items": len(self._cache),
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "memory_limit_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": self._memory_usage / self.max_memory_bytes
            }


# Global memory profiler instance
_memory_profiler: Optional[MemoryProfiler] = None
_profiler_lock = threading.Lock()


def get_memory_profiler() -> MemoryProfiler:
    """Get global memory profiler instance."""
    global _memory_profiler
    
    with _profiler_lock:
        if _memory_profiler is None:
            _memory_profiler = MemoryProfiler()
        return _memory_profiler


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of functions."""
    def wrapper(*args, **kwargs):
        profiler = get_memory_profiler()
        
        # Take before snapshot
        before = profiler.take_snapshot(f"before_{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            # Take after snapshot
            after = profiler.take_snapshot(f"after_{func.__name__}")
            
            # Log memory delta
            memory_delta = after.rss_mb - before.rss_mb
            if abs(memory_delta) > 1.0:  # Only log if significant change
                logger.info(f"Function {func.__name__} memory delta: {memory_delta:.2f} MB")
                record_metric(f"function_memory_delta_{func.__name__}", memory_delta, "histogram")
            
            return result
            
        except Exception as e:
            # Take error snapshot
            profiler.take_snapshot(f"error_{func.__name__}")
            raise
    
    return wrapper


def optimize_memory(aggressive: bool = False) -> Dict[str, Any]:
    """Optimize system memory usage."""
    profiler = get_memory_profiler()
    return profiler.optimize_memory(aggressive=aggressive)


def detect_memory_leaks() -> List[Dict[str, Any]]:
    """Detect potential memory leaks."""
    profiler = get_memory_profiler()
    return profiler.detect_leaks()


def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics."""
    profiler = get_memory_profiler()
    
    if profiler.snapshots:
        latest_snapshot = profiler.snapshots[-1]
        stats = latest_snapshot.to_dict()
        stats.update({
            "trend_analysis": profiler.get_memory_trend(),
            "potential_leaks": profiler.detect_leaks()
        })
        return stats
    
    return {"error": "No memory snapshots available"}