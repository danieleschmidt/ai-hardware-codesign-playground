"""
Caching and performance optimization for AI Hardware Co-Design Playground.

This module provides intelligent caching mechanisms, performance optimization,
and adaptive systems for the platform.
"""

import time
import hashlib
import json
import pickle
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import threading
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import weakref

from ..utils.logging import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


class AdaptiveCache:
    """Adaptive cache with intelligent eviction and performance monitoring."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        enable_stats: bool = True
    ):
        """
        Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            enable_stats: Whether to collect statistics
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_evictions": 0,
            "memory_evictions": 0,
            "ttl_evictions": 0,
            "total_memory_bytes": 0
        }
        
        # Adaptive features
        self._access_patterns = {}  # Track access patterns
        self._performance_history = []
        
        logger.info(
            "Initialized AdaptiveCache",
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            default_ttl=default_ttl
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._record_miss(key)
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats["ttl_evictions"] += 1
                self._record_miss(key)
                return None
            
            # Update access info and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            self._record_hit(key)
            
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        force: bool = False
    ) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
            force: Force insertion even if over limits
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Estimate if serialization fails
            
            # Check if single item is too large
            if not force and size_bytes > self.max_memory_bytes * 0.5:
                logger.warning(
                    "Item too large for cache",
                    key=key,
                    size_mb=size_bytes / (1024 * 1024),
                    max_mb=self.max_memory_bytes / (1024 * 1024)
                )
                return False
            
            # Use provided TTL or default
            actual_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=actual_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["total_memory_bytes"] -= old_entry.size_bytes
                del self._cache[key]
            
            # Make room if necessary
            self._make_room(size_bytes)
            
            # Add entry
            self._cache[key] = entry
            self._stats["total_memory_bytes"] += size_bytes
            
            logger.debug(
                "Cached item",
                key=key,
                size_bytes=size_bytes,
                ttl=actual_ttl,
                cache_size=len(self._cache)
            )
            
            return True
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was present
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats["total_memory_bytes"] -= entry.size_bytes
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats["total_memory_bytes"] = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "memory_usage_mb": self._stats["total_memory_bytes"] / (1024 * 1024),
                "memory_utilization": (
                    self._stats["total_memory_bytes"] / self.max_memory_bytes
                    if self.max_memory_bytes > 0 else 0.0
                )
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._stats["total_memory_bytes"] -= entry.size_bytes
                del self._cache[key]
                self._stats["ttl_evictions"] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def _make_room(self, needed_bytes: int) -> None:
        """Make room in cache for new entry."""
        # Clean up expired entries first
        self.cleanup_expired()
        
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()
            self._stats["size_evictions"] += 1
        
        # Check memory limit
        while (
            self._stats["total_memory_bytes"] + needed_bytes > self.max_memory_bytes
            and len(self._cache) > 0
        ):
            self._evict_lru()
            self._stats["memory_evictions"] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry (first in OrderedDict)
        key = next(iter(self._cache))
        entry = self._cache[key]
        
        self._stats["total_memory_bytes"] -= entry.size_bytes
        del self._cache[key]
        self._stats["evictions"] += 1
        
        logger.debug("Evicted LRU entry", key=key, size_bytes=entry.size_bytes)
    
    def _record_hit(self, key: str) -> None:
        """Record cache hit."""
        if self.enable_stats:
            self._stats["hits"] += 1
            self._update_access_pattern(key, hit=True)
    
    def _record_miss(self, key: str) -> None:
        """Record cache miss."""
        if self.enable_stats:
            self._stats["misses"] += 1
            self._update_access_pattern(key, hit=False)
    
    def _update_access_pattern(self, key: str, hit: bool) -> None:
        """Update access patterns for adaptive behavior."""
        if key not in self._access_patterns:
            self._access_patterns[key] = {"hits": 0, "misses": 0, "last_access": time.time()}
        
        if hit:
            self._access_patterns[key]["hits"] += 1
        else:
            self._access_patterns[key]["misses"] += 1
        
        self._access_patterns[key]["last_access"] = time.time()


class ModelProfileCache(AdaptiveCache):
    """Specialized cache for model profiles."""
    
    def __init__(self):
        super().__init__(
            max_size=500,
            max_memory_mb=50.0,
            default_ttl=7200.0,  # 2 hours
        )
    
    def get_profile_key(self, model_info: Dict[str, Any], input_shape: Tuple[int, ...]) -> str:
        """Generate cache key for model profile."""
        key_data = {
            "model_path": model_info.get("path", ""),
            "framework": model_info.get("framework", ""),
            "input_shape": input_shape,
            "model_hash": model_info.get("hash", "")
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class AcceleratorCache(AdaptiveCache):
    """Specialized cache for accelerator designs."""
    
    def __init__(self):
        super().__init__(
            max_size=1000,
            max_memory_mb=100.0,
            default_ttl=3600.0,  # 1 hour
        )
    
    def get_design_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for accelerator design."""
        # Remove non-deterministic fields
        cache_config = {
            k: v for k, v in config.items()
            if k not in {"rtl_code", "performance_model", "resource_estimates"}
        }
        
        key_str = json.dumps(cache_config, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class ResultCache:
    """Cache for computation results with intelligent invalidation."""
    
    def __init__(self):
        self.model_cache = ModelProfileCache()
        self.accelerator_cache = AcceleratorCache()
        self.exploration_cache = AdaptiveCache(
            max_size=100,
            max_memory_mb=200.0,
            default_ttl=1800.0  # 30 minutes
        )
        self.optimization_cache = AdaptiveCache(
            max_size=200,
            max_memory_mb=150.0,
            default_ttl=1800.0  # 30 minutes
        )
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "model_profiles": self.model_cache.get_stats(),
            "accelerator_designs": self.accelerator_cache.get_stats(),
            "exploration_results": self.exploration_cache.get_stats(),
            "optimization_results": self.optimization_cache.get_stats(),
        }
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches."""
        return {
            "model_profiles": self.model_cache.cleanup_expired(),
            "accelerator_designs": self.accelerator_cache.cleanup_expired(),
            "exploration_results": self.exploration_cache.cleanup_expired(),
            "optimization_results": self.optimization_cache.cleanup_expired(),
        }
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.model_cache.clear()
        self.accelerator_cache.clear()
        self.exploration_cache.clear()
        self.optimization_cache.clear()


# Global cache instance
_result_cache = ResultCache()


def cached(
    cache_type: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    condition_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use
        ttl: Time-to-live override
        key_func: Function to generate cache key
        condition_func: Function to determine if result should be cached
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Select cache
            if cache_type == "model":
                cache = _result_cache.model_cache
            elif cache_type == "accelerator":
                cache = _result_cache.accelerator_cache
            elif cache_type == "exploration":
                cache = _result_cache.exploration_cache
            elif cache_type == "optimization":
                cache = _result_cache.optimization_cache
            else:
                # Use default cache
                if not hasattr(wrapper, "_default_cache"):
                    wrapper._default_cache = AdaptiveCache()
                cache = wrapper._default_cache
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                key_str = json.dumps(key_data, sort_keys=True, default=str)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Try to get from cache
            with perf_logger.timer(f"cache_lookup_{func.__name__}", cache_key=cache_key):
                result = cache.get(cache_key)
            
            if result is not None:
                logger.debug(
                    "Cache hit",
                    function=func.__name__,
                    cache_key=cache_key,
                    cache_type=cache_type
                )
                return result
            
            # Execute function
            with perf_logger.timer(f"function_execution_{func.__name__}", cache_key=cache_key):
                result = func(*args, **kwargs)
            
            # Check if result should be cached
            should_cache = True
            if condition_func:
                should_cache = condition_func(result, *args, **kwargs)
            
            # Cache result
            if should_cache:
                cache.put(cache_key, result, ttl=ttl)
                logger.debug(
                    "Cached result",
                    function=func.__name__,
                    cache_key=cache_key,
                    cache_type=cache_type
                )
            
            return result
        
        return wrapper
    return decorator


class ResourcePool:
    """Thread-safe resource pool with adaptive scaling."""
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        idle_timeout: float = 300.0  # 5 minutes
    ):
        """
        Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            idle_timeout: Timeout for idle resources
        """
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool = []
        self._in_use = set()
        self._lock = threading.Lock()
        self._stats = {
            "created": 0,
            "destroyed": 0,
            "requests": 0,
            "wait_time_total": 0.0
        }
        
        # Pre-populate pool
        for _ in range(min_size):
            resource = self._create_resource()
            if resource:
                self._pool.append((resource, time.time()))
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """
        Acquire resource from pool.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            Resource instance or None if timeout
        """
        start_time = time.time()
        
        with self._lock:
            self._stats["requests"] += 1
            
            # Try to get existing resource
            while self._pool:
                resource, last_used = self._pool.pop(0)
                
                # Check if resource is still valid
                if self._is_resource_valid(resource, last_used):
                    self._in_use.add(resource)
                    return resource
                else:
                    self._destroy_resource(resource)
            
            # Create new resource if under limit
            if len(self._in_use) < self.max_size:
                resource = self._create_resource()
                if resource:
                    self._in_use.add(resource)
                    return resource
        
        # Wait for resource to become available
        if timeout is None:
            timeout = 30.0  # Default 30 second timeout
        
        end_time = start_time + timeout
        while time.time() < end_time:
            time.sleep(0.1)
            
            with self._lock:
                if self._pool:
                    resource, last_used = self._pool.pop(0)
                    if self._is_resource_valid(resource, last_used):
                        self._in_use.add(resource)
                        wait_time = time.time() - start_time
                        self._stats["wait_time_total"] += wait_time
                        return resource
                    else:
                        self._destroy_resource(resource)
        
        logger.warning("Resource pool timeout", timeout=timeout)
        return None
    
    def release(self, resource: Any) -> None:
        """
        Release resource back to pool.
        
        Args:
            resource: Resource to release
        """
        with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                # Return to pool if under min size or pool is empty
                if len(self._pool) < self.min_size or len(self._pool) == 0:
                    self._pool.append((resource, time.time()))
                else:
                    self._destroy_resource(resource)
    
    def cleanup(self) -> int:
        """Remove idle resources and return count removed."""
        with self._lock:
            current_time = time.time()
            active_pool = []
            cleaned_count = 0
            
            for resource, last_used in self._pool:
                if current_time - last_used > self.idle_timeout:
                    self._destroy_resource(resource)
                    cleaned_count += 1
                else:
                    active_pool.append((resource, last_used))
            
            self._pool = active_pool
            return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            avg_wait_time = (
                self._stats["wait_time_total"] / self._stats["requests"]
                if self._stats["requests"] > 0 else 0.0
            )
            
            return {
                **self._stats,
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "average_wait_time": avg_wait_time,
                "utilization": len(self._in_use) / self.max_size if self.max_size > 0 else 0.0
            }
    
    def _create_resource(self) -> Any:
        """Create new resource."""
        try:
            resource = self.factory()
            self._stats["created"] += 1
            logger.debug("Created pool resource")
            return resource
        except Exception as e:
            logger.error("Failed to create pool resource", exception=e)
            return None
    
    def _destroy_resource(self, resource: Any) -> None:
        """Destroy resource."""
        try:
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "cleanup"):
                resource.cleanup()
            
            self._stats["destroyed"] += 1
            logger.debug("Destroyed pool resource")
        except Exception as e:
            logger.warning("Error destroying pool resource", exception=e)
    
    def _is_resource_valid(self, resource: Any, last_used: float) -> bool:
        """Check if resource is still valid."""
        # Check timeout
        if time.time() - last_used > self.idle_timeout:
            return False
        
        # Check resource-specific validity
        if hasattr(resource, "is_valid"):
            return resource.is_valid()
        
        return True


# Global resource pools
_thread_pool = None
_process_pool = None


def get_thread_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """Get shared thread pool executor."""
    global _thread_pool
    
    if _thread_pool is None:
        if max_workers is None:
            import os
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        _thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="codesign_"
        )
        
        logger.info("Initialized thread pool", max_workers=max_workers)
    
    return _thread_pool


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return _result_cache.get_all_stats()


def cleanup_caches() -> Dict[str, int]:
    """Cleanup expired entries in all caches."""
    return _result_cache.cleanup_all()


def clear_all_caches() -> None:
    """Clear all caches."""
    _result_cache.clear_all()
    logger.info("All caches cleared")