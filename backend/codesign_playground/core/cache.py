"""
Caching and performance optimization for AI Hardware Co-Design Playground.

This module provides intelligent caching mechanisms, performance optimization,
and adaptive systems for the platform.
"""

import time
import hashlib
import json
import pickle
import redis
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import threading
from collections import OrderedDict, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import weakref
import sqlite3
import lz4.frame
import gzip
import zlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import io
from contextlib import contextmanager
import psutil

import logging

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger(f"{__name__}.performance")


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    cache_level: int = 1  # L1, L2, L3 cache levels
    compression_type: Optional[str] = None
    access_pattern: List[float] = field(default_factory=list)
    popularity_score: float = 0.0
    eviction_priority: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        current_time = time.time()
        self.last_access = current_time
        
        # Track access pattern for ML prediction
        self.access_pattern.append(current_time)
        if len(self.access_pattern) > 20:  # Keep only recent accesses
            self.access_pattern = self.access_pattern[-20:]
        
        # Update popularity score
        self._update_popularity_score()
    
    def _update_popularity_score(self) -> None:
        """Update popularity score based on access pattern."""
        current_time = time.time()
        recent_accesses = [t for t in self.access_pattern if current_time - t < 3600]  # Last hour
        self.popularity_score = len(recent_accesses) * 0.1 + self.access_count * 0.01
        
        # Calculate eviction priority (lower = more likely to evict)
        time_since_access = current_time - self.last_access
        self.eviction_priority = self.popularity_score / (1 + time_since_access / 3600)


class MLPredictor:
    """Machine learning predictor for cache behavior."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_history = []
        self.target_history = []
        
    def extract_features(self, entry: CacheEntry) -> np.ndarray:
        """Extract features from cache entry for ML prediction."""
        current_time = time.time()
        
        # Access pattern features
        access_frequency = len(entry.access_pattern)
        if access_frequency > 1:
            access_intervals = np.diff(entry.access_pattern)
            avg_interval = np.mean(access_intervals)
            interval_variance = np.var(access_intervals)
        else:
            avg_interval = 0
            interval_variance = 0
        
        # Time-based features
        age = current_time - entry.timestamp
        time_since_access = current_time - entry.last_access
        
        features = np.array([
            entry.access_count,
            access_frequency,
            avg_interval,
            interval_variance,
            age,
            time_since_access,
            entry.size_bytes,
            entry.popularity_score
        ])
        
        return features
    
    def predict_access_probability(self, entry: CacheEntry) -> float:
        """Predict probability of future access."""
        if not self.trained or len(self.feature_history) < 10:
            return 0.5  # Default probability
        
        features = self.extract_features(entry).reshape(1, -1)
        
        try:
            normalized_features = self.scaler.transform(features)
            probability = self.model.predict(normalized_features)[0]
            return max(0.0, min(1.0, probability))
        except Exception:
            return 0.5
    
    def update_model(self, entry: CacheEntry, was_accessed: bool) -> None:
        """Update ML model with new data."""
        features = self.extract_features(entry)
        target = 1.0 if was_accessed else 0.0
        
        self.feature_history.append(features)
        self.target_history.append(target)
        
        # Keep only recent data
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-1000:]
            self.target_history = self.target_history[-1000:]
        
        # Retrain model periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 10 == 0:
            self._retrain_model()
    
    def _retrain_model(self) -> None:
        """Retrain the ML model."""
        try:
            X = np.array(self.feature_history)
            y = np.array(self.target_history)
            
            # Only retrain if we have enough diverse data
            if len(np.unique(y)) > 1 and len(X) >= 20:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.model.fit(X_scaled, y)
                self.trained = True
        except Exception as e:
            logger.warning(f"ML model retraining failed: {e}")


class MultiLevelCache:
    """Multi-level cache hierarchy (L1, L2, L3) with intelligent data movement."""
    
    def __init__(self, l1_size: int = 100, l2_size: int = 500, l3_size: int = 2000):
        self.l1_cache = {}  # Fast in-memory cache
        self.l2_cache = {}  # Compressed in-memory cache
        self.l3_cache = {}  # Disk-based cache
        
        self.l1_max_size = l1_size
        self.l2_max_size = l2_size
        self.l3_max_size = l3_size
        
        self.cache_hits = {'L1': 0, 'L2': 0, 'L3': 0}
        self.cache_misses = 0
        
        self._setup_l3_storage()
    
    def _setup_l3_storage(self) -> None:
        """Setup disk-based L3 cache."""
        self.l3_db_path = '/tmp/cache_l3.db'
        conn = sqlite3.connect(self.l3_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB,
                metadata TEXT,
                timestamp REAL,
                access_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from multi-level cache hierarchy."""
        # Try L1 cache first
        if key in self.l1_cache:
            self.cache_hits['L1'] += 1
            entry = self.l1_cache[key]
            entry.touch()
            return entry
        
        # Try L2 cache
        if key in self.l2_cache:
            self.cache_hits['L2'] += 1
            entry = self.l2_cache[key]
            entry.touch()
            
            # Promote to L1 if frequently accessed
            if entry.access_count > 5:
                self._promote_to_l1(key, entry)
            
            return entry
        
        # Try L3 cache
        entry = self._get_from_l3(key)
        if entry:
            self.cache_hits['L3'] += 1
            entry.touch()
            
            # Promote based on access pattern
            if entry.access_count > 10:
                self._promote_to_l1(key, entry)
            elif entry.access_count > 3:
                self._promote_to_l2(key, entry)
            
            return entry
        
        self.cache_misses += 1
        return None
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Put value in appropriate cache level."""
        # Start with L1 for new entries
        if len(self.l1_cache) >= self.l1_max_size:
            self._evict_from_l1()
        
        entry.cache_level = 1
        self.l1_cache[key] = entry
    
    def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to L1 cache."""
        if len(self.l1_cache) >= self.l1_max_size:
            self._evict_from_l1()
        
        # Remove from current level
        if key in self.l2_cache:
            del self.l2_cache[key]
        else:
            self._remove_from_l3(key)
        
        entry.cache_level = 1
        self.l1_cache[key] = entry
    
    def _promote_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to L2 cache."""
        if len(self.l2_cache) >= self.l2_max_size:
            self._evict_from_l2()
        
        self._remove_from_l3(key)
        
        # Compress entry for L2
        entry = self._compress_entry(entry)
        entry.cache_level = 2
        self.l2_cache[key] = entry
    
    def _evict_from_l1(self) -> None:
        """Evict least valuable entry from L1."""
        if not self.l1_cache:
            return
        
        # Find entry with lowest eviction priority
        min_priority = float('inf')
        evict_key = None
        
        for key, entry in self.l1_cache.items():
            if entry.eviction_priority < min_priority:
                min_priority = entry.eviction_priority
                evict_key = key
        
        if evict_key:
            entry = self.l1_cache.pop(evict_key)
            # Demote to L2
            if len(self.l2_cache) < self.l2_max_size:
                self._demote_to_l2(evict_key, entry)
            else:
                self._demote_to_l3(evict_key, entry)
    
    def _evict_from_l2(self) -> None:
        """Evict least valuable entry from L2."""
        if not self.l2_cache:
            return
        
        min_priority = float('inf')
        evict_key = None
        
        for key, entry in self.l2_cache.items():
            if entry.eviction_priority < min_priority:
                min_priority = entry.eviction_priority
                evict_key = key
        
        if evict_key:
            entry = self.l2_cache.pop(evict_key)
            self._demote_to_l3(evict_key, entry)
    
    def _demote_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Demote entry to L2 cache."""
        entry = self._compress_entry(entry)
        entry.cache_level = 2
        self.l2_cache[key] = entry
    
    def _demote_to_l3(self, key: str, entry: CacheEntry) -> None:
        """Demote entry to L3 cache."""
        entry = self._compress_entry(entry)
        entry.cache_level = 3
        self._put_to_l3(key, entry)
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry to save space."""
        if entry.compression_type is not None:
            return entry  # Already compressed
        
        try:
            serialized = pickle.dumps(entry.value)
            
            # Choose compression based on size
            if len(serialized) > 10000:  # Large data
                compressed = lz4.frame.compress(serialized)
                compression_type = 'lz4'
            elif len(serialized) > 1000:  # Medium data
                compressed = zlib.compress(serialized)
                compression_type = 'zlib'
            else:
                compressed = serialized  # Small data, no compression
                compression_type = None
            
            # Create new entry with compressed data
            new_entry = CacheEntry(
                value=compressed,
                timestamp=entry.timestamp,
                access_count=entry.access_count,
                last_access=entry.last_access,
                ttl=entry.ttl,
                size_bytes=len(compressed),
                cache_level=entry.cache_level,
                compression_type=compression_type,
                access_pattern=entry.access_pattern,
                popularity_score=entry.popularity_score,
                eviction_priority=entry.eviction_priority
            )
            
            return new_entry
        except Exception:
            return entry  # Return original if compression fails
    
    def _decompress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Decompress cache entry."""
        if entry.compression_type is None:
            return entry
        
        try:
            if entry.compression_type == 'lz4':
                decompressed = lz4.frame.decompress(entry.value)
            elif entry.compression_type == 'zlib':
                decompressed = zlib.decompress(entry.value)
            else:
                decompressed = entry.value
            
            original_value = pickle.loads(decompressed)
            
            # Create new entry with decompressed data
            new_entry = CacheEntry(
                value=original_value,
                timestamp=entry.timestamp,
                access_count=entry.access_count,
                last_access=entry.last_access,
                ttl=entry.ttl,
                size_bytes=len(pickle.dumps(original_value)),
                cache_level=entry.cache_level,
                compression_type=None,
                access_pattern=entry.access_pattern,
                popularity_score=entry.popularity_score,
                eviction_priority=entry.eviction_priority
            )
            
            return new_entry
        except Exception:
            return entry
    
    def _get_from_l3(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L3 disk cache."""
        try:
            conn = sqlite3.connect(self.l3_db_path)
            cursor = conn.execute(
                'SELECT value, metadata, timestamp, access_count FROM cache_entries WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value_blob, metadata_json, timestamp, access_count = row
                metadata = json.loads(metadata_json)
                
                entry = CacheEntry(
                    value=pickle.loads(value_blob),
                    timestamp=timestamp,
                    access_count=access_count,
                    cache_level=3,
                    **metadata
                )
                
                return self._decompress_entry(entry)
            
            return None
        except Exception:
            return None
    
    def _put_to_l3(self, key: str, entry: CacheEntry) -> None:
        """Put entry to L3 disk cache."""
        try:
            compressed_entry = self._compress_entry(entry)
            
            metadata = {
                'last_access': compressed_entry.last_access,
                'ttl': compressed_entry.ttl,
                'size_bytes': compressed_entry.size_bytes,
                'compression_type': compressed_entry.compression_type,
                'access_pattern': compressed_entry.access_pattern,
                'popularity_score': compressed_entry.popularity_score,
                'eviction_priority': compressed_entry.eviction_priority
            }
            
            conn = sqlite3.connect(self.l3_db_path)
            conn.execute(
                'INSERT OR REPLACE INTO cache_entries (key, value, metadata, timestamp, access_count) VALUES (?, ?, ?, ?, ?)',
                (key, pickle.dumps(compressed_entry.value), json.dumps(metadata), 
                 compressed_entry.timestamp, compressed_entry.access_count)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
    
    def _remove_from_l3(self, key: str) -> None:
        """Remove entry from L3 disk cache."""
        try:
            conn = sqlite3.connect(self.l3_db_path)
            conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            conn.commit()
            conn.close()
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(self.cache_hits.values())
        total_requests = total_hits + self.cache_misses
        
        return {
            'L1': {
                'size': len(self.l1_cache),
                'max_size': self.l1_max_size,
                'hits': self.cache_hits['L1']
            },
            'L2': {
                'size': len(self.l2_cache),
                'max_size': self.l2_max_size,
                'hits': self.cache_hits['L2']
            },
            'L3': {
                'hits': self.cache_hits['L3']
            },
            'overall': {
                'hit_rate': total_hits / total_requests if total_requests > 0 else 0,
                'total_requests': total_requests,
                'misses': self.cache_misses
            }
        }


class AdaptiveCache:
    """Advanced adaptive cache with ML prediction and multi-level hierarchy."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        enable_stats: bool = True,
        enable_ml_prediction: bool = True,
        enable_distributed: bool = False,
        redis_url: Optional[str] = None
    ):
        """
        Initialize advanced adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            enable_stats: Whether to collect statistics
            enable_ml_prediction: Whether to use ML for prediction
            enable_distributed: Whether to use distributed caching
            redis_url: Redis URL for distributed caching
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        self.enable_ml_prediction = enable_ml_prediction
        self.enable_distributed = enable_distributed
        
        # Multi-level cache hierarchy
        self.multi_level_cache = MultiLevelCache(
            l1_size=max_size // 10,  # 10% for L1
            l2_size=max_size // 2,   # 50% for L2
            l3_size=max_size        # 100% for L3
        )
        
        # Traditional cache for backwards compatibility
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # ML predictor
        self.ml_predictor = MLPredictor() if enable_ml_prediction else None
        
        # Distributed cache
        self.redis_client = None
        if enable_distributed and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for distributed caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        # Enhanced statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_evictions": 0,
            "memory_evictions": 0,
            "ttl_evictions": 0,
            "ml_predictions": 0,
            "ml_correct_predictions": 0,
            "total_memory_bytes": 0,
            "compression_ratio": 0.0,
            "prefetch_hits": 0,
            "cache_warming_events": 0
        }
        
        # Adaptive features
        self._access_patterns = {}  # Track access patterns
        self._performance_history = []
        self._prefetch_queue = []
        self._warming_enabled = True
        
        # Background threads
        self._cleanup_thread = None
        self._prefetch_thread = None
        self._warming_thread = None
        self._shutdown_event = threading.Event()
        
        self._start_background_tasks()
        
        logger.info(
            "Initialized AdaptiveCache",
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            default_ttl=default_ttl,
            ml_prediction=enable_ml_prediction,
            distributed=enable_distributed
        )
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for cache maintenance."""
        # Cleanup thread for expired entries
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="cache_cleanup"
        )
        self._cleanup_thread.start()
        
        # Prefetch thread for predictive caching
        if self.enable_ml_prediction:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                daemon=True,
                name="cache_prefetch"
            )
            self._prefetch_thread.start()
        
        # Cache warming thread
        self._warming_thread = threading.Thread(
            target=self._warming_loop,
            daemon=True,
            name="cache_warming"
        )
        self._warming_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while not self._shutdown_event.is_set():
            try:
                self.cleanup_expired()
                self._shutdown_event.wait(timeout=60)  # Clean every minute
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                self._shutdown_event.wait(timeout=60)
    
    def _prefetch_loop(self) -> None:
        """Background prefetching based on ML predictions."""
        while not self._shutdown_event.is_set():
            try:
                self._process_prefetch_queue()
                self._shutdown_event.wait(timeout=30)  # Process every 30 seconds
            except Exception as e:
                logger.error(f"Cache prefetch error: {e}")
                self._shutdown_event.wait(timeout=30)
    
    def _warming_loop(self) -> None:
        """Background cache warming."""
        while not self._shutdown_event.is_set():
            try:
                if self._warming_enabled:
                    self._warm_cache()
                self._shutdown_event.wait(timeout=300)  # Warm every 5 minutes
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                self._shutdown_event.wait(timeout=300)
    
    def get(self, key: str, use_ml: bool = True, warm_cache: bool = True) -> Optional[Any]:
        """
        Get value from cache with ML prediction and warming.
        
        Args:
            key: Cache key
            use_ml: Whether to use ML predictions
            warm_cache: Whether to trigger cache warming
            
        Returns:
            Cached value or None if not found/expired
        """
        # Try multi-level cache first
        ml_entry = self.multi_level_cache.get(key)
        if ml_entry and not ml_entry.is_expired():
            self._record_hit(key)
            
            # Update ML model if enabled
            if self.enable_ml_prediction and self.ml_predictor:
                self.ml_predictor.update_model(ml_entry, was_accessed=True)
            
            # Trigger prefetching if ML enabled
            if use_ml and self.enable_ml_prediction:
                self._schedule_prefetch(key, ml_entry)
            
            return ml_entry.value
        
        # Try distributed cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"cache:{key}")
                if cached_data:
                    entry_data = pickle.loads(cached_data)
                    entry = CacheEntry(**entry_data)
                    
                    if not entry.is_expired():
                        # Promote to local cache
                        self.multi_level_cache.put(key, entry)
                        self._record_hit(key, cache_type="distributed")
                        return entry.value
            except Exception as e:
                logger.warning(f"Distributed cache error: {e}")
        
        # Try traditional cache for backwards compatibility
        with self._lock:
            if key not in self._cache:
                # Trigger cache warming if enabled
                if warm_cache and self._warming_enabled:
                    self._trigger_warming(key)
                
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
            
            # Update ML model
            if self.enable_ml_prediction and self.ml_predictor:
                self.ml_predictor.update_model(entry, was_accessed=True)
                self._stats["ml_predictions"] += 1
            
            self._record_hit(key)
            
            # Schedule prefetching
            if use_ml and self.enable_ml_prediction:
                self._schedule_prefetch(key, entry)
            
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        force: bool = False,
        cache_level: str = "auto",
        enable_compression: bool = True
    ) -> bool:
        """
        Put value in cache with intelligent placement.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
            force: Force insertion even if over limits
            cache_level: Cache level ("L1", "L2", "L3", "auto", "distributed")
            enable_compression: Whether to enable compression
            
        Returns:
            True if successfully cached
        """
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
        
        # Determine optimal cache level
        if cache_level == "auto":
            cache_level = self._determine_optimal_cache_level(key, entry, size_bytes)
        
        # Place in appropriate cache level
        if cache_level == "distributed" and self.redis_client:
            success = self._put_distributed(key, entry)
            if success:
                # Also cache locally for faster access
                self.multi_level_cache.put(key, entry)
                return True
        
        if cache_level in ["L1", "L2", "L3", "auto"]:
            self.multi_level_cache.put(key, entry)
        
        # Also maintain traditional cache for backwards compatibility
        with self._lock:
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
            cache_level=cache_level,
            cache_size=len(self._cache)
        )
        
        return True
    
    def _determine_optimal_cache_level(self, key: str, entry: CacheEntry, size_bytes: int) -> str:
        """Determine optimal cache level for entry."""
        # Small, frequently accessed items go to L1
        if size_bytes < 10000 and entry.access_count > 5:
            return "L1"
        
        # Medium items with moderate access go to L2
        if size_bytes < 100000 and entry.access_count > 1:
            return "L2"
        
        # Large or infrequently accessed items go to L3
        return "L3"
    
    def _put_distributed(self, key: str, entry: CacheEntry) -> bool:
        """Put entry in distributed cache."""
        try:
            entry_data = {
                'value': entry.value,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'last_access': entry.last_access,
                'ttl': entry.ttl,
                'size_bytes': entry.size_bytes,
                'cache_level': entry.cache_level,
                'compression_type': entry.compression_type,
                'access_pattern': entry.access_pattern,
                'popularity_score': entry.popularity_score,
                'eviction_priority': entry.eviction_priority
            }
            
            serialized = pickle.dumps(entry_data)
            
            # Set with TTL
            if entry.ttl:
                self.redis_client.setex(f"cache:{key}", int(entry.ttl), serialized)
            else:
                self.redis_client.set(f"cache:{key}", serialized)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to put in distributed cache: {e}")
            return False
    
    def _schedule_prefetch(self, key: str, entry: CacheEntry) -> None:
        """Schedule prefetching based on access patterns."""
        if not self.enable_ml_prediction or not self.ml_predictor:
            return
        
        # Predict related keys that might be accessed
        related_keys = self._predict_related_keys(key, entry)
        
        for related_key in related_keys:
            if related_key not in self._prefetch_queue:
                self._prefetch_queue.append(related_key)
    
    def _predict_related_keys(self, key: str, entry: CacheEntry) -> List[str]:
        """Predict related keys based on access patterns."""
        related_keys = []
        
        # Simple pattern-based prediction
        if "_" in key:
            parts = key.split("_")
            if len(parts) > 1:
                # Try variations of the key
                base = "_".join(parts[:-1])
                for i in range(5):  # Try 5 variations
                    related_key = f"{base}_{i}"
                    if related_key != key:
                        related_keys.append(related_key)
        
        return related_keys[:3]  # Limit to 3 related keys
    
    def _process_prefetch_queue(self) -> None:
        """Process prefetch queue."""
        if not self._prefetch_queue:
            return
        
        # Process up to 5 prefetch requests
        for _ in range(min(5, len(self._prefetch_queue))):
            if not self._prefetch_queue:
                break
            
            key = self._prefetch_queue.pop(0)
            
            # Check if key is likely to be accessed
            if self.ml_predictor:
                # Create dummy entry for prediction
                dummy_entry = CacheEntry(
                    value=None,
                    timestamp=time.time(),
                    access_count=0
                )
                
                probability = self.ml_predictor.predict_access_probability(dummy_entry)
                
                if probability > 0.7:  # High probability threshold
                    # Trigger cache warming for this key
                    self._trigger_warming(key)
    
    def _warm_cache(self) -> None:
        """Warm cache with frequently accessed data."""
        if not self._warming_enabled:
            return
        
        # Identify popular keys for warming
        popular_keys = self._identify_popular_keys()
        
        for key in popular_keys[:10]:  # Warm up to 10 keys
            # Check if key is already cached
            if key not in self._cache and not self.multi_level_cache.get(key):
                # Try to generate or fetch the data (this would be application-specific)
                self._attempt_cache_warming(key)
        
        self._stats["cache_warming_events"] += 1
    
    def _identify_popular_keys(self) -> List[str]:
        """Identify popular keys based on access patterns."""
        popular_keys = []
        
        # Analyze access patterns
        for key, pattern in self._access_patterns.items():
            if pattern.get("hits", 0) > 5 and time.time() - pattern.get("last_access", 0) < 3600:
                popular_keys.append(key)
        
        # Sort by popularity
        popular_keys.sort(key=lambda k: self._access_patterns.get(k, {}).get("hits", 0), reverse=True)
        
        return popular_keys
    
    def _attempt_cache_warming(self, key: str) -> None:
        """Attempt to warm cache for a specific key."""
        # This is a placeholder - in a real implementation, this would
        # trigger the computation or fetch of the data for this key
        logger.debug(f"Cache warming attempted for key: {key}")
    
    def _trigger_warming(self, key: str) -> None:
        """Trigger cache warming for a specific key."""
        if self._warming_enabled:
            logger.debug(f"Triggering cache warming for key: {key}")
    
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
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            ml_accuracy = 0.0
            if self._stats["ml_predictions"] > 0:
                ml_accuracy = self._stats["ml_correct_predictions"] / self._stats["ml_predictions"]
            
            # Get multi-level cache stats
            ml_stats = self.multi_level_cache.get_stats()
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "memory_usage_mb": self._stats["total_memory_bytes"] / (1024 * 1024),
                "memory_utilization": (
                    self._stats["total_memory_bytes"] / self.max_memory_bytes
                    if self.max_memory_bytes > 0 else 0.0
                ),
                "ml_prediction_accuracy": ml_accuracy,
                "prefetch_queue_size": len(self._prefetch_queue),
                "warming_enabled": self._warming_enabled,
                "multi_level_cache": ml_stats,
                "distributed_enabled": self.redis_client is not None
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
    
    def _record_hit(self, key: str, cache_type: str = "local") -> None:
        """Record cache hit."""
        if self.enable_stats:
            self._stats["hits"] += 1
            if cache_type == "distributed":
                self._stats["distributed_hits"] = self._stats.get("distributed_hits", 0) + 1
            elif cache_type == "prefetch":
                self._stats["prefetch_hits"] += 1
            
            self._update_access_pattern(key, hit=True)
    
    def _record_miss(self, key: str) -> None:
        """Record cache miss."""
        if self.enable_stats:
            self._stats["misses"] += 1
            self._update_access_pattern(key, hit=False)
    
    def enable_warming(self, enabled: bool = True) -> None:
        """Enable or disable cache warming."""
        self._warming_enabled = enabled
        logger.info(f"Cache warming {'enabled' if enabled else 'disabled'}")
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources."""
        logger.info("Shutting down AdaptiveCache")
        
        # Signal shutdown to background threads
        self._shutdown_event.set()
        
        # Wait for threads to complete
        for thread in [self._cleanup_thread, self._prefetch_thread, self._warming_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass
        
        logger.info("AdaptiveCache shutdown complete")
    
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
            start_time = time.time()
            result = cache.get(cache_key)
            cache_lookup_time = time.time() - start_time
            perf_logger.debug(f"Cache lookup for {func.__name__} took {cache_lookup_time:.4f}s", 
                           extra={"cache_key": cache_key})
            
            if result is not None:
                logger.debug(
                    "Cache hit",
                    function=func.__name__,
                    cache_key=cache_key,
                    cache_type=cache_type
                )
                return result
            
            # Execute function
            exec_start_time = time.time()
            result = func(*args, **kwargs)
            exec_time = time.time() - exec_start_time
            perf_logger.debug(f"Function {func.__name__} execution took {exec_time:.4f}s",
                           extra={"cache_key": cache_key})
            
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