"""
Database and storage optimization for AI Hardware Co-Design Playground.

This module provides:
- Advanced connection pooling with auto-scaling
- Database sharding and partitioning
- Read replicas and write scaling
- Query optimization and caching
- Data archiving and lifecycle management
- Storage optimization and compression
"""

import time
import threading
import asyncio
import uuid
import json
import pickle
import hashlib
import sqlite3
import os
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import logging

# Database drivers
try:
    import psycopg2
    import psycopg2.pool
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import pymongo
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool, StaticPool
    from sqlalchemy.orm import sessionmaker
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Compression libraries
try:
    import lz4.frame
    import zstd
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False

from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


class ShardingStrategy(Enum):
    """Sharding strategies."""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    DIRECTORY_BASED = "directory_based"
    CONSISTENT_HASH = "consistent_hash"


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    ssl_mode: Optional[str] = None
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        if self.db_type == DatabaseType.POSTGRESQL:
            ssl_part = f"?sslmode={self.ssl_mode}" if self.ssl_mode else ""
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_part}"
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class QueryStats:
    """Query performance statistics."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_executed: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def record_execution(self, execution_time: float, success: bool = True) -> None:
        """Record query execution."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.avg_execution_time = self.total_execution_time / self.execution_count
        self.last_executed = time.time()
        
        if not success:
            self.error_count += 1


class ConnectionPool:
    """Advanced database connection pool with monitoring and auto-scaling."""
    
    def __init__(self, config: DatabaseConfig, enable_monitoring: bool = True):
        self.config = config
        self.enable_monitoring = enable_monitoring
        
        # Connection management
        self._connections = deque()
        self._in_use = set()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Pool statistics
        self.stats = {
            'created_connections': 0,
            'destroyed_connections': 0,
            'active_connections': 0,
            'pool_size': 0,
            'acquisitions': 0,
            'releases': 0,
            'timeouts': 0,
            'errors': 0,
            'total_wait_time': 0.0
        }
        
        # Auto-scaling
        self.min_size = max(1, config.pool_size // 4)
        self.max_size = config.pool_size + config.max_overflow
        self.target_utilization = 0.7
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.last_scale_time = 0.0
        self.scale_cooldown = 60.0
        
        # Initialize pool
        self._initialize_pool()
        
        # Start monitoring
        if enable_monitoring:
            self._start_monitoring()
        
        logger.info(f"Initialized connection pool for {config.db_type.value} with {config.pool_size} connections")
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        with self._lock:
            for _ in range(self.min_size):
                conn = self._create_connection()
                if conn:
                    self._connections.append((conn, time.time()))
    
    def _create_connection(self) -> Optional[Any]:
        """Create new database connection."""
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL and HAS_POSTGRESQL:
                conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password
                )
            elif self.config.db_type == DatabaseType.SQLITE:
                conn = sqlite3.connect(self.config.database, check_same_thread=False)
            elif self.config.db_type == DatabaseType.MONGODB and HAS_MONGODB:
                conn = pymongo.MongoClient(
                    host=self.config.host,
                    port=self.config.port,
                    username=self.config.username,
                    password=self.config.password
                )[self.config.database]
            elif self.config.db_type == DatabaseType.REDIS and HAS_REDIS:
                conn = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    db=int(self.config.database)
                )
            else:
                raise ValueError(f"Unsupported database type or missing driver: {self.config.db_type}")
            
            with self._lock:
                self.stats['created_connections'] += 1
            
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            with self._lock:
                self.stats['errors'] += 1
            return None
    
    def _destroy_connection(self, conn: Any) -> None:
        """Destroy database connection."""
        try:
            if hasattr(conn, 'close'):
                conn.close()
            
            with self._lock:
                self.stats['destroyed_connections'] += 1
                
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """Get connection from pool with context manager."""
        start_time = time.time()
        conn = None
        
        try:
            conn = self.acquire_connection(timeout)
            yield conn
        finally:
            if conn:
                self.release_connection(conn)
                
            # Record wait time
            wait_time = time.time() - start_time
            with self._lock:
                self.stats['total_wait_time'] += wait_time
    
    def acquire_connection(self, timeout: Optional[float] = None) -> Any:
        """Acquire connection from pool."""
        if timeout is None:
            timeout = self.config.pool_timeout
        
        start_time = time.time()
        
        with self._condition:
            # Try to get existing connection
            while True:
                if self._connections:
                    conn, last_used = self._connections.popleft()
                    
                    # Check if connection is still valid
                    if self._is_connection_valid(conn, last_used):
                        self._in_use.add(conn)
                        self.stats['acquisitions'] += 1
                        self.stats['active_connections'] = len(self._in_use)
                        return conn
                    else:
                        self._destroy_connection(conn)
                        continue
                
                # Create new connection if under limit
                if len(self._in_use) + len(self._connections) < self.max_size:
                    conn = self._create_connection()
                    if conn:
                        self._in_use.add(conn)
                        self.stats['acquisitions'] += 1
                        self.stats['active_connections'] = len(self._in_use)
                        return conn
                
                # Wait for connection to become available
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.stats['timeouts'] += 1
                    raise TimeoutError(f"Failed to acquire connection within {timeout}s")
                
                remaining_timeout = timeout - elapsed
                if not self._condition.wait(timeout=remaining_timeout):
                    self.stats['timeouts'] += 1
                    raise TimeoutError(f"Failed to acquire connection within {timeout}s")
    
    def release_connection(self, conn: Any) -> None:
        """Release connection back to pool."""
        with self._condition:
            if conn in self._in_use:
                self._in_use.remove(conn)
                
                # Return to pool if under max size and connection is valid
                if (len(self._connections) < self.config.pool_size and 
                    self._is_connection_valid(conn, time.time())):
                    self._connections.append((conn, time.time()))
                else:
                    self._destroy_connection(conn)
                
                self.stats['releases'] += 1
                self.stats['active_connections'] = len(self._in_use)
                self._condition.notify()
    
    def _is_connection_valid(self, conn: Any, last_used: float) -> bool:
        """Check if connection is still valid."""
        # Check recycle time
        if time.time() - last_used > self.config.pool_recycle:
            return False
        
        # Database-specific validation
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL:
                # Check if connection is closed
                return conn.closed == 0
            elif self.config.db_type == DatabaseType.SQLITE:
                # SQLite connections don't really go stale
                return True
            elif self.config.db_type == DatabaseType.MONGODB:
                # MongoDB connections have built-in health checking
                return True
            elif self.config.db_type == DatabaseType.REDIS:
                # Test Redis connection with ping
                conn.ping()
                return True
            
            return True
            
        except Exception:
            return False
    
    def _start_monitoring(self) -> None:
        """Start connection pool monitoring."""
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ConnectionPoolMonitor"
        )
        monitor_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Connection pool monitoring loop."""
        while True:
            try:
                self._cleanup_stale_connections()
                self._auto_scale_pool()
                self._record_metrics()
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection pool monitoring error: {e}")
                time.sleep(30)
    
    def _cleanup_stale_connections(self) -> None:
        """Clean up stale connections."""
        with self._lock:
            active_connections = []
            current_time = time.time()
            
            for conn, last_used in self._connections:
                if self._is_connection_valid(conn, last_used):
                    active_connections.append((conn, last_used))
                else:
                    self._destroy_connection(conn)
            
            self._connections = deque(active_connections)
    
    def _auto_scale_pool(self) -> None:
        """Auto-scale connection pool based on usage."""
        current_time = time.time()
        
        # Respect cooldown
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        with self._lock:
            total_connections = len(self._connections) + len(self._in_use)
            utilization = len(self._in_use) / max(total_connections, 1)
            
            # Scale up if high utilization
            if (utilization > self.scale_up_threshold and 
                total_connections < self.max_size):
                
                new_connections = min(2, self.max_size - total_connections)
                for _ in range(new_connections):
                    conn = self._create_connection()
                    if conn:
                        self._connections.append((conn, current_time))
                
                self.last_scale_time = current_time
                logger.info(f"Scaled up connection pool by {new_connections} connections")
            
            # Scale down if low utilization
            elif (utilization < self.scale_down_threshold and 
                  total_connections > self.min_size):
                
                connections_to_remove = min(
                    len(self._connections),
                    total_connections - self.min_size,
                    2
                )
                
                for _ in range(connections_to_remove):
                    if self._connections:
                        conn, _ = self._connections.popleft()
                        self._destroy_connection(conn)
                
                self.last_scale_time = current_time
                logger.info(f"Scaled down connection pool by {connections_to_remove} connections")
    
    def _record_metrics(self) -> None:
        """Record pool metrics."""
        with self._lock:
            record_metric("db_pool_active_connections", self.stats['active_connections'], "gauge")
            record_metric("db_pool_size", len(self._connections), "gauge")
            record_metric("db_pool_acquisitions", self.stats['acquisitions'], "counter")
            record_metric("db_pool_timeouts", self.stats['timeouts'], "counter")
            record_metric("db_pool_errors", self.stats['errors'], "counter")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            utilization = self.stats['active_connections'] / max(len(self._connections) + len(self._in_use), 1)
            avg_wait_time = self.stats['total_wait_time'] / max(self.stats['acquisitions'], 1)
            
            return {
                **self.stats,
                'pool_size': len(self._connections),
                'utilization': utilization,
                'avg_wait_time_ms': avg_wait_time * 1000,
                'config': {
                    'min_size': self.min_size,
                    'max_size': self.max_size,
                    'target_utilization': self.target_utilization
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown connection pool."""
        with self._lock:
            # Close all connections
            while self._connections:
                conn, _ = self._connections.popleft()
                self._destroy_connection(conn)
            
            for conn in list(self._in_use):
                self._destroy_connection(conn)
            
            self._in_use.clear()
        
        logger.info("Connection pool shutdown complete")


class QueryCache:
    """Query result cache with intelligent invalidation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        self._cache = {}
        self._access_times = {}
        self._ttl_times = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        
        logger.info(f"Initialized QueryCache with max_size={max_size}")
    
    def get(self, query_key: str) -> Optional[Any]:
        """Get cached query result."""
        with self._lock:
            current_time = time.time()
            
            if query_key not in self._cache:
                self.stats['misses'] += 1
                return None
            
            # Check TTL
            if (query_key in self._ttl_times and 
                current_time > self._ttl_times[query_key]):
                del self._cache[query_key]
                del self._ttl_times[query_key]
                if query_key in self._access_times:
                    del self._access_times[query_key]
                self.stats['misses'] += 1
                return None
            
            # Update access time
            self._access_times[query_key] = current_time
            self.stats['hits'] += 1
            
            return self._cache[query_key]
    
    def put(self, query_key: str, result: Any, ttl: Optional[float] = None) -> None:
        """Cache query result."""
        with self._lock:
            current_time = time.time()
            
            # Make room if necessary
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store result
            self._cache[query_key] = result
            self._access_times[query_key] = current_time
            
            # Set TTL
            actual_ttl = ttl if ttl is not None else self.default_ttl
            if actual_ttl > 0:
                self._ttl_times[query_key] = current_time + actual_ttl
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cached queries."""
        with self._lock:
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._access_times.clear()
                self._ttl_times.clear()
            else:
                # Invalidate matching patterns
                keys_to_remove = [
                    key for key in self._cache.keys()
                    if pattern in key
                ]
                count = len(keys_to_remove)
                
                for key in keys_to_remove:
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
                    if key in self._ttl_times:
                        del self._ttl_times[key]
            
            self.stats['invalidations'] += count
            return count
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        # Find LRU key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        if lru_key in self._ttl_times:
            del self._ttl_times[lru_key]
        
        self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'size': len(self._cache),
                'hit_rate': hit_rate
            }


class ShardingManager:
    """Database sharding manager."""
    
    def __init__(self, strategy: ShardingStrategy = ShardingStrategy.HASH_BASED):
        self.strategy = strategy
        self.shards = {}
        self.shard_weights = {}
        self.directory = {}  # For directory-based sharding
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ShardingManager with strategy: {strategy.value}")
    
    def add_shard(self, shard_id: str, config: DatabaseConfig, weight: int = 100) -> None:
        """Add database shard."""
        with self._lock:
            pool = ConnectionPool(config)
            self.shards[shard_id] = pool
            self.shard_weights[shard_id] = weight
        
        logger.info(f"Added shard {shard_id} with weight {weight}")
    
    def remove_shard(self, shard_id: str) -> None:
        """Remove database shard."""
        with self._lock:
            if shard_id in self.shards:
                self.shards[shard_id].shutdown()
                del self.shards[shard_id]
                del self.shard_weights[shard_id]
        
        logger.info(f"Removed shard {shard_id}")
    
    def get_shard(self, key: str) -> Optional[ConnectionPool]:
        """Get shard for given key."""
        if not self.shards:
            return None
        
        with self._lock:
            if self.strategy == ShardingStrategy.HASH_BASED:
                return self._hash_based_shard(key)
            elif self.strategy == ShardingStrategy.RANGE_BASED:
                return self._range_based_shard(key)
            elif self.strategy == ShardingStrategy.DIRECTORY_BASED:
                return self._directory_based_shard(key)
            elif self.strategy == ShardingStrategy.CONSISTENT_HASH:
                return self._consistent_hash_shard(key)
            else:
                # Default to first shard
                return next(iter(self.shards.values()))
    
    def _hash_based_shard(self, key: str) -> ConnectionPool:
        """Hash-based sharding."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % len(self.shards)
        shard_id = list(self.shards.keys())[shard_index]
        return self.shards[shard_id]
    
    def _range_based_shard(self, key: str) -> ConnectionPool:
        """Range-based sharding."""
        # Simple range-based on first character
        if not key:
            return next(iter(self.shards.values()))
        
        first_char = key[0].lower()
        if 'a' <= first_char <= 'h':
            shard_index = 0
        elif 'i' <= first_char <= 'p':
            shard_index = 1
        else:
            shard_index = 2
        
        shard_index = shard_index % len(self.shards)
        shard_id = list(self.shards.keys())[shard_index]
        return self.shards[shard_id]
    
    def _directory_based_shard(self, key: str) -> ConnectionPool:
        """Directory-based sharding."""
        if key in self.directory:
            shard_id = self.directory[key]
            if shard_id in self.shards:
                return self.shards[shard_id]
        
        # Fallback to hash-based
        return self._hash_based_shard(key)
    
    def _consistent_hash_shard(self, key: str) -> ConnectionPool:
        """Consistent hash-based sharding."""
        # Simplified consistent hashing
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find the next shard in the ring
        sorted_shards = sorted(self.shards.keys())
        for shard_id in sorted_shards:
            shard_hash = int(hashlib.md5(shard_id.encode()).hexdigest(), 16)
            if hash_value <= shard_hash:
                return self.shards[shard_id]
        
        # Wrap around to first shard
        return self.shards[sorted_shards[0]]
    
    def set_directory_mapping(self, key: str, shard_id: str) -> None:
        """Set directory mapping for key."""
        with self._lock:
            self.directory[key] = shard_id
    
    def get_shard_stats(self) -> Dict[str, Any]:
        """Get sharding statistics."""
        with self._lock:
            stats = {
                'strategy': self.strategy.value,
                'shard_count': len(self.shards),
                'shards': {}
            }
            
            for shard_id, pool in self.shards.items():
                stats['shards'][shard_id] = {
                    'weight': self.shard_weights.get(shard_id, 100),
                    'pool_stats': pool.get_stats()
                }
            
            return stats


class DataArchiver:
    """Data archiving and lifecycle management."""
    
    def __init__(self, compression: CompressionType = CompressionType.LZ4):
        self.compression = compression
        self.archive_rules = []
        self._lock = threading.RLock()
        
        # Archive statistics
        self.stats = {
            'archived_records': 0,
            'archived_size_bytes': 0,
            'compression_ratio': 0.0,
            'archive_operations': 0
        }
        
        logger.info(f"Initialized DataArchiver with compression: {compression.value}")
    
    def add_archive_rule(self, table_name: str, condition: str, 
                        retention_days: int, archive_table: str) -> None:
        """Add data archiving rule."""
        rule = {
            'table_name': table_name,
            'condition': condition,
            'retention_days': retention_days,
            'archive_table': archive_table,
            'created_at': time.time()
        }
        
        with self._lock:
            self.archive_rules.append(rule)
        
        logger.info(f"Added archive rule for table {table_name}")
    
    def compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """Compress data using configured compression."""
        if self.compression == CompressionType.NONE:
            return data, 1.0
        
        original_size = len(data)
        
        try:
            if self.compression == CompressionType.LZ4 and HAS_COMPRESSION:
                compressed = lz4.frame.compress(data)
            elif self.compression == CompressionType.ZSTD and HAS_COMPRESSION:
                compressed = zstd.compress(data)
            elif self.compression == CompressionType.GZIP:
                import gzip
                compressed = gzip.compress(data)
            else:
                # Fallback to gzip
                import gzip
                compressed = gzip.compress(data)
            
            compression_ratio = len(compressed) / original_size
            return compressed, compression_ratio
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data, 1.0
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data."""
        if self.compression == CompressionType.NONE:
            return compressed_data
        
        try:
            if self.compression == CompressionType.LZ4 and HAS_COMPRESSION:
                return lz4.frame.decompress(compressed_data)
            elif self.compression == CompressionType.ZSTD and HAS_COMPRESSION:
                return zstd.decompress(compressed_data)
            elif self.compression == CompressionType.GZIP:
                import gzip
                return gzip.decompress(compressed_data)
            else:
                # Fallback to gzip
                import gzip
                return gzip.decompress(compressed_data)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compressed_data
    
    def archive_old_data(self, pool: ConnectionPool) -> Dict[str, int]:
        """Archive old data based on rules."""
        archive_results = {}
        
        for rule in self.archive_rules:
            try:
                archived_count = self._archive_table_data(pool, rule)
                archive_results[rule['table_name']] = archived_count
                
                with self._lock:
                    self.stats['archive_operations'] += 1
                    self.stats['archived_records'] += archived_count
                
            except Exception as e:
                logger.error(f"Archive operation failed for {rule['table_name']}: {e}")
                archive_results[rule['table_name']] = 0
        
        return archive_results
    
    def _archive_table_data(self, pool: ConnectionPool, rule: Dict[str, Any]) -> int:
        """Archive data for specific table."""
        with pool.get_connection() as conn:
            if pool.config.db_type == DatabaseType.POSTGRESQL:
                return self._archive_postgresql_data(conn, rule)
            elif pool.config.db_type == DatabaseType.SQLITE:
                return self._archive_sqlite_data(conn, rule)
            else:
                logger.warning(f"Archiving not implemented for {pool.config.db_type}")
                return 0
    
    def _archive_postgresql_data(self, conn, rule: Dict[str, Any]) -> int:
        """Archive PostgreSQL data."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated archiving logic
        
        cursor = conn.cursor()
        
        # Create archive table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {rule['archive_table']} 
            (LIKE {rule['table_name']} INCLUDING ALL)
        """)
        
        # Move old data to archive table
        cutoff_date = time.time() - (rule['retention_days'] * 24 * 3600)
        
        cursor.execute(f"""
            INSERT INTO {rule['archive_table']}
            SELECT * FROM {rule['table_name']}
            WHERE {rule['condition']} AND created_at < %s
        """, (cutoff_date,))
        
        archived_count = cursor.rowcount
        
        # Delete archived data from main table
        cursor.execute(f"""
            DELETE FROM {rule['table_name']}
            WHERE {rule['condition']} AND created_at < %s
        """, (cutoff_date,))
        
        conn.commit()
        return archived_count
    
    def _archive_sqlite_data(self, conn, rule: Dict[str, Any]) -> int:
        """Archive SQLite data."""
        cursor = conn.cursor()
        
        # Create archive table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {rule['archive_table']} 
            AS SELECT * FROM {rule['table_name']} WHERE 0
        """)
        
        # Move old data to archive table
        cutoff_date = time.time() - (rule['retention_days'] * 24 * 3600)
        
        cursor.execute(f"""
            INSERT INTO {rule['archive_table']}
            SELECT * FROM {rule['table_name']}
            WHERE {rule['condition']} AND created_at < ?
        """, (cutoff_date,))
        
        archived_count = cursor.rowcount
        
        # Delete archived data from main table
        cursor.execute(f"""
            DELETE FROM {rule['table_name']}
            WHERE {rule['condition']} AND created_at < ?
        """, (cutoff_date,))
        
        conn.commit()
        return archived_count
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archiving statistics."""
        with self._lock:
            return {
                **self.stats,
                'compression_type': self.compression.value,
                'active_rules': len(self.archive_rules)
            }


class DatabaseOptimizer:
    """Central database optimization manager."""
    
    def __init__(self):
        self.connection_pools = {}
        self.query_cache = QueryCache()
        self.sharding_manager = ShardingManager()
        self.data_archiver = DataArchiver()
        self.query_stats = {}
        self._lock = threading.RLock()
        
        # Performance monitoring
        self.slow_query_threshold = 1.0  # seconds
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("Initialized DatabaseOptimizer")
    
    def add_database(self, name: str, config: DatabaseConfig) -> None:
        """Add database connection pool."""
        with self._lock:
            pool = ConnectionPool(config)
            self.connection_pools[name] = pool
        
        logger.info(f"Added database '{name}' ({config.db_type.value})")
    
    def get_connection(self, database_name: str):
        """Get database connection."""
        with self._lock:
            if database_name not in self.connection_pools:
                raise ValueError(f"Database '{database_name}' not configured")
            
            return self.connection_pools[database_name].get_connection()
    
    def execute_query(self, database_name: str, query: str, params: Optional[Tuple] = None,
                     cache_ttl: Optional[float] = None, enable_cache: bool = True) -> Any:
        """Execute query with caching and monitoring."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(database_name, query, params)
        
        # Try cache first
        if enable_cache:
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                self._record_query_stats(cache_key, query, 0.0, True, cache_hit=True)
                return cached_result
        
        # Execute query
        try:
            with self.get_connection(database_name) as conn:
                if hasattr(conn, 'cursor'):
                    cursor = conn.cursor()
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    if query.strip().upper().startswith('SELECT'):
                        result = cursor.fetchall()
                    else:
                        result = cursor.rowcount
                        conn.commit()
                else:
                    # Handle non-SQL databases
                    result = self._execute_nosql_query(conn, query, params)
                
                execution_time = time.time() - start_time
                
                # Cache result if it's a SELECT query
                if enable_cache and query.strip().upper().startswith('SELECT'):
                    self.query_cache.put(cache_key, result, cache_ttl)
                
                # Record statistics
                self._record_query_stats(cache_key, query, execution_time, True, cache_hit=False)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_stats(cache_key, query, execution_time, False, cache_hit=False)
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _execute_nosql_query(self, conn, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute NoSQL query."""
        # This would need to be implemented based on the specific NoSQL database
        # For now, return a placeholder
        return {"status": "executed", "query": query}
    
    def _generate_cache_key(self, database_name: str, query: str, params: Optional[Tuple] = None) -> str:
        """Generate cache key for query."""
        key_data = f"{database_name}:{query}"
        if params:
            key_data += f":{str(params)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _record_query_stats(self, cache_key: str, query: str, execution_time: float,
                           success: bool, cache_hit: bool) -> None:
        """Record query execution statistics."""
        with self._lock:
            if cache_key not in self.query_stats:
                self.query_stats[cache_key] = QueryStats(cache_key, query)
            
            stats = self.query_stats[cache_key]
            
            if cache_hit:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1
                stats.record_execution(execution_time, success)
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
    
    def start_monitoring(self) -> None:
        """Start database monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="DatabaseMonitor"
        )
        self.monitor_thread.start()
        logger.info("Started database monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop database monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped database monitoring")
    
    def _monitoring_loop(self) -> None:
        """Database monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._perform_maintenance()
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Database monitoring error: {e}")
                time.sleep(60)
    
    def _collect_metrics(self) -> None:
        """Collect database metrics."""
        for name, pool in self.connection_pools.items():
            stats = pool.get_stats()
            
            record_metric(f"db_{name}_active_connections", stats['active_connections'], "gauge")
            record_metric(f"db_{name}_pool_utilization", stats['utilization'], "gauge")
            record_metric(f"db_{name}_avg_wait_time", stats['avg_wait_time_ms'], "gauge")
    
    def _perform_maintenance(self) -> None:
        """Perform database maintenance tasks."""
        # Archive old data
        for name, pool in self.connection_pools.items():
            try:
                archive_results = self.data_archiver.archive_old_data(pool)
                if any(count > 0 for count in archive_results.values()):
                    logger.info(f"Archived data for {name}: {archive_results}")
            except Exception as e:
                logger.error(f"Archiving failed for {name}: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            # Query performance
            slow_queries = [
                stats for stats in self.query_stats.values()
                if stats.avg_execution_time > self.slow_query_threshold
            ]
            
            # Connection pool stats
            pool_stats = {}
            for name, pool in self.connection_pools.items():
                pool_stats[name] = pool.get_stats()
            
            return {
                'query_cache': self.query_cache.get_stats(),
                'connection_pools': pool_stats,
                'sharding': self.sharding_manager.get_shard_stats(),
                'archiving': self.data_archiver.get_archive_stats(),
                'query_performance': {
                    'total_queries': len(self.query_stats),
                    'slow_queries': len(slow_queries),
                    'slow_query_details': [
                        {
                            'query': stats.query_text[:100] + '...' if len(stats.query_text) > 100 else stats.query_text,
                            'avg_execution_time': stats.avg_execution_time,
                            'execution_count': stats.execution_count,
                            'error_rate': stats.error_count / max(stats.execution_count, 1)
                        }
                        for stats in slow_queries[:10]  # Top 10 slow queries
                    ]
                }
            }
    
    def optimize_queries(self) -> List[str]:
        """Generate query optimization recommendations."""
        recommendations = []
        
        with self._lock:
            for stats in self.query_stats.values():
                if stats.execution_count < 10:
                    continue  # Need more data
                
                # High execution time
                if stats.avg_execution_time > self.slow_query_threshold:
                    recommendations.append(
                        f"Query '{stats.query_text[:50]}...' is slow (avg: {stats.avg_execution_time:.2f}s). "
                        "Consider adding indexes or optimizing query structure."
                    )
                
                # High error rate
                error_rate = stats.error_count / stats.execution_count
                if error_rate > 0.1:
                    recommendations.append(
                        f"Query '{stats.query_text[:50]}...' has high error rate ({error_rate:.1%}). "
                        "Review query logic and error handling."
                    )
                
                # Low cache hit rate
                total_requests = stats.cache_hits + stats.cache_misses
                if total_requests > 50 and stats.cache_hits / total_requests < 0.3:
                    recommendations.append(
                        f"Query '{stats.query_text[:50]}...' has low cache hit rate. "
                        "Consider increasing cache TTL or reviewing query pattern."
                    )
        
        return recommendations
    
    def shutdown(self) -> None:
        """Shutdown database optimizer."""
        logger.info("Shutting down DatabaseOptimizer")
        
        self.stop_monitoring()
        
        # Shutdown all connection pools
        with self._lock:
            for pool in self.connection_pools.values():
                pool.shutdown()
            self.connection_pools.clear()
        
        logger.info("DatabaseOptimizer shutdown complete")


# Global database optimizer instance
_global_db_optimizer: Optional[DatabaseOptimizer] = None


def get_database_optimizer() -> DatabaseOptimizer:
    """Get or create global database optimizer."""
    global _global_db_optimizer
    
    if _global_db_optimizer is None:
        _global_db_optimizer = DatabaseOptimizer()
    
    return _global_db_optimizer


def shutdown_database_optimizer() -> None:
    """Shutdown global database optimizer."""
    global _global_db_optimizer
    
    if _global_db_optimizer:
        _global_db_optimizer.shutdown()
        _global_db_optimizer = None