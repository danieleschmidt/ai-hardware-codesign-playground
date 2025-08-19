"""
CDN and edge computing capabilities for AI Hardware Co-Design Playground.

This module provides:
- Edge caching and content delivery
- Edge computing for low-latency processing
- Geographically distributed deployments
- Edge analytics and monitoring
- Edge-to-cloud synchronization
- Mobile and IoT optimizations
"""

import time
import threading
import asyncio
import uuid
import json
import pickle
import hashlib
import gzip
import os
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import socket
import urllib.parse
from urllib.parse import urlparse
# Optional dependency with fallback
try:
    import requests
except ImportError:
    requests = None
from pathlib import Path

# Geographic and network libraries
try:
    import geoip2.database
    import geoip2.errors
    HAS_GEOIP = True
except ImportError:
    HAS_GEOIP = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class EdgeRegion(Enum):
    """Edge computing regions."""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    ASIA_SOUTHEAST = "asia-southeast"
    AUSTRALIA = "australia"
    SOUTH_AMERICA = "south-america"
    AFRICA = "africa"
    MIDDLE_EAST = "middle-east"


class ContentType(Enum):
    """Content types for edge caching."""
    STATIC = "static"           # CSS, JS, images
    DYNAMIC = "dynamic"         # API responses
    STREAMING = "streaming"     # Video, audio
    MODEL = "model"            # ML models
    DATA = "data"              # Datasets
    COMPUTE = "compute"        # Computation results


class CachePolicy(Enum):
    """Cache policies for edge content."""
    NO_CACHE = "no-cache"
    CACHE_FOREVER = "cache-forever"
    TTL_BASED = "ttl-based"
    LRU = "lru"
    SMART = "smart"            # ML-based caching
    GEOGRAPHIC = "geographic"   # Location-based caching


@dataclass
class EdgeLocation:
    """Edge computing location definition."""
    location_id: str
    region: EdgeRegion
    latitude: float
    longitude: float
    city: str
    country: str
    providers: List[str] = field(default_factory=list)
    capacity_score: float = 1.0
    latency_score: float = 1.0
    cost_score: float = 1.0
    active: bool = True
    
    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate distance to coordinates (simplified)."""
        if not HAS_NUMPY:
            # Simplified distance calculation
            return abs(self.latitude - lat) + abs(self.longitude - lon)
        
        # Haversine formula for great circle distance
        lat1, lon1 = np.radians([self.latitude, self.longitude])
        lat2, lon2 = np.radians([lat, lon])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius = 6371
        return earth_radius * c


@dataclass
class EdgeRequest:
    """Edge request with metadata."""
    request_id: str
    client_ip: str
    user_agent: str
    url: str
    method: str
    headers: Dict[str, str]
    body: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    client_location: Optional[Tuple[float, float]] = None
    edge_location: Optional[str] = None
    cache_status: str = "miss"
    processing_time_ms: float = 0.0
    response_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'client_ip': self.client_ip,
            'user_agent': self.user_agent,
            'url': self.url,
            'method': self.method,
            'headers': self.headers,
            'timestamp': self.timestamp,
            'client_location': self.client_location,
            'edge_location': self.edge_location,
            'cache_status': self.cache_status,
            'processing_time_ms': self.processing_time_ms,
            'response_size_bytes': self.response_size_bytes
        }


@dataclass
class CachedContent:
    """Cached content at edge location."""
    content_id: str
    content_type: ContentType
    data: bytes
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    compression_ratio: float = 1.0
    geographic_scope: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.data)
    
    def is_expired(self) -> bool:
        """Check if content is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1


class GeolocationService:
    """Geolocation service for edge routing."""
    
    def __init__(self, geoip_db_path: Optional[str] = None):
        self.geoip_reader = None
        self.fallback_locations = {}
        
        # Initialize GeoIP2 if available
        if HAS_GEOIP and geoip_db_path and os.path.exists(geoip_db_path):
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
                logger.info("Initialized GeoIP2 database")
            except Exception as e:
                logger.warning(f"Failed to initialize GeoIP2: {e}")
        
        # Default edge locations
        self.edge_locations = {
            "us-east-1": EdgeLocation("us-east-1", EdgeRegion.US_EAST, 39.0458, -76.6413, "Baltimore", "US"),
            "us-west-1": EdgeLocation("us-west-1", EdgeRegion.US_WEST, 37.7749, -122.4194, "San Francisco", "US"),
            "eu-west-1": EdgeLocation("eu-west-1", EdgeRegion.EU_WEST, 53.3498, -6.2603, "Dublin", "IE"),
            "eu-central-1": EdgeLocation("eu-central-1", EdgeRegion.EU_CENTRAL, 50.1109, 8.6821, "Frankfurt", "DE"),
            "ap-southeast-1": EdgeLocation("ap-southeast-1", EdgeRegion.ASIA_SOUTHEAST, 1.3521, 103.8198, "Singapore", "SG"),
            "ap-northeast-1": EdgeLocation("ap-northeast-1", EdgeRegion.ASIA_PACIFIC, 35.6762, 139.6503, "Tokyo", "JP"),
        }
        
        logger.info(f"Initialized GeolocationService with {len(self.edge_locations)} edge locations")
    
    def get_client_location(self, ip_address: str) -> Optional[Tuple[float, float, str, str]]:
        """Get client location from IP address."""
        # Handle localhost and private IPs
        if ip_address in ['127.0.0.1', 'localhost', '::1'] or ip_address.startswith('192.168.') or ip_address.startswith('10.'):
            return (37.7749, -122.4194, "San Francisco", "US")  # Default for local testing
        
        # Try GeoIP2 lookup
        if self.geoip_reader:
            try:
                response = self.geoip_reader.city(ip_address)
                return (
                    float(response.location.latitude) if response.location.latitude else 0.0,
                    float(response.location.longitude) if response.location.longitude else 0.0,
                    response.city.name or "Unknown",
                    response.country.iso_code or "Unknown"
                )
            except geoip2.errors.AddressNotFoundError:
                logger.debug(f"IP address {ip_address} not found in GeoIP database")
            except Exception as e:
                logger.warning(f"GeoIP lookup failed for {ip_address}: {e}")
        
        # Fallback location based on IP prefix
        ip_prefix = '.'.join(ip_address.split('.')[:2])
        if ip_prefix in self.fallback_locations:
            return self.fallback_locations[ip_prefix]
        
        # Default fallback
        return (37.7749, -122.4194, "San Francisco", "US")
    
    def find_nearest_edge(self, client_lat: float, client_lon: float) -> EdgeLocation:
        """Find nearest edge location to client."""
        min_distance = float('inf')
        nearest_edge = None
        
        for edge in self.edge_locations.values():
            if not edge.active:
                continue
            
            distance = edge.distance_to(client_lat, client_lon)
            
            # Factor in capacity and latency scores
            weighted_distance = distance / (edge.capacity_score * edge.latency_score)
            
            if weighted_distance < min_distance:
                min_distance = weighted_distance
                nearest_edge = edge
        
        return nearest_edge or list(self.edge_locations.values())[0]
    
    def add_edge_location(self, location: EdgeLocation) -> None:
        """Add new edge location."""
        self.edge_locations[location.location_id] = location
        logger.info(f"Added edge location: {location.location_id}")
    
    def update_edge_scores(self, location_id: str, capacity: float, latency: float, cost: float) -> None:
        """Update edge location performance scores."""
        if location_id in self.edge_locations:
            edge = self.edge_locations[location_id]
            edge.capacity_score = capacity
            edge.latency_score = latency
            edge.cost_score = cost
            logger.debug(f"Updated scores for {location_id}: capacity={capacity}, latency={latency}, cost={cost}")


class EdgeCache:
    """High-performance edge cache with intelligent policies."""
    
    def __init__(self, location_id: str, max_size_mb: float = 1000.0, 
                 default_ttl: float = 3600.0):
        self.location_id = location_id
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        
        # Cache storage
        self._cache = {}
        self._access_order = deque()  # For LRU
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0,
            'total_requests': 0,
            'compression_savings': 0
        }
        
        # Content policies
        self.content_policies = {
            ContentType.STATIC: CachePolicy.CACHE_FOREVER,
            ContentType.DYNAMIC: CachePolicy.TTL_BASED,
            ContentType.STREAMING: CachePolicy.LRU,
            ContentType.MODEL: CachePolicy.SMART,
            ContentType.DATA: CachePolicy.GEOGRAPHIC,
            ContentType.COMPUTE: CachePolicy.TTL_BASED
        }
        
        # Geographic restrictions
        self.geographic_rules = {}
        
        logger.info(f"Initialized EdgeCache for {location_id} with {max_size_mb}MB capacity")
    
    def get(self, content_id: str, client_location: Optional[Tuple[float, float]] = None) -> Optional[CachedContent]:
        """Get content from edge cache."""
        with self._lock:
            self.stats['total_requests'] += 1
            
            if content_id not in self._cache:
                self.stats['misses'] += 1
                return None
            
            content = self._cache[content_id]
            
            # Check expiration
            if content.is_expired():
                self._remove_content(content_id)
                self.stats['misses'] += 1
                return None
            
            # Check geographic restrictions
            if not self._check_geographic_access(content, client_location):
                self.stats['misses'] += 1
                return None
            
            # Update access info
            content.touch()
            self._update_access_order(content_id)
            
            self.stats['hits'] += 1
            return content
    
    def put(self, content: CachedContent, force: bool = False) -> bool:
        """Put content in edge cache."""
        with self._lock:
            # Check if content should be cached based on policy
            if not force and not self._should_cache(content):
                return False
            
            # Compress content if beneficial
            compressed_content = self._compress_content(content)
            
            # Make room if necessary
            if not self._make_room(compressed_content.size_bytes):
                return False
            
            # Store content
            self._cache[compressed_content.content_id] = compressed_content
            self._access_order.append(compressed_content.content_id)
            self.stats['size_bytes'] += compressed_content.size_bytes
            
            logger.debug(f"Cached content {compressed_content.content_id} ({compressed_content.size_bytes} bytes)")
            return True
    
    def invalidate(self, content_id: str) -> bool:
        """Invalidate specific content."""
        with self._lock:
            if content_id in self._cache:
                self._remove_content(content_id)
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate content matching pattern."""
        with self._lock:
            matching_ids = [
                content_id for content_id in self._cache.keys()
                if pattern in content_id
            ]
            
            for content_id in matching_ids:
                self._remove_content(content_id)
            
            return len(matching_ids)
    
    def _should_cache(self, content: CachedContent) -> bool:
        """Determine if content should be cached based on policy."""
        policy = self.content_policies.get(content.content_type, CachePolicy.TTL_BASED)
        
        if policy == CachePolicy.NO_CACHE:
            return False
        elif policy == CachePolicy.CACHE_FOREVER:
            return True
        elif policy == CachePolicy.TTL_BASED:
            return content.ttl_seconds is not None and content.ttl_seconds > 0
        elif policy == CachePolicy.LRU:
            return content.size_bytes < self.max_size_bytes * 0.1  # Only cache small items
        elif policy == CachePolicy.SMART:
            return self._smart_cache_decision(content)
        elif policy == CachePolicy.GEOGRAPHIC:
            return content.geographic_scope is not None
        
        return True
    
    def _smart_cache_decision(self, content: CachedContent) -> bool:
        """ML-based cache decision (simplified)."""
        # In a real implementation, this would use ML models
        # For now, use simple heuristics
        
        # Cache frequently accessed content
        if content.access_count > 5:
            return True
        
        # Cache recent content
        if time.time() - content.created_at < 3600:  # 1 hour
            return True
        
        # Cache based on content type
        if content.content_type in [ContentType.MODEL, ContentType.STATIC]:
            return True
        
        return False
    
    def _compress_content(self, content: CachedContent) -> CachedContent:
        """Compress content if beneficial."""
        if len(content.data) < 1024:  # Don't compress small content
            return content
        
        try:
            compressed_data = gzip.compress(content.data)
            compression_ratio = len(compressed_data) / len(content.data)
            
            if compression_ratio < 0.9:  # Only use if significant compression
                new_content = CachedContent(
                    content_id=content.content_id,
                    content_type=content.content_type,
                    data=compressed_data,
                    metadata={**content.metadata, 'compressed': True},
                    created_at=content.created_at,
                    last_accessed=content.last_accessed,
                    access_count=content.access_count,
                    ttl_seconds=content.ttl_seconds,
                    size_bytes=len(compressed_data),
                    compression_ratio=compression_ratio,
                    geographic_scope=content.geographic_scope
                )
                
                self.stats['compression_savings'] += len(content.data) - len(compressed_data)
                return new_content
        
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        
        return content
    
    def _decompress_content(self, content: CachedContent) -> bytes:
        """Decompress content if compressed."""
        if content.metadata.get('compressed', False):
            try:
                return gzip.decompress(content.data)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                return content.data
        
        return content.data
    
    def _make_room(self, needed_bytes: int) -> bool:
        """Make room in cache for new content."""
        # Check if we have enough space
        if self.stats['size_bytes'] + needed_bytes <= self.max_size_bytes:
            return True
        
        # Evict content using LRU policy
        bytes_to_free = (self.stats['size_bytes'] + needed_bytes) - self.max_size_bytes
        bytes_freed = 0
        
        while bytes_freed < bytes_to_free and self._access_order:
            content_id = self._access_order.popleft()
            if content_id in self._cache:
                content = self._cache[content_id]
                bytes_freed += content.size_bytes
                self._remove_content(content_id, update_order=False)
                self.stats['evictions'] += 1
        
        return bytes_freed >= bytes_to_free
    
    def _remove_content(self, content_id: str, update_order: bool = True) -> None:
        """Remove content from cache."""
        if content_id in self._cache:
            content = self._cache[content_id]
            self.stats['size_bytes'] -= content.size_bytes
            del self._cache[content_id]
            
            if update_order and content_id in self._access_order:
                self._access_order.remove(content_id)
    
    def _update_access_order(self, content_id: str) -> None:
        """Update LRU access order."""
        if content_id in self._access_order:
            self._access_order.remove(content_id)
        self._access_order.append(content_id)
    
    def _check_geographic_access(self, content: CachedContent, 
                                client_location: Optional[Tuple[float, float]]) -> bool:
        """Check if client can access content based on geographic rules."""
        if not content.geographic_scope:
            return True
        
        if not client_location:
            return True  # Allow if location unknown
        
        # Simple geographic check (in practice, this would be more sophisticated)
        client_lat, client_lon = client_location
        
        for region in content.geographic_scope:
            if region == "global":
                return True
            # Add more sophisticated geographic checking here
        
        return True
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self.stats['hits'] / max(self.stats['total_requests'], 1)
            utilization = self.stats['size_bytes'] / self.max_size_bytes
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'utilization': utilization,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'current_size_mb': self.stats['size_bytes'] / (1024 * 1024),
                'item_count': len(self._cache)
            }


class EdgeCompute:
    """Edge computing service for low-latency processing."""
    
    def __init__(self, location_id: str):
        self.location_id = location_id
        self.functions = {}
        self.compute_stats = defaultdict(int)
        self._lock = threading.RLock()
        
        # Resource limits
        self.max_cpu_time = 30.0  # seconds
        self.max_memory_mb = 512.0
        self.max_concurrent_jobs = 10
        
        # Job management
        self.active_jobs = {}
        self.job_history = deque(maxlen=1000)
        
        logger.info(f"Initialized EdgeCompute for {location_id}")
    
    def register_function(self, function_name: str, function: Callable, 
                         cpu_limit: float = 10.0, memory_limit: float = 128.0) -> None:
        """Register edge function."""
        with self._lock:
            self.functions[function_name] = {
                'function': function,
                'cpu_limit': min(cpu_limit, self.max_cpu_time),
                'memory_limit': min(memory_limit, self.max_memory_mb),
                'registered_at': time.time(),
                'invocations': 0,
                'total_duration': 0.0,
                'error_count': 0
            }
        
        logger.info(f"Registered edge function: {function_name}")
    
    def invoke_function(self, function_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Invoke edge function."""
        if function_name not in self.functions:
            return {
                'success': False,
                'error': f"Function '{function_name}' not found",
                'execution_time': 0.0
            }
        
        job_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check resource limits
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return {
                'success': False,
                'error': 'Resource limit exceeded - too many concurrent jobs',
                'execution_time': 0.0
            }
        
        try:
            func_info = self.functions[function_name]
            
            # Record job start
            with self._lock:
                self.active_jobs[job_id] = {
                    'function_name': function_name,
                    'start_time': start_time,
                    'args': args,
                    'kwargs': kwargs
                }
            
            # Execute function with timeout
            result = self._execute_with_limits(
                func_info['function'],
                func_info['cpu_limit'],
                func_info['memory_limit'],
                *args,
                **kwargs
            )
            
            execution_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                func_info['invocations'] += 1
                func_info['total_duration'] += execution_time
                self.compute_stats[f"{function_name}_invocations"] += 1
                self.compute_stats[f"{function_name}_total_time"] += execution_time
                
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            
            # Record job completion
            self.job_history.append({
                'job_id': job_id,
                'function_name': function_name,
                'execution_time': execution_time,
                'success': True,
                'timestamp': time.time()
            })
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'job_id': job_id
            }
            
        except TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Function execution timed out after {execution_time:.2f}s"
            
            with self._lock:
                func_info['error_count'] += 1
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            
            self.job_history.append({
                'job_id': job_id,
                'function_name': function_name,
                'execution_time': execution_time,
                'success': False,
                'error': error_msg,
                'timestamp': time.time()
            })
            
            return {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            with self._lock:
                func_info['error_count'] += 1
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            
            self.job_history.append({
                'job_id': job_id,
                'function_name': function_name,
                'execution_time': execution_time,
                'success': False,
                'error': error_msg,
                'timestamp': time.time()
            })
            
            return {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time
            }
    
    def _execute_with_limits(self, function: Callable, cpu_limit: float, 
                           memory_limit: float, *args, **kwargs) -> Any:
        """Execute function with resource limits."""
        # In a real implementation, this would use process isolation
        # For now, just execute with basic timeout
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {cpu_limit}s")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(cpu_limit))
        
        try:
            result = function(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        except Exception:
            signal.alarm(0)  # Cancel timeout
            raise
    
    def get_function_stats(self) -> Dict[str, Any]:
        """Get edge function statistics."""
        with self._lock:
            function_stats = {}
            
            for name, info in self.functions.items():
                avg_duration = (
                    info['total_duration'] / info['invocations']
                    if info['invocations'] > 0 else 0.0
                )
                
                error_rate = (
                    info['error_count'] / info['invocations']
                    if info['invocations'] > 0 else 0.0
                )
                
                function_stats[name] = {
                    'invocations': info['invocations'],
                    'avg_duration_ms': avg_duration * 1000,
                    'error_rate': error_rate,
                    'cpu_limit': info['cpu_limit'],
                    'memory_limit': info['memory_limit']
                }
            
            return {
                'functions': function_stats,
                'active_jobs': len(self.active_jobs),
                'total_jobs_processed': len(self.job_history),
                'resource_utilization': len(self.active_jobs) / self.max_concurrent_jobs
            }


class EdgeNode:
    """Complete edge node with caching and computing capabilities."""
    
    def __init__(self, location_id: str, region: EdgeRegion, 
                 cache_size_mb: float = 1000.0):
        self.location_id = location_id
        self.region = region
        
        # Core components
        self.cache = EdgeCache(location_id, cache_size_mb)
        self.compute = EdgeCompute(location_id)
        
        # Request handling
        self.request_stats = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        self.error_rates = deque(maxlen=1000)
        
        # Health monitoring
        self.health_status = "healthy"
        self.last_health_check = time.time()
        
        logger.info(f"Initialized EdgeNode {location_id} in region {region.value}")
    
    def handle_request(self, request: EdgeRequest) -> Dict[str, Any]:
        """Handle incoming request at edge."""
        start_time = time.time()
        
        try:
            # Update request statistics
            self.request_stats['total_requests'] += 1
            self.request_stats[f"method_{request.method}"] += 1
            
            # Try cache first for GET requests
            if request.method == "GET":
                cached_response = self._handle_cached_request(request)
                if cached_response:
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    self.error_rates.append(0)  # Success
                    return cached_response
            
            # Handle compute requests
            if self._is_compute_request(request):
                compute_response = self._handle_compute_request(request)
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self.error_rates.append(0 if compute_response.get('success', False) else 1)
                return compute_response
            
            # Forward to origin if not cacheable or computable
            origin_response = self._forward_to_origin(request)
            
            # Cache response if appropriate
            if origin_response.get('cacheable', False):
                self._cache_response(request, origin_response)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.error_rates.append(0 if origin_response.get('success', True) else 1)
            
            return origin_response
            
        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.error_rates.append(1)  # Error
            
            return {
                'success': False,
                'error': str(e),
                'status_code': 500,
                'response_time': response_time
            }
    
    def _handle_cached_request(self, request: EdgeRequest) -> Optional[Dict[str, Any]]:
        """Handle request from cache."""
        content_id = self._generate_content_id(request)
        cached_content = self.cache.get(content_id, request.client_location)
        
        if cached_content:
            request.cache_status = "hit"
            self.request_stats['cache_hits'] += 1
            
            # Decompress if needed
            response_data = self.cache._decompress_content(cached_content)
            
            return {
                'success': True,
                'data': response_data,
                'status_code': 200,
                'cache_status': 'hit',
                'content_type': cached_content.content_type.value,
                'size_bytes': len(response_data)
            }
        
        request.cache_status = "miss"
        self.request_stats['cache_misses'] += 1
        return None
    
    def _is_compute_request(self, request: EdgeRequest) -> bool:
        """Check if request should be handled by edge compute."""
        # Check URL patterns that indicate compute requests
        compute_patterns = ['/compute/', '/edge-function/', '/process/']
        return any(pattern in request.url for pattern in compute_patterns)
    
    def _handle_compute_request(self, request: EdgeRequest) -> Dict[str, Any]:
        """Handle edge compute request."""
        try:
            # Extract function name from URL
            url_parts = request.url.split('/')
            if 'edge-function' in url_parts:
                func_index = url_parts.index('edge-function') + 1
                if func_index < len(url_parts):
                    function_name = url_parts[func_index]
                else:
                    return {'success': False, 'error': 'Function name not specified'}
            else:\n                return {'success': False, 'error': 'Invalid compute request format'}\n            \n            # Extract parameters\n            params = {}\n            if request.body:\n                try:\n                    params = json.loads(request.body.decode('utf-8'))\n                except Exception:\n                    pass\n            \n            # Invoke edge function\n            result = self.compute.invoke_function(function_name, **params)\n            \n            return {\n                'success': result['success'],\n                'data': result.get('result'),\n                'error': result.get('error'),\n                'execution_time': result['execution_time'],\n                'job_id': result.get('job_id'),\n                'status_code': 200 if result['success'] else 500\n            }\n            \n        except Exception as e:\n            return {\n                'success': False,\n                'error': str(e),\n                'status_code': 500\n            }\n    \n    def _forward_to_origin(self, request: EdgeRequest) -> Dict[str, Any]:\n        \"\"\"Forward request to origin server.\"\"\"\n        # This is a simplified implementation\n        # In practice, this would make actual HTTP requests to origin servers\n        \n        try:\n            # Simulate origin request\n            time.sleep(0.1)  # Simulate network latency\n            \n            # Mock response based on URL\n            if '/api/' in request.url:\n                response_data = {\n                    'message': 'API response from origin',\n                    'timestamp': time.time(),\n                    'request_id': request.request_id\n                }\n                return {\n                    'success': True,\n                    'data': json.dumps(response_data).encode('utf-8'),\n                    'status_code': 200,\n                    'cacheable': True,\n                    'content_type': 'application/json'\n                }\n            else:\n                return {\n                    'success': True,\n                    'data': b'Static content from origin',\n                    'status_code': 200,\n                    'cacheable': True,\n                    'content_type': 'text/plain'\n                }\n                \n        except Exception as e:\n            return {\n                'success': False,\n                'error': str(e),\n                'status_code': 500\n            }\n    \n    def _cache_response(self, request: EdgeRequest, response: Dict[str, Any]) -> None:\n        \"\"\"Cache response data.\"\"\"\n        try:\n            content_id = self._generate_content_id(request)\n            \n            # Determine content type\n            content_type = ContentType.DYNAMIC\n            if '/static/' in request.url:\n                content_type = ContentType.STATIC\n            elif '/api/' in request.url:\n                content_type = ContentType.DYNAMIC\n            elif '/model/' in request.url:\n                content_type = ContentType.MODEL\n            \n            # Create cached content\n            cached_content = CachedContent(\n                content_id=content_id,\n                content_type=content_type,\n                data=response.get('data', b''),\n                metadata={\n                    'url': request.url,\n                    'method': request.method,\n                    'content_type': response.get('content_type', 'text/plain'),\n                    'status_code': response.get('status_code', 200)\n                },\n                created_at=time.time(),\n                last_accessed=time.time(),\n                ttl_seconds=self._get_cache_ttl(content_type)\n            )\n            \n            # Store in cache\n            self.cache.put(cached_content)\n            \n        except Exception as e:\n            logger.warning(f\"Failed to cache response: {e}\")\n    \n    def _generate_content_id(self, request: EdgeRequest) -> str:\n        \"\"\"Generate unique content ID for request.\"\"\"\n        # Include URL, method, and relevant headers\n        key_data = f\"{request.method}:{request.url}\"\n        \n        # Include cache-relevant headers\n        cache_headers = ['accept', 'accept-language', 'authorization']\n        for header in cache_headers:\n            if header in request.headers:\n                key_data += f\":{header}={request.headers[header]}\"\n        \n        return hashlib.md5(key_data.encode()).hexdigest()\n    \n    def _get_cache_ttl(self, content_type: ContentType) -> float:\n        \"\"\"Get cache TTL based on content type.\"\"\"\n        ttl_map = {\n            ContentType.STATIC: 86400.0,      # 24 hours\n            ContentType.DYNAMIC: 300.0,       # 5 minutes\n            ContentType.STREAMING: 60.0,      # 1 minute\n            ContentType.MODEL: 3600.0,        # 1 hour\n            ContentType.DATA: 1800.0,         # 30 minutes\n            ContentType.COMPUTE: 0.0          # No caching\n        }\n        return ttl_map.get(content_type, 300.0)\n    \n    def get_node_stats(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive node statistics.\"\"\"\n        # Calculate performance metrics\n        recent_response_times = list(self.response_times)[-100:]  # Last 100 requests\n        recent_error_rates = list(self.error_rates)[-100:]\n        \n        avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0\n        error_rate = sum(recent_error_rates) / len(recent_error_rates) if recent_error_rates else 0\n        \n        return {\n            'location_id': self.location_id,\n            'region': self.region.value,\n            'health_status': self.health_status,\n            'request_stats': dict(self.request_stats),\n            'performance': {\n                'avg_response_time_ms': avg_response_time * 1000,\n                'error_rate': error_rate,\n                'total_requests': len(self.response_times)\n            },\n            'cache_stats': self.cache.get_cache_stats(),\n            'compute_stats': self.compute.get_function_stats()\n        }\n    \n    def health_check(self) -> Dict[str, Any]:\n        \"\"\"Perform health check.\"\"\"\n        try:\n            current_time = time.time()\n            \n            # Check cache health\n            cache_stats = self.cache.get_cache_stats()\n            cache_healthy = cache_stats['utilization'] < 0.95  # Not over 95% full\n            \n            # Check compute health\n            compute_stats = self.compute.get_function_stats()\n            compute_healthy = compute_stats['resource_utilization'] < 0.9  # Not over 90% utilized\n            \n            # Check recent error rate\n            recent_errors = list(self.error_rates)[-50:]  # Last 50 requests\n            error_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0\n            error_healthy = error_rate < 0.1  # Less than 10% error rate\n            \n            # Overall health\n            overall_healthy = cache_healthy and compute_healthy and error_healthy\n            \n            self.health_status = \"healthy\" if overall_healthy else \"degraded\"\n            self.last_health_check = current_time\n            \n            return {\n                'status': self.health_status,\n                'timestamp': current_time,\n                'checks': {\n                    'cache': cache_healthy,\n                    'compute': compute_healthy,\n                    'error_rate': error_healthy\n                },\n                'metrics': {\n                    'cache_utilization': cache_stats['utilization'],\n                    'compute_utilization': compute_stats['resource_utilization'],\n                    'error_rate': error_rate\n                }\n            }\n            \n        except Exception as e:\n            self.health_status = \"unhealthy\"\n            return {\n                'status': 'unhealthy',\n                'error': str(e),\n                'timestamp': time.time()\n            }\n\n\nclass EdgeCDN:\n    \"\"\"Complete Edge CDN system with global distribution.\"\"\"\n    \n    def __init__(self, geoip_db_path: Optional[str] = None):\n        self.geolocation = GeolocationService(geoip_db_path)\n        self.edge_nodes = {}\n        self.routing_stats = defaultdict(int)\n        self._lock = threading.RLock()\n        \n        # Global statistics\n        self.global_stats = {\n            'total_requests': 0,\n            'cache_hit_rate': 0.0,\n            'avg_response_time': 0.0,\n            'error_rate': 0.0\n        }\n        \n        # Initialize edge nodes for major regions\n        self._initialize_default_nodes()\n        \n        logger.info(\"Initialized EdgeCDN with global distribution\")\n    \n    def _initialize_default_nodes(self) -> None:\n        \"\"\"Initialize default edge nodes.\"\"\"\n        default_configs = [\n            (\"us-east-1\", EdgeRegion.US_EAST, 1000.0),\n            (\"us-west-1\", EdgeRegion.US_WEST, 1000.0),\n            (\"eu-west-1\", EdgeRegion.EU_WEST, 800.0),\n            (\"ap-southeast-1\", EdgeRegion.ASIA_SOUTHEAST, 600.0),\n        ]\n        \n        for location_id, region, cache_size in default_configs:\n            self.add_edge_node(location_id, region, cache_size)\n    \n    def add_edge_node(self, location_id: str, region: EdgeRegion, \n                     cache_size_mb: float = 1000.0) -> None:\n        \"\"\"Add edge node to CDN.\"\"\"\n        with self._lock:\n            node = EdgeNode(location_id, region, cache_size_mb)\n            self.edge_nodes[location_id] = node\n        \n        logger.info(f\"Added edge node: {location_id} ({region.value})\")\n    \n    def route_request(self, request: EdgeRequest) -> Dict[str, Any]:\n        \"\"\"Route request to optimal edge node.\"\"\"\n        start_time = time.time()\n        \n        try:\n            # Get client location\n            if not request.client_location:\n                location_data = self.geolocation.get_client_location(request.client_ip)\n                if location_data:\n                    request.client_location = (location_data[0], location_data[1])\n            \n            # Find optimal edge node\n            edge_node = self._select_edge_node(request)\n            \n            if not edge_node:\n                return {\n                    'success': False,\n                    'error': 'No available edge nodes',\n                    'status_code': 503\n                }\n            \n            # Update routing stats\n            self.routing_stats[f\"routed_to_{edge_node.location_id}\"] += 1\n            self.routing_stats['total_requests'] += 1\n            \n            # Set edge location in request\n            request.edge_location = edge_node.location_id\n            \n            # Handle request at edge\n            response = edge_node.handle_request(request)\n            \n            # Update request metadata\n            response_time = time.time() - start_time\n            request.processing_time_ms = response_time * 1000\n            request.response_size_bytes = response.get('size_bytes', 0)\n            \n            # Update global stats\n            self._update_global_stats(response, response_time)\n            \n            # Add edge metadata to response\n            response['edge_location'] = edge_node.location_id\n            response['edge_region'] = edge_node.region.value\n            response['processing_time_ms'] = request.processing_time_ms\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Request routing failed: {e}\")\n            return {\n                'success': False,\n                'error': str(e),\n                'status_code': 500,\n                'processing_time_ms': (time.time() - start_time) * 1000\n            }\n    \n    def _select_edge_node(self, request: EdgeRequest) -> Optional[EdgeNode]:\n        \"\"\"Select optimal edge node for request.\"\"\"\n        if not self.edge_nodes:\n            return None\n        \n        # If client location is available, use geographic routing\n        if request.client_location:\n            client_lat, client_lon = request.client_location\n            nearest_edge = self.geolocation.find_nearest_edge(client_lat, client_lon)\n            \n            if nearest_edge and nearest_edge.location_id in self.edge_nodes:\n                node = self.edge_nodes[nearest_edge.location_id]\n                if node.health_status in ['healthy', 'degraded']:\n                    return node\n        \n        # Fallback to load-based selection\n        healthy_nodes = [\n            node for node in self.edge_nodes.values()\n            if node.health_status in ['healthy', 'degraded']\n        ]\n        \n        if not healthy_nodes:\n            # Return any available node as last resort\n            return next(iter(self.edge_nodes.values())) if self.edge_nodes else None\n        \n        # Select node with lowest load\n        min_load = float('inf')\n        selected_node = None\n        \n        for node in healthy_nodes:\n            # Calculate load score based on active requests and cache utilization\n            cache_stats = node.cache.get_cache_stats()\n            compute_stats = node.compute.get_function_stats()\n            \n            load_score = (\n                cache_stats['utilization'] * 0.4 +\n                compute_stats['resource_utilization'] * 0.6\n            )\n            \n            if load_score < min_load:\n                min_load = load_score\n                selected_node = node\n        \n        return selected_node\n    \n    def _update_global_stats(self, response: Dict[str, Any], response_time: float) -> None:\n        \"\"\"Update global CDN statistics.\"\"\"\n        with self._lock:\n            self.global_stats['total_requests'] += 1\n            \n            # Update running averages\n            total = self.global_stats['total_requests']\n            \n            # Response time\n            current_avg_rt = self.global_stats['avg_response_time']\n            self.global_stats['avg_response_time'] = (\n                (current_avg_rt * (total - 1) + response_time) / total\n            )\n            \n            # Error rate\n            is_error = not response.get('success', True)\n            current_error_rate = self.global_stats['error_rate']\n            self.global_stats['error_rate'] = (\n                (current_error_rate * (total - 1) + (1 if is_error else 0)) / total\n            )\n            \n            # Cache hit rate (approximate)\n            is_cache_hit = response.get('cache_status') == 'hit'\n            current_hit_rate = self.global_stats['cache_hit_rate']\n            self.global_stats['cache_hit_rate'] = (\n                (current_hit_rate * (total - 1) + (1 if is_cache_hit else 0)) / total\n            )\n    \n    def register_edge_function(self, function_name: str, function: Callable,\n                              locations: Optional[List[str]] = None) -> None:\n        \"\"\"Register function across edge nodes.\"\"\"\n        target_locations = locations or list(self.edge_nodes.keys())\n        \n        for location_id in target_locations:\n            if location_id in self.edge_nodes:\n                self.edge_nodes[location_id].compute.register_function(\n                    function_name, function\n                )\n        \n        logger.info(f\"Registered edge function '{function_name}' at {len(target_locations)} locations\")\n    \n    def invalidate_cache(self, pattern: str, locations: Optional[List[str]] = None) -> Dict[str, int]:\n        \"\"\"Invalidate cache across edge nodes.\"\"\"\n        target_locations = locations or list(self.edge_nodes.keys())\n        results = {}\n        \n        for location_id in target_locations:\n            if location_id in self.edge_nodes:\n                count = self.edge_nodes[location_id].cache.invalidate_pattern(pattern)\n                results[location_id] = count\n        \n        total_invalidated = sum(results.values())\n        logger.info(f\"Invalidated {total_invalidated} cache entries matching '{pattern}'\")\n        \n        return results\n    \n    def get_cdn_stats(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive CDN statistics.\"\"\"\n        node_stats = {}\n        \n        for location_id, node in self.edge_nodes.items():\n            node_stats[location_id] = node.get_node_stats()\n        \n        return {\n            'global_stats': self.global_stats,\n            'routing_stats': dict(self.routing_stats),\n            'edge_nodes': node_stats,\n            'total_edge_nodes': len(self.edge_nodes),\n            'healthy_nodes': sum(\n                1 for node in self.edge_nodes.values()\n                if node.health_status == 'healthy'\n            )\n        }\n    \n    def health_check_all(self) -> Dict[str, Dict[str, Any]]:\n        \"\"\"Perform health check on all edge nodes.\"\"\"\n        results = {}\n        \n        for location_id, node in self.edge_nodes.items():\n            results[location_id] = node.health_check()\n        \n        return results\n\n\n# Global CDN instance\n_global_edge_cdn: Optional[EdgeCDN] = None\n\n\ndef get_edge_cdn(geoip_db_path: Optional[str] = None) -> EdgeCDN:\n    \"\"\"Get or create global edge CDN.\"\"\"\n    global _global_edge_cdn\n    \n    if _global_edge_cdn is None:\n        _global_edge_cdn = EdgeCDN(geoip_db_path)\n    \n    return _global_edge_cdn\n\n\ndef shutdown_edge_cdn() -> None:\n    \"\"\"Shutdown global edge CDN.\"\"\"\n    global _global_edge_cdn\n    \n    if _global_edge_cdn:\n        logger.info(\"Shutting down EdgeCDN\")\n        _global_edge_cdn = None"}, "function_name": "/api/", "cache_status": "miss", "content_type": "application/json"}
            \n            return result\n            \n        except Exception as e:\n            return {'success': False, 'error': str(e), 'status_code': 500}"}, {"old_string": "            elif '/edge-function/' in request.url:\n                function_name = url_parts[func_index]", "new_string": "        if 'edge-function' in url_parts:\n            func_index = url_parts.index('edge-function') + 1\n            if func_index < len(url_parts):\n                function_name = url_parts[func_index]"}]