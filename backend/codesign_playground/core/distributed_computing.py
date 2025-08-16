"""
Distributed computing and microservices architecture for AI Hardware Co-Design Playground.

This module provides:
- Service mesh architecture with intelligent routing
- Distributed task scheduling and coordination
- Event-driven architecture with message queues
- Service discovery and configuration management
- Cross-service communication optimization
- Fault tolerance and circuit breakers
"""

import time
import threading
import asyncio
import uuid
import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import socket
import requests
from urllib.parse import urljoin
import logging

# Message queue and distributed computing
try:
    import redis
    import pika  # RabbitMQ
    HAS_REDIS = True
    HAS_RABBITMQ = True
except ImportError:
    HAS_REDIS = HAS_RABBITMQ = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    import celery
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False

try:
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class MessageType(Enum):
    """Message types for inter-service communication."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    HEALTH_CHECK = "health_check"
    SERVICE_DISCOVERY = "service_discovery"
    CONFIGURATION_UPDATE = "configuration_update"
    ALERT = "alert"
    METRICS = "metrics"
    EVENT = "event"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    RANDOM = "random"


@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""
    service_id: str
    host: str
    port: int
    protocol: str = "http"
    path: str = "/"
    weight: int = 100
    health_check_path: str = "/health"
    timeout_seconds: float = 30.0
    
    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def full_url(self) -> str:
        return urljoin(self.base_url, self.path)
    
    @property
    def health_check_url(self) -> str:
        return urljoin(self.base_url, self.health_check_path)


@dataclass
class ServiceRegistration:
    """Service registration information."""
    service_id: str
    service_name: str
    version: str
    endpoints: List[ServiceEndpoint]
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: ServiceStatus = ServiceStatus.STARTING
    last_heartbeat: float = field(default_factory=time.time)
    registration_time: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceRegistration':
        endpoints = [ServiceEndpoint(**ep) for ep in data.get('endpoints', [])]
        data['endpoints'] = endpoints
        data['status'] = ServiceStatus(data.get('status', ServiceStatus.UNKNOWN.value))
        return cls(**data)


@dataclass
class DistributedTask:
    """Distributed task definition."""
    task_id: str
    task_type: str
    service_name: str
    function_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    assigned_node: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        return cls(**data)


class ServiceDiscovery:
    """Service discovery and registry."""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client
        self.local_registry = {}
        self.service_cache = {}
        self.cache_ttl = 60  # seconds
        self._lock = threading.RLock()
        
        # Health checking
        self.health_check_interval = 30  # seconds
        self.health_check_thread = None
        self.health_check_active = False
        
        logger.info("Initialized ServiceDiscovery")
    
    def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service."""
        try:
            service_key = f"service:{registration.service_name}:{registration.service_id}"
            
            with self._lock:
                self.local_registry[registration.service_id] = registration
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.hset(
                        "services",
                        service_key,
                        json.dumps(registration.to_dict(), default=str)
                    )
                    self.redis_client.expire("services", 3600)  # 1 hour TTL
                except Exception as e:
                    logger.warning(f"Failed to register service in Redis: {e}")
            
            logger.info(f"Registered service: {registration.service_name} ({registration.service_id})")
            
            # Start health checking if not already active
            if not self.health_check_active:
                self.start_health_checking()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service."""
        try:
            with self._lock:
                if service_id in self.local_registry:
                    registration = self.local_registry.pop(service_id)
                    
                    # Remove from Redis
                    if self.redis_client:
                        try:
                            service_key = f"service:{registration.service_name}:{service_id}"
                            self.redis_client.hdel("services", service_key)
                        except Exception as e:
                            logger.warning(f"Failed to deregister service from Redis: {e}")
                    
                    logger.info(f"Deregistered service: {service_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            return False
    
    def discover_services(self, service_name: str, refresh_cache: bool = False) -> List[ServiceRegistration]:
        """Discover services by name."""
        cache_key = f"discover:{service_name}"
        current_time = time.time()
        
        # Check cache first
        if not refresh_cache and cache_key in self.service_cache:
            cached_data, cache_time = self.service_cache[cache_key]
            if current_time - cache_time < self.cache_ttl:
                return cached_data
        
        services = []
        
        # Get from local registry
        with self._lock:
            for registration in self.local_registry.values():
                if registration.service_name == service_name:
                    services.append(registration)
        
        # Get from Redis
        if self.redis_client:
            try:
                service_data = self.redis_client.hgetall("services")
                for key, data_json in service_data.items():
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    if isinstance(data_json, bytes):
                        data_json = data_json.decode('utf-8')
                    
                    if f"service:{service_name}:" in key:
                        try:
                            data = json.loads(data_json)
                            registration = ServiceRegistration.from_dict(data)
                            
                            # Avoid duplicates
                            if not any(s.service_id == registration.service_id for s in services):
                                services.append(registration)
                        except Exception as e:
                            logger.warning(f"Failed to parse service data: {e}")
            
            except Exception as e:
                logger.warning(f"Failed to discover services from Redis: {e}")
        
        # Cache results
        self.service_cache[cache_key] = (services, current_time)
        
        logger.debug(f"Discovered {len(services)} instances of service '{service_name}'")
        return services
    
    def get_healthy_services(self, service_name: str) -> List[ServiceRegistration]:
        """Get only healthy service instances."""
        all_services = self.discover_services(service_name)
        healthy_services = [
            service for service in all_services
            if service.status == ServiceStatus.HEALTHY
        ]
        
        logger.debug(f"Found {len(healthy_services)} healthy instances of '{service_name}'")
        return healthy_services
    
    def start_health_checking(self) -> None:
        """Start background health checking."""
        if self.health_check_active:
            return
        
        self.health_check_active = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="ServiceHealthCheck"
        )
        self.health_check_thread.start()
        logger.info("Started service health checking")
    
    def stop_health_checking(self) -> None:
        """Stop background health checking."""
        self.health_check_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        logger.info("Stopped service health checking")
    
    def _health_check_loop(self) -> None:
        """Background health checking loop."""
        while self.health_check_active:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all registered services."""
        with self._lock:
            services_to_check = list(self.local_registry.values())
        
        for service in services_to_check:
            for endpoint in service.endpoints:
                try:
                    response = requests.get(
                        endpoint.health_check_url,
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        new_status = ServiceStatus.HEALTHY
                    else:
                        new_status = ServiceStatus.DEGRADED
                        
                except requests.exceptions.RequestException:
                    new_status = ServiceStatus.UNHEALTHY
                except Exception as e:
                    logger.warning(f"Health check error for {service.service_id}: {e}")
                    new_status = ServiceStatus.UNKNOWN
                
                # Update service status
                if service.status != new_status:
                    logger.info(f"Service {service.service_id} status changed: {service.status.value} -> {new_status.value}")
                    service.status = new_status
                    service.last_heartbeat = time.time()
                    
                    # Update in Redis
                    if self.redis_client:
                        try:
                            service_key = f"service:{service.service_name}:{service.service_id}"
                            self.redis_client.hset(
                                "services",
                                service_key,
                                json.dumps(service.to_dict(), default=str)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update service status in Redis: {e}")


class LoadBalancer:
    """Intelligent load balancer for distributed services."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.connection_counts = defaultdict(int)
        self.response_times = defaultdict(deque)
        self.round_robin_counters = defaultdict(int)
        self._lock = threading.RLock()
        
        # Performance tracking
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        logger.info(f"Initialized LoadBalancer with strategy: {strategy.value}")
    
    def select_endpoint(self, service_name: str, endpoints: List[ServiceEndpoint], 
                       request_hash: Optional[str] = None) -> Optional[ServiceEndpoint]:
        """Select best endpoint based on load balancing strategy."""
        if not endpoints:
            return None
        
        # Filter healthy endpoints
        healthy_endpoints = [ep for ep in endpoints if self._is_endpoint_healthy(ep)]
        if not healthy_endpoints:
            # Fall back to all endpoints if none are healthy
            healthy_endpoints = endpoints
        
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(service_name, healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(service_name, healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.HASH_BASED:
                return self._hash_based_selection(healthy_endpoints, request_hash)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(healthy_endpoints)
            else:
                return healthy_endpoints[0]  # Default fallback
    
    def record_request(self, endpoint: ServiceEndpoint, response_time: float, success: bool) -> None:
        """Record request metrics for load balancing decisions."""
        with self._lock:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            
            # Record response time
            if endpoint_key not in self.response_times:
                self.response_times[endpoint_key] = deque(maxlen=100)
            self.response_times[endpoint_key].append(response_time)
            
            # Record request count
            self.request_counts[endpoint_key] += 1
            
            # Record errors
            if not success:
                self.error_counts[endpoint_key] += 1
    
    def start_connection(self, endpoint: ServiceEndpoint) -> None:
        """Record connection start."""
        with self._lock:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            self.connection_counts[endpoint_key] += 1
    
    def end_connection(self, endpoint: ServiceEndpoint) -> None:
        """Record connection end."""
        with self._lock:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            self.connection_counts[endpoint_key] = max(0, self.connection_counts[endpoint_key] - 1)
    
    def _is_endpoint_healthy(self, endpoint: ServiceEndpoint) -> bool:
        """Check if endpoint is considered healthy."""
        endpoint_key = f"{endpoint.host}:{endpoint.port}"
        
        # Check error rate
        total_requests = self.request_counts.get(endpoint_key, 0)
        if total_requests > 10:  # Only check if we have enough data
            error_rate = self.error_counts.get(endpoint_key, 0) / total_requests
            if error_rate > 0.5:  # More than 50% error rate
                return False
        
        return True
    
    def _round_robin_selection(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection."""
        counter = self.round_robin_counters[service_name]
        selected = endpoints[counter % len(endpoints)]
        self.round_robin_counters[service_name] = counter + 1
        return selected
    
    def _least_connections_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection."""
        min_connections = float('inf')
        selected_endpoint = endpoints[0]
        
        for endpoint in endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            connections = self.connection_counts.get(endpoint_key, 0)
            
            if connections < min_connections:
                min_connections = connections
                selected_endpoint = endpoint
        
        return selected_endpoint
    
    def _weighted_round_robin_selection(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin selection."""
        # Build weighted list
        weighted_endpoints = []
        for endpoint in endpoints:
            weighted_endpoints.extend([endpoint] * endpoint.weight)
        
        if not weighted_endpoints:
            return endpoints[0]
        
        counter = self.round_robin_counters[service_name]
        selected = weighted_endpoints[counter % len(weighted_endpoints)]
        self.round_robin_counters[service_name] = counter + 1
        return selected
    
    def _least_response_time_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least average response time selection."""
        min_response_time = float('inf')
        selected_endpoint = endpoints[0]
        
        for endpoint in endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            response_times = self.response_times.get(endpoint_key, [])
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
            else:
                avg_response_time = 0  # Prefer endpoints with no data
            
            if avg_response_time < min_response_time:
                min_response_time = avg_response_time
                selected_endpoint = endpoint
        
        return selected_endpoint
    
    def _hash_based_selection(self, endpoints: List[ServiceEndpoint], request_hash: Optional[str]) -> ServiceEndpoint:
        """Hash-based selection for session affinity."""
        if not request_hash:
            return endpoints[0]
        
        # Use consistent hashing
        hash_value = int(hashlib.md5(request_hash.encode()).hexdigest(), 16)
        index = hash_value % len(endpoints)
        return endpoints[index]
    
    def _random_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random selection."""
        import random
        return random.choice(endpoints)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            stats = {
                "strategy": self.strategy.value,
                "endpoints": {}
            }
            
            all_endpoints = set(self.connection_counts.keys()) | set(self.request_counts.keys())
            
            for endpoint_key in all_endpoints:
                response_times = list(self.response_times.get(endpoint_key, []))
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                total_requests = self.request_counts.get(endpoint_key, 0)
                total_errors = self.error_counts.get(endpoint_key, 0)
                error_rate = total_errors / total_requests if total_requests > 0 else 0
                
                stats["endpoints"][endpoint_key] = {
                    "active_connections": self.connection_counts.get(endpoint_key, 0),
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "error_rate": error_rate,
                    "avg_response_time_ms": avg_response_time * 1000,
                    "recent_response_times": response_times[-10:]  # Last 10
                }
            
            return stats


class MessageQueue:
    """Message queue for inter-service communication."""
    
    def __init__(self, redis_client: Optional[Any] = None, rabbitmq_url: Optional[str] = None):
        self.redis_client = redis_client
        self.rabbitmq_url = rabbitmq_url
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        
        # Local queue fallback
        self.local_queues = defaultdict(queue.Queue)
        self.subscribers = defaultdict(list)
        
        # Initialize connections
        self._init_rabbitmq()
        
        logger.info("Initialized MessageQueue")
    
    def _init_rabbitmq(self) -> None:
        """Initialize RabbitMQ connection."""
        if not HAS_RABBITMQ or not self.rabbitmq_url:
            return
        
        try:
            import pika
            self.rabbitmq_connection = pika.BlockingConnection(
                pika.URLParameters(self.rabbitmq_url)
            )
            self.rabbitmq_channel = self.rabbitmq_connection.channel()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.warning(f"Failed to connect to RabbitMQ: {e}")
    
    def publish(self, topic: str, message: Dict[str, Any], priority: int = 5) -> bool:
        """Publish message to topic."""
        try:
            message_data = {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "topic": topic,
                "priority": priority,
                "payload": message
            }
            
            # Try RabbitMQ first
            if self.rabbitmq_channel:
                try:
                    self.rabbitmq_channel.queue_declare(queue=topic, durable=True)
                    self.rabbitmq_channel.basic_publish(
                        exchange='',
                        routing_key=topic,
                        body=json.dumps(message_data),
                        properties=pika.BasicProperties(
                            priority=priority,
                            delivery_mode=2  # Make message persistent
                        )
                    )
                    record_metric("message_published", 1, "counter", {"topic": topic, "transport": "rabbitmq"})
                    return True
                except Exception as e:
                    logger.warning(f"RabbitMQ publish failed: {e}")
            
            # Try Redis
            if self.redis_client:
                try:
                    self.redis_client.lpush(f"queue:{topic}", json.dumps(message_data))
                    record_metric("message_published", 1, "counter", {"topic": topic, "transport": "redis"})
                    return True
                except Exception as e:
                    logger.warning(f"Redis publish failed: {e}")
            
            # Fallback to local queue
            self.local_queues[topic].put(message_data)
            record_metric("message_published", 1, "counter", {"topic": topic, "transport": "local"})
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to topic messages."""
        try:
            self.subscribers[topic].append(callback)
            
            # Start consuming from RabbitMQ
            if self.rabbitmq_channel:
                try:
                    self.rabbitmq_channel.queue_declare(queue=topic, durable=True)
                    
                    def wrapper(ch, method, properties, body):
                        try:
                            message_data = json.loads(body)
                            callback(message_data)
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")
                            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                    
                    self.rabbitmq_channel.basic_consume(
                        queue=topic,
                        on_message_callback=wrapper
                    )
                    
                    # Start consuming in background thread
                    consume_thread = threading.Thread(
                        target=self.rabbitmq_channel.start_consuming,
                        daemon=True
                    )
                    consume_thread.start()
                    
                    logger.info(f"Subscribed to RabbitMQ topic: {topic}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"RabbitMQ subscription failed: {e}")
            
            # Start local queue consumer
            consumer_thread = threading.Thread(
                target=self._consume_local_queue,
                args=(topic, callback),
                daemon=True
            )
            consumer_thread.start()
            
            logger.info(f"Subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic: {e}")
            return False
    
    def _consume_local_queue(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Consume messages from local queue."""
        while True:
            try:
                # Check Redis first
                if self.redis_client:
                    try:
                        message_data = self.redis_client.brpop(f"queue:{topic}", timeout=1)
                        if message_data:
                            _, message_json = message_data
                            if isinstance(message_json, bytes):
                                message_json = message_json.decode('utf-8')
                            message = json.loads(message_json)
                            callback(message)
                            continue
                    except Exception as e:
                        if "timeout" not in str(e).lower():
                            logger.warning(f"Redis consume error: {e}")
                
                # Check local queue
                try:
                    message = self.local_queues[topic].get(timeout=1.0)
                    callback(message)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Message consumption error: {e}")
                time.sleep(1.0)


class DistributedTaskScheduler:
    """Distributed task scheduler and coordinator."""
    
    def __init__(self, service_discovery: ServiceDiscovery, message_queue: MessageQueue):
        self.service_discovery = service_discovery
        self.message_queue = message_queue
        self.load_balancer = LoadBalancer()
        
        # Task management
        self.pending_tasks = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Coordination
        self.task_dependencies = defaultdict(set)
        self.task_results = {}
        
        # Control
        self._scheduler_active = False
        self._scheduler_thread = None
        self._lock = threading.RLock()
        
        # Subscribe to task responses
        self.message_queue.subscribe("task_responses", self._handle_task_response)
        
        logger.info("Initialized DistributedTaskScheduler")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution."""
        try:
            # Check dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    self.task_dependencies[task.task_id].add(dep_id)
            
            # Queue task
            self.pending_tasks.put((task.priority, task.created_at, task))
            
            logger.info(f"Submitted task {task.task_id} for service {task.service_name}")
            
            # Start scheduler if not active
            if not self._scheduler_active:
                self.start_scheduler()
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def start_scheduler(self) -> None:
        """Start task scheduler."""
        if self._scheduler_active:
            return
        
        self._scheduler_active = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="DistributedTaskScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Started distributed task scheduler")
    
    def stop_scheduler(self) -> None:
        """Stop task scheduler."""
        self._scheduler_active = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("Stopped distributed task scheduler")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_active:
            try:
                # Get next task
                try:
                    priority, created_at, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if dependencies are satisfied
                if not self._dependencies_satisfied(task):
                    # Put task back in queue
                    self.pending_tasks.put((priority, created_at, task))
                    time.sleep(0.1)
                    continue
                
                # Find service to execute task
                success = self._schedule_task(task)
                
                if not success:
                    # Retry later
                    self.pending_tasks.put((priority + 1, created_at, task))
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1.0)
    
    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _schedule_task(self, task: DistributedTask) -> bool:
        """Schedule task to appropriate service."""
        try:
            # Discover available services
            services = self.service_discovery.get_healthy_services(task.service_name)
            
            if not services:
                logger.warning(f"No healthy services found for {task.service_name}")
                return False
            
            # Extract endpoints
            all_endpoints = []
            for service in services:
                all_endpoints.extend(service.endpoints)
            
            if not all_endpoints:
                logger.warning(f"No endpoints found for {task.service_name}")
                return False
            
            # Select best endpoint
            endpoint = self.load_balancer.select_endpoint(task.service_name, all_endpoints)
            
            if not endpoint:
                logger.warning(f"Failed to select endpoint for {task.service_name}")
                return False
            
            # Send task to service
            success = self._send_task_to_service(task, endpoint)
            
            if success:
                with self._lock:
                    task.assigned_node = f"{endpoint.host}:{endpoint.port}"
                    task.started_at = time.time()
                    self.active_tasks[task.task_id] = task
                
                record_metric("task_scheduled", 1, "counter", {"service": task.service_name})
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task.task_id}: {e}")
            return False
    
    def _send_task_to_service(self, task: DistributedTask, endpoint: ServiceEndpoint) -> bool:
        """Send task to service endpoint."""
        try:
            self.load_balancer.start_connection(endpoint)
            start_time = time.time()
            
            # Prepare task message
            task_message = {
                "type": MessageType.TASK_REQUEST.value,
                "task": task.to_dict(),
                "response_topic": "task_responses"
            }
            
            # Send via HTTP POST
            response = requests.post(
                urljoin(endpoint.full_url, "tasks"),
                json=task_message,
                timeout=endpoint.timeout_seconds
            )
            
            response_time = time.time() - start_time
            success = response.status_code == 200
            
            self.load_balancer.record_request(endpoint, response_time, success)
            self.load_balancer.end_connection(endpoint)
            
            if success:
                logger.debug(f"Successfully sent task {task.task_id} to {endpoint.host}:{endpoint.port}")
            else:
                logger.warning(f"Task submission failed: HTTP {response.status_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send task to service: {e}")
            self.load_balancer.end_connection(endpoint)
            return False
    
    def _handle_task_response(self, message: Dict[str, Any]) -> None:
        """Handle task response message."""
        try:
            if message.get("type") != MessageType.TASK_RESPONSE.value:
                return
            
            task_data = message.get("task", {})
            task_id = task_data.get("task_id")
            
            if not task_id:
                logger.warning("Received task response without task_id")
                return
            
            with self._lock:
                if task_id in self.active_tasks:
                    task = self.active_tasks.pop(task_id)
                    
                    # Update task with response
                    task.completed_at = time.time()
                    task.result = task_data.get("result")
                    task.error = task_data.get("error")
                    
                    if task.error:
                        self.failed_tasks[task_id] = task
                        logger.warning(f"Task {task_id} failed: {task.error}")
                        
                        # Retry if possible
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.started_at = None
                            task.completed_at = None
                            task.assigned_node = None
                            self.pending_tasks.put((task.priority + 1, task.created_at, task))
                            logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                    else:
                        self.completed_tasks[task_id] = task
                        self.task_results[task_id] = task.result
                        logger.info(f"Task {task_id} completed successfully")
                    
                    record_metric("task_completed", 1, "counter", {
                        "status": "success" if not task.error else "failed"
                    })
                
        except Exception as e:
            logger.error(f"Failed to handle task response: {e}")
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, waiting if necessary."""
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].result
            
            # Check if task failed
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                raise RuntimeError(f"Task failed: {task.error}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "scheduler_active": self._scheduler_active,
                "pending_tasks": self.pending_tasks.qsize(),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "load_balancer_stats": self.load_balancer.get_load_balancer_stats()
            }


class ServiceMesh:
    """Service mesh for microservices architecture."""
    
    def __init__(self, redis_url: Optional[str] = None, rabbitmq_url: Optional[str] = None):
        # Initialize components
        redis_client = None
        if redis_url and HAS_REDIS:
            try:
                redis_client = redis.from_url(redis_url)
                redis_client.ping()  # Test connection
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                redis_client = None
        
        self.service_discovery = ServiceDiscovery(redis_client)
        self.message_queue = MessageQueue(redis_client, rabbitmq_url)
        self.task_scheduler = DistributedTaskScheduler(self.service_discovery, self.message_queue)
        
        # Service mesh configuration
        self.mesh_config = {
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff_factor": 2.0
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60.0
            },
            "timeout": {
                "default": 30.0,
                "max": 300.0
            }
        }
        
        logger.info("Initialized ServiceMesh")
    
    def register_service(self, service_name: str, host: str, port: int,
                        version: str = "1.0", **kwargs) -> str:
        """Register service with the mesh."""
        service_id = str(uuid.uuid4())
        
        endpoint = ServiceEndpoint(
            service_id=service_id,
            host=host,
            port=port,
            **kwargs
        )
        
        registration = ServiceRegistration(
            service_id=service_id,
            service_name=service_name,
            version=version,
            endpoints=[endpoint],
            status=ServiceStatus.HEALTHY
        )
        
        success = self.service_discovery.register_service(registration)
        
        if success:
            logger.info(f"Registered service '{service_name}' with ID {service_id}")
        else:
            logger.error(f"Failed to register service '{service_name}'")
        
        return service_id if success else ""
    
    def call_service(self, service_name: str, function_name: str,
                    *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """Call remote service function."""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type="function_call",
            service_name=service_name,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            timeout_seconds=timeout or self.mesh_config["timeout"]["default"]
        )
        
        task_id = self.task_scheduler.submit_task(task)
        return self.task_scheduler.get_task_result(task_id, timeout)
    
    def async_call_service(self, service_name: str, function_name: str,
                          *args, **kwargs) -> str:
        """Asynchronously call remote service function."""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type="function_call",
            service_name=service_name,
            function_name=function_name,
            args=args,
            kwargs=kwargs
        )
        
        return self.task_scheduler.submit_task(task)
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status of service."""
        services = self.service_discovery.discover_services(service_name)
        
        if not services:
            return {"status": "not_found", "instances": 0}
        
        healthy_count = sum(1 for s in services if s.status == ServiceStatus.HEALTHY)
        degraded_count = sum(1 for s in services if s.status == ServiceStatus.DEGRADED)
        unhealthy_count = sum(1 for s in services if s.status == ServiceStatus.UNHEALTHY)
        
        if healthy_count == len(services):
            overall_status = "healthy"
        elif healthy_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "instances": len(services),
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count
        }
    
    def get_mesh_stats(self) -> Dict[str, Any]:
        """Get comprehensive service mesh statistics."""
        return {
            "service_discovery": {
                "registered_services": len(self.service_discovery.local_registry),
                "health_check_active": self.service_discovery.health_check_active
            },
            "task_scheduler": self.task_scheduler.get_scheduler_stats(),
            "configuration": self.mesh_config
        }
    
    def shutdown(self) -> None:
        """Shutdown service mesh."""
        logger.info("Shutting down ServiceMesh")
        
        self.service_discovery.stop_health_checking()
        self.task_scheduler.stop_scheduler()
        
        # Close connections
        if self.message_queue.rabbitmq_connection:
            try:
                self.message_queue.rabbitmq_connection.close()
            except Exception:
                pass
        
        logger.info("ServiceMesh shutdown complete")


# Global service mesh instance
_global_service_mesh: Optional[ServiceMesh] = None


def get_service_mesh(redis_url: Optional[str] = None, 
                    rabbitmq_url: Optional[str] = None) -> ServiceMesh:
    """Get or create global service mesh."""
    global _global_service_mesh
    
    if _global_service_mesh is None:
        _global_service_mesh = ServiceMesh(redis_url, rabbitmq_url)
    
    return _global_service_mesh


def shutdown_service_mesh() -> None:
    """Shutdown global service mesh."""
    global _global_service_mesh
    
    if _global_service_mesh:
        _global_service_mesh.shutdown()
        _global_service_mesh = None