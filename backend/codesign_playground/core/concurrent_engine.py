"""
Advanced concurrent processing engine with work stealing, resource pooling, and GPU acceleration.

This module provides high-performance concurrent processing capabilities including:
- Work stealing algorithms for dynamic load balancing
- Advanced resource pooling with auto-scaling
- GPU acceleration support
- Distributed computing integration
- Asynchronous processing pipelines
"""

import time
import threading
import asyncio
import queue
import uuid
import gc
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import psutil
import numpy as np

try:
    import cupy as cp
    import numba
    from numba import cuda
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    import dask
    from dask.distributed import Client
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

import logging
from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class Task:
    """Enhanced task with metadata and dependencies."""
    
    id: str
    func: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 1.0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    max_retries: int = 3
    timeout: Optional[float] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    gpu_required: bool = False
    memory_intensive: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    current_load: float = 0.0
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    steal_attempts: int = 0
    successful_steals: int = 0
    last_activity: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 1.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average task execution time."""
        return self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0.0


class WorkStealingQueue:
    """Work stealing queue implementation."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self._deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
    def put_task(self, task: Task) -> None:
        """Add task to the queue (LIFO for better cache locality)."""
        with self._not_empty:
            self._deque.append(task)
            self._not_empty.notify()
    
    def get_task(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get task from own queue (LIFO)."""
        with self._not_empty:
            while not self._deque:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            
            return self._deque.pop()  # LIFO
    
    def steal_task(self) -> Optional[Task]:
        """Steal task from queue (FIFO for work stealing)."""
        with self._lock:
            if self._deque:
                return self._deque.popleft()  # FIFO
            return None
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._deque)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._deque) == 0


class ResourcePool:
    """Advanced resource pool with auto-scaling and monitoring."""
    
    def __init__(
        self,
        resource_type: ResourceType,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 32,
        scaling_factor: float = 1.5,
        idle_timeout: float = 300.0
    ):
        self.resource_type = resource_type
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.scaling_factor = scaling_factor
        self.idle_timeout = idle_timeout
        
        self._available = queue.Queue()
        self._in_use = set()
        self._all_resources = weakref.WeakSet()
        self._lock = threading.RLock()
        self._stats = {
            'created': 0,
            'destroyed': 0,
            'acquisitions': 0,
            'wait_time_total': 0.0
        }
        
        # Initialize minimum resources
        for _ in range(min_size):
            resource = self._create_resource()
            if resource:
                self._available.put((resource, time.time()))
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire resource with intelligent scaling."""
        start_time = time.time()
        
        with self._lock:
            self._stats['acquisitions'] += 1
        
        # Try to get available resource
        try:
            while True:
                try:
                    resource, last_used = self._available.get_nowait()
                    
                    # Check if resource is still valid
                    if self._is_resource_valid(resource, last_used):
                        with self._lock:
                            self._in_use.add(resource)
                        return resource
                    else:
                        self._destroy_resource(resource)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Resource acquisition error: {e}")
        
        # Create new resource if under limit
        with self._lock:
            if len(self._in_use) + self._available.qsize() < self.max_size:
                resource = self._create_resource()
                if resource:
                    self._in_use.add(resource)
                    return resource
        
        # Wait for resource with timeout
        if timeout is None:
            timeout = 30.0
        
        end_time = start_time + timeout
        while time.time() < end_time:
            try:
                resource, last_used = self._available.get(timeout=0.1)
                if self._is_resource_valid(resource, last_used):
                    with self._lock:
                        self._in_use.add(resource)
                        wait_time = time.time() - start_time
                        self._stats['wait_time_total'] += wait_time
                    return resource
                else:
                    self._destroy_resource(resource)
            except queue.Empty:
                continue
        
        raise TimeoutError(f"Failed to acquire {self.resource_type.value} resource within {timeout}s")
    
    def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                # Return to pool if under max size
                if self._available.qsize() < self.max_size:
                    self._available.put((resource, time.time()))
                else:
                    self._destroy_resource(resource)
    
    def _create_resource(self) -> Any:
        """Create new resource."""
        try:
            resource = self.factory()
            self._all_resources.add(resource)
            with self._lock:
                self._stats['created'] += 1
            logger.debug(f"Created {self.resource_type.value} resource")
            return resource
        except Exception as e:
            logger.error(f"Failed to create {self.resource_type.value} resource: {e}")
            return None
    
    def _destroy_resource(self, resource: Any) -> None:
        """Destroy resource."""
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
            
            with self._lock:
                self._stats['destroyed'] += 1
            logger.debug(f"Destroyed {self.resource_type.value} resource")
        except Exception as e:
            logger.warning(f"Error destroying {self.resource_type.value} resource: {e}")
    
    def _is_resource_valid(self, resource: Any, last_used: float) -> bool:
        """Check if resource is still valid."""
        # Check timeout
        if time.time() - last_used > self.idle_timeout:
            return False
        
        # Resource-specific validation
        if hasattr(resource, 'is_valid'):
            return resource.is_valid()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            avg_wait_time = (
                self._stats['wait_time_total'] / self._stats['acquisitions']
                if self._stats['acquisitions'] > 0 else 0.0
            )
            
            return {
                'resource_type': self.resource_type.value,
                'available': self._available.qsize(),
                'in_use': len(self._in_use),
                'total_created': self._stats['created'],
                'total_destroyed': self._stats['destroyed'],
                'utilization': len(self._in_use) / self.max_size,
                'avg_wait_time_ms': avg_wait_time * 1000
            }


class WorkStealingExecutor:
    """Advanced work stealing executor with GPU support and resource management."""
    
    def __init__(
        self,
        max_workers: int = None,
        enable_gpu: bool = False,
        enable_work_stealing: bool = True,
        steal_ratio: float = 0.5,
        enable_distributed: bool = False
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) * 2)
        self.enable_gpu = enable_gpu and HAS_GPU
        self.enable_work_stealing = enable_work_stealing
        self.steal_ratio = steal_ratio
        self.enable_distributed = enable_distributed
        
        # Worker management
        self.workers = {}
        self.worker_queues = {}
        self.worker_stats = {}
        self.worker_threads = {}
        
        # Task management
        self.pending_tasks = queue.PriorityQueue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_dependencies = defaultdict(set)
        
        # Resource pools
        self.resource_pools = {}
        self.gpu_resources = []
        
        # Control
        self._shutdown_event = threading.Event()
        self._stats_lock = threading.RLock()
        self._scheduler_thread = None
        
        # Performance monitoring
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'steal_attempts': 0,
            'successful_steals': 0,
            'avg_execution_time': 0.0,
            'gpu_utilization': 0.0
        }
        
        self._initialize_resources()
        self._start_workers()
        self._start_scheduler()
        
        logger.info(
            f"Initialized WorkStealingExecutor with {self.max_workers} workers",
            gpu_enabled=self.enable_gpu,
            work_stealing=enable_work_stealing,
            distributed=enable_distributed
        )
    
    def _initialize_resources(self) -> None:
        """Initialize resource pools."""
        # CPU thread pool
        self.resource_pools[ResourceType.CPU] = ResourcePool(
            ResourceType.CPU,
            lambda: ThreadPoolExecutor(max_workers=1),
            min_size=2,
            max_size=self.max_workers
        )
        
        # GPU resources
        if self.enable_gpu:
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                for i in range(gpu_count):
                    self.gpu_resources.append({
                        'device_id': i,
                        'memory_total': cp.cuda.Device(i).mem_info[1],
                        'in_use': False
                    })
                logger.info(f"Initialized {gpu_count} GPU resources")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU resources: {e}")
                self.enable_gpu = False
        
        # Distributed computing
        if self.enable_distributed:
            self._initialize_distributed()
    
    def _initialize_distributed(self) -> None:
        """Initialize distributed computing backends."""
        # Try Ray first
        if HAS_RAY and not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Initialized Ray for distributed computing")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
        
        # Try Dask as alternative
        if HAS_DASK:
            try:
                self.dask_client = Client(processes=False)
                logger.info("Initialized Dask for distributed computing")
            except Exception as e:
                logger.warning(f"Failed to initialize Dask: {e}")
    
    def _start_workers(self) -> None:
        """Start worker threads."""
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            
            # Create worker queue
            worker_queue = WorkStealingQueue(worker_id)
            self.worker_queues[worker_id] = worker_queue
            
            # Create worker stats
            self.worker_stats[worker_id] = WorkerStats(worker_id)
            
            # Start worker thread
            worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"WorkStealingWorker-{i}"
            )
            worker_thread.start()
            self.worker_threads[worker_id] = worker_thread
    
    def _start_scheduler(self) -> None:
        """Start task scheduler thread."""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TaskScheduler"
        )
        self._scheduler_thread.start()
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop for task distribution."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from pending queue
                try:
                    priority, task_id, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check dependencies
                if not self._dependencies_satisfied(task):
                    # Put back in queue with slight delay
                    self.pending_tasks.put((priority, task_id, task))
                    time.sleep(0.1)
                    continue
                
                # Find best worker for task
                worker_id = self._select_worker(task)
                
                if worker_id:
                    # Assign task to worker
                    self.worker_queues[worker_id].put_task(task)
                    task.worker_id = worker_id
                    
                    record_metric("task_assigned", 1, "counter", {"worker": worker_id})
                else:
                    # No available worker, put back in queue
                    self.pending_tasks.put((priority, task_id, task))
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop with work stealing."""
        stats = self.worker_stats[worker_id]
        worker_queue = self.worker_queues[worker_id]
        
        while not self._shutdown_event.is_set():
            task = None
            
            try:
                # Try to get task from own queue
                task = worker_queue.get_task(timeout=1.0)
                
                # If no task, try work stealing
                if task is None and self.enable_work_stealing:
                    task = self._attempt_work_stealing(worker_id)
                
                if task is None:
                    continue
                
                # Execute task
                self._execute_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if task:
                    self._handle_task_failure(task, e)
                time.sleep(1.0)
    
    def _attempt_work_stealing(self, worker_id: str) -> Optional[Task]:
        """Attempt to steal work from other workers."""
        stats = self.worker_stats[worker_id]
        stats.steal_attempts += 1
        
        # Find workers with tasks
        candidates = []
        for other_worker_id, other_queue in self.worker_queues.items():
            if other_worker_id != worker_id and other_queue.size() > 1:
                candidates.append((other_worker_id, other_queue.size()))
        
        if not candidates:
            return None
        
        # Sort by queue size (steal from busiest)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try to steal from top candidates
        for other_worker_id, _ in candidates[:3]:
            other_queue = self.worker_queues[other_worker_id]
            stolen_task = other_queue.steal_task()
            
            if stolen_task:
                stats.successful_steals += 1
                
                with self._stats_lock:
                    self.performance_metrics['successful_steals'] += 1
                
                record_metric("work_stealing_success", 1, "counter", {
                    "stealer": worker_id,
                    "victim": other_worker_id
                })
                
                logger.debug(f"Worker {worker_id} stole task from {other_worker_id}")
                return stolen_task
        
        return None
    
    def _execute_task(self, task: Task, worker_id: str) -> None:
        """Execute a task with resource management."""
        stats = self.worker_stats[worker_id]
        start_time = time.time()
        task.started_at = start_time
        
        try:
            # Acquire required resources
            resources = self._acquire_task_resources(task)
            
            # Execute on GPU if required and available
            if task.gpu_required and self.enable_gpu:
                result = self._execute_gpu_task(task, resources)
            else:
                result = self._execute_cpu_task(task, resources)
            
            # Task completed successfully
            task.completed_at = time.time()
            task.result = result
            
            execution_time = task.completed_at - start_time
            
            # Update statistics
            stats.tasks_completed += 1
            stats.total_execution_time += execution_time
            stats.last_activity = time.time()
            
            with self._stats_lock:
                self.performance_metrics['completed_tasks'] += 1
                self.performance_metrics['avg_execution_time'] = (
                    (self.performance_metrics['avg_execution_time'] * (stats.tasks_completed - 1) + execution_time)
                    / stats.tasks_completed
                )
            
            # Store result
            self.completed_tasks[task.id] = task
            
            # Release resources
            self._release_task_resources(resources)
            
            # Notify dependent tasks
            self._notify_task_completion(task)
            
            record_metric("task_completed", 1, "counter", {"worker": worker_id})
            record_metric("task_execution_time", execution_time, "timer")
            
            logger.debug(f"Task {task.id} completed by {worker_id} in {execution_time:.3f}s")
            
        except Exception as e:
            self._handle_task_failure(task, e)
    
    def _acquire_task_resources(self, task: Task) -> Dict[ResourceType, Any]:
        """Acquire resources required for task execution."""
        resources = {}
        
        # Acquire CPU resource
        if ResourceType.CPU in task.resource_requirements:
            cpu_pool = self.resource_pools[ResourceType.CPU]
            resources[ResourceType.CPU] = cpu_pool.acquire(timeout=30.0)
        
        # Acquire GPU resource if required
        if task.gpu_required and self.enable_gpu:
            gpu_device = self._acquire_gpu_device()
            if gpu_device is not None:
                resources[ResourceType.GPU] = gpu_device
        
        return resources
    
    def _release_task_resources(self, resources: Dict[ResourceType, Any]) -> None:
        """Release task resources."""
        for resource_type, resource in resources.items():
            if resource_type == ResourceType.CPU:
                cpu_pool = self.resource_pools[ResourceType.CPU]
                cpu_pool.release(resource)
            elif resource_type == ResourceType.GPU:
                self._release_gpu_device(resource)
    
    def _acquire_gpu_device(self) -> Optional[int]:
        """Acquire available GPU device."""
        for gpu in self.gpu_resources:
            if not gpu['in_use']:
                gpu['in_use'] = True
                return gpu['device_id']
        return None
    
    def _release_gpu_device(self, device_id: int) -> None:
        """Release GPU device."""
        for gpu in self.gpu_resources:
            if gpu['device_id'] == device_id:
                gpu['in_use'] = False
                break
    
    def _execute_cpu_task(self, task: Task, resources: Dict[ResourceType, Any]) -> Any:
        """Execute task on CPU."""
        try:
            # Set timeout if specified
            if task.timeout:
                # This is a simplified timeout implementation
                # In practice, you'd want more sophisticated timeout handling
                result = task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            return result
        except Exception as e:
            logger.error(f"CPU task execution failed: {e}")
            raise
    
    def _execute_gpu_task(self, task: Task, resources: Dict[ResourceType, Any]) -> Any:
        """Execute task on GPU."""
        if not self.enable_gpu:
            raise RuntimeError("GPU execution requested but GPU support not available")
        
        device_id = resources.get(ResourceType.GPU)
        if device_id is None:
            raise RuntimeError("No GPU device available for task")
        
        try:
            # Set GPU device
            cp.cuda.Device(device_id).use()
            
            # Execute task
            result = task.func(*task.args, **task.kwargs)
            
            # Synchronize and clear GPU memory
            cp.cuda.Stream.null.synchronize()
            
            return result
        except Exception as e:
            logger.error(f"GPU task execution failed: {e}")
            raise
        finally:
            # Clean up GPU memory
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
    
    def _handle_task_failure(self, task: Task, error: Exception) -> None:
        """Handle task failure with retry logic."""
        task.error = error
        task.retry_count += 1
        
        stats = self.worker_stats.get(task.worker_id)
        if stats:
            stats.tasks_failed += 1
        
        with self._stats_lock:
            self.performance_metrics['failed_tasks'] += 1
        
        # Check if task should be retried
        if task.retry_count <= task.max_retries:
            logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {error}")
            
            # Reset task state for retry
            task.started_at = None
            task.completed_at = None
            task.result = None
            task.worker_id = None
            
            # Re-queue task with lower priority
            priority = min(task.priority.value + 1, TaskPriority.BACKGROUND.value)
            self.pending_tasks.put((priority, task.id, task))
        else:
            logger.error(f"Task {task.id} failed permanently after {task.retry_count} retries: {error}")
            self.failed_tasks[task.id] = task
        
        record_metric("task_failed", 1, "counter", {"retries": task.retry_count})
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _select_worker(self, task: Task) -> Optional[str]:
        """Select best worker for task based on load and capabilities."""
        candidates = []
        
        for worker_id, stats in self.worker_stats.items():
            queue_size = self.worker_queues[worker_id].size()
            
            # Calculate worker score (lower is better)
            score = queue_size + stats.current_load
            
            # Prefer workers with good success rate
            score *= (2.0 - stats.success_rate)
            
            candidates.append((worker_id, score))
        
        if not candidates:
            return None
        
        # Sort by score and return best worker
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def _notify_task_completion(self, task: Task) -> None:
        """Notify completion to dependent tasks."""
        # This is handled by the scheduler checking dependencies
        pass
    
    def submit_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        gpu_required: bool = False,
        estimated_duration: float = 1.0,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit task for execution."""
        task = Task(
            id=str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies or set(),
            gpu_required=gpu_required,
            estimated_duration=estimated_duration,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Add to pending queue
        self.pending_tasks.put((priority.value, task.id, task))
        
        with self._stats_lock:
            self.performance_metrics['total_tasks'] += 1
        
        record_metric("task_submitted", 1, "counter", {"priority": priority.name})
        
        logger.debug(f"Submitted task {task.id} with priority {priority.name}")
        return task.id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return task.result
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                raise task.error
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive executor statistics."""
        with self._stats_lock:
            stats = self.performance_metrics.copy()
        
        # Worker statistics
        worker_stats = {}
        for worker_id, worker_stat in self.worker_stats.items():
            worker_stats[worker_id] = {
                'tasks_completed': worker_stat.tasks_completed,
                'tasks_failed': worker_stat.tasks_failed,
                'success_rate': worker_stat.success_rate,
                'queue_size': self.worker_queues[worker_id].size(),
                'avg_execution_time': worker_stat.average_execution_time
            }
        
        # Resource pool statistics
        resource_stats = {}
        for resource_type, pool in self.resource_pools.items():
            resource_stats[resource_type.value] = pool.get_stats()
        
        # GPU statistics
        gpu_stats = []
        for gpu in self.gpu_resources:
            gpu_stats.append({
                'device_id': gpu['device_id'],
                'memory_total': gpu['memory_total'],
                'in_use': gpu['in_use']
            })
        
        return {
            'performance': stats,
            'workers': worker_stats,
            'resource_pools': resource_stats,
            'gpu_resources': gpu_stats,
            'pending_tasks': self.pending_tasks.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }
    
    def shutdown(self) -> None:
        """Shutdown executor and cleanup resources."""
        logger.info("Shutting down WorkStealingExecutor")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for threads to complete
        for thread in self.worker_threads.values():
            thread.join(timeout=5.0)
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        
        # Shutdown resource pools
        for pool in self.resource_pools.values():
            if hasattr(pool, 'shutdown'):
                pool.shutdown()
        
        # Cleanup distributed resources
        if self.enable_distributed:
            if HAS_RAY and ray.is_initialized():
                ray.shutdown()
            
            if hasattr(self, 'dask_client'):
                self.dask_client.close()
        
        logger.info("WorkStealingExecutor shutdown complete")


# Global executor instance
_global_executor: Optional[WorkStealingExecutor] = None


def get_concurrent_executor(
    max_workers: Optional[int] = None,
    enable_gpu: bool = False,
    enable_work_stealing: bool = True,
    enable_distributed: bool = False
) -> WorkStealingExecutor:
    """Get or create global concurrent executor."""
    global _global_executor
    
    if _global_executor is None:
        _global_executor = WorkStealingExecutor(
            max_workers=max_workers,
            enable_gpu=enable_gpu,
            enable_work_stealing=enable_work_stealing,
            enable_distributed=enable_distributed
        )
    
    return _global_executor


def shutdown_concurrent_executor() -> None:
    """Shutdown global concurrent executor."""
    global _global_executor
    
    if _global_executor:
        _global_executor.shutdown()
        _global_executor = None


def parallel_map(
    func: Callable,
    iterable: List[Any],
    max_workers: Optional[int] = None,
    enable_gpu: bool = False,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """Parallel map function with work stealing."""
    executor = get_concurrent_executor(
        max_workers=max_workers,
        enable_gpu=enable_gpu
    )
    
    # Submit tasks
    task_ids = []
    for item in iterable:
        task_id = executor.submit_task(
            func,
            item,
            gpu_required=enable_gpu
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        try:
            result = executor.get_result(task_id, timeout=300.0)
            results.append(result)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            results.append(None)
    
    return results


def gpu_accelerated_batch(
    func: Callable,
    batch_data: List[Any],
    batch_size: int = 32
) -> List[Any]:
    """GPU-accelerated batch processing."""
    if not HAS_GPU:
        raise RuntimeError("GPU support not available")
    
    executor = get_concurrent_executor(enable_gpu=True)
    
    # Split into batches
    batches = [
        batch_data[i:i + batch_size]
        for i in range(0, len(batch_data), batch_size)
    ]
    
    # Submit batch tasks
    task_ids = []
    for batch in batches:
        task_id = executor.submit_task(
            func,
            batch,
            gpu_required=True,
            priority=TaskPriority.HIGH
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        batch_results = executor.get_result(task_id, timeout=600.0)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results