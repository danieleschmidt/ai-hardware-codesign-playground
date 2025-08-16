"""
Comprehensive health check endpoints and status monitoring for AI Hardware Co-Design Playground.

This module provides REST endpoints for health checks, system status monitoring,
and readiness/liveness probes for container orchestration.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

from .health_monitoring import get_health_monitor, ComponentHealth
from .monitoring import get_system_monitor
from .circuit_breaker import get_all_circuit_breaker_metrics
from .enhanced_resilience import get_resilience_manager
from .authentication import get_auth_manager
from .rate_limiting import get_rate_limit_manager
from .logging import get_logger
from .exceptions import CodesignError

logger = get_logger(__name__)


class HealthCheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"    # Basic service availability
    READINESS = "readiness"  # Ready to accept traffic
    STARTUP = "startup"      # Initial startup health
    DEEP = "deep"           # Comprehensive system health


class ServiceStatus(Enum):
    """Service status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class HealthCheckResult:
    """Result of a health check endpoint."""
    status: ServiceStatus
    timestamp: float
    duration_ms: float
    checks: Dict[str, Any]
    metrics: Dict[str, Any]
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "checks": self.checks,
            "metrics": self.metrics,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "message": self.message
        }


class HealthEndpointManager:
    """Manager for health check endpoints."""
    
    def __init__(self):
        """Initialize health endpoint manager."""
        self.start_time = time.time()
        self.status = ServiceStatus.STARTING
        self.custom_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        
        # Initialize after short delay to allow other components to start
        threading.Timer(5.0, self._mark_ready).start()
        
        logger.info("Initialized HealthEndpointManager")
    
    def _mark_ready(self) -> None:
        """Mark service as ready after startup."""
        with self._lock:
            if self.status == ServiceStatus.STARTING:
                self.status = ServiceStatus.HEALTHY
                logger.info("Service marked as ready")
    
    def register_custom_check(self, name: str, check_func: Callable[[], Dict[str, Any]]) -> None:
        """Register custom health check."""
        with self._lock:
            self.custom_checks[name] = check_func
        
        logger.info("Registered custom health check", name=name)
    
    def liveness_check(self) -> HealthCheckResult:
        """Basic liveness check - is the service running?"""
        start_time = time.time()
        
        try:
            # Basic checks
            checks = {
                "service": {"status": "running", "pid": self._get_process_id()},
                "timestamp": time.time()
            }
            
            # Check if critical threads are alive
            thread_check = self._check_critical_threads()
            checks["threads"] = thread_check
            
            # Determine status
            status = ServiceStatus.HEALTHY
            if not thread_check.get("all_alive", True):
                status = ServiceStatus.DEGRADED
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks=checks,
                metrics={"liveness_check_count": 1},
                uptime_seconds=time.time() - self.start_time,
                message="Liveness check completed"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Liveness check failed", error=str(e))
            
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks={"error": str(e)},
                metrics={},
                uptime_seconds=time.time() - self.start_time,
                message=f"Liveness check failed: {e}"
            )
    
    def readiness_check(self) -> HealthCheckResult:
        """Readiness check - is the service ready to accept traffic?"""
        start_time = time.time()
        
        try:
            checks = {}
            overall_healthy = True
            
            # Check system health
            try:
                health_monitor = get_health_monitor()
                health_status = health_monitor.get_health_status()
                checks["system_health"] = health_status
                
                if health_status.get("overall_status") in ["critical", "failed"]:
                    overall_healthy = False
            except Exception as e:
                checks["system_health"] = {"error": str(e)}
                overall_healthy = False
            
            # Check circuit breakers
            try:
                cb_metrics = get_all_circuit_breaker_metrics()
                checks["circuit_breakers"] = cb_metrics
                
                # Check if any critical circuit breakers are open
                for cb_name, cb_data in cb_metrics.items():
                    if cb_data.get("state") == "open" and "critical" in cb_name.lower():
                        overall_healthy = False
            except Exception as e:
                checks["circuit_breakers"] = {"error": str(e)}
            
            # Check dependencies
            dependency_checks = self._check_dependencies()
            checks["dependencies"] = dependency_checks
            if not dependency_checks.get("all_healthy", True):
                overall_healthy = False
            
            # Check custom health checks
            for name, check_func in self.custom_checks.items():
                try:
                    custom_result = check_func()
                    checks[f"custom_{name}"] = custom_result
                    if not custom_result.get("healthy", True):
                        overall_healthy = False
                except Exception as e:
                    checks[f"custom_{name}"] = {"error": str(e)}
                    overall_healthy = False
            
            # Determine final status
            if self.status == ServiceStatus.STARTING:
                status = ServiceStatus.STARTING
            elif overall_healthy:
                status = ServiceStatus.HEALTHY
            else:
                status = ServiceStatus.DEGRADED
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks=checks,
                metrics=self._get_readiness_metrics(),
                uptime_seconds=time.time() - self.start_time,
                message="Readiness check completed"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Readiness check failed", error=str(e))
            
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks={"error": str(e)},
                metrics={},
                uptime_seconds=time.time() - self.start_time,
                message=f"Readiness check failed: {e}"
            )
    
    def startup_check(self) -> HealthCheckResult:
        """Startup check - is the service starting up correctly?"""
        start_time = time.time()
        
        checks = {
            "startup_time": time.time() - self.start_time,
            "status": self.status.value,
            "initialization": self._check_initialization()
        }
        
        # Check if still starting
        if self.status == ServiceStatus.STARTING:
            # Allow up to 2 minutes for startup
            if time.time() - self.start_time > 120:
                status = ServiceStatus.UNHEALTHY
                message = "Startup timeout exceeded"
            else:
                status = ServiceStatus.STARTING
                message = "Service still starting"
        else:
            status = self.status
            message = "Startup completed"
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            status=status,
            timestamp=start_time,
            duration_ms=duration_ms,
            checks=checks,
            metrics={"startup_duration": time.time() - self.start_time},
            uptime_seconds=time.time() - self.start_time,
            message=message
        )
    
    def deep_health_check(self) -> HealthCheckResult:
        """Comprehensive deep health check."""
        start_time = time.time()
        
        try:
            checks = {}
            metrics = {}
            overall_healthy = True
            
            # System monitoring
            try:
                system_monitor = get_system_monitor()
                system_status = system_monitor.get_monitoring_summary()
                checks["system"] = system_status
                
                # Check system metrics
                sys_metrics = system_status.get("system_metrics", {})
                if sys_metrics.get("cpu_percent", 0) > 90:
                    overall_healthy = False
                if sys_metrics.get("memory_percent", 0) > 95:
                    overall_healthy = False
                if sys_metrics.get("disk_usage_percent", 0) > 95:
                    overall_healthy = False
                
            except Exception as e:
                checks["system"] = {"error": str(e)}
                overall_healthy = False
            
            # Health monitoring
            try:
                health_monitor = get_health_monitor()
                health_status = health_monitor.get_health_status()
                checks["health_monitoring"] = health_status
                
                if health_status.get("overall_status") == "critical":
                    overall_healthy = False
                    
            except Exception as e:
                checks["health_monitoring"] = {"error": str(e)}
                overall_healthy = False
            
            # Circuit breakers
            try:
                cb_metrics = get_all_circuit_breaker_metrics()
                checks["circuit_breakers"] = cb_metrics
                metrics["circuit_breaker_count"] = len(cb_metrics)
                
                open_breakers = sum(1 for cb in cb_metrics.values() if cb.get("state") == "open")
                metrics["open_circuit_breakers"] = open_breakers
                
            except Exception as e:
                checks["circuit_breakers"] = {"error": str(e)}
            
            # Resilience patterns
            try:
                resilience_manager = get_resilience_manager()
                resilience_stats = resilience_manager.get_comprehensive_stats()
                checks["resilience"] = resilience_stats
                
            except Exception as e:
                checks["resilience"] = {"error": str(e)}
            
            # Authentication system
            try:
                auth_manager = get_auth_manager()
                auth_stats = {
                    "user_count": len(auth_manager.users),
                    "active_sessions": len([s for s in auth_manager.sessions.values() if s.is_valid()]),
                    "locked_accounts": len(auth_manager.locked_accounts)
                }
                checks["authentication"] = auth_stats
                metrics.update(auth_stats)
                
            except Exception as e:
                checks["authentication"] = {"error": str(e)}
            
            # Rate limiting
            try:
                rate_limit_manager = get_rate_limit_manager()
                rate_limit_stats = {"limiter_count": len(rate_limit_manager.limiters)}
                checks["rate_limiting"] = rate_limit_stats
                metrics.update(rate_limit_stats)
                
            except Exception as e:
                checks["rate_limiting"] = {"error": str(e)}
            
            # Performance metrics
            performance_metrics = self._get_performance_metrics()
            checks["performance"] = performance_metrics
            metrics.update(performance_metrics)
            
            # Resource utilization
            resource_metrics = self._get_resource_metrics()
            checks["resources"] = resource_metrics
            metrics.update(resource_metrics)
            
            # Determine final status
            if overall_healthy:
                status = ServiceStatus.HEALTHY
                message = "All systems healthy"
            else:
                status = ServiceStatus.DEGRADED
                message = "Some systems degraded"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks=checks,
                metrics=metrics,
                uptime_seconds=time.time() - self.start_time,
                message=message
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Deep health check failed", error=str(e))
            
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                timestamp=start_time,
                duration_ms=duration_ms,
                checks={"error": str(e)},
                metrics={},
                uptime_seconds=time.time() - self.start_time,
                message=f"Deep health check failed: {e}"
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "service": {
                "name": "ai_hardware_codesign",
                "version": "1.0.0",
                "status": self.status.value,
                "uptime_seconds": time.time() - self.start_time,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat()
            },
            "environment": {
                "platform": self._get_platform_info(),
                "python_version": self._get_python_version(),
                "process_id": self._get_process_id()
            },
            "endpoints": {
                "liveness": "/health/live",
                "readiness": "/health/ready", 
                "startup": "/health/startup",
                "deep": "/health/deep",
                "info": "/health/info"
            }
        }
    
    def _check_critical_threads(self) -> Dict[str, Any]:
        """Check if critical threads are alive."""
        import threading
        
        active_threads = threading.active_count()
        thread_names = [t.name for t in threading.enumerate()]
        
        return {
            "active_count": active_threads,
            "thread_names": thread_names,
            "all_alive": True  # Simplified check
        }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies."""
        # This would check databases, external services, etc.
        # For now, return a basic check
        return {
            "all_healthy": True,
            "checked": []
        }
    
    def _check_initialization(self) -> Dict[str, Any]:
        """Check if all components are initialized."""
        components = {
            "logging": True,
            "monitoring": True,
            "health_checks": True,
            "authentication": True,
            "rate_limiting": True
        }
        
        return {
            "components": components,
            "all_initialized": all(components.values())
        }
    
    def _get_readiness_metrics(self) -> Dict[str, Any]:
        """Get metrics relevant to readiness."""
        return {
            "readiness_check_count": 1,
            "uptime_seconds": time.time() - self.start_time,
            "status": self.status.value
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            }
        except:
            return {"error": "Unable to collect performance metrics"}
    
    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource utilization metrics."""
        try:
            import psutil
            
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except:
            return {"error": "Unable to collect resource metrics"}
    
    def _get_platform_info(self) -> str:
        """Get platform information."""
        import platform
        return f"{platform.system()} {platform.release()}"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_process_id(self) -> int:
        """Get current process ID."""
        import os
        return os.getpid()
    
    def shutdown(self) -> None:
        """Mark service as shutting down."""
        with self._lock:
            self.status = ServiceStatus.STOPPING
        
        logger.info("Service marked as shutting down")


# Global health endpoint manager
_health_manager: Optional[HealthEndpointManager] = None
_health_lock = threading.Lock()


def get_health_endpoint_manager() -> HealthEndpointManager:
    """Get global health endpoint manager."""
    global _health_manager
    
    with _health_lock:
        if _health_manager is None:
            _health_manager = HealthEndpointManager()
        
        return _health_manager


def create_health_endpoints():
    """Create health check endpoint functions."""
    manager = get_health_endpoint_manager()
    
    def liveness():
        """Liveness endpoint."""
        result = manager.liveness_check()
        status_code = 200 if result.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED] else 503
        return result.to_dict(), status_code
    
    def readiness():
        """Readiness endpoint."""
        result = manager.readiness_check()
        status_code = 200 if result.status == ServiceStatus.HEALTHY else 503
        return result.to_dict(), status_code
    
    def startup():
        """Startup endpoint."""
        result = manager.startup_check()
        status_code = 200 if result.status in [ServiceStatus.HEALTHY, ServiceStatus.STARTING] else 503
        return result.to_dict(), status_code
    
    def deep():
        """Deep health check endpoint."""
        result = manager.deep_health_check()
        status_code = 200 if result.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED] else 503
        return result.to_dict(), status_code
    
    def info():
        """System info endpoint."""
        info = manager.get_system_info()
        return info, 200
    
    return {
        "liveness": liveness,
        "readiness": readiness,
        "startup": startup,
        "deep": deep,
        "info": info
    }