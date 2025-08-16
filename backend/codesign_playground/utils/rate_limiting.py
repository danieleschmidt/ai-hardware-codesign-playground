"""
Advanced rate limiting and throttling for AI Hardware Co-Design Playground.

This module provides comprehensive rate limiting with multiple algorithms,
distributed support, and integration with security monitoring.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from functools import wraps
import math

from .logging import get_logger, get_audit_logger
from .monitoring import record_metric
from .exceptions import SecurityError

logger = get_logger(__name__)
audit_logger = get_audit_logger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limit scope levels."""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    limit: int  # Number of requests
    window_seconds: int  # Time window
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    scope: RateLimitScope = RateLimitScope.USER
    burst_limit: Optional[int] = None  # Allow burst beyond normal limit
    priority: int = 0  # Higher priority rules checked first
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "name": self.name,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "algorithm": self.algorithm.value,
            "scope": self.scope.value,
            "burst_limit": self.burst_limit,
            "priority": self.priority
        }


@dataclass
class RateLimitState:
    """State tracking for rate limiting."""
    requests: deque = field(default_factory=deque)  # Request timestamps
    tokens: float = 0.0  # Available tokens (for token bucket)
    last_refill: float = field(default_factory=time.time)  # Last token refill
    total_requests: int = 0
    blocked_requests: int = 0
    last_request: float = 0.0
    
    def reset(self) -> None:
        """Reset rate limit state."""
        self.requests.clear()
        self.tokens = 0.0
        self.last_refill = time.time()
        self.total_requests = 0
        self.blocked_requests = 0
        self.last_request = 0.0


class RateLimiter:
    """Advanced rate limiter with multiple algorithms."""
    
    def __init__(self, rules: List[RateLimitRule]):
        """Initialize rate limiter with rules."""
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.state: Dict[str, Dict[str, RateLimitState]] = defaultdict(
            lambda: defaultdict(RateLimitState)
        )
        self._lock = threading.Lock()
        
        logger.info("Initialized RateLimiter", rule_count=len(rules))
        
        for rule in rules:
            logger.debug("Loaded rate limit rule", 
                        name=rule.name, 
                        limit=rule.limit, 
                        window=rule.window_seconds,
                        algorithm=rule.algorithm.value)
    
    def check_rate_limit(self, identifier: str, scope: RateLimitScope,
                        endpoint: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        with self._lock:
            for rule in self.rules:
                if rule.scope != scope:
                    continue
                
                # Get key for this rule
                key = self._get_state_key(identifier, rule, endpoint)
                state = self.state[rule.name][key]
                
                # Check rate limit based on algorithm
                allowed, info = self._check_algorithm(rule, state, current_time)
                
                if not allowed:
                    # Record blocked request
                    state.blocked_requests += 1
                    state.total_requests += 1
                    
                    # Log rate limit violation
                    audit_logger.log_security_event("rate_limit_exceeded",
                                                   f"Rate limit exceeded for {identifier}",
                                                   "medium", 
                                                   identifier=identifier,
                                                   rule_name=rule.name,
                                                   scope=scope.value,
                                                   endpoint=endpoint)
                    
                    record_metric("rate_limit_exceeded", 1, "counter",
                                 {"rule": rule.name, "scope": scope.value})
                    
                    return False, {
                        "rule": rule.name,
                        "limit": rule.limit,
                        "window_seconds": rule.window_seconds,
                        "algorithm": rule.algorithm.value,
                        "reset_time": info.get("reset_time"),
                        "retry_after": info.get("retry_after")
                    }
            
            # All rules passed - allow request
            for rule in self.rules:
                if rule.scope == scope:
                    key = self._get_state_key(identifier, rule, endpoint)
                    state = self.state[rule.name][key]
                    self._record_request(rule, state, current_time)
            
            record_metric("rate_limit_passed", 1, "counter", {"scope": scope.value})
            return True, {}
    
    def _get_state_key(self, identifier: str, rule: RateLimitRule, 
                      endpoint: Optional[str]) -> str:
        """Get state key for rule and identifier."""
        if rule.scope == RateLimitScope.GLOBAL:
            return "global"
        elif rule.scope == RateLimitScope.ENDPOINT and endpoint:
            return f"{identifier}:{endpoint}"
        else:
            return identifier
    
    def _check_algorithm(self, rule: RateLimitRule, state: RateLimitState,
                        current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using specified algorithm."""
        if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._check_sliding_window(rule, state, current_time)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._check_fixed_window(rule, state, current_time)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(rule, state, current_time)
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return self._check_leaky_bucket(rule, state, current_time)
        else:
            return True, {}
    
    def _check_sliding_window(self, rule: RateLimitRule, state: RateLimitState,
                             current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using sliding window algorithm."""
        window_start = current_time - rule.window_seconds
        
        # Remove old requests
        while state.requests and state.requests[0] <= window_start:
            state.requests.popleft()
        
        # Check if under limit
        current_count = len(state.requests)
        limit = rule.burst_limit or rule.limit
        
        if current_count >= limit:
            # Calculate retry after
            if state.requests:
                oldest_request = state.requests[0]
                retry_after = oldest_request + rule.window_seconds - current_time
            else:
                retry_after = rule.window_seconds
            
            return False, {
                "current_count": current_count,
                "retry_after": max(0, retry_after),
                "reset_time": current_time + retry_after
            }
        
        return True, {"current_count": current_count, "remaining": limit - current_count}
    
    def _check_fixed_window(self, rule: RateLimitRule, state: RateLimitState,
                           current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using fixed window algorithm."""
        window_start = math.floor(current_time / rule.window_seconds) * rule.window_seconds
        
        # Reset if we're in a new window
        if not state.requests or state.requests[0] < window_start:
            state.requests.clear()
        
        current_count = len(state.requests)
        limit = rule.burst_limit or rule.limit
        
        if current_count >= limit:
            next_window = window_start + rule.window_seconds
            retry_after = next_window - current_time
            
            return False, {
                "current_count": current_count,
                "retry_after": retry_after,
                "reset_time": next_window
            }
        
        return True, {"current_count": current_count, "remaining": limit - current_count}
    
    def _check_token_bucket(self, rule: RateLimitRule, state: RateLimitState,
                           current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using token bucket algorithm."""
        # Refill tokens
        time_passed = current_time - state.last_refill
        tokens_to_add = time_passed * (rule.limit / rule.window_seconds)
        state.tokens = min(rule.limit, state.tokens + tokens_to_add)
        state.last_refill = current_time
        
        # Check if token available
        if state.tokens >= 1.0:
            return True, {"tokens_remaining": state.tokens - 1.0}
        
        # Calculate retry after
        tokens_needed = 1.0 - state.tokens
        retry_after = tokens_needed / (rule.limit / rule.window_seconds)
        
        return False, {
            "tokens_remaining": state.tokens,
            "retry_after": retry_after,
            "reset_time": current_time + retry_after
        }
    
    def _check_leaky_bucket(self, rule: RateLimitRule, state: RateLimitState,
                           current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using leaky bucket algorithm."""
        # Leak requests from bucket
        time_passed = current_time - state.last_refill
        requests_to_leak = time_passed * (rule.limit / rule.window_seconds)
        
        # Remove leaked requests
        current_count = len(state.requests)
        leak_count = min(current_count, int(requests_to_leak))
        for _ in range(leak_count):
            if state.requests:
                state.requests.popleft()
        
        state.last_refill = current_time
        
        # Check if bucket has capacity
        current_count = len(state.requests)
        capacity = rule.burst_limit or rule.limit
        
        if current_count >= capacity:
            # Bucket is full
            retry_after = 1.0 / (rule.limit / rule.window_seconds)  # Time for one request to leak
            
            return False, {
                "bucket_size": current_count,
                "retry_after": retry_after,
                "reset_time": current_time + retry_after
            }
        
        return True, {"bucket_size": current_count, "capacity_remaining": capacity - current_count}
    
    def _record_request(self, rule: RateLimitRule, state: RateLimitState,
                       current_time: float) -> None:
        """Record successful request."""
        state.total_requests += 1
        state.last_request = current_time
        
        if rule.algorithm in [RateLimitAlgorithm.SLIDING_WINDOW, 
                             RateLimitAlgorithm.FIXED_WINDOW,
                             RateLimitAlgorithm.LEAKY_BUCKET]:
            state.requests.append(current_time)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            state.tokens -= 1.0
    
    def get_stats(self, identifier: str, scope: RateLimitScope) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        stats = {}
        
        with self._lock:
            for rule in self.rules:
                if rule.scope == scope:
                    key = self._get_state_key(identifier, rule, None)
                    state = self.state[rule.name][key]
                    
                    stats[rule.name] = {
                        "total_requests": state.total_requests,
                        "blocked_requests": state.blocked_requests,
                        "last_request": state.last_request,
                        "current_usage": len(state.requests),
                        "limit": rule.limit,
                        "window_seconds": rule.window_seconds,
                        "algorithm": rule.algorithm.value
                    }
        
        return stats
    
    def reset_limits(self, identifier: str, scope: RateLimitScope) -> None:
        """Reset rate limits for identifier."""
        with self._lock:
            for rule in self.rules:
                if rule.scope == scope:
                    key = self._get_state_key(identifier, rule, None)
                    if key in self.state[rule.name]:
                        self.state[rule.name][key].reset()
        
        logger.info("Reset rate limits", identifier=identifier, scope=scope.value)


class RateLimitManager:
    """Manager for multiple rate limiters with different scopes."""
    
    def __init__(self):
        """Initialize rate limit manager."""
        self.limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
        
        # Default rate limiting rules
        self._setup_default_rules()
        
        logger.info("Initialized RateLimitManager")
    
    def _setup_default_rules(self) -> None:
        """Setup default rate limiting rules."""
        default_rules = [
            # API rate limits
            RateLimitRule("api_per_user", 1000, 3600, RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.USER),
            RateLimitRule("api_per_ip", 5000, 3600, RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.IP),
            
            # Login rate limits
            RateLimitRule("login_per_ip", 5, 300, RateLimitAlgorithm.FIXED_WINDOW, RateLimitScope.IP),
            RateLimitRule("login_per_user", 3, 300, RateLimitAlgorithm.FIXED_WINDOW, RateLimitScope.USER),
            
            # Model operations
            RateLimitRule("model_upload", 10, 3600, RateLimitAlgorithm.TOKEN_BUCKET, RateLimitScope.USER),
            RateLimitRule("hardware_design", 50, 3600, RateLimitAlgorithm.LEAKY_BUCKET, RateLimitScope.USER),
            
            # Global limits
            RateLimitRule("global_api", 100000, 3600, RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.GLOBAL),
        ]
        
        self.add_limiter("default", RateLimiter(default_rules))
    
    def add_limiter(self, name: str, limiter: RateLimiter) -> None:
        """Add rate limiter."""
        with self._lock:
            self.limiters[name] = limiter
        
        logger.info("Added rate limiter", name=name)
    
    def check_limits(self, identifier: str, scope: RateLimitScope,
                    endpoint: Optional[str] = None,
                    limiter_name: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """Check rate limits across all applicable limiters."""
        limiter = self.limiters.get(limiter_name)
        if not limiter:
            logger.warning("Rate limiter not found", name=limiter_name)
            return True, {}
        
        return limiter.check_rate_limit(identifier, scope, endpoint)
    
    def get_all_stats(self, identifier: str, scope: RateLimitScope) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all limiters."""
        all_stats = {}
        
        with self._lock:
            for name, limiter in self.limiters.items():
                all_stats[name] = limiter.get_stats(identifier, scope)
        
        return all_stats


# Global rate limit manager
_rate_limit_manager: Optional[RateLimitManager] = None
_rate_limit_lock = threading.Lock()


def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager."""
    global _rate_limit_manager
    
    with _rate_limit_lock:
        if _rate_limit_manager is None:
            _rate_limit_manager = RateLimitManager()
        
        return _rate_limit_manager


def rate_limit(scope: RateLimitScope = RateLimitScope.USER,
              identifier_key: str = "user_id",
              limiter_name: str = "default"):
    """Decorator for rate limiting functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract identifier
            identifier = kwargs.get(identifier_key)
            if not identifier and hasattr(threading.current_thread(), '_current_user'):
                user = threading.current_thread()._current_user
                if scope == RateLimitScope.USER:
                    identifier = user.user_id
            
            if not identifier:
                # Use IP address as fallback
                identifier = kwargs.get('ip_address', 'unknown')
            
            # Check rate limits
            manager = get_rate_limit_manager()
            allowed, info = manager.check_limits(
                identifier, scope, 
                endpoint=f"{func.__module__}.{func.__name__}",
                limiter_name=limiter_name
            )
            
            if not allowed:
                raise SecurityError(
                    f"Rate limit exceeded: {info.get('rule', 'unknown')}",
                    "RATE_LIMIT_EXCEEDED",
                    info
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_rate_limit(identifier: str, scope: RateLimitScope,
                    endpoint: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Check rate limit for identifier and scope."""
    manager = get_rate_limit_manager()
    return manager.check_limits(identifier, scope, endpoint)