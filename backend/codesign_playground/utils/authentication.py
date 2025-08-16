"""
Authentication and authorization framework for AI Hardware Co-Design Playground.

This module provides comprehensive authentication, authorization, and session
management with enterprise-grade security features.
"""

import time
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import threading
import json

from .logging import get_logger, get_audit_logger
from .monitoring import record_metric
from .exceptions import SecurityError
from .security import constant_time_compare, generate_secure_token

logger = get_logger(__name__)
audit_logger = get_audit_logger(__name__)


class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    # Model operations
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write" 
    MODEL_DELETE = "model:delete"
    
    # Hardware operations
    HARDWARE_READ = "hardware:read"
    HARDWARE_WRITE = "hardware:write"
    HARDWARE_DESIGN = "hardware:design"
    
    # Workflow operations
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_EXECUTE = "workflow:execute"
    WORKFLOW_MANAGE = "workflow:manage"
    
    # System operations
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIGURE = "system:configure"
    SYSTEM_ADMIN = "system:admin"
    
    # User management
    USER_READ = "user:read"
    USER_MANAGE = "user:manage"


@dataclass
class User:
    """User account with roles and permissions."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    password_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or self._role_has_permission(permission)
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def _role_has_permission(self, permission: Permission) -> bool:
        """Check if any user role has the permission."""
        role_permissions = {
            UserRole.ADMIN: list(Permission),  # Admin has all permissions
            UserRole.DEVELOPER: [
                Permission.MODEL_READ, Permission.MODEL_WRITE,
                Permission.HARDWARE_READ, Permission.HARDWARE_WRITE, Permission.HARDWARE_DESIGN,
                Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE,
                Permission.SYSTEM_MONITOR
            ],
            UserRole.RESEARCHER: [
                Permission.MODEL_READ, Permission.MODEL_WRITE,
                Permission.HARDWARE_READ, Permission.HARDWARE_DESIGN,
                Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE,
                Permission.SYSTEM_MONITOR
            ],
            UserRole.VIEWER: [
                Permission.MODEL_READ, Permission.HARDWARE_READ,
                Permission.WORKFLOW_READ, Permission.SYSTEM_MONITOR
            ],
            UserRole.GUEST: [
                Permission.MODEL_READ, Permission.HARDWARE_READ
            ]
        }
        
        for role in self.roles:
            if permission in role_permissions.get(role, []):
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [role.value for role in self.roles],
            "permissions": [perm.value for perm in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """User session with expiration and security tracking."""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid and active."""
        return self.is_active and not self.is_expired()
    
    def refresh(self, extend_hours: int = 24) -> None:
        """Refresh session expiration."""
        self.last_activity = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=extend_hours)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


class AuthenticationManager:
    """Comprehensive authentication and session management."""
    
    def __init__(self, secret_key: Optional[str] = None, 
                 session_timeout_hours: int = 24,
                 max_failed_attempts: int = 5,
                 lockout_duration_minutes: int = 30):
        """Initialize authentication manager."""
        self.secret_key = secret_key or generate_secure_token(32)
        self.session_timeout_hours = session_timeout_hours
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        
        # In-memory storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        
        self._lock = threading.Lock()
        
        # Create default admin user
        self._create_default_admin()
        
        logger.info("Initialized AuthenticationManager",
                   session_timeout_hours=session_timeout_hours,
                   max_failed_attempts=max_failed_attempts)
    
    def _create_default_admin(self) -> None:
        """Create default admin user for initial setup."""
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@localhost",
            roles=[UserRole.ADMIN],
            password_hash=self._hash_password("admin123!")  # Change in production!
        )
        self.users["admin"] = admin_user
        
        logger.warning("Created default admin user - CHANGE PASSWORD IN PRODUCTION!")
    
    def create_user(self, username: str, email: str, password: str,
                   roles: Optional[List[UserRole]] = None,
                   permissions: Optional[List[Permission]] = None) -> User:
        """Create a new user account."""
        user_id = generate_secure_token(16)
        
        # Validate inputs
        if len(password) < 8:
            raise SecurityError("Password must be at least 8 characters")
        
        if username in [u.username for u in self.users.values()]:
            raise SecurityError("Username already exists")
        
        if email in [u.email for u in self.users.values()]:
            raise SecurityError("Email already registered")
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or [UserRole.VIEWER],
            permissions=permissions or [],
            password_hash=self._hash_password(password)
        )
        
        with self._lock:
            self.users[user_id] = user
        
        audit_logger.log_security_event("user_created",
                                       f"User {username} created",
                                       "low", user_id=user_id, username=username)
        
        record_metric("user_created", 1, "counter")
        
        logger.info("Created user account", username=username, user_id=user_id)
        return user
    
    def authenticate(self, username: str, password: str,
                    ip_address: Optional[str] = None,
                    user_agent: Optional[str] = None) -> Optional[Session]:
        """Authenticate user and create session."""
        # Check if account is locked
        if self._is_account_locked(username):
            audit_logger.log_security_event("locked_account_access",
                                           f"Attempt to access locked account: {username}",
                                           "high", username=username, ip_address=ip_address)
            raise SecurityError("Account is temporarily locked due to failed login attempts")
        
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break
        
        if not user:
            self._record_failed_attempt(username)
            audit_logger.log_security_event("invalid_username",
                                           f"Login attempt with invalid username: {username}",
                                           "medium", username=username, ip_address=ip_address)
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._record_failed_attempt(username)
            audit_logger.log_security_event("invalid_password",
                                           f"Invalid password for user: {username}",
                                           "medium", username=username, ip_address=ip_address,
                                           user_id=user.user_id)
            return None
        
        # Clear failed attempts on successful login
        self._clear_failed_attempts(username)
        
        # Create session
        session = self._create_session(user.user_id, ip_address, user_agent)
        
        # Update user last login
        user.last_login = datetime.utcnow()
        
        audit_logger.log_security_event("user_login",
                                       f"User {username} logged in successfully",
                                       "low", username=username, user_id=user.user_id,
                                       session_id=session.session_id, ip_address=ip_address)
        
        record_metric("user_login_success", 1, "counter", {"username": username})
        
        logger.info("User authenticated successfully", username=username, user_id=user.user_id)
        return session
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user if valid."""
        with self._lock:
            session = self.sessions.get(session_id)
            
            if not session or not session.is_valid():
                if session:
                    # Remove invalid session
                    del self.sessions[session_id]
                return None
            
            # Refresh session activity
            session.refresh()
            
            # Get user
            user = self.users.get(session.user_id)
            if not user or not user.is_active:
                # Remove session for inactive user
                del self.sessions[session_id]
                return None
            
            return user
    
    def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        with self._lock:
            session = self.sessions.get(session_id)
            
            if session:
                session.is_active = False
                user = self.users.get(session.user_id)
                
                audit_logger.log_security_event("user_logout",
                                               f"User logged out",
                                               "low", user_id=session.user_id,
                                               session_id=session_id)
                
                record_metric("user_logout", 1, "counter")
                
                logger.info("User logged out", user_id=session.user_id, session_id=session_id)
                return True
            
            return False
    
    def generate_jwt_token(self, user: User, expires_hours: int = 24) -> str:
        """Generate JWT token for API access."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=expires_hours)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        audit_logger.log_security_event("jwt_token_generated",
                                       f"JWT token generated for user {user.username}",
                                       "low", user_id=user.user_id)
        
        return token
    
    def validate_jwt_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if user_id:
                user = self.users.get(user_id)
                if user and user.is_active:
                    return user
            
        except jwt.ExpiredSignatureError:
            audit_logger.log_security_event("jwt_token_expired",
                                           "Attempt to use expired JWT token",
                                           "low")
        except jwt.InvalidTokenError:
            audit_logger.log_security_event("jwt_token_invalid",
                                           "Attempt to use invalid JWT token",
                                           "medium")
        
        return None
    
    def _create_session(self, user_id: str, ip_address: Optional[str],
                       user_agent: Optional[str]) -> Session:
        """Create a new session."""
        session_id = generate_secure_token(32)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(hours=self.session_timeout_hours),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self._lock:
            self.sessions[session_id] = session
        
        return session
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                           password.encode('utf-8'), 
                                           salt.encode('utf-8'), 
                                           100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, stored_hash = password_hash.split(':', 1)
            password_hash_check = hashlib.pbkdf2_hmac('sha256',
                                                     password.encode('utf-8'),
                                                     salt.encode('utf-8'),
                                                     100000)
            return constant_time_compare(stored_hash, password_hash_check.hex())
        except:
            return False
    
    def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        current_time = datetime.utcnow()
        
        with self._lock:
            if username not in self.failed_attempts:
                self.failed_attempts[username] = []
            
            self.failed_attempts[username].append(current_time)
            
            # Clean old attempts (older than lockout duration)
            cutoff_time = current_time - timedelta(minutes=self.lockout_duration_minutes)
            self.failed_attempts[username] = [
                attempt for attempt in self.failed_attempts[username]
                if attempt > cutoff_time
            ]
            
            # Check if account should be locked
            if len(self.failed_attempts[username]) >= self.max_failed_attempts:
                self.locked_accounts[username] = current_time + timedelta(minutes=self.lockout_duration_minutes)
                
                audit_logger.log_security_event("account_locked",
                                               f"Account {username} locked due to failed attempts",
                                               "high", username=username)
                
                record_metric("account_locked", 1, "counter", {"username": username})
    
    def _clear_failed_attempts(self, username: str) -> None:
        """Clear failed login attempts for user."""
        with self._lock:
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            if username in self.locked_accounts:
                del self.locked_accounts[username]
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked."""
        with self._lock:
            if username in self.locked_accounts:
                unlock_time = self.locked_accounts[username]
                if datetime.utcnow() >= unlock_time:
                    # Unlock account
                    del self.locked_accounts[username]
                    if username in self.failed_attempts:
                        del self.failed_attempts[username]
                    return False
                return True
            return False


# Global authentication manager
_auth_manager: Optional[AuthenticationManager] = None
_auth_lock = threading.Lock()


def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager instance."""
    global _auth_manager
    
    with _auth_lock:
        if _auth_manager is None:
            _auth_manager = AuthenticationManager()
        
        return _auth_manager


def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user from context or kwargs
            user = kwargs.get('_user') or getattr(threading.current_thread(), '_current_user', None)
            
            if not user:
                raise SecurityError("Authentication required", "AUTHENTICATION_REQUIRED")
            
            if not user.has_permission(permission):
                audit_logger.log_security_event("permission_denied",
                                               f"User {user.username} denied access to {func.__name__}",
                                               "medium", user_id=user.user_id,
                                               permission=permission.value)
                raise SecurityError(f"Permission denied: {permission.value}", "PERMISSION_DENIED")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: UserRole):
    """Decorator to require specific role."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = kwargs.get('_user') or getattr(threading.current_thread(), '_current_user', None)
            
            if not user:
                raise SecurityError("Authentication required", "AUTHENTICATION_REQUIRED")
            
            if not user.has_role(role):
                audit_logger.log_security_event("role_denied",
                                               f"User {user.username} denied access to {func.__name__}",
                                               "medium", user_id=user.user_id,
                                               required_role=role.value)
                raise SecurityError(f"Role required: {role.value}", "ROLE_REQUIRED")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def authenticate_request(session_id: Optional[str] = None, 
                        jwt_token: Optional[str] = None) -> Optional[User]:
    """Authenticate request using session ID or JWT token."""
    auth_manager = get_auth_manager()
    
    if session_id:
        return auth_manager.validate_session(session_id)
    elif jwt_token:
        return auth_manager.validate_jwt_token(jwt_token)
    
    return None