"""
Security utilities for AI Hardware Co-Design Playground.

This module provides security measures including input sanitization,
file validation, and access control mechanisms.
"""

import re
import os
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import mimetypes
from urllib.parse import urlparse

from .exceptions import SecurityError


class SecurityManager:
    """Central security manager for the platform."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or {}
        self.allowed_file_extensions = self.config.get(
            "allowed_file_extensions",
            {".onnx", ".pb", ".pt", ".pth", ".h5", ".tflite", ".json", ".yaml", ".yml"}
        )
        self.max_file_size_mb = self.config.get("max_file_size_mb", 500)
        self.allowed_directories = set(self.config.get("allowed_directories", []))
        self.blocked_patterns = self.config.get(
            "blocked_patterns",
            [r"\.\.\/", r"__pycache__", r"\.git", r"\.ssh", r"\/etc\/", r"\/root\/"]
        )
    
    def validate_file_access(self, file_path: str) -> bool:
        """
        Validate that file access is allowed.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if access is allowed
            
        Raises:
            SecurityError: If access is denied
        """
        try:
            path = Path(file_path).resolve()
            
            # Check for path traversal
            if self._contains_path_traversal(str(path)):
                raise SecurityError(
                    "Path traversal detected",
                    violation_type="path_traversal",
                    resource=file_path
                )
            
            # Check blocked patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, str(path), re.IGNORECASE):
                    raise SecurityError(
                        f"Path matches blocked pattern: {pattern}",
                        violation_type="blocked_pattern",
                        resource=file_path
                    )
            
            # Check allowed directories
            if self.allowed_directories:
                allowed = any(
                    path.is_relative_to(Path(allowed_dir).resolve())
                    for allowed_dir in self.allowed_directories
                )
                if not allowed:
                    raise SecurityError(
                        "File outside allowed directories",
                        violation_type="directory_restriction",
                        resource=file_path
                    )
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_file_extensions:
                raise SecurityError(
                    f"File extension not allowed: {path.suffix}",
                    violation_type="extension_restriction",
                    resource=file_path
                )
            
            # Check file size if exists
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    raise SecurityError(
                        f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB",
                        violation_type="size_limit",
                        resource=file_path
                    )
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"File validation failed: {e}", resource=file_path)
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Replace dangerous characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Remove multiple dots and underscores
        sanitized = re.sub(r'\.{2,}', '.', sanitized)
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        
        # Ensure reasonable length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:250] + ext
        
        # Ensure not empty
        if not sanitized or sanitized == '.':
            sanitized = f"file_{secrets.token_hex(4)}"
        
        return sanitized
    
    def validate_mime_type(self, file_path: str) -> bool:
        """
        Validate file MIME type.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if MIME type is allowed
        """
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            
            allowed_mime_types = {
                "application/octet-stream",  # ONNX, binary models
                "application/x-protobuf",    # TensorFlow protobuf
                "application/json",          # JSON configs
                "text/yaml",                 # YAML configs
                "text/plain",                # Text files
            }
            
            return mime_type in allowed_mime_types or mime_type is None
            
        except Exception:
            return False
    
    def compute_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """
        Compute file hash for integrity checking.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm
            
        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _contains_path_traversal(self, path: str) -> bool:
        """Check if path contains traversal sequences."""
        dangerous_patterns = [
            "..", "~", "//", "\\\\",
            "%2e%2e", "%2f", "%5c",
            "..%2f", "..\\", "../",
        ]
        
        path_lower = path.lower()
        return any(pattern in path_lower for pattern in dangerous_patterns)


def sanitize_input(value: Any, input_type: str = "string") -> Any:
    """
    Sanitize user input based on type.
    
    Args:
        value: Input value to sanitize
        input_type: Type of input ("string", "filename", "number", "boolean")
        
    Returns:
        Sanitized value
    """
    if value is None:
        return None
    
    if input_type == "string":
        return sanitize_string(str(value))
    elif input_type == "filename":
        return sanitize_filename(str(value))
    elif input_type == "number":
        return sanitize_number(value)
    elif input_type == "boolean":
        return sanitize_boolean(value)
    else:
        return str(value)


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'expression\s*\(',           # CSS expressions
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename input.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    manager = SecurityManager()
    return manager.sanitize_filename(filename)


def sanitize_number(value: Any) -> Union[int, float, None]:
    """
    Sanitize numeric input.
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized number or None if invalid
    """
    try:
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d\.-]', '', value)
            
            if '.' in cleaned:
                return float(cleaned)
            else:
                return int(cleaned)
        
        return None
        
    except (ValueError, TypeError):
        return None


def sanitize_boolean(value: Any) -> bool:
    """
    Sanitize boolean input.
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized boolean
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return False


def validate_file_path(file_path: str, must_exist: bool = False) -> bool:
    """
    Validate file path for security.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if path is valid and safe
        
    Raises:
        SecurityError: If path is invalid or unsafe
    """
    manager = SecurityManager()
    
    try:
        manager.validate_file_access(file_path)
        
        if must_exist:
            path = Path(file_path)
            if not path.exists():
                raise SecurityError(f"File does not exist: {file_path}")
            if not path.is_file():
                raise SecurityError(f"Path is not a file: {file_path}")
        
        return True
        
    except SecurityError:
        raise
    except Exception as e:
        raise SecurityError(f"Path validation failed: {e}")


def validate_url(url: str) -> bool:
    """
    Validate URL for security.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid and safe
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Check for localhost/private IPs (basic check)
        hostname = parsed.hostname or ''
        if hostname in ('localhost', '127.0.0.1', '::1'):
            return False
        
        # Check for private IP ranges (basic)
        if (hostname.startswith('192.168.') or 
            hostname.startswith('10.') or 
            hostname.startswith('172.')):
            return False
        
        return True
        
    except Exception:
        return False


def generate_secure_token(length: int = 32) -> str:
    """
    Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hex-encoded secure token
    """
    return secrets.token_hex(length)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare strings in constant time to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> list of timestamps
        
        # Import the new rate limiting system
        try:
            from .rate_limiting import get_rate_limit_manager, RateLimitScope
            self._advanced_limiter = get_rate_limit_manager()
            self._use_advanced = True
        except ImportError:
            self._use_advanced = False
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        # Use advanced rate limiter if available
        if self._use_advanced:
            try:
                from .rate_limiting import RateLimitScope
                allowed, _ = self._advanced_limiter.check_limits(client_id, RateLimitScope.IP)
                return allowed
            except:
                pass  # Fall back to simple implementation
        
        # Simple rate limiting implementation
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True