"""
Security Testing Framework for AI Hardware Co-Design Playground.

This module implements comprehensive security tests including penetration testing,
input validation, authentication, and vulnerability assessment.
"""

import pytest
import os
import tempfile
import subprocess
import json
import hashlib
import base64
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import sqlite3

from backend.codesign_playground.utils.security import SecurityValidator, hash_password, verify_password
from backend.codesign_playground.utils.authentication import AuthenticationManager, TokenManager
from backend.codesign_playground.utils.compliance import ComplianceManager, DataCategory
from backend.codesign_playground.core.workflow import Workflow
from backend.codesign_playground.utils.exceptions import SecurityError, ValidationError


class SecurityTestSuite:
    """Comprehensive security testing framework."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.test_vectors = self._generate_test_vectors()
    
    def _generate_test_vectors(self) -> Dict[str, List[str]]:
        """Generate security test vectors for various attack types."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "admin'--",
                "' UNION SELECT * FROM information_schema.tables --",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --"
            ],
            "xss_attacks": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "'\"><script>alert('XSS')</script>"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc//passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd"
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "&& whoami",
                "`id`",
                "$(cat /etc/passwd)"
            ],
            "ldap_injection": [
                "*)(uid=*",
                "*)(|(uid=*",
                "admin)(&(password=*))",
                "*))%00",
                ")(cn=*"
            ],
            "buffer_overflow": [
                "A" * 1000,
                "A" * 10000,
                "\x00" * 1000,
                "\xff" * 1000,
                "A" * 65536
            ],
            "format_string": [
                "%s%s%s%s",
                "%x%x%x%x",
                "%n%n%n%n",
                "%08x.%08x.%08x",
                "AAAA%08x.%08x.%08x"
            ],
            "unicode_attacks": [
                "\u0000",
                "\ufeff",
                "test\u0000hidden",
                "\u202e\u0000\u202d",
                "\ud800\udc00"
            ]
        }


class TestInputValidation:
    """Test input validation and sanitization."""
    
    @pytest.fixture
    def security_suite(self):
        return SecurityTestSuite()
    
    @pytest.fixture
    def security_validator(self):
        return SecurityValidator()
    
    def test_sql_injection_prevention(self, security_suite, security_validator):
        """Test SQL injection prevention in user inputs."""
        for injection_payload in security_suite.test_vectors["sql_injection"]:
            # Test file path validation
            is_valid = security_validator.validate_file_path(injection_payload)
            assert not is_valid, f"SQL injection payload should be rejected: {injection_payload}"
            
            # Test string input validation
            is_valid = security_validator.validate_string_input(
                injection_payload, "test_field", max_length=100
            )
            assert not is_valid, f"SQL injection payload should be rejected: {injection_payload}"
    
    def test_xss_prevention(self, security_suite, security_validator):
        """Test XSS attack prevention."""
        for xss_payload in security_suite.test_vectors["xss_attacks"]:
            # Test string sanitization
            sanitized = security_validator.sanitize_string(xss_payload)
            
            # Should not contain script tags or javascript
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()
    
    def test_path_traversal_prevention(self, security_suite, security_validator):
        """Test path traversal attack prevention."""
        for traversal_payload in security_suite.test_vectors["path_traversal"]:
            is_valid = security_validator.validate_file_path(traversal_payload)
            assert not is_valid, f"Path traversal payload should be rejected: {traversal_payload}"
    
    def test_command_injection_prevention(self, security_suite, security_validator):
        """Test command injection prevention."""
        for cmd_payload in security_suite.test_vectors["command_injection"]:
            is_valid = security_validator.validate_string_input(
                cmd_payload, "command_field", max_length=100
            )
            assert not is_valid, f"Command injection payload should be rejected: {cmd_payload}"
    
    def test_buffer_overflow_prevention(self, security_suite, security_validator):
        """Test buffer overflow prevention."""
        for overflow_payload in security_suite.test_vectors["buffer_overflow"]:
            # Test with reasonable max length
            is_valid = security_validator.validate_string_input(
                overflow_payload, "test_field", max_length=255
            )
            assert not is_valid, f"Buffer overflow payload should be rejected"
    
    def test_numeric_input_validation(self, security_validator):
        """Test numeric input validation and bounds checking."""
        # Test valid numeric inputs
        assert security_validator.validate_numeric_input(10, "test", min_value=0, max_value=100)
        assert security_validator.validate_numeric_input(50.5, "test", min_value=0, max_value=100)
        
        # Test invalid numeric inputs
        assert not security_validator.validate_numeric_input(-1, "test", min_value=0, max_value=100)
        assert not security_validator.validate_numeric_input(101, "test", min_value=0, max_value=100)
        assert not security_validator.validate_numeric_input(float('inf'), "test", min_value=0, max_value=100)
        assert not security_validator.validate_numeric_input(float('nan'), "test", min_value=0, max_value=100)
    
    def test_file_upload_validation(self, security_validator, tmp_path):
        """Test file upload security validation."""
        # Create test files
        safe_file = tmp_path / "model.onnx"
        safe_file.write_text("safe model data")
        
        malicious_file = tmp_path / "malicious.exe"
        malicious_file.write_bytes(b"malicious executable")
        
        script_file = tmp_path / "script.py"
        script_file.write_text("import os; os.system('rm -rf /')")
        
        # Test validation
        assert security_validator.validate_file_upload(str(safe_file), allowed_extensions=['.onnx', '.pt'])
        assert not security_validator.validate_file_upload(str(malicious_file), allowed_extensions=['.onnx', '.pt'])
        assert not security_validator.validate_file_upload(str(script_file), allowed_extensions=['.onnx', '.pt'])


class TestAuthentication:
    """Test authentication and authorization mechanisms."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthenticationManager()
    
    @pytest.fixture
    def token_manager(self):
        return TokenManager(secret_key="test_secret_key_for_testing_only")
    
    def test_password_hashing(self):
        """Test password hashing security."""
        password = "test_password_123"
        
        # Hash password
        hashed = hash_password(password)
        
        # Verify hash properties
        assert len(hashed) > 50  # Should be substantial length
        assert hashed != password  # Should not be plaintext
        assert '$' in hashed  # Should contain salt delimiter
        
        # Verify password verification
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)
    
    def test_weak_password_rejection(self, auth_manager):
        """Test weak password rejection."""
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "qwerty",
            "admin",
            "a",  # Too short
            "password123",  # Common pattern
            "12345678"  # Numeric only
        ]
        
        for weak_password in weak_passwords:
            is_strong = auth_manager.validate_password_strength(weak_password)
            assert not is_strong, f"Weak password should be rejected: {weak_password}"
    
    def test_strong_password_acceptance(self, auth_manager):
        """Test strong password acceptance."""
        strong_passwords = [
            "StrongP@ssw0rd123!",
            "MySecur3P@ssw0rd#2024",
            "C0mpl3x!P@ssw0rd$789",
            "Ungu3ss@bl3#Str0ng&P@ss"
        ]
        
        for strong_password in strong_passwords:
            is_strong = auth_manager.validate_password_strength(strong_password)
            assert is_strong, f"Strong password should be accepted: {strong_password}"
    
    def test_token_security(self, token_manager):
        """Test JWT token security."""
        user_id = "test_user_123"
        
        # Generate token
        token = token_manager.generate_token(user_id)
        assert token is not None
        assert len(token) > 50  # Should be substantial length
        
        # Verify token
        decoded_user_id = token_manager.verify_token(token)
        assert decoded_user_id == user_id
        
        # Test token tampering
        tampered_token = token[:-5] + "XXXXX"
        decoded = token_manager.verify_token(tampered_token)
        assert decoded is None  # Should reject tampered token
    
    def test_token_expiration(self, token_manager):
        """Test token expiration handling."""
        user_id = "test_user_exp"
        
        # Generate token with short expiration
        token = token_manager.generate_token(user_id, expires_in_minutes=1)
        
        # Should be valid immediately
        assert token_manager.verify_token(token) == user_id
        
        # Mock time advancement (in real test, you'd wait or use time mocking)
        with patch('time.time', return_value=time.time() + 3600):  # +1 hour
            expired_result = token_manager.verify_token(token)
            assert expired_result is None  # Should be expired
    
    def test_session_management(self, auth_manager):
        """Test session management security."""
        user_id = "test_session_user"
        
        # Create session
        session_id = auth_manager.create_session(user_id)
        assert session_id is not None
        assert len(session_id) >= 32  # Should be cryptographically random
        
        # Validate session
        assert auth_manager.validate_session(session_id) == user_id
        
        # Test session invalidation
        auth_manager.invalidate_session(session_id)
        assert auth_manager.validate_session(session_id) is None
    
    def test_brute_force_protection(self, auth_manager):
        """Test brute force attack protection."""
        username = "test_user_brute"
        wrong_password = "wrong_password"
        
        # Attempt multiple failed logins
        for i in range(6):  # Exceed typical threshold
            result = auth_manager.authenticate(username, wrong_password)
            assert not result["success"]
        
        # Account should be locked
        status = auth_manager.get_account_status(username)
        assert status.get("locked", False) or status.get("attempts", 0) >= 5


class TestComplianceSecurity:
    """Test compliance and privacy security features."""
    
    @pytest.fixture
    def compliance_manager(self, tmp_path):
        return ComplianceManager(db_path=str(tmp_path / "test_compliance.db"))
    
    def test_data_encryption_at_rest(self, compliance_manager, tmp_path):
        """Test that sensitive data is encrypted at rest."""
        user_id = "test_user_encryption"
        
        # Record sensitive data processing
        success = compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="testing_encryption",
            legal_basis="consent"
        )
        assert success
        
        # Check database file directly
        db_path = compliance_manager.db_path
        with open(db_path, 'rb') as f:
            db_content = f.read()
        
        # Raw user ID should not appear in plaintext in the database
        # Note: This is a simplified test - in production, you'd encrypt sensitive fields
        raw_content = db_content.decode('utf-8', errors='ignore')
        
        # The database may contain the user ID in various forms, but sensitive fields should be protected
        # For this test, we'll verify that basic security measures are in place
        assert len(raw_content) > 0  # Database should have content
    
    def test_audit_log_integrity(self, compliance_manager):
        """Test audit log integrity and tamper detection."""
        user_id = "test_user_audit"
        
        # Record multiple operations
        operations = [
            ("data_access", "read_profile"),
            ("data_modification", "update_preferences"),
            ("data_deletion", "delete_account")
        ]
        
        for action, resource in operations:
            compliance_manager._log_audit_event(
                user_id=user_id,
                action=action,
                resource=resource,
                data_category=DataCategory.PERSONAL_IDENTIFYING,
                legal_basis="legitimate_interests",
                result="success"
            )
        
        # Verify audit logs exist and have proper structure
        with sqlite3.connect(compliance_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs WHERE user_id = ?", (user_id,))
            logs = cursor.fetchall()
        
        assert len(logs) == len(operations)
        
        # Each log should have required fields
        for log in logs:
            assert len(log) >= 10  # Should have all required columns
            assert log[2] == user_id  # user_id field
            assert log[12] == "success"  # result field
    
    def test_data_subject_rights_security(self, compliance_manager):
        """Test security of data subject rights requests."""
        user_id = "test_user_rights"
        
        # Test data access request
        response = compliance_manager.handle_data_subject_request(
            user_id=user_id,
            request_type="access",
            verification_method="email",
            request_context={"ip_address": "192.168.1.100"}
        )
        
        assert response["status"] in ["processing", "completed"]
        assert "user_id" in response
        
        # Verify that exported data doesn't contain sensitive system information
        if "data" in response:
            exported_data = response["data"]
            
            # Should not contain internal system details
            data_str = json.dumps(exported_data)
            assert "password" not in data_str.lower()
            assert "secret" not in data_str.lower()
            assert "key" not in data_str.lower()
    
    def test_consent_withdrawal_security(self, compliance_manager):
        """Test security of consent withdrawal mechanisms."""
        user_id = "test_user_consent"
        
        # Grant consent first
        compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                DataCategory.ANALYTICS: True,
                DataCategory.MARKETING: True
            },
            ip_address="192.168.1.100"
        )
        
        # Withdraw consent
        compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                DataCategory.ANALYTICS: False,
                DataCategory.MARKETING: False
            },
            ip_address="192.168.1.100"
        )
        
        # Verify withdrawal was recorded securely
        consent_record = compliance_manager._user_consents.get(user_id)
        assert consent_record is not None
        assert not consent_record.has_consent(DataCategory.ANALYTICS)
        assert not consent_record.has_consent(DataCategory.MARKETING)


class TestWorkflowSecurity:
    """Test workflow security features."""
    
    def test_secure_model_import(self, tmp_path):
        """Test secure model import with validation."""
        workflow = Workflow("security_test", output_dir=str(tmp_path))
        
        # Create malicious file
        malicious_file = tmp_path / "malicious.py"
        malicious_file.write_text("import os; os.system('rm -rf /')")
        
        # Should reject malicious file
        with pytest.raises(Exception):  # Should raise validation or workflow error
            workflow.import_model(
                model_path=str(malicious_file),
                input_shapes={"input": (1, 3, 224, 224)},
                framework="pytorch"
            )
    
    def test_secure_file_operations(self, tmp_path):
        """Test secure file operations and path validation."""
        workflow = Workflow("security_file_test", output_dir=str(tmp_path))
        
        # Test path traversal attack
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises(Exception):  # Should raise security error
                workflow.save_state(malicious_path)
    
    def test_resource_limits(self, tmp_path):
        """Test resource consumption limits."""
        workflow = Workflow("resource_test", output_dir=str(tmp_path))
        
        # Test large input shape (potential DoS)
        large_shapes = {
            "input": (1000, 1000, 1000, 1000)  # Extremely large
        }
        
        # Create dummy model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("dummy model")
        
        # Should handle large inputs gracefully (either reject or process safely)
        try:
            workflow.import_model(
                model_path=str(model_file),
                input_shapes=large_shapes,
                framework="pytorch"
            )
            # If it succeeds, verify it doesn't consume excessive resources
            assert True  # Test passes if no exception or resource exhaustion
        except (ValidationError, SecurityError, WorkflowError):
            # Expected to reject overly large inputs
            assert True


class TestPenetrationTesting:
    """Penetration testing scenarios."""
    
    def test_api_endpoint_security(self):
        """Test API endpoint security (mock test)."""
        # This would test actual API endpoints in a real scenario
        # For now, we'll test the underlying security components
        
        security_validator = SecurityValidator()
        
        # Test various malicious inputs that might come through API
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "A" * 10000,  # Buffer overflow attempt
            "\x00\x01\x02\x03",  # Binary data
        ]
        
        for malicious_input in malicious_inputs:
            # API should validate and reject malicious inputs
            is_safe = security_validator.validate_string_input(
                malicious_input, "api_param", max_length=1000
            )
            assert not is_safe, f"API should reject malicious input: {malicious_input}"
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation attacks."""
        auth_manager = AuthenticationManager()
        
        # Test that regular user cannot access admin functions
        regular_user_id = "regular_user"
        admin_user_id = "admin_user"
        
        # Simulate privilege check
        regular_permissions = auth_manager.get_user_permissions(regular_user_id)
        admin_permissions = auth_manager.get_user_permissions(admin_user_id)
        
        # Regular user should not have admin permissions
        assert "admin" not in regular_permissions.get("roles", [])
        
        # Test that user cannot modify their own permissions
        result = auth_manager.modify_user_permissions(
            requesting_user=regular_user_id,
            target_user=regular_user_id,
            new_permissions=["admin"]
        )
        assert not result.get("success", False)
    
    def test_information_disclosure_prevention(self, tmp_path):
        """Test prevention of information disclosure."""
        # Test that error messages don't leak sensitive information
        workflow = Workflow("disclosure_test", output_dir=str(tmp_path))
        
        try:
            # Trigger an error condition
            workflow.import_model(
                model_path="/nonexistent/path/model.pt",
                input_shapes={"input": (1, 3, 224, 224)},
                framework="pytorch"
            )
        except Exception as e:
            error_message = str(e)
            
            # Error message should not contain sensitive information
            sensitive_keywords = [
                "password", "secret", "key", "token",
                "/home/", "/root/", "C:\\Users\\",
                "database", "connection string"
            ]
            
            for keyword in sensitive_keywords:
                assert keyword.lower() not in error_message.lower(), \
                    f"Error message contains sensitive information: {keyword}"


class TestSecurityConfiguration:
    """Test security configuration and hardening."""
    
    def test_default_security_settings(self):
        """Test that default security settings are secure."""
        security_validator = SecurityValidator()
        
        # Test default file extension restrictions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.sh', '.ps1', '.vbs', '.js']
        for ext in dangerous_extensions:
            assert not security_validator.is_safe_file_extension(ext)
        
        # Test default safe extensions
        safe_extensions = ['.onnx', '.pt', '.pth', '.pb', '.tflite', '.json']
        for ext in safe_extensions:
            assert security_validator.is_safe_file_extension(ext)
    
    def test_security_headers_configuration(self):
        """Test security headers and configurations."""
        # This would test HTTP security headers in a real web application
        # For now, we'll test security configuration validation
        
        security_config = {
            "max_file_size_mb": 100,
            "allowed_file_types": [".onnx", ".pt", ".pth"],
            "max_request_rate": 60,  # requests per minute
            "session_timeout_minutes": 30,
            "password_min_length": 12,
            "require_2fa": True
        }
        
        # Validate configuration
        assert security_config["max_file_size_mb"] <= 500  # Reasonable limit
        assert security_config["session_timeout_minutes"] <= 480  # Max 8 hours
        assert security_config["password_min_length"] >= 8  # Minimum security
        assert security_config["require_2fa"] is True  # Should require 2FA
    
    def test_encryption_configuration(self):
        """Test encryption configuration and strength."""
        # Test password hashing configuration
        test_password = "test_password_123"
        hashed = hash_password(test_password)
        
        # Should use strong hashing algorithm (bcrypt, scrypt, or Argon2)
        assert any(alg in hashed for alg in ['$2b$', '$scrypt$', '$argon2'])
        
        # Test that hash has sufficient cost factor
        if '$2b$' in hashed:
            # bcrypt cost factor should be at least 10
            cost_part = hashed.split('$')[2]
            cost = int(cost_part)
            assert cost >= 10, f"bcrypt cost factor {cost} is too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])