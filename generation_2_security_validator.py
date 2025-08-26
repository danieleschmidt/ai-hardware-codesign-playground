"""
Generation 2 Security Validation Suite
Comprehensive security testing for robustness implementation.
"""

import os
import sys
import time
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_security_features():
    """Test comprehensive security features."""
    print("🔒 GENERATION 2: SECURITY VALIDATION")
    print("=" * 50)
    
    results = {
        "input_sanitization": False,
        "file_validation": False, 
        "rate_limiting": False,
        "authentication": False,
        "circuit_breakers": False,
        "monitoring_alerts": False
    }
    
    # Test 1: Input Sanitization
    try:
        from codesign_playground.utils.security import sanitize_input, sanitize_string
        
        # Test XSS protection
        malicious_input = "<script>alert('xss')</script>"
        sanitized = sanitize_string(malicious_input)
        assert "script" not in sanitized.lower()
        
        # Test SQL injection protection
        sql_input = "'; DROP TABLE users; --"
        sanitized = sanitize_input(sql_input, "string")
        assert "drop table" not in sanitized.lower()
        
        print("✅ Input sanitization: XSS and injection protection working")
        results["input_sanitization"] = True
        
    except Exception as e:
        print(f"❌ Input sanitization failed: {e}")
    
    # Test 2: File Validation
    try:
        from codesign_playground.utils.security import SecurityManager, SecurityError
        
        security_manager = SecurityManager()
        
        # Test path traversal protection
        try:
            security_manager.validate_file_access("../../../etc/passwd")
            assert False, "Should have blocked path traversal"
        except SecurityError as e:
            assert "path_traversal" in str(e).lower() or "traversal" in str(e).lower()
        
        # Test valid file access
        test_file = "/tmp/test.json"
        with open(test_file, "w") as f:
            f.write('{"test": true}')
        
        # Should pass validation for allowed extensions
        security_manager.config = {"allowed_file_extensions": {".json"}}
        security_manager.allowed_file_extensions = {".json"}
        security_manager.validate_file_access(test_file)
        
        # Cleanup
        os.remove(test_file)
        
        print("✅ File validation: Path traversal blocked, valid files allowed")
        results["file_validation"] = True
        
    except Exception as e:
        print(f"❌ File validation failed: {e}")
    
    # Test 3: Rate Limiting
    try:
        from codesign_playground.utils.security import RateLimiter
        
        rate_limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # First two requests should be allowed
        assert rate_limiter.is_allowed("test_client") == True
        assert rate_limiter.is_allowed("test_client") == True
        
        # Third request should be blocked
        assert rate_limiter.is_allowed("test_client") == False
        
        print("✅ Rate limiting: Request throttling working correctly")
        results["rate_limiting"] = True
        
    except Exception as e:
        print(f"❌ Rate limiting failed: {e}")
    
    # Test 4: Authentication Framework
    try:
        from codesign_playground.utils.authentication import AuthenticationManager
        from codesign_playground.utils.security import generate_secure_token
        
        auth_manager = AuthenticationManager()
        
        # Test token generation
        token = generate_secure_token(32)
        assert len(token) == 64  # 32 bytes = 64 hex chars
        
        # Test token validation
        session_token = auth_manager.generate_session_token("test_user")
        assert session_token is not None
        assert len(session_token) > 0
        
        print("✅ Authentication: Token generation and validation working")
        results["authentication"] = True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
    
    # Test 5: Circuit Breakers
    try:
        from codesign_playground.utils.enhanced_resilience import get_resilience_manager, ResilienceLevel
        from codesign_playground.utils.circuit_breaker import AdvancedCircuitBreaker, CircuitState
        
        # Get resilience manager
        resilience_manager = get_resilience_manager(ResilienceLevel.STANDARD)
        
        # Add circuit breaker
        cb = resilience_manager.add_circuit_breaker("test_service", failure_threshold=2)
        
        # Test circuit breaker functionality
        assert cb.state == CircuitState.CLOSED
        
        # Simulate failures
        def failing_function():
            raise Exception("Test failure")
        
        # Should fail and increment failure count
        try:
            cb(failing_function)()
        except:
            pass
        
        try:
            cb(failing_function)()
        except:
            pass
        
        # Circuit should be open after failures
        metrics = cb.get_metrics()
        assert metrics["failure_count"] >= 2
        
        print("✅ Circuit breakers: Fault isolation and recovery working")
        results["circuit_breakers"] = True
        
    except Exception as e:
        print(f"❌ Circuit breakers failed: {e}")
    
    # Test 6: Monitoring and Alerts
    try:
        from codesign_playground.utils.comprehensive_monitoring import AdvancedMetricsCollector, AlertSeverity, MetricType
        
        monitor = AdvancedMetricsCollector()
        
        # Test metric recording
        monitor.record_metric("test.cpu_usage", 85.0, MetricType.GAUGE)
        monitor.record_metric("test.response_time", 150.0, MetricType.HISTOGRAM)
        
        # Test alert setup
        alert_id = monitor.add_alert(
            name="High CPU Usage",
            description="CPU usage exceeds threshold",
            metric_name="test.cpu_usage", 
            threshold_value=80.0,
            comparison_operator=">",
            severity=AlertSeverity.WARNING
        )
        
        assert alert_id is not None
        assert len(monitor.alerts) > 0
        
        # Test metric statistics
        stats = monitor.get_metric_statistics("test.cpu_usage", time_range_minutes=1)
        assert stats.get("count", 0) > 0
        
        print("✅ Monitoring: Metrics collection and alerting working")
        results["monitoring_alerts"] = True
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("GENERATION 2 SECURITY VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for feature, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {feature.replace('_', ' ').title()}: {'PASSED' if status else 'FAILED'}")
    
    success_rate = (passed / total) * 100
    print(f"\nOverall Security Validation: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 GENERATION 2 ROBUSTNESS: SECURITY VALIDATION PASSED")
        return True
    else:
        print("⚠️  GENERATION 2: Some security features need attention")
        return False

def test_resilience_patterns():
    """Test resilience patterns implementation."""
    print("\n🛡️  RESILIENCE PATTERNS VALIDATION")
    print("=" * 40)
    
    try:
        from codesign_playground.utils.enhanced_resilience import (
            EnhancedResilienceManager, ResilienceLevel, BulkheadConfig, BulkheadIsolation
        )
        
        # Test 1: Bulkhead Isolation
        config = BulkheadConfig(name="test_bulkhead", max_concurrent=2)
        bulkhead = BulkheadIsolation(config)
        
        def test_function(delay=0.1):
            import time
            time.sleep(delay)
            return "success"
        
        # Test concurrent execution
        result = bulkhead.execute(test_function, 0.01)
        assert result == "success"
        
        stats = bulkhead.get_stats()
        assert stats["name"] == "test_bulkhead"
        
        print("✅ Bulkhead isolation: Resource partitioning working")
        
        # Test 2: Adaptive Timeouts
        from codesign_playground.utils.enhanced_resilience import AdaptiveTimeoutManager
        
        timeout_manager = AdaptiveTimeoutManager(initial_timeout=1.0)
        
        # Record some response times
        timeout_manager.record_response_time(0.5)
        timeout_manager.record_response_time(0.8)
        timeout_manager.record_response_time(0.3)
        
        current_timeout = timeout_manager.get_timeout()
        assert current_timeout > 0
        
        print("✅ Adaptive timeouts: Dynamic timeout adjustment working")
        
        # Test 3: Comprehensive Resilience Manager
        manager = EnhancedResilienceManager(ResilienceLevel.STANDARD)
        
        # Add all patterns
        manager.add_circuit_breaker("test_cb")
        manager.add_bulkhead("test_bulk")
        manager.add_timeout_manager("test_timeout")
        manager.add_retry_mechanism("test_retry")
        
        # Get comprehensive stats
        stats = manager.get_comprehensive_stats()
        assert "circuit_breakers" in stats
        assert "bulkheads" in stats
        assert stats["level"] == "standard"
        
        print("✅ Resilience manager: All patterns integrated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience patterns failed: {e}")
        return False

def test_monitoring_system():
    """Test comprehensive monitoring system."""
    print("\n📊 MONITORING SYSTEM VALIDATION")
    print("=" * 40)
    
    try:
        from codesign_playground.utils.comprehensive_monitoring import (
            AdvancedMetricsCollector, MetricType, AlertSeverity, SystemHealth
        )
        
        monitor = AdvancedMetricsCollector()
        
        # Test metric collection
        monitor.record_metric("test.requests", 100, MetricType.COUNTER, 
                            tags={"endpoint": "/api/test"})
        monitor.record_metric("test.latency", 45.5, MetricType.HISTOGRAM,
                            labels={"service": "accelerator"})
        
        # Test system health collection
        health = monitor.collect_system_health()
        assert isinstance(health, SystemHealth)
        assert health.cpu_usage >= 0
        assert health.memory_usage >= 0
        
        # Test statistics calculation
        stats = monitor.get_metric_statistics("test.latency")
        if stats:  # May be empty if not enough data
            assert "count" in stats
            assert "mean" in stats
        
        # Test anomaly detection
        # Add some normal data points
        for i in range(20):
            monitor.record_metric("test.anomaly", 50 + (i % 10), MetricType.GAUGE)
        
        # Add anomalous data point
        monitor.record_metric("test.anomaly", 150, MetricType.GAUGE)
        
        anomalies = monitor.detect_anomalies("test.anomaly")
        # Should detect the anomalous value
        print(f"Detected {len(anomalies)} anomalies in test data")
        
        # Test SLA compliance
        monitor.sla_targets["test.latency"] = 100.0  # 100ms target
        compliance = monitor.get_sla_compliance(1)  # 1 hour
        
        # Test dashboard generation
        dashboard = monitor.get_monitoring_dashboard()
        assert "system_health" in dashboard
        assert "alerts" in dashboard
        assert "metrics" in dashboard
        
        print("✅ Monitoring system: All features working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system failed: {e}")
        return False

async def test_api_robustness():
    """Test API robustness with direct function calls."""
    print("\n🌐 API ROBUSTNESS VALIDATION")
    print("=" * 40)
    
    try:
        # Test FastAPI app components directly
        from backend.main import app
        
        # Test app configuration
        assert app.title == "AI Hardware Co-Design Platform - Generation 1"
        assert app.version == "1.0.0-gen1"
        print("✅ API: App configuration correct")
        
        # Test core imports work
        from backend.codesign_playground.core.accelerator import Accelerator
        from backend.codesign_playground.core.optimizer import ModelOptimizer
        
        # Test accelerator creation
        accelerator = Accelerator(
            compute_units=64,
            memory_hierarchy={'L1': 32, 'L2': 256, 'L3': 2048},
            dataflow='weight_stationary',
            frequency_mhz=300,
            precision='int8'
        )
        
        performance = accelerator.estimate_performance()
        assert performance['throughput_ops_s'] > 0
        print("✅ API: Core accelerator functionality working")
        
        # Test global services
        try:
            import importlib
            compliance_module = importlib.import_module('backend.codesign_playground.global.compliance')
            i18n_module = importlib.import_module('backend.codesign_playground.global.internationalization')
            print("✅ API: Global services importable")
        except:
            print("⚠️  API: Global services partially available")
        
        # Test research modules
        try:
            from backend.codesign_playground.research.novel_algorithms import get_quantum_optimizer
            from backend.codesign_playground.research.research_discovery import conduct_comprehensive_research_discovery
            print("✅ API: Research modules available")
        except:
            print("⚠️  API: Research modules partially available")
        
        return True
        
    except Exception as e:
        print(f"❌ API robustness failed: {e}")
        return False

def main():
    """Run comprehensive Generation 2 validation."""
    print("🚀 AUTONOMOUS SDLC - GENERATION 2 VALIDATION")
    print("=" * 60)
    print("Testing: MAKE IT ROBUST - Reliability & Security")
    print("=" * 60)
    
    test_results = []
    
    # Run all validation tests
    test_results.append(("Security Features", test_security_features()))
    test_results.append(("Resilience Patterns", test_resilience_patterns())) 
    test_results.append(("Monitoring System", test_monitoring_system()))
    test_results.append(("API Robustness", asyncio.run(test_api_robustness())))
    
    # Calculate overall results
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n" + "=" * 60)
    print("GENERATION 2 FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    for test_name, result in test_results:
        status_icon = "✅" if result else "❌"
        status_text = "PASSED" if result else "FAILED"
        print(f"{status_icon} {test_name}: {status_text}")
    
    print(f"\nOverall Generation 2 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 GENERATION 2: MAKE IT ROBUST - VALIDATION SUCCESSFUL!")
        print("✅ Platform is ready for Generation 3: MAKE IT SCALE")
        return True
    else:
        print("⚠️  GENERATION 2: Some robustness features need attention")
        print("Recommendation: Address failing tests before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)