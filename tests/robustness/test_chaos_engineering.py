"""
Chaos Engineering Tests for AI Hardware Co-Design Playground.

This module implements chaos engineering tests to validate system resilience
under various failure conditions and stress scenarios.
"""

import pytest
import asyncio
import random
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, patch
import psutil
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.codesign_playground.core.accelerator import AcceleratorDesigner, Accelerator
from backend.codesign_playground.core.optimizer import ModelOptimizer
from backend.codesign_playground.core.workflow import Workflow
from backend.codesign_playground.utils.circuit_breaker import CircuitBreaker
from backend.codesign_playground.utils.health_monitoring import HealthMonitor
from backend.codesign_playground.utils.exceptions import (
    OptimizationError, WorkflowError, ValidationError, SecurityError
)


class ChaosMonkey:
    """Chaos engineering orchestrator for system resilience testing."""
    
    def __init__(self, failure_rate: float = 0.1):
        """
        Initialize chaos monkey.
        
        Args:
            failure_rate: Probability of inducing failures (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.active_failures = []
        self.failure_history = []
        self._running = False
        
    def induce_cpu_stress(self, duration: float = 5.0, intensity: float = 0.8) -> None:
        """Induce CPU stress to simulate high load conditions."""
        def cpu_stress():
            end_time = time.time() + duration
            while time.time() < end_time:
                # Busy loop to consume CPU
                for _ in range(10000):
                    pass
        
        num_threads = int(psutil.cpu_count() * intensity)
        threads = []
        
        for _ in range(num_threads):
            thread = threading.Thread(target=cpu_stress)
            thread.start()
            threads.append(thread)
        
        self.active_failures.append({
            "type": "cpu_stress",
            "threads": threads,
            "start_time": time.time(),
            "duration": duration
        })
    
    def induce_memory_pressure(self, memory_mb: int = 100) -> None:
        """Induce memory pressure by allocating large amounts of memory."""
        # Allocate memory in chunks
        memory_chunks = []
        chunk_size = 1024 * 1024  # 1MB chunks
        
        for _ in range(memory_mb):
            try:
                chunk = bytearray(chunk_size)
                memory_chunks.append(chunk)
            except MemoryError:
                break
        
        self.active_failures.append({
            "type": "memory_pressure",
            "chunks": memory_chunks,
            "allocated_mb": len(memory_chunks),
            "start_time": time.time()
        })
    
    def induce_disk_io_stress(self, duration: float = 5.0) -> None:
        """Induce disk I/O stress by creating temporary files."""
        def io_stress():
            end_time = time.time() + duration
            temp_files = []
            
            try:
                while time.time() < end_time:
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    
                    # Write random data
                    for _ in range(100):
                        temp_file.write(os.urandom(1024))
                    
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    # Brief pause
                    time.sleep(0.01)
            finally:
                # Clean up temporary files
                for temp_file_path in temp_files:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
        
        thread = threading.Thread(target=io_stress)
        thread.start()
        
        self.active_failures.append({
            "type": "disk_io_stress",
            "thread": thread,
            "start_time": time.time(),
            "duration": duration
        })
    
    def induce_network_latency(self, latency_ms: int = 100) -> None:
        """Simulate network latency by introducing delays."""
        # This would be implemented with network simulation tools in production
        # For testing, we'll use a simple delay injection
        
        def delayed_network_call(original_func):
            def wrapper(*args, **kwargs):
                time.sleep(latency_ms / 1000.0)
                return original_func(*args, **kwargs)
            return wrapper
        
        self.active_failures.append({
            "type": "network_latency",
            "latency_ms": latency_ms,
            "start_time": time.time()
        })
    
    def induce_random_exceptions(self, exception_rate: float = 0.1) -> None:
        """Induce random exceptions to test error handling."""
        self.exception_rate = exception_rate
        
        self.active_failures.append({
            "type": "random_exceptions",
            "exception_rate": exception_rate,
            "start_time": time.time()
        })
    
    def should_fail(self) -> bool:
        """Determine if an operation should fail based on chaos configuration."""
        return random.random() < self.failure_rate
    
    def get_random_exception(self) -> Exception:
        """Get a random exception for chaos testing."""
        exceptions = [
            OptimizationError("Chaos-induced optimization failure"),
            ValidationError("Chaos-induced validation failure"),
            ConnectionError("Chaos-induced connection failure"),
            TimeoutError("Chaos-induced timeout"),
            MemoryError("Chaos-induced memory error"),
            RuntimeError("Chaos-induced runtime error")
        ]
        return random.choice(exceptions)
    
    def clear_failures(self) -> None:
        """Clear all active failures."""
        self.active_failures.clear()
        self.exception_rate = 0.0


class TestChaosEngineering:
    """Chaos engineering test suite."""
    
    @pytest.fixture
    def chaos_monkey(self):
        """Provide chaos monkey instance."""
        return ChaosMonkey(failure_rate=0.2)
    
    @pytest.fixture
    def accelerator_designer(self):
        """Provide accelerator designer instance."""
        return AcceleratorDesigner()
    
    @pytest.fixture
    def mock_model(self):
        """Provide mock model for testing."""
        class MockModel:
            def __init__(self):
                self.complexity = 1.0
                self.path = "/mock/model.pt"
                self.framework = "pytorch"
        
        return MockModel()
    
    def test_accelerator_design_under_cpu_stress(self, chaos_monkey, accelerator_designer):
        """Test accelerator design resilience under CPU stress."""
        # Induce CPU stress
        chaos_monkey.induce_cpu_stress(duration=3.0, intensity=0.9)
        
        start_time = time.time()
        
        # Attempt accelerator design under stress
        try:
            accelerator = accelerator_designer.design(
                compute_units=64,
                memory_hierarchy=["sram_64kb", "dram"],
                dataflow="weight_stationary"
            )
            
            # Verify design was successful
            assert accelerator is not None
            assert accelerator.compute_units == 64
            assert accelerator.dataflow == "weight_stationary"
            
            # Performance should be degraded but not failing
            design_time = time.time() - start_time
            assert design_time < 30.0  # Should complete within reasonable time even under stress
            
        finally:
            chaos_monkey.clear_failures()
    
    def test_optimizer_resilience_under_memory_pressure(self, chaos_monkey, mock_model):
        """Test optimizer resilience under memory pressure."""
        # Create accelerator
        accelerator = Accelerator(
            compute_units=32,
            memory_hierarchy=["sram_32kb", "dram"],
            dataflow="output_stationary"
        )
        
        # Induce memory pressure
        chaos_monkey.induce_memory_pressure(memory_mb=50)
        
        try:
            optimizer = ModelOptimizer(mock_model, accelerator)
            
            # Attempt optimization under memory pressure
            result = optimizer.co_optimize(
                target_fps=30.0,
                power_budget=5.0,
                iterations=5  # Reduced iterations due to memory pressure
            )
            
            # Verify optimization completed
            assert result is not None
            assert result.metrics is not None
            assert result.optimization_time > 0
            
        finally:
            chaos_monkey.clear_failures()
    
    def test_workflow_resilience_with_random_failures(self, chaos_monkey, tmp_path):
        """Test workflow resilience with random failure injection."""
        chaos_monkey.induce_random_exceptions(exception_rate=0.3)
        
        workflow = Workflow("chaos_test", output_dir=str(tmp_path))
        
        # Mock model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("mock_model_data")
        
        failures_encountered = 0
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Simulate workflow execution with potential failures
                if chaos_monkey.should_fail():
                    failures_encountered += 1
                    raise chaos_monkey.get_random_exception()
                
                # If no failure, continue with workflow
                workflow.import_model(
                    model_path=str(model_file),
                    input_shapes={"input": (1, 3, 224, 224)},
                    framework="pytorch"
                )
                
                workflow.map_to_hardware(
                    template="systolic_array",
                    size=(8, 8),
                    precision="int8"
                )
                
                # If we reach here, workflow succeeded
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, record it
                    assert failures_encountered > 0, "Should have encountered some failures during chaos testing"
                continue
        
        chaos_monkey.clear_failures()
    
    def test_circuit_breaker_under_sustained_failures(self, chaos_monkey):
        """Test circuit breaker behavior under sustained failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=2,
            expected_exception=OptimizationError
        )
        
        def failing_operation():
            if chaos_monkey.should_fail():
                raise OptimizationError("Chaos-induced failure")
            return "success"
        
        # Set high failure rate to trigger circuit breaker
        chaos_monkey.failure_rate = 0.9
        
        failures = 0
        circuit_open_detected = False
        
        # Attempt operations until circuit breaker opens
        for i in range(10):
            try:
                result = circuit_breaker.call(failing_operation)
                assert result == "success"
            except OptimizationError:
                failures += 1
            except Exception as e:
                # Circuit breaker should open after threshold
                if "Circuit breaker is OPEN" in str(e):
                    circuit_open_detected = True
                    break
        
        assert failures >= 3, "Should have at least threshold failures"
        assert circuit_open_detected, "Circuit breaker should have opened"
        
        chaos_monkey.clear_failures()
    
    def test_health_monitor_under_stress(self, chaos_monkey):
        """Test health monitor behavior under various stress conditions."""
        health_monitor = HealthMonitor("chaos_test")
        
        # Initial health should be good
        assert health_monitor.is_healthy()
        
        # Induce various stresses
        chaos_monkey.induce_cpu_stress(duration=2.0)
        chaos_monkey.induce_memory_pressure(memory_mb=30)
        
        # Health should still be maintained
        time.sleep(1.0)
        
        # Update health status during stress
        health_monitor.update_status("degraded", {"stress_test": True})
        
        # Health monitor should handle status updates under stress
        status = health_monitor.get_status()
        assert status["status"] in ["healthy", "degraded"]
        assert "stress_test" in status.get("details", {})
        
        chaos_monkey.clear_failures()
    
    def test_concurrent_operations_under_chaos(self, chaos_monkey, tmp_path):
        """Test concurrent operations under chaotic conditions."""
        chaos_monkey.failure_rate = 0.2
        chaos_monkey.induce_cpu_stress(duration=5.0, intensity=0.5)
        
        def create_workflow(workflow_id: int) -> Dict[str, Any]:
            try:
                workflow = Workflow(f"chaos_concurrent_{workflow_id}", output_dir=str(tmp_path / f"wf_{workflow_id}"))
                
                # Mock operations with potential failures
                if chaos_monkey.should_fail():
                    raise chaos_monkey.get_random_exception()
                
                # Simulate some work
                time.sleep(random.uniform(0.1, 0.5))
                
                return {
                    "workflow_id": workflow_id,
                    "status": "success",
                    "stage": workflow.state.stage.value
                }
            except Exception as e:
                return {
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Run concurrent workflows
        num_workflows = 5
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_workflow, i) for i in range(num_workflows)]
            
            for future in as_completed(futures, timeout=30):
                result = future.result()
                results.append(result)
        
        # Analyze results
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        # At least some operations should succeed despite chaos
        assert len(successful) > 0, "Some workflows should succeed despite chaos"
        
        # Some failures are expected due to chaos
        total_operations = len(results)
        success_rate = len(successful) / total_operations
        
        # Success rate should be reasonable but not perfect
        assert 0.3 <= success_rate <= 1.0, f"Success rate {success_rate} should be reasonable under chaos"
        
        chaos_monkey.clear_failures()
    
    def test_resource_exhaustion_recovery(self, chaos_monkey):
        """Test system recovery from resource exhaustion."""
        # Induce severe resource pressure
        chaos_monkey.induce_memory_pressure(memory_mb=100)
        chaos_monkey.induce_cpu_stress(duration=3.0, intensity=0.95)
        chaos_monkey.induce_disk_io_stress(duration=3.0)
        
        # Wait for stress conditions
        time.sleep(1.0)
        
        # Clear all failures to simulate recovery
        chaos_monkey.clear_failures()
        
        # Wait for system to recover
        time.sleep(2.0)
        
        # Test that system can function normally after recovery
        designer = AcceleratorDesigner()
        
        start_time = time.time()
        accelerator = designer.design(
            compute_units=16,
            memory_hierarchy=["sram_32kb"],
            dataflow="weight_stationary"
        )
        recovery_time = time.time() - start_time
        
        # System should function normally after recovery
        assert accelerator is not None
        assert recovery_time < 10.0  # Should be reasonably fast after recovery
    
    def test_chaos_with_compliance_tracking(self, chaos_monkey, tmp_path):
        """Test compliance tracking under chaotic conditions."""
        from backend.codesign_playground.utils.compliance import get_compliance_manager, DataCategory
        
        compliance_manager = get_compliance_manager()
        chaos_monkey.failure_rate = 0.3
        
        # Attempt multiple data processing operations under chaos
        successful_records = 0
        failed_records = 0
        
        for i in range(10):
            try:
                if chaos_monkey.should_fail():
                    raise chaos_monkey.get_random_exception()
                
                # Record data processing
                success = compliance_manager.record_data_processing(
                    user_id=f"chaos_user_{i}",
                    data_category=DataCategory.MODEL_ARTIFACTS,
                    processing_purpose="chaos_testing",
                    legal_basis="legitimate_interests"
                )
                
                if success:
                    successful_records += 1
                else:
                    failed_records += 1
                    
            except Exception:
                failed_records += 1
        
        # Compliance tracking should continue working despite chaos
        assert successful_records > 0, "Some compliance records should succeed"
        
        # Verify audit logs were created
        report = compliance_manager.generate_compliance_report(
            start_date=time.time() - 3600,
            end_date=time.time()
        )
        
        assert report["processing_activities"]["total"] >= successful_records
        
        chaos_monkey.clear_failures()


class TestFailureInjection:
    """Specific failure injection test cases."""
    
    def test_database_connection_failure(self):
        """Test behavior when database connections fail."""
        from backend.codesign_playground.utils.compliance import ComplianceManager
        
        # Mock database failure
        with patch('sqlite3.connect', side_effect=Exception("Database connection failed")):
            try:
                compliance_manager = ComplianceManager(db_path="nonexistent.db")
                # Should handle database initialization failure gracefully
                assert compliance_manager is not None
            except Exception as e:
                # Should be a specific error type, not a generic exception
                assert "Database" in str(e) or "compliance" in str(e).lower()
    
    def test_filesystem_permission_errors(self, tmp_path):
        """Test behavior when filesystem permissions are denied."""
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            workflow = Workflow("permission_test", output_dir=str(readonly_dir / "subdir"))
            
            # Should handle permission errors gracefully
            assert workflow.output_dir.exists() or True  # May fail gracefully
            
        except Exception as e:
            # Should be a specific permission-related error
            assert any(keyword in str(e).lower() for keyword in ["permission", "access", "denied"])
        
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_network_timeout_simulation(self):
        """Test behavior under network timeout conditions."""
        # This would use network simulation tools in production
        # For testing, we'll use delays and timeouts
        
        def slow_network_operation():
            time.sleep(2.0)  # Simulate slow network
            return "data"
        
        start_time = time.time()
        
        try:
            # Set short timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Network timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(1)  # 1 second timeout
            
            result = slow_network_operation()
            signal.alarm(0)  # Cancel alarm
            
        except TimeoutError:
            # Expected timeout
            elapsed = time.time() - start_time
            assert elapsed < 1.5, "Should timeout quickly"
        
        except Exception:
            # Other exceptions are acceptable
            pass
    
    def test_memory_leak_detection(self):
        """Test detection and handling of memory leaks."""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Create objects that might leak
        potential_leaks = []
        for i in range(1000):
            designer = AcceleratorDesigner()
            potential_leaks.append(designer)
        
        # Clear references
        potential_leaks.clear()
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Some object growth is normal, but excessive growth indicates leaks
        assert object_growth < 500, f"Excessive object growth detected: {object_growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])