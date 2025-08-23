"""
Comprehensive Test Suite for Autonomous SDLC Implementation.

This test suite validates all three generations of autonomous SDLC development
with comprehensive coverage of functionality, robustness, and optimization.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import all the new modules we created
from backend.codesign_playground.core.quantum_enhanced_optimizer import (
    QuantumEnhancedOptimizer, QuantumState, OptimizationResult
)
from backend.codesign_playground.core.autonomous_design_agent import (
    AutonomousDesignAgent, DesignGoal, AgentState, AutonomousDesignResult
)
from backend.codesign_playground.research.breakthrough_algorithms import (
    BreakthroughResearchManager, NeuroEvolutionaryOptimizer, 
    SwarmIntelligenceOptimizer, ResearchHypothesis, AlgorithmType
)
from backend.codesign_playground.utils.advanced_error_handling import (
    ErrorRecoveryManager, ErrorSeverity, RecoveryStrategy, robust_error_handler
)
from backend.codesign_playground.utils.comprehensive_monitoring import (
    AdvancedMetricsCollector, MetricType, AlertSeverity, SystemHealth
)
from backend.codesign_playground.utils.security_fortress import (
    AdvancedSecurityManager, SecurityEventType, ThreatLevel, UserSession
)
from backend.codesign_playground.core.hyperscale_optimizer import (
    HyperscaleOptimizer, OptimizationLevel, ScalingStrategy, PerformanceMetrics
)
from backend.codesign_playground.core.accelerator import (
    AcceleratorDesigner, ModelProfile
)


class TestQuantumEnhancedOptimizer:
    """Test suite for quantum-enhanced optimization algorithms."""
    
    @pytest.fixture
    def quantum_optimizer(self):
        """Create quantum optimizer instance for testing."""
        return QuantumEnhancedOptimizer(
            population_size=20,
            max_generations=10,
            quantum_coherence_length=5
        )
    
    @pytest.fixture
    def sample_design_space(self):
        """Sample design space for testing."""
        return {
            "compute_units": [16, 32, 64],
            "memory_hierarchy": [["sram_32kb", "dram"], ["sram_64kb", "dram"]],
            "dataflow": ["weight_stationary", "output_stationary"]
        }
    
    @pytest.fixture
    def simple_fitness_function(self):
        """Simple fitness function for testing."""
        def fitness_func(config: Dict[str, Any]) -> float:
            # Mock fitness based on compute units
            return config.get("compute_units", 16) * 0.1
        return fitness_func
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization and properties."""
        amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
        phases = np.array([0.0, 0.0, 0.0, 0.0])
        entanglement = np.zeros((4, 4))
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement,
            coherence_time=10.0
        )
        
        assert state.coherence_time == 10.0
        assert state.measurement_count == 0
        assert len(state.amplitudes) == 4
    
    def test_quantum_state_measurement(self):
        """Test quantum state measurement and decoherence."""
        amplitudes = np.array([0.5, 0.5, 0.5, 0.5])
        phases = np.zeros(4)
        entanglement = np.zeros((4, 4))
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement,
            coherence_time=10.0
        )
        
        # Test measurement
        measured_values = state.measure()
        
        assert len(measured_values) == 4
        assert state.measurement_count == 1
        assert state.coherence_time < 10.0  # Decoherence should reduce coherence time
    
    def test_quantum_gate_operations(self):
        """Test quantum gate operations on states."""
        amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
        phases = np.zeros(4)
        entanglement = np.zeros((4, 4))
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement,
            coherence_time=10.0
        )
        
        # Apply Hadamard gate
        state.apply_quantum_gate("hadamard", [0])
        
        # Check that amplitude changed
        assert state.amplitudes[0] != 1.0
        
        # Apply phase gate
        original_phase = state.phases[1]
        state.apply_quantum_gate("phase", [1])
        assert state.phases[1] != original_phase
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_basic(self, quantum_optimizer, sample_design_space, simple_fitness_function):
        """Test basic quantum optimization functionality."""
        # Run optimization with small parameters for testing
        result = await quantum_optimizer.optimize_async(
            design_space=sample_design_space,
            fitness_function=simple_fitness_function
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_configuration is not None
        assert result.best_fitness > 0
        assert result.total_evaluations > 0
        assert result.convergence_generations >= 0
        assert result.quantum_advantage > 0
    
    def test_quantum_optimizer_statistics(self, quantum_optimizer):
        """Test quantum optimizer statistics collection."""
        stats = quantum_optimizer.get_optimization_statistics()
        
        assert "total_evaluations" in stats
        assert "best_fitness" in stats
        assert "population_size" in stats
        assert "quantum_coherence" in stats
        assert "quantum_entanglement" in stats


class TestAutonomousDesignAgent:
    """Test suite for autonomous design agent."""
    
    @pytest.fixture
    def design_agent(self):
        """Create design agent for testing."""
        return AutonomousDesignAgent(
            expertise_level="expert",
            creativity_factor=0.7,
            risk_tolerance=0.3,
            max_design_iterations=5
        )
    
    @pytest.fixture
    def sample_model_profile(self):
        """Sample model profile for testing."""
        return ModelProfile(
            peak_gflops=50.0,
            bandwidth_gb_s=25.6,
            operations={"conv2d": 1000000, "dense": 500000},
            parameters=10000000,
            memory_mb=40.0,
            compute_intensity=2.0,
            layer_types=["conv2d", "dense", "activation"],
            model_size_mb=40.0
        )
    
    @pytest.fixture
    def sample_design_goal(self):
        """Sample design goal for testing."""
        return DesignGoal(
            target_throughput_ops_s=1e9,
            max_power_w=10.0,
            max_area_mm2=20.0,
            target_latency_ms=10.0,
            precision_requirements=["int8"],
            compatibility_targets=["edge_device"]
        )
    
    def test_design_agent_initialization(self, design_agent):
        """Test design agent initialization."""
        assert design_agent.expertise_level == "expert"
        assert design_agent.creativity_factor == 0.7
        assert design_agent.risk_tolerance == 0.3
        assert design_agent.current_state == AgentState.IDLE
        assert len(design_agent.design_history) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_design_process(self, design_agent, sample_model_profile, sample_design_goal):
        """Test complete autonomous design process."""
        result = await design_agent.design_accelerator_autonomously(
            model_profile=sample_model_profile,
            design_goal=sample_design_goal,
            context={"test_context": True}
        )
        
        assert isinstance(result, AutonomousDesignResult)
        assert result.final_design is not None
        assert result.performance_metrics is not None
        assert result.design_confidence >= 0.0
        assert result.total_design_time > 0
        assert len(result.design_decisions) > 0
    
    def test_agent_statistics(self, design_agent):
        """Test agent statistics collection."""
        stats = design_agent.get_agent_statistics()
        
        assert "total_designs_created" in stats
        assert "successful_designs" in stats
        assert "success_rate" in stats
        assert "expertise_level" in stats
        assert "creativity_factor" in stats
        assert "current_state" in stats


class TestBreakthroughResearchManager:
    """Test suite for breakthrough research algorithms."""
    
    @pytest.fixture
    def research_manager(self):
        """Create research manager for testing."""
        return BreakthroughResearchManager()
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Sample research hypothesis."""
        return ResearchHypothesis(
            hypothesis_id="test_001",
            title="Neural Evolution Optimization",
            description="Test neural evolutionary algorithms for hardware design",
            expected_improvement=1.5,
            baseline_algorithm="genetic_algorithm",
            novel_algorithm="neural_evolution",
            success_criteria={"improvement_factor": 1.2, "statistical_significance": 0.05}
        )
    
    @pytest.fixture
    def simple_design_space(self):
        """Simple design space for testing."""
        return {
            "param1": [1, 2, 3],
            "param2": ["a", "b", "c"]
        }
    
    @pytest.fixture
    def simple_fitness(self):
        """Simple fitness function."""
        def fitness_func(config: Dict[str, Any]) -> float:
            return config.get("param1", 1) * 0.5
        return fitness_func
    
    def test_neuroevolutionary_optimizer(self):
        """Test neuroevolutionary optimizer initialization."""
        optimizer = NeuroEvolutionaryOptimizer(population_size=10, generations=5)
        
        assert optimizer.population_size == 10
        assert optimizer.generations == 5
        assert len(optimizer.neural_controllers) == 0
    
    def test_swarm_intelligence_optimizer(self):
        """Test swarm intelligence optimizer initialization."""
        optimizer = SwarmIntelligenceOptimizer(swarm_size=10, max_iterations=5)
        
        assert optimizer.swarm_size == 10
        assert optimizer.max_iterations == 5
        assert len(optimizer.particles) == 0
    
    @pytest.mark.asyncio
    async def test_neuroevolution_experiment(self, simple_design_space, simple_fitness):
        """Test neuroevolutionary algorithm experiment."""
        optimizer = NeuroEvolutionaryOptimizer(population_size=5, generations=3)
        
        result = await optimizer.evolve_design_strategies(
            design_space=simple_design_space,
            fitness_function=simple_fitness,
            runs=2
        )
        
        assert result.algorithm_name == "NeuroEvolutionaryOptimizer"
        assert result.hypothesis_id == "neuroevolution_001"
        assert "best_fitness" in result.performance_metrics
        assert result.reproducibility_score >= 0.0
    
    def test_research_statistics(self, research_manager):
        """Test research statistics collection."""
        stats = research_manager.get_research_statistics()
        
        assert "total_experiments" in stats
        assert "successful_experiments" in stats
        assert "success_rate" in stats
        assert "average_improvement_factor" in stats


class TestAdvancedErrorHandling:
    """Test suite for advanced error handling and recovery."""
    
    @pytest.fixture
    def error_manager(self):
        """Create error recovery manager."""
        return ErrorRecoveryManager()
    
    def test_error_manager_initialization(self, error_manager):
        """Test error manager initialization."""
        assert len(error_manager.error_history) == 0
        assert len(error_manager.recovery_strategies) > 0
        assert error_manager.learning_enabled is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_retry(self, error_manager):
        """Test retry recovery strategy."""
        test_error = ValueError("Test error")
        context = {
            "function_name": "test_function",
            "module_name": "test_module",
            "original_function": lambda: "success"
        }
        
        # Should succeed with retry recovery
        result = await error_manager.handle_error_with_recovery(
            error=test_error,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=1
        )
        
        assert result == "success"
        assert len(error_manager.error_history) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_fallback(self, error_manager):
        """Test fallback recovery strategy."""
        test_error = RuntimeError("Test error")
        context = {
            "function_name": "test_function",
            "module_name": "test_module",
            "fallback_function": lambda: "fallback_result"
        }
        
        result = await error_manager.handle_error_with_recovery(
            error=test_error,
            context=context,
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        
        assert result == "fallback_result"
    
    def test_robust_error_handler_decorator(self):
        """Test robust error handler decorator."""
        
        @robust_error_handler(
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_function=lambda: "fallback"
        )
        def test_function():
            raise ValueError("Test error")
        
        # Should return fallback value
        result = test_function()
        assert result == "fallback"
    
    def test_error_statistics(self, error_manager):
        """Test error statistics collection."""
        stats = error_manager.get_error_statistics()
        
        assert "total_errors" in stats
        assert "successful_recoveries" in stats
        assert "recovery_rate" in stats
        assert "error_patterns_learned" in stats


class TestComprehensiveMonitoring:
    """Test suite for comprehensive monitoring system."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        return AdvancedMetricsCollector(retention_hours=1, collection_interval=1)
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector.retention_hours == 1
        assert metrics_collector.collection_interval == 1
        assert len(metrics_collector.metrics) == 0
        assert not metrics_collector.is_running
    
    def test_record_metric(self, metrics_collector):
        """Test metric recording."""
        metrics_collector.record_metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            tags={"environment": "test"}
        )
        
        assert "test_metric" in metrics_collector.metrics
        assert len(metrics_collector.metrics["test_metric"]) == 1
        assert metrics_collector.metrics["test_metric"][0].value == 42.0
    
    def test_metric_statistics(self, metrics_collector):
        """Test metric statistics calculation."""
        # Record some test metrics
        for i in range(10):
            metrics_collector.record_metric(
                name="test_metric",
                value=float(i),
                metric_type=MetricType.GAUGE
            )
        
        stats = metrics_collector.get_metric_statistics("test_metric", 60)
        
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert stats["mean"] == 4.5
    
    def test_alert_system(self, metrics_collector):
        """Test alert system."""
        alert_id = metrics_collector.add_alert(
            name="High CPU Usage",
            description="CPU usage is too high",
            metric_name="cpu_usage",
            threshold_value=80.0,
            comparison_operator=">",
            severity=AlertSeverity.WARNING
        )
        
        assert alert_id in metrics_collector.alerts
        assert metrics_collector.alerts[alert_id].name == "High CPU Usage"
    
    def test_system_health_collection(self, metrics_collector):
        """Test system health metrics collection."""
        health = metrics_collector.collect_system_health()
        
        assert isinstance(health, SystemHealth)
        assert health.timestamp > 0
        assert health.cpu_usage >= 0
        assert health.memory_usage >= 0
    
    def test_monitoring_dashboard(self, metrics_collector):
        """Test monitoring dashboard generation."""
        # Record some metrics first
        metrics_collector.record_metric("test_metric", 100.0)
        
        dashboard = metrics_collector.get_monitoring_dashboard()
        
        assert "timestamp" in dashboard
        assert "system_health" in dashboard
        assert "alerts" in dashboard
        assert "metrics" in dashboard
        assert "monitoring_status" in dashboard


class TestSecurityFortress:
    """Test suite for advanced security system."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager."""
        return AdvancedSecurityManager()
    
    def test_security_manager_initialization(self, security_manager):
        """Test security manager initialization."""
        assert len(security_manager.security_events) == 0
        assert len(security_manager.active_sessions) == 0
        assert security_manager.jwt_secret is not None
        assert security_manager.encryption_key is not None
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, security_manager):
        """Test user authentication process."""
        token = await security_manager.authenticate_user(
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        assert token is not None
        assert len(security_manager.active_sessions) > 0
    
    @pytest.mark.asyncio
    async def test_request_authorization(self, security_manager):
        """Test request authorization."""
        # First authenticate
        token = await security_manager.authenticate_user(
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        # Then authorize request
        authorized, user_id = await security_manager.authorize_request(
            token=token,
            resource="test_resource",
            action="read",
            ip_address="192.168.1.1"
        )
        
        assert authorized is True
        assert user_id == "testuser"
    
    def test_data_encryption(self, security_manager):
        """Test data encryption and decryption."""
        test_data = "sensitive information"
        
        # Encrypt
        encrypted = security_manager.encrypt_sensitive_data(test_data)
        assert encrypted != test_data
        
        # Decrypt
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_vulnerability_scanning(self, security_manager):
        """Test vulnerability scanning."""
        request_data = {
            "query": "SELECT * FROM users WHERE id = 1 OR 1=1",
            "comment": "<script>alert('xss')</script>"
        }
        
        vulnerabilities = await security_manager.scan_for_vulnerabilities(request_data)
        
        assert len(vulnerabilities) > 0
        assert any(vuln["type"] == "sql_injection" for vuln in vulnerabilities)
    
    def test_security_report(self, security_manager):
        """Test security report generation."""
        report = security_manager.generate_security_report(24)
        
        assert "report_period" in report
        assert "summary" in report
        assert "event_breakdown" in report
        assert "recommendations" in report


class TestHyperscaleOptimizer:
    """Test suite for hyperscale performance optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create hyperscale optimizer."""
        return HyperscaleOptimizer(
            optimization_level=OptimizationLevel.MODERATE,
            auto_scaling_enabled=True
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimization_level == OptimizationLevel.MODERATE
        assert optimizer.auto_scaling_enabled is True
        assert len(optimizer.thread_pools) > 0
        assert not optimizer.optimizer_active
    
    def test_performance_metrics_collection(self, optimizer):
        """Test performance metrics collection."""
        metrics = optimizer._collect_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.timestamp > 0
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
    
    @pytest.mark.asyncio
    async def test_auto_scaling_decision(self, optimizer):
        """Test auto-scaling decision making."""
        # Create high-load metrics
        high_load_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=90.0,  # High CPU usage
            memory_usage=85.0,  # High memory usage
            disk_io={}, network_io={}, thread_count=10, process_count=5,
            cache_hit_ratio=0.8, throughput_ops_per_sec=1000.0,
            latency_p95_ms=100.0, latency_p99_ms=200.0,
            error_rate=1.0, queue_depth=50, active_connections=100
        )
        
        decision = await optimizer.auto_scale_resources(high_load_metrics)
        
        if decision:  # May be None if no scaling needed
            assert decision.action in ["scale_up", "scale_down", "scale_out", "scale_in"]
            assert decision.confidence > 0
    
    def test_cache_optimization(self, optimizer):
        """Test cache performance optimization."""
        cache_result = optimizer.optimize_cache_performance(
            cache_size_mb=512,
            eviction_policy="lru",
            prefetch_enabled=True
        )
        
        assert "cache_config" in cache_result
        assert "estimated_hit_ratio" in cache_result
        assert cache_result["cache_config"]["size_mb"] == 512
    
    def test_capacity_prediction(self, optimizer):
        """Test capacity requirement prediction."""
        # Create sample historical metrics
        historical_metrics = []
        for i in range(25):  # Need at least 24 hours
            metrics = PerformanceMetrics(
                timestamp=time.time() - (i * 3600),  # Hourly data
                cpu_usage=50.0 + (i % 10),
                memory_usage=60.0 + (i % 15),
                disk_io={}, network_io={}, thread_count=10, process_count=5,
                cache_hit_ratio=0.8, throughput_ops_per_sec=1000.0,
                latency_p95_ms=100.0, latency_p99_ms=200.0,
                error_rate=1.0, queue_depth=50, active_connections=100
            )
            historical_metrics.append(metrics)
        
        predictions = optimizer.predict_capacity_requirements(
            historical_metrics=historical_metrics,
            forecast_horizon_hours=24
        )
        
        assert "predictions" in predictions
        assert "trends" in predictions
        assert "recommendations" in predictions
    
    def test_optimization_statistics(self, optimizer):
        """Test optimization statistics collection."""
        stats = optimizer.get_optimization_statistics()
        
        assert "optimization_level" in stats
        assert "auto_scaling_enabled" in stats
        assert "thread_pools" in stats
        assert "cache_statistics" in stats
        assert "is_active" in stats


class TestIntegrationScenarios:
    """Integration tests for complete autonomous SDLC scenarios."""
    
    @pytest.fixture
    def full_system(self):
        """Setup complete system for integration testing."""
        return {
            "accelerator_designer": AcceleratorDesigner(),
            "quantum_optimizer": QuantumEnhancedOptimizer(population_size=10, max_generations=5),
            "design_agent": AutonomousDesignAgent(max_design_iterations=3),
            "error_manager": ErrorRecoveryManager(),
            "security_manager": AdvancedSecurityManager(),
            "performance_optimizer": HyperscaleOptimizer(optimization_level=OptimizationLevel.BASIC)
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_design_flow(self, full_system):
        """Test complete end-to-end design flow."""
        # Create model profile
        model_profile = ModelProfile(
            peak_gflops=100.0,
            bandwidth_gb_s=50.0,
            operations={"conv2d": 2000000, "dense": 1000000},
            parameters=50000000,
            memory_mb=200.0,
            compute_intensity=2.5,
            layer_types=["conv2d", "dense"],
            model_size_mb=200.0
        )
        
        # Create design goal
        design_goal = DesignGoal(
            target_throughput_ops_s=1e10,
            max_power_w=15.0,
            max_area_mm2=50.0,
            target_latency_ms=5.0,
            precision_requirements=["int8"],
            compatibility_targets=["high_performance"]
        )
        
        # Run autonomous design
        result = await full_system["design_agent"].design_accelerator_autonomously(
            model_profile=model_profile,
            design_goal=design_goal
        )
        
        assert result.final_design is not None
        assert result.design_confidence > 0
        assert len(result.design_decisions) > 0
    
    def test_error_resilience_integration(self, full_system):
        """Test error resilience across system components."""
        error_manager = full_system["error_manager"]
        
        # Test that error manager can handle various error types
        test_errors = [
            ValueError("Invalid input"),
            RuntimeError("System failure"),
            TimeoutError("Operation timeout")
        ]
        
        for error in test_errors:
            # Should not raise exception due to error handling
            try:
                context = {"function_name": "test", "module_name": "test"}
                asyncio.run(error_manager.handle_error_with_recovery(error, context))
            except:
                pass  # Expected for some error types
        
        # Should have recorded error events
        assert len(error_manager.error_history) > 0
    
    def test_security_integration(self, full_system):
        """Test security integration across components."""
        security_manager = full_system["security_manager"]
        
        # Test data encryption
        test_data = "sensitive design parameters"
        encrypted = security_manager.encrypt_sensitive_data(test_data)
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        
        assert decrypted == test_data
        
        # Test vulnerability scanning
        request_data = {"design": "normal data"}
        vulnerabilities = asyncio.run(
            security_manager.scan_for_vulnerabilities(request_data)
        )
        
        # Should not detect vulnerabilities in normal data
        assert len(vulnerabilities) == 0
    
    def test_performance_optimization_integration(self, full_system):
        """Test performance optimization integration."""
        optimizer = full_system["performance_optimizer"]
        
        # Test optimization statistics
        stats = optimizer.get_optimization_statistics()
        assert "optimization_level" in stats
        
        # Test cache optimization
        cache_result = optimizer.optimize_cache_performance()
        assert "cache_config" in cache_result
    
    @pytest.mark.asyncio
    async def test_scalability_under_load(self, full_system):
        """Test system behavior under high load conditions."""
        # Simulate concurrent design requests
        design_tasks = []
        
        for i in range(5):  # Reduced for test performance
            model_profile = ModelProfile(
                peak_gflops=10.0 * (i + 1),
                bandwidth_gb_s=5.0 * (i + 1),
                operations={"conv2d": 100000 * (i + 1)},
                parameters=1000000 * (i + 1),
                memory_mb=10.0 * (i + 1),
                compute_intensity=1.5,
                layer_types=["conv2d"],
                model_size_mb=10.0 * (i + 1)
            )
            
            design_goal = DesignGoal(
                target_throughput_ops_s=1e8 * (i + 1),
                max_power_w=5.0 + i,
                max_area_mm2=10.0 + i * 5,
                target_latency_ms=10.0 - i,
                precision_requirements=["int8"],
                compatibility_targets=["edge"]
            )
            
            task = full_system["design_agent"].design_accelerator_autonomously(
                model_profile=model_profile,
                design_goal=design_goal
            )
            design_tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*design_tasks, return_exceptions=True)
        
        # Check that most tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 3  # At least 60% success rate


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])