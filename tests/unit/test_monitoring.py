"""
Unit tests for monitoring and health check functionality.

This module tests the comprehensive monitoring system including metrics collection,
health checks, circuit breakers, and system monitoring.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from codesign_playground.utils.monitoring import (
    MetricCollector,
    HealthChecker,
    SystemMonitor,
    MetricValue,
    MetricType,
    HealthStatus,
    SystemMetrics,
    HealthCheckResult,
    record_metric,
    get_health_status,
    MonitoringDecorator
)


class TestMetricValue:
    """Test MetricValue dataclass."""
    
    def test_metric_value_creation(self):
        """Test MetricValue creation with valid data."""
        timestamp = time.time()
        metric = MetricValue(
            name="test_metric",
            value=42.5,
            type=MetricType.GAUGE,
            timestamp=timestamp,
            labels={"service": "test"},
            unit="ms",
            description="Test metric"
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.type == MetricType.GAUGE
        assert metric.timestamp == timestamp
        assert metric.labels["service"] == "test"
        assert metric.unit == "ms"
        assert metric.description == "Test metric"
    
    def test_metric_value_to_dict(self):
        """Test MetricValue serialization to dictionary."""
        metric = MetricValue(
            name="test_metric",
            value=100,
            type=MetricType.COUNTER,
            timestamp=1234567890.0,
            labels={"env": "test"},
            unit="count",
            description="Test counter"
        )
        
        result = metric.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test_metric"
        assert result["value"] == 100
        assert result["type"] == "counter"
        assert result["timestamp"] == 1234567890.0
        assert result["labels"]["env"] == "test"


class TestMetricCollector:
    """Test MetricCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricCollector(max_history=100)
    
    def test_collector_initialization(self):
        """Test MetricCollector initialization."""
        assert self.collector.max_history == 100
        assert len(self.collector._metrics) == 0
    
    def test_record_counter(self):
        """Test recording counter metrics."""
        self.collector.record_counter("test_counter", 5)
        
        history = self.collector.get_metric_history("test_counter")
        assert len(history) == 1
        assert history[0].name == "test_counter"
        assert history[0].value == 5
        assert history[0].type == MetricType.COUNTER
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        self.collector.record_gauge("test_gauge", 42.5, labels={"env": "test"})
        
        history = self.collector.get_metric_history("test_gauge")
        assert len(history) == 1
        assert history[0].value == 42.5
        assert history[0].type == MetricType.GAUGE
        assert history[0].labels["env"] == "test"
    
    def test_record_histogram(self):
        """Test recording histogram metrics."""
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.collector.record_histogram("test_histogram", value)
        
        history = self.collector.get_metric_history("test_histogram")
        assert len(history) == 5
        
        recorded_values = [h.value for h in history]
        assert recorded_values == values
    
    def test_record_timer(self):
        """Test recording timer metrics."""
        self.collector.record_timer("test_timer", 0.125, unit="seconds")
        
        history = self.collector.get_metric_history("test_timer")
        assert len(history) == 1
        assert history[0].value == 0.125
        assert history[0].type == MetricType.TIMER
        assert history[0].unit == "seconds"
    
    def test_metric_history_limit(self):
        """Test that metric history respects max limit."""
        # Record more metrics than the limit
        for i in range(150):
            self.collector.record_counter("test_limited", 1)
        
        history = self.collector.get_metric_history("test_limited")
        assert len(history) == 100  # Should be limited to max_history
    
    def test_get_metric_history_nonexistent(self):
        """Test getting history for non-existent metric."""
        history = self.collector.get_metric_history("nonexistent")
        assert len(history) == 0
    
    def test_get_metric_history_with_limit(self):
        """Test getting metric history with limit parameter."""
        # Record 20 metrics
        for i in range(20):
            self.collector.record_gauge("test_limit", i)
        
        # Get only last 5
        history = self.collector.get_metric_history("test_limit", limit=5)
        assert len(history) == 5
        
        # Should be the last 5 values (15-19)
        values = [h.value for h in history]
        assert values == [15, 16, 17, 18, 19]
    
    def test_get_metric_stats(self):
        """Test getting metric statistics."""
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.collector.record_gauge("test_stats", value)
        
        stats = self.collector.get_metric_stats("test_stats")
        
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["latest"] == 50
        assert stats["std"] > 0
    
    def test_get_metric_stats_nonexistent(self):
        """Test getting stats for non-existent metric."""
        stats = self.collector.get_metric_stats("nonexistent")
        assert "error" in stats
    
    def test_get_metric_stats_with_window(self):
        """Test getting stats within time window."""
        # Record metrics with delays
        for i in range(5):
            self.collector.record_gauge("test_window", i * 10)
            if i < 4:  # Don't sleep after last one
                time.sleep(0.1)
        
        # Get stats for last 0.3 seconds (should include last 3-4 metrics)
        stats = self.collector.get_metric_stats("test_window", window_seconds=0.3)
        
        assert stats["count"] >= 3
        assert "latest" in stats
    
    def test_get_all_metrics(self):
        """Test getting all metrics summary."""
        self.collector.record_counter("counter1", 5)
        self.collector.record_gauge("gauge1", 42.5)
        self.collector.record_timer("timer1", 0.125)
        
        all_metrics = self.collector.get_all_metrics()
        
        assert "counter1" in all_metrics
        assert "gauge1" in all_metrics
        assert "timer1" in all_metrics
        
        assert all_metrics["counter1"]["latest_value"] == 5
        assert all_metrics["gauge1"]["latest_value"] == 42.5
        assert all_metrics["timer1"]["latest_value"] == 0.125
    
    def test_export_metrics_json(self):
        """Test exporting metrics in JSON format."""
        self.collector.record_counter("test_export", 10)
        self.collector.record_gauge("test_gauge_export", 25.5)
        
        json_export = self.collector.export_metrics("json")
        
        assert isinstance(json_export, str)
        assert "test_export" in json_export
        assert "test_gauge_export" in json_export
        assert "10" in json_export
        assert "25.5" in json_export
    
    def test_export_metrics_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        self.collector.record_counter("test_prom", 5, labels={"env": "test"})
        
        prom_export = self.collector.export_metrics("prometheus")
        
        assert isinstance(prom_export, str)
        assert "# HELP test_prom" in prom_export
        assert "# TYPE test_prom counter" in prom_export
        assert 'test_prom{env="test"}' in prom_export
    
    def test_export_metrics_invalid_format(self):
        """Test exporting with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.collector.export_metrics("invalid_format")


class TestHealthChecker:
    """Test HealthChecker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.health_checker = HealthChecker()
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_health_checker_initialization(self, mock_uniform):
        """Test HealthChecker initialization."""
        # Mock CPU and memory values for consistent testing
        mock_uniform.side_effect = [25.0, 45.0]  # CPU, Memory
        
        # Should have default health checks registered
        assert len(self.health_checker._checks) > 0
        assert "cpu_usage" in self.health_checker._checks
        assert "memory_usage" in self.health_checker._checks
    
    def test_register_custom_check(self):
        """Test registering custom health check."""
        def custom_check():
            return HealthCheckResult(
                name="custom_test",
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
                timestamp=time.time(),
                duration_ms=1.0
            )
        
        initial_count = len(self.health_checker._checks)
        self.health_checker.register_check("custom_test", custom_check)
        
        assert len(self.health_checker._checks) == initial_count + 1
        assert "custom_test" in self.health_checker._checks
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_run_check_success(self, mock_uniform):
        """Test running a successful health check."""
        mock_uniform.return_value = 25.0  # Low CPU usage
        
        result = self.health_checker.run_check("cpu_usage")
        
        assert isinstance(result, HealthCheckResult)
        assert result.name == "cpu_usage"
        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms >= 0
        assert "CPU usage normal" in result.message
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_run_check_warning(self, mock_uniform):
        """Test running health check that returns warning."""
        mock_uniform.return_value = 85.0  # High CPU usage
        
        result = self.health_checker.run_check("cpu_usage")
        
        assert result.status == HealthStatus.WARNING
        assert "High CPU usage" in result.message
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_run_check_critical(self, mock_uniform):
        """Test running health check that returns critical."""
        mock_uniform.return_value = 95.0  # Very high CPU usage
        
        result = self.health_checker.run_check("cpu_usage")
        
        assert result.status == HealthStatus.CRITICAL
        assert "Critical CPU usage" in result.message
    
    def test_run_check_nonexistent(self):
        """Test running non-existent health check."""
        result = self.health_checker.run_check("nonexistent_check")
        
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message
    
    def test_run_check_exception(self):
        """Test health check that raises exception."""
        def failing_check():
            raise RuntimeError("Test failure")
        
        self.health_checker.register_check("failing_test", failing_check)
        result = self.health_checker.run_check("failing_test")
        
        assert result.status == HealthStatus.CRITICAL
        assert "Health check failed" in result.message
        assert "Test failure" in result.message
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_run_all_checks(self, mock_uniform):
        """Test running all health checks."""
        mock_uniform.return_value = 25.0  # Normal values
        
        results = self.health_checker.run_all_checks()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "cpu_usage" in results
        assert "memory_usage" in results
        
        for name, result in results.items():
            assert isinstance(result, HealthCheckResult)
            assert result.name == name
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_get_overall_status_healthy(self, mock_uniform):
        """Test overall status when all checks are healthy."""
        mock_uniform.return_value = 25.0  # Low values
        
        status = self.health_checker.get_overall_status()
        assert status == HealthStatus.HEALTHY
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_get_overall_status_critical(self, mock_uniform):
        """Test overall status when some checks are critical."""
        mock_uniform.side_effect = [95.0, 25.0, 25.0, 25.0, 25.0]  # Critical CPU, normal others
        
        status = self.health_checker.get_overall_status()
        assert status == HealthStatus.CRITICAL


class TestSystemMonitor:
    """Test SystemMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = SystemMonitor(collection_interval=0.1)  # Fast interval for testing
    
    def test_monitor_initialization(self):
        """Test SystemMonitor initialization."""
        assert self.monitor.collection_interval == 0.1
        assert isinstance(self.monitor.metric_collector, MetricCollector)
        assert isinstance(self.monitor.health_checker, HealthChecker)
        assert self.monitor._monitoring == False
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    @patch('codesign_playground.utils.monitoring.random.randint')
    def test_collect_system_metrics(self, mock_randint, mock_uniform):
        """Test collecting system metrics."""
        # Mock random values for consistent testing
        mock_uniform.side_effect = [25.0, 60.0, 75.0, 1.5, 1.0, 0.5]  # CPU, memory%, disk%, load avg
        mock_randint.side_effect = [5000000, 8000000, 30]  # network bytes, connections
        
        metrics = self.monitor.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 25.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_usage_percent == 75.0
        assert metrics.uptime_seconds >= 0
        assert len(metrics.load_average) == 3
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    def test_get_health_status(self, mock_uniform):
        """Test getting health status."""
        mock_uniform.return_value = 25.0  # Normal values
        
        health_status = self.monitor.get_health_status()
        
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        assert "timestamp" in health_status
        assert "checks" in health_status
        assert health_status["overall_status"] == "healthy"
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    @patch('codesign_playground.utils.monitoring.random.randint')
    def test_get_monitoring_summary(self, mock_randint, mock_uniform):
        """Test getting monitoring summary."""
        mock_uniform.side_effect = [25.0] * 10  # Normal values
        mock_randint.side_effect = [5000000, 8000000, 30]
        
        summary = self.monitor.get_monitoring_summary()
        
        assert isinstance(summary, dict)
        assert "system_metrics" in summary
        assert "health_status" in summary
        assert "monitoring" in summary
        assert "metrics_summary" in summary
        
        monitoring_info = summary["monitoring"]
        assert "uptime_seconds" in monitoring_info
        assert "monitoring_active" in monitoring_info
        assert "collection_interval" in monitoring_info
    
    @patch('codesign_playground.utils.monitoring.random.uniform')
    @patch('codesign_playground.utils.monitoring.random.randint')
    def test_export_monitoring_data(self, mock_randint, mock_uniform):
        """Test exporting monitoring data."""
        mock_uniform.side_effect = [25.0] * 10
        mock_randint.side_effect = [5000000, 8000000, 30]
        
        exported_data = self.monitor.export_monitoring_data(format="json")
        
        assert isinstance(exported_data, str)
        assert "system_metrics" in exported_data
        assert "health_status" in exported_data
        assert "uptime_seconds" in exported_data
    
    def test_export_monitoring_data_invalid_format(self):
        """Test exporting with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.monitor.export_monitoring_data(format="invalid")
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert self.monitor._monitoring == False
        
        self.monitor.start_monitoring()
        assert self.monitor._monitoring == True
        assert self.monitor._monitor_thread is not None
        assert self.monitor._monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        self.monitor.stop_monitoring()
        assert self.monitor._monitoring == False
        
        # Wait for thread to finish
        time.sleep(0.1)


class TestMonitoringDecorator:
    """Test MonitoringDecorator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.decorator = MonitoringDecorator("test_function")
    
    @patch('codesign_playground.utils.monitoring.record_metric')
    def test_decorator_success(self, mock_record_metric):
        """Test monitoring decorator with successful function."""
        
        @self.decorator
        def test_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        result = test_function()
        
        assert result == "success"
        
        # Check that metrics were recorded
        mock_record_metric.assert_called()
        
        # Should record success and duration
        call_args = [call[0] for call in mock_record_metric.call_args_list]
        metric_names = [args[0] for args in call_args]
        
        assert any("test_function_calls" in name for name in metric_names)
        assert any("test_function_duration_seconds" in name for name in metric_names)
    
    @patch('codesign_playground.utils.monitoring.record_metric')
    def test_decorator_exception(self, mock_record_metric):
        """Test monitoring decorator with function that raises exception."""
        
        @self.decorator
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Should record error metrics
        mock_record_metric.assert_called()
        
        call_args = [call[0] for call in mock_record_metric.call_args_list]
        metric_names = [args[0] for args in call_args]
        
        assert any("test_function_calls" in name for name in metric_names)
        assert any("test_function_errors" in name for name in metric_names)
    
    def test_decorator_no_duration_recording(self):
        """Test decorator with duration recording disabled."""
        decorator = MonitoringDecorator("test_no_duration", record_duration=False)
        
        @decorator
        def test_function():
            return "success"
        
        with patch('codesign_playground.utils.monitoring.record_metric') as mock_record:
            test_function()
            
            # Should not record duration
            call_args = [call[0] for call in mock_record.call_args_list]
            metric_names = [args[0] for args in call_args]
            
            assert not any("duration_seconds" in name for name in metric_names)
    
    def test_decorator_no_error_recording(self):
        """Test decorator with error recording disabled."""
        decorator = MonitoringDecorator("test_no_errors", record_errors=False)
        
        @decorator
        def failing_function():
            raise ValueError("Test error")
        
        with patch('codesign_playground.utils.monitoring.record_metric') as mock_record:
            with pytest.raises(ValueError):
                failing_function()
            
            # Should not record error metrics
            call_args = [call[0] for call in mock_record.call_args_list]
            metric_names = [args[0] for args in call_args]
            
            assert not any("errors" in name for name in metric_names)


class TestMonitoringUtilities:
    """Test monitoring utility functions."""
    
    @patch('codesign_playground.utils.monitoring.get_system_monitor')
    def test_record_metric_function(self, mock_get_monitor):
        """Test global record_metric function."""
        mock_monitor = Mock()
        mock_collector = Mock()
        mock_monitor.metric_collector = mock_collector
        mock_get_monitor.return_value = mock_monitor
        
        # Test different metric types
        record_metric("test_counter", 1, "counter")
        mock_collector.record_counter.assert_called_with("test_counter", 1, None)
        
        record_metric("test_gauge", 42.5, "gauge", labels={"env": "test"})
        mock_collector.record_gauge.assert_called_with("test_gauge", 42.5, {"env": "test"})
        
        record_metric("test_histogram", 10, "histogram")
        mock_collector.record_histogram.assert_called_with("test_histogram", 10, None)
        
        record_metric("test_timer", 0.125, "timer")
        mock_collector.record_timer.assert_called_with("test_timer", 0.125, None)
    
    def test_record_metric_invalid_type(self):
        """Test record_metric with invalid metric type."""
        with pytest.raises(ValueError, match="Unknown metric type"):
            record_metric("test_invalid", 1, "invalid_type")
    
    @patch('codesign_playground.utils.monitoring.get_system_monitor')
    def test_get_health_status_function(self, mock_get_monitor):
        """Test global get_health_status function."""
        mock_monitor = Mock()
        mock_health_status = {"overall_status": "healthy", "checks": {}}
        mock_monitor.get_health_status.return_value = mock_health_status
        mock_get_monitor.return_value = mock_monitor
        
        result = get_health_status()
        
        assert result == mock_health_status
        mock_monitor.get_health_status.assert_called_once()


@pytest.fixture
def sample_metric_collector():
    """Fixture for metric collector with some data."""
    collector = MetricCollector(max_history=50)
    
    # Add some test metrics
    for i in range(10):
        collector.record_counter("test_counter", 1)
        collector.record_gauge("test_gauge", i * 10)
        time.sleep(0.001)  # Small delay for different timestamps
    
    return collector


@pytest.fixture
def sample_health_checker():
    """Fixture for health checker with custom checks."""
    checker = HealthChecker()
    
    # Add custom checks
    def always_healthy():
        return HealthCheckResult(
            name="always_healthy",
            status=HealthStatus.HEALTHY,
            message="Always healthy test",
            timestamp=time.time(),
            duration_ms=1.0
        )
    
    def sometimes_warning():
        import random
        status = HealthStatus.WARNING if random.random() > 0.5 else HealthStatus.HEALTHY
        return HealthCheckResult(
            name="sometimes_warning",
            status=status,
            message="Random warning test",
            timestamp=time.time(),
            duration_ms=2.0
        )
    
    checker.register_check("always_healthy", always_healthy)
    checker.register_check("sometimes_warning", sometimes_warning)
    
    return checker


class TestMonitoringFixtures:
    """Test monitoring functionality with fixtures."""
    
    def test_with_sample_collector(self, sample_metric_collector):
        """Test using sample metric collector fixture."""
        stats = sample_metric_collector.get_metric_stats("test_counter")
        assert stats["count"] == 10
        assert stats["latest"] == 1
        
        gauge_stats = sample_metric_collector.get_metric_stats("test_gauge")
        assert gauge_stats["count"] == 10
        assert gauge_stats["max"] == 90
    
    def test_with_sample_health_checker(self, sample_health_checker):
        """Test using sample health checker fixture."""
        result = sample_health_checker.run_check("always_healthy")
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "always_healthy"
        
        # Run all checks
        results = sample_health_checker.run_all_checks()
        assert "always_healthy" in results
        assert "sometimes_warning" in results
        
        # Check overall status
        overall_status = sample_health_checker.get_overall_status()
        assert overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring system integration."""
        # Initialize monitor
        monitor = SystemMonitor(collection_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Let it collect some data
            time.sleep(0.3)
            
            # Get comprehensive status
            summary = monitor.get_monitoring_summary()
            
            # Verify all components working
            assert "system_metrics" in summary
            assert "health_status" in summary
            assert "monitoring" in summary
            
            # Check that metrics are being collected
            metrics_summary = summary["metrics_summary"]
            assert len(metrics_summary) > 0
            
            # Export data
            exported = monitor.export_monitoring_data(include_history=True)
            assert "metrics_history" in exported
            
        finally:
            monitor.stop_monitoring()
    
    @patch('codesign_playground.utils.monitoring.record_metric')
    def test_decorator_integration_with_monitor(self, mock_record_metric):
        """Test monitoring decorator integration with system monitor."""
        
        @MonitoringDecorator("integration_test")
        def test_integration():
            # Simulate some work
            time.sleep(0.01)
            return "integration_success"
        
        result = test_integration()
        
        assert result == "integration_success"
        
        # Verify metrics were recorded
        mock_record_metric.assert_called()
        
        # Check for expected metric types
        call_args = [call[0] for call in mock_record_metric.call_args_list]
        recorded_metrics = [args[0] for args in call_args]
        
        assert any("integration_test_calls" in metric for metric in recorded_metrics)
        assert any("integration_test_duration_seconds" in metric for metric in recorded_metrics)