"""
Unit tests for memory management functionality.

This module tests the memory profiling, optimization, and management components.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from codesign_playground.utils.memory_management import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryEfficientCache,
    profile_memory,
    optimize_memory,
    detect_memory_leaks,
    get_memory_stats
)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""
    
    def test_memory_snapshot_creation(self):
        """Test MemorySnapshot creation with valid data."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=150.5,
            vms_mb=250.0,
            percent=10.5,
            available_mb=4096.0,
            cached_objects=10000,
            gc_collections={"gen_0": 100, "gen_1": 10, "gen_2": 1},
            top_allocations=[("test.py:10", 1024, 5)]
        )
        
        assert snapshot.rss_mb == 150.5
        assert snapshot.percent == 10.5
        assert snapshot.cached_objects == 10000
        assert len(snapshot.top_allocations) == 1
    
    def test_memory_snapshot_to_dict(self):
        """Test MemorySnapshot serialization to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            rss_mb=150.5,
            vms_mb=250.0,
            percent=10.5,
            available_mb=4096.0,
            cached_objects=10000,
            gc_collections={"gen_0": 100},
            top_allocations=[("test.py:10", 1024, 5)]
        )
        
        result = snapshot.to_dict()
        
        assert isinstance(result, dict)
        assert result["timestamp"] == 1234567890.0
        assert result["rss_mb"] == 150.5
        assert result["cached_objects"] == 10000
        assert len(result["top_allocations"]) == 1
        assert result["top_allocations"][0]["traceback"] == "test.py:10"


class TestMemoryProfiler:
    """Test MemoryProfiler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(enable_tracemalloc=False, max_snapshots=10)
    
    def test_profiler_initialization(self):
        """Test MemoryProfiler initialization."""
        assert self.profiler.max_snapshots == 10
        assert self.profiler.enable_tracemalloc == False
        assert len(self.profiler.snapshots) == 0
    
    @patch('codesign_playground.utils.memory_management.record_metric')
    def test_take_snapshot(self, mock_record_metric):
        """Test taking memory snapshots."""
        snapshot = self.profiler.take_snapshot("test_snapshot")
        
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.rss_mb > 0
        assert snapshot.cached_objects > 0
        assert len(self.profiler.snapshots) == 1
        
        # Check metrics were recorded
        mock_record_metric.assert_called()
    
    def test_multiple_snapshots(self):
        """Test taking multiple snapshots."""
        snapshots = []
        
        for i in range(5):
            snapshot = self.profiler.take_snapshot(f"snapshot_{i}")
            snapshots.append(snapshot)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        assert len(self.profiler.snapshots) == 5
        
        # Check timestamps are in order
        for i in range(1, len(snapshots)):
            assert snapshots[i].timestamp >= snapshots[i-1].timestamp
    
    def test_max_snapshots_limit(self):
        """Test that snapshot count respects max limit."""
        # Take more snapshots than the limit
        for i in range(15):
            self.profiler.take_snapshot(f"snapshot_{i}")
        
        # Should only keep the last 10 (max_snapshots)
        assert len(self.profiler.snapshots) == 10
    
    def test_detect_leaks_no_data(self):
        """Test leak detection with insufficient data."""
        leaks = self.profiler.detect_leaks()
        assert len(leaks) == 0
    
    def test_detect_leaks_with_growth(self):
        """Test leak detection with memory growth."""
        # Take baseline snapshot
        self.profiler.take_snapshot("baseline")
        
        # Simulate memory growth by modifying the snapshot
        baseline = self.profiler.snapshots[0]
        
        # Take another snapshot with higher memory usage
        current_snapshot = self.profiler.take_snapshot("current")
        current_snapshot.rss_mb = baseline.rss_mb + 100.0  # Add 100MB growth
        current_snapshot.cached_objects = baseline.cached_objects + 20000
        
        leaks = self.profiler.detect_leaks(threshold_mb=50.0)
        
        assert len(leaks) > 0
        
        # Check for memory growth detection
        memory_growth_leak = next((leak for leak in leaks if leak["type"] == "memory_growth"), None)
        assert memory_growth_leak is not None
        assert memory_growth_leak["growth_mb"] == 100.0
    
    def test_memory_trend_analysis(self):
        """Test memory trend analysis."""
        # Take multiple snapshots with increasing memory
        base_memory = 100.0
        for i in range(5):
            snapshot = self.profiler.take_snapshot(f"trend_{i}")
            snapshot.rss_mb = base_memory + (i * 10)  # Increasing trend
            time.sleep(0.01)
        
        trend = self.profiler.get_memory_trend(window_minutes=60)
        
        assert "memory_trend_mb_per_hour" in trend
        assert "trend_direction" in trend
        assert trend["snapshots_count"] == 5
        assert trend["trend_direction"] in ["increasing", "stable", "decreasing"]
    
    def test_memory_trend_insufficient_data(self):
        """Test memory trend with insufficient data."""
        trend = self.profiler.get_memory_trend()
        assert "error" in trend
    
    @patch('gc.collect')
    def test_optimize_memory(self, mock_gc_collect):
        """Test memory optimization."""
        mock_gc_collect.return_value = 5  # Mock objects collected
        
        # Take baseline snapshot
        self.profiler.take_snapshot("before_optimization")
        
        result = self.profiler.optimize_memory(aggressive=False)
        
        assert "before_mb" in result
        assert "after_mb" in result
        assert "memory_freed_mb" in result
        assert "optimizations_performed" in result
        assert len(result["optimizations_performed"]) > 0
        
        # GC should have been called
        assert mock_gc_collect.called
    
    def test_export_profile(self):
        """Test profile export functionality."""
        # Take some snapshots
        self.profiler.take_snapshot("test_export_1")
        self.profiler.take_snapshot("test_export_2")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            result_path = self.profiler.export_profile(export_path)
            assert result_path == export_path
            
            # Check file exists and has content
            export_file = Path(export_path)
            assert export_file.exists()
            
            content = export_file.read_text()
            assert len(content) > 0
            assert "snapshots" in content
            assert "profiler_config" in content
            
        finally:
            Path(export_path).unlink(missing_ok=True)


class TestMemoryEfficientCache:
    """Test MemoryEfficientCache class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MemoryEfficientCache(max_memory_mb=1.0, cleanup_threshold=0.8)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_memory_bytes == 1024 * 1024  # 1MB in bytes
        assert self.cache.cleanup_threshold == 0.8
        assert len(self.cache._cache) == 0
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        test_data = "test_value"
        
        success = self.cache.put("test_key", test_data)
        assert success == True
        
        result = self.cache.get("test_key")
        assert result == test_data
    
    def test_get_nonexistent_key(self):
        """Test getting non-existent key."""
        result = self.cache.get("nonexistent_key")
        assert result is None
    
    def test_put_large_value(self):
        """Test putting value that's too large."""
        large_data = "x" * (2 * 1024 * 1024)  # 2MB string, larger than limit
        
        success = self.cache.put("large_key", large_data)
        assert success == False
        
        result = self.cache.get("large_key")
        assert result is None
    
    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced."""
        # Fill cache with multiple small items
        for i in range(100):
            data = "x" * 1000  # 1KB each
            self.cache.put(f"key_{i}", data)
        
        stats = self.cache.get_stats()
        
        # Should not exceed memory limit significantly
        assert stats["memory_utilization"] <= 1.2  # Allow some overhead
    
    def test_lru_eviction(self):
        """Test LRU eviction behavior."""
        # Add items until cache is full
        for i in range(50):
            data = "x" * 1000  # 1KB each
            self.cache.put(f"key_{i}", data)
        
        # Access some items to change their LRU order
        self.cache.get("key_10")
        self.cache.get("key_20")
        
        # Add more items to trigger eviction
        for i in range(50, 100):
            data = "x" * 1000
            self.cache.put(f"key_{i}", data)
        
        # Recently accessed items should still be there
        assert self.cache.get("key_10") is not None
        assert self.cache.get("key_20") is not None
        
        # Some early items should have been evicted
        early_items_found = sum(1 for i in range(10) if self.cache.get(f"key_{i}") is not None)
        assert early_items_found < 10  # Some should be evicted
    
    def test_get_stats(self):
        """Test cache statistics."""
        # Add some items
        for i in range(5):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        stats = self.cache.get_stats()
        
        assert "items" in stats
        assert "memory_usage_mb" in stats
        assert "memory_limit_mb" in stats
        assert "memory_utilization" in stats
        
        assert stats["items"] == 5
        assert stats["memory_usage_mb"] > 0
        assert stats["memory_limit_mb"] == 1.0
        assert 0 <= stats["memory_utilization"] <= 1


class TestMemoryManagementDecorators:
    """Test memory management decorators and utilities."""
    
    @patch('codesign_playground.utils.memory_management.get_memory_profiler')
    def test_profile_memory_decorator(self, mock_get_profiler):
        """Test memory profiling decorator."""
        mock_profiler = Mock()
        mock_profiler.take_snapshot.return_value = Mock(rss_mb=100.0)
        mock_get_profiler.return_value = mock_profiler
        
        @profile_memory
        def test_function():
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert mock_profiler.take_snapshot.call_count == 2  # Before and after
    
    @patch('codesign_playground.utils.memory_management.get_memory_profiler')
    def test_optimize_memory_function(self, mock_get_profiler):
        """Test global optimize_memory function."""
        mock_profiler = Mock()
        mock_optimization_result = {
            "before_mb": 100.0,
            "after_mb": 90.0,
            "memory_freed_mb": 10.0,
            "optimizations_performed": ["gc_collection"]
        }
        mock_profiler.optimize_memory.return_value = mock_optimization_result
        mock_get_profiler.return_value = mock_profiler
        
        result = optimize_memory(aggressive=False)
        
        assert result == mock_optimization_result
        mock_profiler.optimize_memory.assert_called_once_with(aggressive=False)
    
    @patch('codesign_playground.utils.memory_management.get_memory_profiler')
    def test_detect_memory_leaks_function(self, mock_get_profiler):
        """Test global detect_memory_leaks function."""
        mock_profiler = Mock()
        mock_leaks = [{"type": "memory_growth", "growth_mb": 50.0}]
        mock_profiler.detect_leaks.return_value = mock_leaks
        mock_get_profiler.return_value = mock_profiler
        
        result = detect_memory_leaks()
        
        assert result == mock_leaks
        mock_profiler.detect_leaks.assert_called_once()
    
    @patch('codesign_playground.utils.memory_management.get_memory_profiler')
    def test_get_memory_stats_function(self, mock_get_profiler):
        """Test global get_memory_stats function."""
        mock_profiler = Mock()
        mock_snapshot = Mock()
        mock_snapshot.to_dict.return_value = {"rss_mb": 100.0, "percent": 5.0}
        mock_profiler.snapshots = [mock_snapshot]
        mock_profiler.get_memory_trend.return_value = {"trend": "stable"}
        mock_profiler.detect_leaks.return_value = []
        mock_get_profiler.return_value = mock_profiler
        
        result = get_memory_stats()
        
        assert "rss_mb" in result
        assert "trend_analysis" in result
        assert "potential_leaks" in result
    
    @patch('codesign_playground.utils.memory_management.get_memory_profiler')
    def test_get_memory_stats_no_data(self, mock_get_profiler):
        """Test get_memory_stats with no snapshots."""
        mock_profiler = Mock()
        mock_profiler.snapshots = []
        mock_get_profiler.return_value = mock_profiler
        
        result = get_memory_stats()
        
        assert "error" in result


@pytest.fixture
def sample_memory_snapshot():
    """Fixture for sample memory snapshot."""
    return MemorySnapshot(
        timestamp=time.time(),
        rss_mb=150.0,
        vms_mb=250.0,
        percent=8.5,
        available_mb=4096.0,
        cached_objects=15000,
        gc_collections={"gen_0": 100, "gen_1": 10, "gen_2": 1},
        top_allocations=[
            ("test.py:10", 2048, 10),
            ("test.py:20", 1024, 5)
        ]
    )


@pytest.fixture
def populated_profiler():
    """Fixture for profiler with some snapshots."""
    profiler = MemoryProfiler(enable_tracemalloc=False)
    
    # Add some test snapshots
    for i in range(3):
        snapshot = profiler.take_snapshot(f"test_{i}")
        snapshot.rss_mb = 100.0 + (i * 10)  # Increasing memory
        time.sleep(0.01)
    
    return profiler


class TestMemoryManagementFixtures:
    """Test memory management functionality with fixtures."""
    
    def test_with_sample_snapshot(self, sample_memory_snapshot):
        """Test using sample memory snapshot fixture."""
        assert sample_memory_snapshot.rss_mb == 150.0
        assert sample_memory_snapshot.cached_objects == 15000
        assert len(sample_memory_snapshot.top_allocations) == 2
        
        # Test serialization
        snapshot_dict = sample_memory_snapshot.to_dict()
        assert snapshot_dict["rss_mb"] == 150.0
    
    def test_with_populated_profiler(self, populated_profiler):
        """Test using populated profiler fixture."""
        assert len(populated_profiler.snapshots) == 3
        
        # Test trend analysis
        trend = populated_profiler.get_memory_trend()
        assert "snapshots_count" in trend
        assert trend["snapshots_count"] == 3
        
        # Test leak detection
        leaks = populated_profiler.detect_leaks(threshold_mb=5.0)
        assert len(leaks) > 0  # Should detect growth


class TestMemoryManagementIntegration:
    """Integration tests for memory management components."""
    
    def test_end_to_end_memory_monitoring(self):
        """Test complete memory monitoring workflow."""
        # Initialize profiler
        profiler = MemoryProfiler(enable_tracemalloc=False)
        
        # Take baseline snapshot
        baseline = profiler.take_snapshot("baseline")
        
        # Simulate some work that uses memory
        test_data = []
        for i in range(1000):
            test_data.append(f"test_data_{i}" * 10)
        
        # Take another snapshot
        after_work = profiler.take_snapshot("after_work")
        
        # Analyze memory usage
        trend = profiler.get_memory_trend()
        leaks = profiler.detect_leaks(threshold_mb=1.0)
        
        # Optimize memory
        optimization_result = profiler.optimize_memory()
        
        # Take final snapshot
        final = profiler.take_snapshot("final")
        
        # Verify workflow
        assert len(profiler.snapshots) >= 3
        assert baseline.timestamp < after_work.timestamp
        assert after_work.timestamp < final.timestamp
        assert "optimizations_performed" in optimization_result
        
        # Clean up
        del test_data
    
    def test_cache_with_memory_profiling(self):
        """Test cache functionality with memory profiling."""
        cache = MemoryEfficientCache(max_memory_mb=2.0)
        
        @profile_memory
        def use_cache():
            # Fill cache
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}" * 100)
            
            # Access some items
            for i in range(0, 100, 10):
                cache.get(f"key_{i}")
            
            return cache.get_stats()
        
        stats = use_cache()
        
        assert stats["items"] > 0
        assert stats["memory_usage_mb"] > 0
        assert stats["memory_utilization"] <= 1.0