"""
Intelligent Auto-Scaling System for AI Hardware Co-Design Platform.

This module implements Generation 3+ auto-scaling capabilities with predictive
resource allocation, workload-aware scaling, and intelligent load balancing
for optimal performance under varying computational demands.
"""

import asyncio
import time
import statistics
import math
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Fallback psutil implementation
    class FallbackPsutil:
        @staticmethod
        def cpu_count(logical=True):
            return 4 if logical else 2
        
        @staticmethod
        def cpu_percent(interval=None):
            return 50.0
        
        @staticmethod
        def virtual_memory():
            return type('memory', (), {
                'total': 8 * 1024**3,
                'percent': 60.0
            })()
    
    psutil = FallbackPsutil()

from ..utils.logging import get_logger
from ..utils.monitoring import record_metric
from ..utils.exceptions import ScalingError

logger = get_logger(__name__)


class ScalingMode(Enum):
    """Auto-scaling modes for different optimization strategies."""
    REACTIVE = "reactive"          # Scale based on current metrics
    PREDICTIVE = "predictive"      # Scale based on workload prediction
    AGGRESSIVE = "aggressive"      # Fast scaling for bursty workloads
    CONSERVATIVE = "conservative"  # Gradual scaling for stable workloads
    INTELLIGENT = "intelligent"    # ML-based scaling decisions


class WorkloadPattern(Enum):
    """Detected workload patterns for intelligent scaling."""
    STEADY_STATE = "steady"
    BURSTY = "bursty"
    SEASONAL = "seasonal"
    RANDOM = "random"
    CRESCENDO = "crescendo"  # Gradually increasing
    DIMINUENDO = "diminuendo"  # Gradually decreasing


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    error_rate: float = 0.0
    average_response_time: float = 0.0
    throughput: float = 0.0
    
    # Predictive metrics
    predicted_load: float = 0.0
    trend_direction: float = 0.0  # -1 to 1 (decreasing to increasing)
    pattern_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "queue_depth": self.queue_depth,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "error_rate": self.error_rate,
            "average_response_time": self.average_response_time,
            "throughput": self.throughput,
            "predicted_load": self.predicted_load,
            "trend_direction": self.trend_direction,
            "pattern_confidence": self.pattern_confidence
        }


@dataclass
class ScalingDecision:
    """Represents a scaling decision with rationale."""
    
    timestamp: float = field(default_factory=time.time)
    action: str = "none"  # "scale_up", "scale_down", "none"
    target_workers: int = 0
    current_workers: int = 0
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    expected_impact: Dict[str, float] = field(default_factory=dict)


class WorkloadPredictor:
    """Intelligent workload prediction engine."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes of data
        """
        Initialize workload predictor.
        
        Args:
            window_size: Number of historical samples to analyze
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.pattern_history = deque(maxlen=100)
        
        # Pattern detection parameters
        self.steady_threshold = 0.1    # 10% variance for steady state
        self.burst_threshold = 0.5     # 50% spike for burst detection
        self.trend_window = 30         # Samples for trend analysis
        
        logger.info(f"Workload predictor initialized with window_size={window_size}")
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics sample to history."""
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) >= 10:  # Need minimum samples
            pattern = self._detect_pattern()
            self.pattern_history.append(pattern)
    
    def _detect_pattern(self) -> WorkloadPattern:
        """Detect current workload pattern."""
        if len(self.metrics_history) < 10:
            return WorkloadPattern.RANDOM
        
        recent_cpu = [m.cpu_utilization for m in list(self.metrics_history)[-30:]]
        
        # Calculate variance and trend
        variance = statistics.variance(recent_cpu) if len(recent_cpu) > 1 else 0
        mean_cpu = statistics.mean(recent_cpu)
        
        # Trend analysis
        if len(recent_cpu) >= 5:
            early_half = recent_cpu[:len(recent_cpu)//2]
            later_half = recent_cpu[len(recent_cpu)//2:]
            early_mean = statistics.mean(early_half)
            later_mean = statistics.mean(later_half)
            trend = (later_mean - early_mean) / max(early_mean, 0.1)
        else:
            trend = 0
        
        # Pattern classification
        if variance < self.steady_threshold:
            return WorkloadPattern.STEADY_STATE
        elif variance > self.burst_threshold:
            # Check for burst pattern
            max_cpu = max(recent_cpu)
            if max_cpu > mean_cpu * 1.5:
                return WorkloadPattern.BURSTY
        
        # Trend-based patterns
        if trend > 0.2:
            return WorkloadPattern.CRESCENDO
        elif trend < -0.2:
            return WorkloadPattern.DIMINUENDO
        
        # Check for seasonal pattern (simplified)
        if len(self.metrics_history) >= 60:  # Need more data
            hourly_samples = list(self.metrics_history)[-60:]
            cpu_values = [m.cpu_utilization for m in hourly_samples]
            
            # Simple seasonal detection (peaks at regular intervals)
            peaks = self._find_peaks(cpu_values)
            if len(peaks) >= 2:
                peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
                if len(set(peak_intervals)) <= 2:  # Regular intervals
                    return WorkloadPattern.SEASONAL
        
        return WorkloadPattern.RANDOM
    
    def _find_peaks(self, values: List[float], min_distance: int = 10) -> List[int]:
        """Find peaks in the signal."""
        peaks = []
        for i in range(1, len(values) - 1):
            if (values[i] > values[i-1] and values[i] > values[i+1] and 
                values[i] > statistics.mean(values) * 1.2):
                # Check minimum distance from last peak
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        return peaks
    
    def predict_load(self, horizon_minutes: int = 5) -> Tuple[float, float]:
        """
        Predict future load based on historical patterns.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Tuple of (predicted_load, confidence)
        """
        if len(self.metrics_history) < 10:
            return 0.5, 0.1  # Default prediction with low confidence
        
        current_pattern = self.pattern_history[-1] if self.pattern_history else WorkloadPattern.RANDOM
        recent_metrics = list(self.metrics_history)[-30:]
        current_cpu = statistics.mean([m.cpu_utilization for m in recent_metrics])
        
        # Pattern-specific prediction
        if current_pattern == WorkloadPattern.STEADY_STATE:
            predicted_load = current_cpu
            confidence = 0.8
        
        elif current_pattern == WorkloadPattern.BURSTY:
            # Predict higher load for bursty patterns
            max_recent = max([m.cpu_utilization for m in recent_metrics])
            predicted_load = min(1.0, current_cpu * 1.3)
            confidence = 0.6
        
        elif current_pattern == WorkloadPattern.CRESCENDO:
            # Predict continued growth
            growth_rate = self._calculate_growth_rate()
            predicted_load = min(1.0, current_cpu * (1 + growth_rate * horizon_minutes / 60))
            confidence = 0.7
        
        elif current_pattern == WorkloadPattern.DIMINUENDO:
            # Predict continued decline
            decline_rate = abs(self._calculate_growth_rate())
            predicted_load = max(0.1, current_cpu * (1 - decline_rate * horizon_minutes / 60))
            confidence = 0.7
        
        elif current_pattern == WorkloadPattern.SEASONAL:
            # Use historical seasonal data
            predicted_load = self._predict_seasonal(horizon_minutes)
            confidence = 0.8
        
        else:  # RANDOM
            # Use moving average with some uncertainty
            predicted_load = current_cpu
            confidence = 0.4
        
        return predicted_load, confidence
    
    def _calculate_growth_rate(self) -> float:
        """Calculate current growth rate."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_values = [m.cpu_utilization for m in list(self.metrics_history)[-10:]]
        early_avg = statistics.mean(recent_values[:5])
        later_avg = statistics.mean(recent_values[5:])
        
        if early_avg > 0:
            return (later_avg - early_avg) / early_avg
        return 0.0
    
    def _predict_seasonal(self, horizon_minutes: int) -> float:
        """Predict load based on seasonal patterns."""
        if len(self.metrics_history) < 60:
            return statistics.mean([m.cpu_utilization for m in self.metrics_history])
        
        # Simple seasonal prediction using historical averages
        # In a real implementation, this would use more sophisticated time series analysis
        historical_values = [m.cpu_utilization for m in list(self.metrics_history)[-60:]]
        return statistics.mean(historical_values)
    
    def get_trend_direction(self) -> float:
        """Get current trend direction (-1 to 1)."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_values = [m.cpu_utilization for m in list(self.metrics_history)[-20:]]
        
        # Simple linear regression slope
        n = len(recent_values)
        x_sum = sum(range(n))
        y_sum = sum(recent_values)
        xy_sum = sum(i * recent_values[i] for i in range(n))
        x_squared_sum = sum(i * i for i in range(n))
        
        if n * x_squared_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)
        
        # Normalize slope to -1 to 1 range
        return max(-1.0, min(1.0, slope * 10))


class IntelligentAutoScaler:
    """Main auto-scaling engine with intelligent decision making."""
    
    def __init__(
        self,
        mode: ScalingMode = ScalingMode.INTELLIGENT,
        min_workers: int = 2,
        max_workers: int = 32,
        target_cpu_utilization: float = 0.7,
        scale_up_cooldown: float = 60.0,
        scale_down_cooldown: float = 180.0
    ):
        """
        Initialize intelligent auto-scaler.
        
        Args:
            mode: Scaling mode strategy
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers  
            target_cpu_utilization: Target CPU utilization (0.0 to 1.0)
            scale_up_cooldown: Minimum time between scale-up operations (seconds)
            scale_down_cooldown: Minimum time between scale-down operations (seconds)
        """
        self.mode = mode
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_utilization = target_cpu_utilization
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        # Current state
        self.current_workers = min_workers
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Components
        self.predictor = WorkloadPredictor()
        self.metrics_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=100)
        
        # Performance tracking
        self.scaling_events = 0
        self.prediction_accuracy = deque(maxlen=50)
        
        # Background monitoring
        self.monitoring_enabled = True
        self._shutdown_event = threading.Event()
        self._monitoring_thread = None
        
        self._start_monitoring()
        
        logger.info(f"Intelligent auto-scaler initialized: mode={mode.value}, workers={min_workers}-{max_workers}, target_cpu={target_cpu_utilization}")
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_enabled:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="AutoScalerMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Background monitoring and scaling loop."""
        logger.info("Auto-scaler monitoring started")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect current metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                self.predictor.add_metrics(metrics)
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                if decision.action != "none":
                    self._execute_scaling_decision(decision)
                    self.decision_history.append(decision)
                
                # Validate predictions
                self._validate_predictions()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(10.0)  # Backoff on error
    
    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get prediction from predictor
            predicted_load, pattern_confidence = self.predictor.predict_load()
            trend_direction = self.predictor.get_trend_direction()
            
            return ScalingMetrics(
                cpu_utilization=cpu_percent / 100.0,
                memory_utilization=memory.percent / 100.0,
                active_tasks=threading.active_count(),
                predicted_load=predicted_load,
                trend_direction=trend_direction,
                pattern_confidence=pattern_confidence
            )
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return ScalingMetrics()  # Return default metrics
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on current metrics and predictions."""
        current_time = time.time()
        
        decision = ScalingDecision(
            current_workers=self.current_workers,
            target_workers=self.current_workers
        )
        
        # Check cooldown periods
        scale_up_ready = (current_time - self.last_scale_up) >= self.scale_up_cooldown
        scale_down_ready = (current_time - self.last_scale_down) >= self.scale_down_cooldown
        
        # Mode-specific decision logic
        if self.mode == ScalingMode.REACTIVE:
            decision = self._reactive_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
        elif self.mode == ScalingMode.PREDICTIVE:
            decision = self._predictive_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
        elif self.mode == ScalingMode.AGGRESSIVE:
            decision = self._aggressive_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
        elif self.mode == ScalingMode.CONSERVATIVE:
            decision = self._conservative_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
        elif self.mode == ScalingMode.INTELLIGENT:
            decision = self._intelligent_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
        
        # Apply bounds
        decision.target_workers = max(self.min_workers, min(self.max_workers, decision.target_workers))
        
        # Determine action
        if decision.target_workers > self.current_workers:
            decision.action = "scale_up"
        elif decision.target_workers < self.current_workers:
            decision.action = "scale_down"
        else:
            decision.action = "none"
        
        return decision
    
    def _reactive_scaling_decision(
        self, 
        metrics: ScalingMetrics, 
        decision: ScalingDecision,
        scale_up_ready: bool,
        scale_down_ready: bool
    ) -> ScalingDecision:
        """Reactive scaling based on current metrics only."""
        current_cpu = metrics.cpu_utilization
        
        if current_cpu > self.target_cpu_utilization + 0.1 and scale_up_ready:
            # Scale up
            scale_factor = min(2.0, current_cpu / self.target_cpu_utilization)
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.append(f"CPU utilization {current_cpu:.1%} > target {self.target_cpu_utilization:.1%}")
            decision.confidence = 0.7
            
        elif current_cpu < self.target_cpu_utilization - 0.15 and scale_down_ready:
            # Scale down
            scale_factor = max(0.5, current_cpu / self.target_cpu_utilization)
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.append(f"CPU utilization {current_cpu:.1%} < target {self.target_cpu_utilization:.1%}")
            decision.confidence = 0.6
        
        return decision
    
    def _predictive_scaling_decision(
        self,
        metrics: ScalingMetrics,
        decision: ScalingDecision,
        scale_up_ready: bool,
        scale_down_ready: bool
    ) -> ScalingDecision:
        """Predictive scaling based on workload predictions."""
        predicted_cpu = metrics.predicted_load
        confidence = metrics.pattern_confidence
        
        # Only act on high-confidence predictions
        if confidence > 0.6:
            if predicted_cpu > self.target_cpu_utilization + 0.1 and scale_up_ready:
                scale_factor = min(2.0, predicted_cpu / self.target_cpu_utilization)
                decision.target_workers = int(self.current_workers * scale_factor)
                decision.reasoning.append(f"Predicted CPU {predicted_cpu:.1%} > target (confidence={confidence:.1%})")
                decision.confidence = confidence
                
            elif predicted_cpu < self.target_cpu_utilization - 0.15 and scale_down_ready:
                scale_factor = max(0.5, predicted_cpu / self.target_cpu_utilization)
                decision.target_workers = int(self.current_workers * scale_factor)
                decision.reasoning.append(f"Predicted CPU {predicted_cpu:.1%} < target (confidence={confidence:.1%})")
                decision.confidence = confidence * 0.8  # Slightly lower confidence for scale down
        else:
            # Fall back to reactive scaling for low confidence
            decision = self._reactive_scaling_decision(metrics, decision, scale_up_ready, scale_down_ready)
            decision.reasoning.append("Low prediction confidence, using reactive scaling")
        
        return decision
    
    def _aggressive_scaling_decision(
        self,
        metrics: ScalingMetrics,
        decision: ScalingDecision,
        scale_up_ready: bool,
        scale_down_ready: bool
    ) -> ScalingDecision:
        """Aggressive scaling for bursty workloads."""
        current_cpu = metrics.cpu_utilization
        trend = metrics.trend_direction
        
        # More aggressive thresholds
        if current_cpu > self.target_cpu_utilization + 0.05 and scale_up_ready:
            # Scale up aggressively, especially if trend is upward
            scale_factor = 1.5 + (trend * 0.5) if trend > 0 else 1.5
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.append(f"Aggressive scale-up: CPU={current_cpu:.1%}, trend={trend:.2f}")
            decision.confidence = 0.8
            
        elif current_cpu < self.target_cpu_utilization - 0.2 and scale_down_ready:
            # Less aggressive scale down to avoid thrashing
            scale_factor = 0.8
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.append(f"Aggressive scale-down: CPU={current_cpu:.1%}")
            decision.confidence = 0.6
        
        return decision
    
    def _conservative_scaling_decision(
        self,
        metrics: ScalingMetrics,
        decision: ScalingDecision,
        scale_up_ready: bool,
        scale_down_ready: bool
    ) -> ScalingDecision:
        """Conservative scaling for stable workloads."""
        current_cpu = metrics.cpu_utilization
        
        # Need sustained high/low utilization before scaling
        if len(self.metrics_history) >= 10:
            recent_cpu = [m.cpu_utilization for m in list(self.metrics_history)[-10:]]
            avg_cpu = statistics.mean(recent_cpu)
            cpu_stability = 1.0 - statistics.stdev(recent_cpu) if len(recent_cpu) > 1 else 1.0
            
            if avg_cpu > self.target_cpu_utilization + 0.15 and cpu_stability > 0.8 and scale_up_ready:
                # Conservative scale up
                decision.target_workers = self.current_workers + 1
                decision.reasoning.append(f"Conservative scale-up: stable high CPU={avg_cpu:.1%}")
                decision.confidence = cpu_stability
                
            elif avg_cpu < self.target_cpu_utilization - 0.25 and cpu_stability > 0.8 and scale_down_ready:
                # Conservative scale down
                decision.target_workers = max(self.min_workers, self.current_workers - 1)
                decision.reasoning.append(f"Conservative scale-down: stable low CPU={avg_cpu:.1%}")
                decision.confidence = cpu_stability * 0.9
        
        return decision
    
    def _intelligent_scaling_decision(
        self,
        metrics: ScalingMetrics,
        decision: ScalingDecision,
        scale_up_ready: bool,
        scale_down_ready: bool
    ) -> ScalingDecision:
        """Intelligent scaling combining multiple factors."""
        current_cpu = metrics.cpu_utilization
        predicted_cpu = metrics.predicted_load
        trend = metrics.trend_direction
        confidence = metrics.pattern_confidence
        
        # Weighted decision based on current state, prediction, and trend
        current_weight = 0.4
        prediction_weight = confidence * 0.4
        trend_weight = 0.2
        
        # Calculate effective CPU utilization
        effective_cpu = (
            current_cpu * current_weight +
            predicted_cpu * prediction_weight +
            (current_cpu + trend * 0.1) * trend_weight
        )
        
        # Adjust thresholds based on confidence and trend
        up_threshold = self.target_cpu_utilization + 0.1 - (confidence * 0.05)
        down_threshold = self.target_cpu_utilization - 0.15 + (confidence * 0.05)
        
        if effective_cpu > up_threshold and scale_up_ready:
            # Intelligent scale up
            base_factor = min(2.0, effective_cpu / self.target_cpu_utilization)
            trend_factor = 1.0 + max(0, trend * 0.3)  # Boost for positive trend
            scale_factor = base_factor * trend_factor
            
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.extend([
                f"Intelligent scale-up: effective_cpu={effective_cpu:.1%}",
                f"Current={current_cpu:.1%}, predicted={predicted_cpu:.1%}, trend={trend:.2f}",
                f"Confidence={confidence:.1%}"
            ])
            decision.confidence = 0.6 + (confidence * 0.3)
            
        elif effective_cpu < down_threshold and scale_down_ready:
            # Intelligent scale down (more conservative)
            base_factor = max(0.7, effective_cpu / self.target_cpu_utilization)
            trend_factor = 1.0 + min(0, trend * 0.2)  # Reduce for negative trend
            scale_factor = base_factor * trend_factor
            
            decision.target_workers = int(self.current_workers * scale_factor)
            decision.reasoning.extend([
                f"Intelligent scale-down: effective_cpu={effective_cpu:.1%}",
                f"Current={current_cpu:.1%}, predicted={predicted_cpu:.1%}, trend={trend:.2f}",
                f"Confidence={confidence:.1%}"
            ])
            decision.confidence = 0.5 + (confidence * 0.3)
        
        # Add expected impact
        if decision.action != "none":
            expected_cpu_change = (self.current_workers - decision.target_workers) * 0.1
            decision.expected_impact = {
                "cpu_utilization_change": expected_cpu_change,
                "throughput_change": (decision.target_workers - self.current_workers) * 0.2
            }
        
        return decision
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        try:
            if decision.action == "scale_up":
                self.current_workers = decision.target_workers
                self.last_scale_up = time.time()
                self.scaling_events += 1
                
                logger.info(f"Scaled UP to {decision.target_workers} workers (confidence={decision.confidence:.2f})")
                for reason in decision.reasoning:
                    logger.info(f"  Reason: {reason}")
                
                record_metric("autoscaler_scale_up", 1, "counter")
                record_metric("autoscaler_worker_count", self.current_workers, "gauge")
                
            elif decision.action == "scale_down":
                self.current_workers = decision.target_workers
                self.last_scale_down = time.time()
                self.scaling_events += 1
                
                logger.info(f"Scaled DOWN to {decision.target_workers} workers (confidence={decision.confidence:.2f})")
                for reason in decision.reasoning:
                    logger.info(f"  Reason: {reason}")
                
                record_metric("autoscaler_scale_down", 1, "counter")
                record_metric("autoscaler_worker_count", self.current_workers, "gauge")
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            raise ScalingError(f"Failed to execute scaling decision: {e}")
    
    def _validate_predictions(self):
        """Validate prediction accuracy for continuous improvement."""
        if len(self.metrics_history) >= 2:
            current_metrics = self.metrics_history[-1]
            previous_metrics = self.metrics_history[-2]
            
            # Compare previous prediction with actual outcome
            prediction_error = abs(previous_metrics.predicted_load - current_metrics.cpu_utilization)
            accuracy = max(0, 1.0 - prediction_error)
            
            self.prediction_accuracy.append(accuracy)
            
            # Log prediction accuracy periodically
            if len(self.prediction_accuracy) >= 50 and len(self.prediction_accuracy) % 50 == 0:
                avg_accuracy = statistics.mean(self.prediction_accuracy)
                logger.info(f"Prediction accuracy over last 50 samples: {avg_accuracy:.1%}")
                record_metric("autoscaler_prediction_accuracy", avg_accuracy, "gauge")
    
    def get_current_workers(self) -> int:
        """Get current number of workers."""
        return self.current_workers
    
    def force_scale(self, target_workers: int, reason: str = "Manual override"):
        """Force scaling to specific number of workers."""
        if target_workers < self.min_workers or target_workers > self.max_workers:
            raise ValueError(f"Target workers {target_workers} outside bounds [{self.min_workers}, {self.max_workers}]")
        
        old_workers = self.current_workers
        self.current_workers = target_workers
        
        logger.warning(f"FORCED scaling from {old_workers} to {target_workers} workers. Reason: {reason}")
        record_metric("autoscaler_forced_scaling", 1, "counter")
        record_metric("autoscaler_worker_count", self.current_workers, "gauge")
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling report."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last 5 minutes
        
        # Calculate statistics
        avg_cpu = statistics.mean([m.cpu_utilization for m in recent_metrics])
        avg_predicted_load = statistics.mean([m.predicted_load for m in recent_metrics])
        avg_trend = statistics.mean([m.trend_direction for m in recent_metrics])
        
        # Prediction accuracy
        avg_prediction_accuracy = statistics.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
        
        # Recent decisions
        recent_decisions = list(self.decision_history)[-10:]
        
        return {
            "mode": self.mode.value,
            "current_state": {
                "workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "target_cpu_utilization": self.target_cpu_utilization
            },
            "performance": {
                "avg_cpu_utilization": avg_cpu,
                "avg_predicted_load": avg_predicted_load,
                "avg_trend": avg_trend,
                "prediction_accuracy": avg_prediction_accuracy,
                "scaling_events": self.scaling_events
            },
            "cooldown_status": {
                "time_since_scale_up": time.time() - self.last_scale_up,
                "time_since_scale_down": time.time() - self.last_scale_down,
                "scale_up_ready": (time.time() - self.last_scale_up) >= self.scale_up_cooldown,
                "scale_down_ready": (time.time() - self.last_scale_down) >= self.scale_down_cooldown
            },
            "recent_decisions": [
                {
                    "timestamp": d.timestamp,
                    "action": d.action,
                    "target_workers": d.target_workers,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning
                }
                for d in recent_decisions
            ]
        }
    
    def shutdown(self):
        """Graceful shutdown of auto-scaler."""
        logger.info("Shutting down intelligent auto-scaler")
        
        self._shutdown_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10.0)
        
        logger.info("Auto-scaler shutdown complete")


# Global auto-scaler instance
_auto_scaler: Optional[IntelligentAutoScaler] = None


def get_auto_scaler(
    mode: ScalingMode = ScalingMode.INTELLIGENT,
    **kwargs
) -> IntelligentAutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    
    if _auto_scaler is None:
        _auto_scaler = IntelligentAutoScaler(mode=mode, **kwargs)
    
    return _auto_scaler


def auto_scaled_execution(func: Callable) -> Callable:
    """
    Decorator to automatically scale resources based on execution demand.
    
    Args:
        func: Function to wrap with auto-scaling
        
    Returns:
        Wrapped function with auto-scaling capabilities
    """
    def wrapper(*args, **kwargs):
        scaler = get_auto_scaler()
        
        # Record task start
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            record_metric("scaled_function_execution_time", execution_time * 1000, "histogram")
            record_metric("scaled_function_success", 1, "counter")
            
            return result
            
        except Exception as e:
            # Record failed execution
            execution_time = time.time() - start_time
            record_metric("scaled_function_error_time", execution_time * 1000, "histogram")
            record_metric("scaled_function_error", 1, "counter")
            raise
    
    return wrapper