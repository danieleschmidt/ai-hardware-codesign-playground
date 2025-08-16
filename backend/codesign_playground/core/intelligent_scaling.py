"""
Intelligent auto-scaling system with advanced ML predictions and cost optimization.

This module provides next-generation scaling capabilities including:
- ML-powered demand prediction with seasonality detection
- Anomaly detection and proactive response
- Cost optimization with usage forecasting
- Multi-metric decision making
- Horizontal and vertical scaling strategies
"""

import time
import threading
import asyncio
import math
import statistics
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import json
import pickle
import psutil

# ML and prediction libraries
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib

import logging
from ..utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for different scenarios."""
    REACTIVE = "reactive"           # React to current load
    PREDICTIVE = "predictive"       # Scale based on predictions
    PROACTIVE = "proactive"         # Scale ahead of demand
    COST_OPTIMIZED = "cost_optimized"  # Minimize costs
    PERFORMANCE_FIRST = "performance_first"  # Prioritize performance
    BALANCED = "balanced"           # Balance cost and performance
    ADAPTIVE = "adaptive"           # Adaptive strategy selection


class ScalingDirection(Enum):
    """Scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"    # Horizontal scaling
    SCALE_IN = "scale_in"      # Horizontal scaling
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """Comprehensive resource metrics for scaling decisions."""
    
    # Timestamps
    timestamp: float
    collection_duration_ms: float = 0.0
    
    # System resources
    cpu_percent: float = 0.0
    cpu_count: int = 0
    memory_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_io_read_mb_per_s: float = 0.0
    disk_io_write_mb_per_s: float = 0.0
    
    # Network
    network_in_mb_per_s: float = 0.0
    network_out_mb_per_s: float = 0.0
    tcp_connections: int = 0
    udp_connections: int = 0
    
    # Load and processes
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    context_switches_per_s: float = 0.0
    
    # Application metrics
    active_workers: int = 0
    queue_size: int = 0
    pending_tasks: int = 0
    completed_tasks_per_minute: int = 0
    failed_tasks_per_minute: int = 0
    
    # Performance metrics
    response_time_ms: float = 0.0
    response_time_p50_ms: float = 0.0
    response_time_p95_ms: float = 0.0
    response_time_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 1.0
    
    # Business metrics
    concurrent_users: int = 0
    requests_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    active_sessions: int = 0
    
    # Cost metrics
    estimated_cost_per_hour: float = 0.0
    resource_efficiency: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to ML feature vector."""
        return np.array([
            self.cpu_percent,
            self.memory_percent,
            self.disk_usage_percent,
            self.load_avg_1m,
            self.load_avg_5m,
            self.load_avg_15m,
            self.active_workers,
            self.queue_size,
            self.pending_tasks,
            self.response_time_ms,
            self.response_time_p95_ms,
            self.throughput_rps,
            self.error_rate,
            self.concurrent_users,
            self.requests_per_second,
            self.cache_hit_rate,
            self.network_in_mb_per_s,
            self.network_out_mb_per_s,
            self.disk_io_read_mb_per_s,
            self.disk_io_write_mb_per_s
        ])


@dataclass
class ScalingDecision:
    """Enhanced scaling decision with comprehensive analysis."""
    
    # Basic decision
    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    confidence: float
    
    # Analysis
    reasoning: List[str]
    risk_factors: List[str]
    predicted_metrics: Dict[str, float]
    
    # ML insights
    demand_forecast: float = 0.0
    anomaly_detected: bool = False
    seasonality_factor: float = 1.0
    trend_direction: str = "stable"
    
    # Cost analysis
    cost_impact_per_hour: float = 0.0
    cost_benefit_ratio: float = 0.0
    estimated_savings: float = 0.0
    
    # Performance impact
    performance_impact: float = 0.0
    sla_risk: float = 0.0
    capacity_buffer: float = 0.0
    
    # Timing
    recommended_timing: str = "immediate"
    forecast_horizon_minutes: int = 15
    execution_time_estimate_s: float = 30.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    strategy_used: ScalingStrategy = ScalingStrategy.BALANCED
    model_version: str = "1.0"


class SeasonalityDetector:
    """Detect and model seasonal patterns in workload."""
    
    def __init__(self):
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.weekly_patterns = defaultdict(list)
        self.monthly_patterns = defaultdict(list)
        
        self.min_samples_for_pattern = 10
        self.max_samples_per_pattern = 100
        
    def update_patterns(self, timestamp: float, value: float) -> None:
        """Update seasonal patterns with new data."""
        dt = time.localtime(timestamp)
        
        # Update patterns
        self.hourly_patterns[dt.tm_hour].append(value)
        self.daily_patterns[dt.tm_wday].append(value)
        self.weekly_patterns[dt.tm_wday].append(value)
        self.monthly_patterns[dt.tm_mon].append(value)
        
        # Limit memory usage
        for patterns in [self.hourly_patterns, self.daily_patterns, 
                        self.weekly_patterns, self.monthly_patterns]:
            for key in patterns:
                if len(patterns[key]) > self.max_samples_per_pattern:
                    patterns[key] = patterns[key][-self.max_samples_per_pattern:]
    
    def get_seasonal_factor(self, timestamp: float) -> float:
        """Get seasonal adjustment factor for given timestamp."""
        dt = time.localtime(timestamp)
        factors = []
        
        # Hourly seasonality
        if (dt.tm_hour in self.hourly_patterns and 
            len(self.hourly_patterns[dt.tm_hour]) >= self.min_samples_for_pattern):
            
            hourly_avg = np.mean(self.hourly_patterns[dt.tm_hour])
            overall_avg = np.mean([v for values in self.hourly_patterns.values() 
                                 for v in values])
            if overall_avg > 0:
                factors.append(hourly_avg / overall_avg)
        
        # Daily seasonality
        if (dt.tm_wday in self.daily_patterns and 
            len(self.daily_patterns[dt.tm_wday]) >= self.min_samples_for_pattern):
            
            daily_avg = np.mean(self.daily_patterns[dt.tm_wday])
            overall_avg = np.mean([v for values in self.daily_patterns.values() 
                                 for v in values])
            if overall_avg > 0:
                factors.append(daily_avg / overall_avg)
        
        # Return weighted average of factors
        if factors:
            return np.mean(factors)
        return 1.0
    
    def predict_pattern(self, future_timestamp: float) -> float:
        """Predict expected value based on seasonal patterns."""
        factor = self.get_seasonal_factor(future_timestamp)
        
        # Get baseline from recent data
        all_values = [v for values in self.hourly_patterns.values() for v in values]
        baseline = np.mean(all_values) if all_values else 1.0
        
        return baseline * factor


class AnomalyDetector:
    """Detect anomalies in system behavior."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.training_data = []
        self.anomaly_history = deque(maxlen=1000)
        
    def update_training_data(self, metrics: ResourceMetrics) -> None:
        """Update training data with new metrics."""
        features = metrics.to_feature_vector()
        self.training_data.append(features)
        
        # Limit training data size
        if len(self.training_data) > 5000:
            self.training_data = self.training_data[-2500:]
        
        # Retrain periodically
        if len(self.training_data) % 100 == 0 and len(self.training_data) >= 100:
            self._retrain_model()
    
    def detect_anomaly(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Detect if current metrics represent an anomaly."""
        if not self.trained or len(self.training_data) < 50:
            return {"is_anomaly": False, "confidence": 0.0, "anomaly_score": 0.0}
        
        try:
            features = metrics.to_feature_vector().reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
            
            # Convert score to confidence (higher score = less anomalous)
            confidence = max(0.0, min(1.0, (anomaly_score + 0.5) * 2))
            
            result = {
                "is_anomaly": is_anomaly,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "timestamp": metrics.timestamp
            }
            
            self.anomaly_history.append(result)
            return result
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0, "anomaly_score": 0.0}
    
    def _retrain_model(self) -> None:
        """Retrain anomaly detection model."""
        try:
            if len(self.training_data) < 50:
                return
            
            X = np.array(self.training_data)
            
            # Fit scaler and model
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.isolation_forest.fit(X_scaled)
            
            self.trained = True
            logger.info("Retrained anomaly detection model")
            
        except Exception as e:
            logger.error(f"Anomaly model retraining failed: {e}")
    
    def get_recent_anomalies(self, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get anomalies from recent time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        return [
            anomaly for anomaly in self.anomaly_history
            if anomaly.get("timestamp", 0) > cutoff_time and anomaly.get("is_anomaly", False)
        ]


class DemandPredictor:
    """Predict future resource demand using ML."""
    
    def __init__(self):
        self.models = {
            "short_term": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "medium_term": RandomForestRegressor(n_estimators=100, random_state=42),
            "long_term": RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        self.scalers = {
            "features": StandardScaler(),
            "targets": MinMaxScaler()
        }
        
        self.training_data = []
        self.prediction_history = deque(maxlen=1000)
        self.models_trained = {key: False for key in self.models.keys()}
        self.last_training = 0.0
        
        # Feature engineering
        self.feature_windows = [5, 15, 30, 60]  # minutes
        
    def extract_time_features(self, timestamp: float) -> np.ndarray:
        """Extract time-based features."""
        dt = time.localtime(timestamp)
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * dt.tm_hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.tm_hour / 24)
        
        day_sin = np.sin(2 * np.pi * dt.tm_wday / 7)
        day_cos = np.cos(2 * np.pi * dt.tm_wday / 7)
        
        month_sin = np.sin(2 * np.pi * dt.tm_mon / 12)
        month_cos = np.cos(2 * np.pi * dt.tm_mon / 12)
        
        # Additional time features
        is_weekend = float(dt.tm_wday >= 5)
        is_business_hours = float(9 <= dt.tm_hour <= 17)
        is_peak_hours = float(dt.tm_hour in [9, 10, 11, 14, 15, 16])
        
        return np.array([
            hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos,
            is_weekend, is_business_hours, is_peak_hours
        ])
    
    def extract_lag_features(self, current_metrics: ResourceMetrics) -> np.ndarray:
        """Extract lagged features from historical data."""
        if len(self.training_data) < max(self.feature_windows):
            return np.zeros(len(self.feature_windows) * 3)  # 3 metrics per window
        
        features = []
        for window in self.feature_windows:
            if len(self.training_data) >= window:
                recent_data = self.training_data[-window:]
                
                # Average metrics over window
                avg_cpu = np.mean([d["metrics"].cpu_percent for d in recent_data])
                avg_throughput = np.mean([d["metrics"].throughput_rps for d in recent_data])
                avg_response_time = np.mean([d["metrics"].response_time_ms for d in recent_data])
                
                features.extend([avg_cpu, avg_throughput, avg_response_time])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def predict_demand(self, current_metrics: ResourceMetrics, 
                      horizon_minutes: int = 15) -> Dict[str, float]:
        """Predict future demand at different time horizons."""
        predictions = {}
        
        # Determine which model to use based on horizon
        if horizon_minutes <= 30:
            model_key = "short_term"
        elif horizon_minutes <= 120:
            model_key = "medium_term"
        else:
            model_key = "long_term"
        
        if not self.models_trained[model_key]:
            # Fallback to simple prediction
            return self._fallback_prediction(current_metrics, horizon_minutes)
        
        try:
            # Prepare features
            metric_features = current_metrics.to_feature_vector()
            time_features = self.extract_time_features(current_metrics.timestamp + horizon_minutes * 60)
            lag_features = self.extract_lag_features(current_metrics)
            
            features = np.concatenate([metric_features, time_features, lag_features]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scalers["features"].transform(features)
            
            # Make prediction
            prediction_scaled = self.models[model_key].predict(features_scaled)[0]
            
            # Inverse transform prediction
            prediction = self.scalers["targets"].inverse_transform([[prediction_scaled]])[0][0]
            
            predictions[f"{horizon_minutes}min"] = max(0, prediction)
            
            # Store prediction for validation
            self.prediction_history.append({
                "timestamp": current_metrics.timestamp,
                "horizon_minutes": horizon_minutes,
                "prediction": prediction,
                "actual_throughput": current_metrics.throughput_rps,
                "model_used": model_key
            })
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Demand prediction failed: {e}")
            return self._fallback_prediction(current_metrics, horizon_minutes)
    
    def _fallback_prediction(self, current_metrics: ResourceMetrics, 
                           horizon_minutes: int) -> Dict[str, float]:
        """Fallback prediction using simple heuristics."""
        # Use current throughput with slight trend adjustment
        base_throughput = current_metrics.throughput_rps
        
        # Simple trend estimation
        if len(self.training_data) >= 10:
            recent_throughputs = [d["metrics"].throughput_rps for d in self.training_data[-10:]]
            trend = np.polyfit(range(len(recent_throughputs)), recent_throughputs, 1)[0]
            predicted_throughput = base_throughput + (trend * horizon_minutes)
        else:
            predicted_throughput = base_throughput
        
        return {f"{horizon_minutes}min": max(0, predicted_throughput)}
    
    def update_model(self, metrics: ResourceMetrics, 
                    actual_demand: Optional[float] = None) -> None:
        """Update model with new training data."""
        training_point = {
            "timestamp": metrics.timestamp,
            "metrics": metrics,
            "target": actual_demand or metrics.throughput_rps
        }
        
        self.training_data.append(training_point)
        
        # Limit training data size
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-5000:]
        
        # Retrain periodically
        if (time.time() - self.last_training > 3600 and  # Every hour
            len(self.training_data) >= 200):  # Minimum data
            self._retrain_models()
    
    def _retrain_models(self) -> None:
        """Retrain all prediction models."""
        try:
            if len(self.training_data) < 100:
                return
            
            # Prepare training data
            X_list = []
            y_list = []
            
            for i, data_point in enumerate(self.training_data):
                if i < max(self.feature_windows):
                    continue  # Skip early points without enough history
                
                metrics = data_point["metrics"]
                
                # Extract features
                metric_features = metrics.to_feature_vector()
                time_features = self.extract_time_features(metrics.timestamp)
                
                # Create lag features using previous data
                lag_features = []
                for window in self.feature_windows:
                    if i >= window:
                        window_data = self.training_data[i-window:i]
                        avg_cpu = np.mean([d["metrics"].cpu_percent for d in window_data])
                        avg_throughput = np.mean([d["metrics"].throughput_rps for d in window_data])
                        avg_response_time = np.mean([d["metrics"].response_time_ms for d in window_data])
                        lag_features.extend([avg_cpu, avg_throughput, avg_response_time])
                    else:
                        lag_features.extend([0.0, 0.0, 0.0])
                
                features = np.concatenate([metric_features, time_features, lag_features])
                target = data_point["target"]
                
                X_list.append(features)
                y_list.append(target)
            
            if len(X_list) < 50:
                return
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Fit scalers
            self.scalers["features"].fit(X)
            self.scalers["targets"].fit(y.reshape(-1, 1))
            
            # Transform data
            X_scaled = self.scalers["features"].transform(X)
            y_scaled = self.scalers["targets"].transform(y.reshape(-1, 1)).ravel()
            
            # Train models
            for model_key, model in self.models.items():
                model.fit(X_scaled, y_scaled)
                self.models_trained[model_key] = True
                
                # Evaluate model
                cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=3, scoring='neg_mean_squared_error')
                logger.info(f"Retrained {model_key} model. CV RMSE: {np.sqrt(-cv_scores.mean()):.3f}")
            
            self.last_training = time.time()
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        if len(self.prediction_history) < 10:
            return {"error": "Insufficient prediction history"}
        
        accuracies = {}
        
        for horizon in [5, 15, 30, 60]:
            horizon_predictions = [
                p for p in self.prediction_history 
                if p["horizon_minutes"] == horizon
            ]
            
            if len(horizon_predictions) >= 5:
                errors = []
                for pred in horizon_predictions:
                    if pred["actual_throughput"] > 0:
                        error = abs(pred["prediction"] - pred["actual_throughput"]) / pred["actual_throughput"]
                        errors.append(error)
                
                if errors:
                    accuracies[f"{horizon}min"] = {
                        "mae": np.mean(errors),
                        "rmse": np.sqrt(np.mean([e**2 for e in errors])),
                        "accuracy": 1.0 - np.mean(errors)
                    }
        
        return accuracies


class IntelligentScaler:
    """Next-generation intelligent auto-scaler with ML and cost optimization."""
    
    def __init__(self, 
                 min_capacity: int = 2,
                 max_capacity: int = 100,
                 target_cpu_utilization: float = 70.0,
                 target_response_time_ms: float = 500.0,
                 cost_per_unit_per_hour: float = 0.10,
                 strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE):
        
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.current_capacity = min_capacity
        self.target_cpu_utilization = target_cpu_utilization
        self.target_response_time_ms = target_response_time_ms
        self.cost_per_unit_per_hour = cost_per_unit_per_hour
        self.strategy = strategy
        
        # ML components
        self.demand_predictor = DemandPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.seasonality_detector = SeasonalityDetector()
        
        # State management
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.scaling_history = deque(maxlen=1000)
        self.last_scaling_time = 0.0
        self.scaling_cooldown = 60.0  # seconds
        
        # Performance tracking
        self.decision_accuracy = deque(maxlen=100)
        self.cost_history = deque(maxlen=1440)
        
        # Control
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        self._shutdown_event = threading.Event()
        
        logger.info(f"Initialized IntelligentScaler with capacity range {min_capacity}-{max_capacity}")
    
    def start_monitoring(self, interval_seconds: float = 60.0) -> None:
        """Start continuous monitoring and scaling."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
            name="IntelligentScaler"
        )
        self._monitor_thread.start()
        logger.info(f"Started intelligent scaling with {interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring and scaling."""
        self._monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        
        logger.info("Stopped intelligent scaling")
    
    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Main monitoring and scaling loop."""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_comprehensive_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Update ML models
                self.demand_predictor.update_model(metrics)
                self.anomaly_detector.update_training_data(metrics)
                self.seasonality_detector.update_patterns(metrics.timestamp, metrics.throughput_rps)
                
                # Make scaling decision
                if self._should_make_scaling_decision():
                    decision = self._make_intelligent_scaling_decision(metrics)
                    
                    if decision.direction != ScalingDirection.MAINTAIN:
                        success = self._execute_scaling_decision(decision)
                        
                        # Track decision accuracy
                        self.decision_accuracy.append({
                            "timestamp": time.time(),
                            "decision": decision,
                            "executed": success
                        })
                
                # Wait for next iteration
                self._shutdown_event.wait(timeout=interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self._shutdown_event.wait(timeout=interval_seconds)
    
    def _collect_comprehensive_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system and application metrics."""
        start_time = time.time()
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network and I/O
            net_io = psutil.net_io_counters()
            disk_io = psutil.disk_io_counters()
            
            # Load average
            load_avg = psutil.getloadavg()
            
            # Process info
            process_count = len(psutil.pids())
            
            # Connection counts
            tcp_connections = len([conn for conn in psutil.net_connections() if conn.type == 1])  # TCP
            udp_connections = len([conn for conn in psutil.net_connections() if conn.type == 2])  # UDP
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            # Fallback values
            cpu_percent = 50.0
            cpu_count = 4
            memory_percent = 60.0
            memory_total_gb = 8.0
            memory_available_gb = 3.2
            disk_usage_percent = 50.0
            load_avg = (1.0, 1.0, 1.0)
            process_count = 100
            tcp_connections = udp_connections = 50
            net_io = disk_io = None
        
        # Application metrics (these would be collected from your application)
        app_metrics = self._collect_application_metrics()
        
        collection_duration = (time.time() - start_time) * 1000
        
        return ResourceMetrics(
            timestamp=time.time(),
            collection_duration_ms=collection_duration,
            
            # System resources
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_percent=memory_percent,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            disk_io_read_mb_per_s=getattr(disk_io, 'read_bytes', 0) / (1024**2) if disk_io else 0,
            disk_io_write_mb_per_s=getattr(disk_io, 'write_bytes', 0) / (1024**2) if disk_io else 0,
            
            # Network
            network_in_mb_per_s=getattr(net_io, 'bytes_recv', 0) / (1024**2) if net_io else 0,
            network_out_mb_per_s=getattr(net_io, 'bytes_sent', 0) / (1024**2) if net_io else 0,
            tcp_connections=tcp_connections,
            udp_connections=udp_connections,
            
            # Load
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2],
            process_count=process_count,
            
            # Application metrics
            **app_metrics
        )
    
    def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        # This would integrate with your application's metrics
        # For now, return mock data
        return {
            "active_workers": self.current_capacity,
            "queue_size": max(0, int(np.random.normal(10, 5))),
            "pending_tasks": max(0, int(np.random.normal(5, 3))),
            "completed_tasks_per_minute": max(0, int(np.random.normal(50, 15))),
            "failed_tasks_per_minute": max(0, int(np.random.normal(2, 1))),
            
            "response_time_ms": max(10, np.random.normal(200, 50)),
            "response_time_p50_ms": max(10, np.random.normal(150, 30)),
            "response_time_p95_ms": max(50, np.random.normal(400, 100)),
            "response_time_p99_ms": max(100, np.random.normal(800, 200)),
            "throughput_rps": max(0, np.random.normal(20, 5)),
            "error_rate": max(0, min(1, np.random.normal(0.02, 0.01))),
            "success_rate": max(0, min(1, 1 - np.random.normal(0.02, 0.01))),
            
            "concurrent_users": max(0, int(np.random.normal(100, 30))),
            "requests_per_second": max(0, np.random.normal(25, 8)),
            "cache_hit_rate": max(0, min(1, np.random.normal(0.85, 0.1))),
            "database_connections": max(0, int(np.random.normal(20, 5))),
            "active_sessions": max(0, int(np.random.normal(80, 20))),
            
            "estimated_cost_per_hour": self.current_capacity * self.cost_per_unit_per_hour,
            "resource_efficiency": min(1.0, max(0.0, np.random.normal(0.75, 0.1)))
        }
    
    def _should_make_scaling_decision(self) -> bool:
        """Determine if a scaling decision should be made."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        # Need sufficient metrics history
        with self._lock:
            return len(self.metrics_history) >= 5
    
    def _make_intelligent_scaling_decision(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """Make intelligent scaling decision using ML and multi-factor analysis."""
        
        # Collect recent metrics for trend analysis
        with self._lock:
            recent_metrics = list(self.metrics_history)[-10:]
        
        if len(recent_metrics) < 3:
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                current_capacity=self.current_capacity,
                target_capacity=self.current_capacity,
                confidence=0.0,
                reasoning=["Insufficient metrics history"],
                risk_factors=[],
                predicted_metrics={}
            )
        
        # Anomaly detection
        anomaly_result = self.anomaly_detector.detect_anomaly(current_metrics)
        
        # Demand prediction
        demand_predictions = self.demand_predictor.predict_demand(current_metrics, horizon_minutes=15)
        predicted_demand = demand_predictions.get("15min", current_metrics.throughput_rps)
        
        # Seasonality adjustment
        seasonality_factor = self.seasonality_detector.get_seasonal_factor(current_metrics.timestamp)
        adjusted_demand = predicted_demand * seasonality_factor
        
        # Performance analysis
        performance_score = self._calculate_performance_score(current_metrics)
        resource_utilization = self._calculate_resource_utilization(current_metrics)
        
        # Cost analysis
        current_cost = self.current_capacity * self.cost_per_unit_per_hour
        
        # Decision logic
        scaling_signals = []
        risk_factors = []
        confidence_factors = []
        
        # CPU-based signals
        if current_metrics.cpu_percent > self.target_cpu_utilization + 20:
            scaling_signals.append(f"High CPU: {current_metrics.cpu_percent:.1f}%")
            confidence_factors.append(0.9)
        elif current_metrics.cpu_percent < self.target_cpu_utilization - 30:
            scaling_signals.append(f"Low CPU: {current_metrics.cpu_percent:.1f}%")
            confidence_factors.append(0.7)
        
        # Response time signals
        if current_metrics.response_time_ms > self.target_response_time_ms * 2:
            scaling_signals.append(f"High response time: {current_metrics.response_time_ms:.0f}ms")
            confidence_factors.append(0.95)
            risk_factors.append("SLA violation risk")
        
        # Queue-based signals
        optimal_queue_size = self.current_capacity * 2
        if current_metrics.queue_size > optimal_queue_size:
            scaling_signals.append(f"Large queue: {current_metrics.queue_size}")
            confidence_factors.append(0.8)
        
        # Predictive signals
        demand_change = (adjusted_demand - current_metrics.throughput_rps) / max(current_metrics.throughput_rps, 1)
        if demand_change > 0.5:  # 50% increase predicted
            scaling_signals.append(f"Demand spike predicted: +{demand_change:.1%}")
            confidence_factors.append(0.7)
        elif demand_change < -0.3:  # 30% decrease predicted
            scaling_signals.append(f"Demand drop predicted: {demand_change:.1%}")
            confidence_factors.append(0.6)
        
        # Anomaly-based signals
        if anomaly_result["is_anomaly"]:
            scaling_signals.append("Anomaly detected")
            confidence_factors.append(0.8)
            risk_factors.append("Unusual system behavior")
        
        # Make decision
        if not scaling_signals:
            direction = ScalingDirection.MAINTAIN
            target_capacity = self.current_capacity
            reasoning = ["System operating within normal parameters"]
        else:
            # Determine scaling direction and magnitude
            scale_up_signals = [s for s in scaling_signals if any(keyword in s.lower() 
                                                                for keyword in ["high", "large", "spike", "anomaly"])]
            scale_down_signals = [s for s in scaling_signals if any(keyword in s.lower() 
                                                                  for keyword in ["low", "drop"])]
            
            if scale_up_signals and len(scale_up_signals) >= len(scale_down_signals):
                direction = ScalingDirection.SCALE_UP
                target_capacity = self._calculate_target_capacity(current_metrics, adjusted_demand, "up")
                reasoning = scale_up_signals
            elif scale_down_signals:
                direction = ScalingDirection.SCALE_DOWN
                target_capacity = self._calculate_target_capacity(current_metrics, adjusted_demand, "down")
                reasoning = scale_down_signals
            else:
                direction = ScalingDirection.MAINTAIN
                target_capacity = self.current_capacity
                reasoning = ["Mixed signals - maintaining current capacity"]
        
        # Calculate confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Cost impact analysis
        capacity_change = target_capacity - self.current_capacity
        cost_impact = capacity_change * self.cost_per_unit_per_hour
        
        return ScalingDecision(
            direction=direction,
            current_capacity=self.current_capacity,
            target_capacity=target_capacity,
            confidence=confidence,
            reasoning=reasoning,
            risk_factors=risk_factors,
            predicted_metrics={
                "predicted_demand": adjusted_demand,
                "current_demand": current_metrics.throughput_rps,
                "seasonality_factor": seasonality_factor,
                "performance_score": performance_score,
                "resource_utilization": resource_utilization
            },
            demand_forecast=adjusted_demand,
            anomaly_detected=anomaly_result["is_anomaly"],
            seasonality_factor=seasonality_factor,
            cost_impact_per_hour=cost_impact,
            strategy_used=self.strategy
        )
    
    def _calculate_performance_score(self, metrics: ResourceMetrics) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        scores = []
        
        # Response time score
        if metrics.response_time_ms <= self.target_response_time_ms:
            rt_score = 1.0
        else:
            rt_score = max(0.0, 1.0 - (metrics.response_time_ms - self.target_response_time_ms) / self.target_response_time_ms)
        scores.append(rt_score)
        
        # Error rate score
        error_score = max(0.0, 1.0 - metrics.error_rate * 10)  # 10% error = 0 score
        scores.append(error_score)
        
        # Resource utilization score (prefer 70-80% CPU)
        if 70 <= metrics.cpu_percent <= 80:
            cpu_score = 1.0
        elif metrics.cpu_percent < 70:
            cpu_score = metrics.cpu_percent / 70
        else:
            cpu_score = max(0.0, 1.0 - (metrics.cpu_percent - 80) / 20)
        scores.append(cpu_score)
        
        return np.mean(scores)
    
    def _calculate_resource_utilization(self, metrics: ResourceMetrics) -> float:
        """Calculate overall resource utilization (0-1)."""
        utilizations = [
            metrics.cpu_percent / 100,
            metrics.memory_percent / 100,
            min(1.0, metrics.load_avg_1m / metrics.cpu_count) if metrics.cpu_count > 0 else 0
        ]
        return np.mean(utilizations)
    
    def _calculate_target_capacity(self, metrics: ResourceMetrics, 
                                 predicted_demand: float, direction: str) -> int:
        """Calculate optimal target capacity."""
        if direction == "up":
            # Calculate capacity needed for predicted demand
            current_capacity_utilization = metrics.cpu_percent / 100
            if current_capacity_utilization > 0:
                capacity_per_unit = metrics.throughput_rps / (self.current_capacity * current_capacity_utilization)
            else:
                capacity_per_unit = 1.0  # fallback
            
            needed_capacity = predicted_demand / capacity_per_unit
            target_capacity = int(np.ceil(needed_capacity * 1.2))  # 20% buffer
            
            # Conservative scaling
            max_increase = max(1, int(self.current_capacity * 0.5))
            target_capacity = min(target_capacity, self.current_capacity + max_increase)
            
        else:  # scale down
            # More conservative scale down
            target_capacity = max(self.min_capacity, int(self.current_capacity * 0.8))
        
        return max(self.min_capacity, min(self.max_capacity, target_capacity))
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        logger.info(
            f"Executing scaling decision: {decision.direction.value} "
            f"from {decision.current_capacity} to {decision.target_capacity} "
            f"(confidence: {decision.confidence:.2f})"
        )
        logger.info(f"Reasoning: {'; '.join(decision.reasoning)}")
        
        try:
            # This is where you would integrate with your infrastructure
            # For now, just update the internal state
            old_capacity = self.current_capacity
            self.current_capacity = decision.target_capacity
            self.last_scaling_time = time.time()
            
            # Record scaling event
            with self._lock:
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "decision": decision,
                    "success": True,
                    "old_capacity": old_capacity,
                    "new_capacity": self.current_capacity
                })
            
            # Record metrics
            record_metric("scaling_action", 1, "counter", {
                "direction": decision.direction.value,
                "old_capacity": old_capacity,
                "new_capacity": self.current_capacity
            })
            
            logger.info(f"Successfully scaled to {self.current_capacity} units")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-60:]  # Last hour
            recent_scaling = list(self.scaling_history)[-10:]   # Last 10 decisions
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_response_time = np.mean([m.response_time_ms for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_rps for m in recent_metrics])
        
        # Scaling statistics
        scale_up_count = sum(1 for s in recent_scaling if s["decision"].direction == ScalingDirection.SCALE_UP)
        scale_down_count = sum(1 for s in recent_scaling if s["decision"].direction == ScalingDirection.SCALE_DOWN)
        
        # Cost analysis
        current_hourly_cost = self.current_capacity * self.cost_per_unit_per_hour
        
        return {
            "current_capacity": self.current_capacity,
            "capacity_limits": {
                "min": self.min_capacity,
                "max": self.max_capacity
            },
            "strategy": self.strategy.value,
            
            "performance": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "avg_response_time_ms": avg_response_time,
                "avg_throughput_rps": avg_throughput
            },
            
            "scaling_activity": {
                "scale_up_count": scale_up_count,
                "scale_down_count": scale_down_count,
                "last_scaling_time": self.last_scaling_time,
                "cooldown_remaining": max(0, self.scaling_cooldown - (time.time() - self.last_scaling_time))
            },
            
            "ml_models": {
                "demand_prediction_accuracy": self.demand_predictor.get_prediction_accuracy(),
                "anomaly_detection_trained": self.anomaly_detector.trained,
                "seasonality_patterns": {
                    "hourly": len(self.seasonality_detector.hourly_patterns),
                    "daily": len(self.seasonality_detector.daily_patterns),
                    "weekly": len(self.seasonality_detector.weekly_patterns)
                }
            },
            
            "cost": {
                "current_hourly_cost": current_hourly_cost,
                "cost_per_unit": self.cost_per_unit_per_hour,
                "estimated_monthly_cost": current_hourly_cost * 24 * 30
            }
        }
    
    def get_capacity_forecast(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get capacity forecast for the next N hours."""
        if not self.metrics_history:
            return {"error": "No metrics history available"}
        
        current_metrics = self.metrics_history[-1]
        forecasts = []
        
        for hour in range(1, hours_ahead + 1):
            future_timestamp = current_metrics.timestamp + (hour * 3600)
            
            # Predict demand
            future_metrics = ResourceMetrics(
                timestamp=future_timestamp,
                cpu_percent=current_metrics.cpu_percent,
                memory_percent=current_metrics.memory_percent,
                throughput_rps=current_metrics.throughput_rps,
                response_time_ms=current_metrics.response_time_ms,
                error_rate=current_metrics.error_rate,
                active_workers=self.current_capacity
            )
            
            demand_prediction = self.demand_predictor.predict_demand(future_metrics, horizon_minutes=60)
            predicted_demand = demand_prediction.get("60min", current_metrics.throughput_rps)
            
            # Apply seasonality
            seasonality_factor = self.seasonality_detector.get_seasonal_factor(future_timestamp)
            adjusted_demand = predicted_demand * seasonality_factor
            
            # Estimate required capacity
            required_capacity = self._calculate_target_capacity(future_metrics, adjusted_demand, "up")
            
            forecasts.append({
                "hour": hour,
                "timestamp": future_timestamp,
                "predicted_demand": adjusted_demand,
                "required_capacity": required_capacity,
                "seasonality_factor": seasonality_factor,
                "estimated_cost": required_capacity * self.cost_per_unit_per_hour
            })
        
        total_cost = sum(f["estimated_cost"] for f in forecasts)
        peak_capacity = max(f["required_capacity"] for f in forecasts)
        
        return {
            "forecast_horizon_hours": hours_ahead,
            "forecasts": forecasts,
            "summary": {
                "peak_capacity": peak_capacity,
                "total_estimated_cost": total_cost,
                "avg_capacity": np.mean([f["required_capacity"] for f in forecasts])
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown the intelligent scaler."""
        logger.info("Shutting down IntelligentScaler")
        self.stop_monitoring()
        logger.info("IntelligentScaler shutdown complete")


# Global scaler instance
_global_intelligent_scaler: Optional[IntelligentScaler] = None


def get_intelligent_scaler(**kwargs) -> IntelligentScaler:
    """Get or create global intelligent scaler."""
    global _global_intelligent_scaler
    
    if _global_intelligent_scaler is None:
        _global_intelligent_scaler = IntelligentScaler(**kwargs)
    
    return _global_intelligent_scaler


def shutdown_intelligent_scaler() -> None:
    """Shutdown global intelligent scaler."""
    global _global_intelligent_scaler
    
    if _global_intelligent_scaler:
        _global_intelligent_scaler.shutdown()
        _global_intelligent_scaler = None


def create_cost_optimized_scaler(cost_per_unit: float = 0.10, **kwargs) -> IntelligentScaler:
    """Create cost-optimized intelligent scaler."""
    return IntelligentScaler(
        cost_per_unit_per_hour=cost_per_unit,
        strategy=ScalingStrategy.COST_OPTIMIZED,
        **kwargs
    )


def create_performance_first_scaler(**kwargs) -> IntelligentScaler:
    """Create performance-first intelligent scaler."""
    return IntelligentScaler(
        strategy=ScalingStrategy.PERFORMANCE_FIRST,
        target_cpu_utilization=60.0,  # Lower for better performance
        target_response_time_ms=200.0,  # Stricter SLA
        **kwargs
    )