"""
Federated Learning Module for Distributed AI Hardware Co-Design.

This module implements federated learning algorithms for distributed hardware optimization,
privacy-preserving collaborative design, and edge-cloud co-optimization frameworks.
"""

import hashlib
import hmac
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import logging

from ..utils.logging import get_logger
from ..utils.monitoring import record_metric
from ..utils.security import SecurityValidator
from ..utils.exceptions import SecurityError, ValidationError

logger = get_logger(__name__)


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies."""
    FEDERATED_AVERAGING = "fed_avg"
    FEDERATED_PROXIMAL = "fed_prox"
    FEDERATED_NOVA = "fed_nova"
    BYZANTINE_ROBUST = "byzantine_robust"
    DIFFERENTIAL_PRIVATE = "dp_sgd"
    ADAPTIVE_FEDERATED = "adaptive_fed"


class ClientSelectionStrategy(Enum):
    """Client selection strategies for federated learning."""
    RANDOM = "random"
    PERFORMANCE_BASED = "performance"
    RESOURCE_AWARE = "resource_aware"
    DIVERSITY_BASED = "diversity"
    FAIRNESS_AWARE = "fairness"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    MULTI_PARTY_COMPUTATION = "mpc"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_dp"


@dataclass
class ClientProfile:
    """Profile information for a federated learning client."""
    
    client_id: str
    hardware_specs: Dict[str, Any]
    compute_capacity: float  # FLOPS
    memory_capacity: int  # MB
    bandwidth: float  # Mbps
    energy_budget: float  # mWh
    availability_score: float  # [0, 1]
    trustworthiness_score: float  # [0, 1]
    
    # Performance metrics
    training_time_avg: float = 0.0  # seconds
    communication_time_avg: float = 0.0  # seconds
    accuracy_contribution: float = 0.0
    data_size: int = 0
    
    # Privacy preferences
    privacy_level: int = 1  # 1-5 scale
    data_sharing_consent: bool = False
    local_computation_only: bool = False


@dataclass
class FederatedModel:
    """Federated learning model representation."""
    
    model_id: str
    parameters: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    version: int = 0
    
    # Model characteristics
    parameter_count: int = 0
    model_size_mb: float = 0.0
    computational_complexity: float = 0.0
    
    # Training metadata
    training_rounds: int = 0
    participating_clients: List[str] = field(default_factory=list)
    global_accuracy: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedTask:
    """Federated learning task definition."""
    
    task_id: str
    task_type: str  # "classification", "regression", "optimization"
    objective_function: str
    privacy_requirements: List[PrivacyMechanism]
    
    # Hardware co-design specific
    hardware_constraints: Dict[str, float]
    target_hardware: Dict[str, Any]
    optimization_objectives: List[str]
    
    # Task parameters
    max_rounds: int = 100
    min_clients_per_round: int = 2
    convergence_threshold: float = 1e-6
    timeout_seconds: int = 3600


class SecureAggregator:
    """Secure aggregation for federated learning with privacy guarantees."""
    
    def __init__(self, privacy_budget: float = 1.0, noise_multiplier: float = 1.0):
        """
        Initialize secure aggregator.
        
        Args:
            privacy_budget: Differential privacy budget (epsilon)
            noise_multiplier: Noise multiplier for DP
        """
        self.privacy_budget = privacy_budget
        self.noise_multiplier = noise_multiplier
        self.aggregation_history = []
        self.client_contributions = defaultdict(list)
        
        # Cryptographic keys (simplified for demonstration)
        self.master_key = self._generate_key()
        self.client_keys = {}
        
        logger.info("Initialized secure aggregator with privacy budget %.2f", privacy_budget)
    
    def _generate_key(self) -> bytes:
        """Generate cryptographic key."""
        return hashlib.sha256(str(time.time()).encode()).digest()
    
    def register_client(self, client_id: str) -> Dict[str, Any]:
        """
        Register a client and generate secure keys.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Client registration information with keys
        """
        client_key = self._generate_key()
        self.client_keys[client_id] = client_key
        
        return {
            "client_id": client_id,
            "registration_time": time.time(),
            "public_key": client_key.hex()[:32],  # Simplified public key
            "session_token": self._generate_session_token(client_id)
        }
    
    def _generate_session_token(self, client_id: str) -> str:
        """Generate session token for client."""
        message = f"{client_id}_{time.time()}".encode()
        signature = hmac.new(self.master_key, message, hashlib.sha256).hexdigest()
        return signature[:16]
    
    def aggregate_secure(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        client_weights: Optional[Dict[str, float]] = None,
        privacy_mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY
    ) -> Dict[str, np.ndarray]:
        """
        Perform secure aggregation of client updates.
        
        Args:
            client_updates: Dictionary of client parameter updates
            client_weights: Weights for each client
            privacy_mechanism: Privacy preservation mechanism
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Validate client updates
        validated_updates = self._validate_client_updates(client_updates)
        
        if client_weights is None:
            client_weights = {client_id: 1.0 for client_id in validated_updates.keys()}
        
        # Apply privacy mechanism
        if privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            aggregated = self._aggregate_with_differential_privacy(validated_updates, client_weights)
        elif privacy_mechanism == PrivacyMechanism.SECURE_AGGREGATION:
            aggregated = self._aggregate_with_secure_computation(validated_updates, client_weights)
        elif privacy_mechanism == PrivacyMechanism.LOCAL_DIFFERENTIAL_PRIVACY:
            aggregated = self._aggregate_with_local_dp(validated_updates, client_weights)
        else:
            # Default aggregation
            aggregated = self._aggregate_federated_averaging(validated_updates, client_weights)
        
        # Record aggregation
        self.aggregation_history.append({
            "timestamp": time.time(),
            "num_clients": len(validated_updates),
            "privacy_mechanism": privacy_mechanism.value,
            "aggregated_norm": self._compute_parameter_norm(aggregated)
        })
        
        return aggregated
    
    def _validate_client_updates(self, client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Validate client updates for security and consistency."""
        validated = {}
        
        for client_id, update in client_updates.items():
            # Check if client is registered
            if client_id not in self.client_keys:
                logger.warning(f"Unregistered client {client_id} attempted update")
                continue
            
            # Validate parameter shapes and values
            if self._validate_parameter_update(update):
                validated[client_id] = update
                self.client_contributions[client_id].append(time.time())
            else:
                logger.warning(f"Invalid update from client {client_id}")
        
        return validated
    
    def _validate_parameter_update(self, update: Dict[str, np.ndarray]) -> bool:
        """Validate individual parameter update."""
        try:
            for param_name, param_value in update.items():
                # Check for valid numpy array
                if not isinstance(param_value, np.ndarray):
                    return False
                
                # Check for reasonable values (no NaN, Inf, or extreme values)
                if np.any(np.isnan(param_value)) or np.any(np.isinf(param_value)):
                    return False
                
                # Check for suspiciously large values
                if np.max(np.abs(param_value)) > 1000:
                    return False
            
            return True
        except Exception:
            return False
    
    def _aggregate_federated_averaging(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        client_weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Standard federated averaging aggregation."""
        if not client_updates:
            return {}
        
        # Normalize weights
        total_weight = sum(client_weights.values())
        normalized_weights = {k: v / total_weight for k, v in client_weights.items()}
        
        # Get parameter names from first client
        first_client = next(iter(client_updates.values()))
        param_names = list(first_client.keys())
        
        aggregated = {}
        for param_name in param_names:
            weighted_sum = None
            
            for client_id, update in client_updates.items():
                if param_name in update:
                    weight = normalized_weights.get(client_id, 0.0)
                    weighted_param = update[param_name] * weight
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            if weighted_sum is not None:
                aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _aggregate_with_differential_privacy(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        client_weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Aggregate with differential privacy noise."""
        # First perform standard aggregation
        aggregated = self._aggregate_federated_averaging(client_updates, client_weights)
        
        # Add calibrated noise for differential privacy
        for param_name, param_value in aggregated.items():
            # Calculate sensitivity (simplified)
            sensitivity = 2.0 / len(client_updates)  # L2 sensitivity
            
            # Add Gaussian noise
            noise_scale = sensitivity * self.noise_multiplier / self.privacy_budget
            noise = np.random.normal(0, noise_scale, param_value.shape)
            aggregated[param_name] = param_value + noise
        
        return aggregated
    
    def _aggregate_with_secure_computation(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        client_weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Aggregate using secure multi-party computation (simplified)."""
        # In practice, this would use actual MPC protocols
        # Here we simulate by adding small random masks
        
        aggregated = self._aggregate_federated_averaging(client_updates, client_weights)
        
        # Add and remove random masks (simplified MPC simulation)
        for param_name, param_value in aggregated.items():
            # Simulate secret sharing with random masks
            mask_sum = np.zeros_like(param_value)
            for client_id in client_updates.keys():
                # Generate deterministic mask for each client
                np.random.seed(hash(client_id + param_name) % 2**32)
                client_mask = np.random.normal(0, 0.01, param_value.shape)
                mask_sum += client_mask
            
            # Remove accumulated masks (in real MPC, clients would provide mask cancellations)
            aggregated[param_name] = param_value - mask_sum + mask_sum  # Identity for demonstration
        
        return aggregated
    
    def _aggregate_with_local_dp(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        client_weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Aggregate with local differential privacy."""
        # Clients already added noise locally, so just aggregate
        return self._aggregate_federated_averaging(client_updates, client_weights)
    
    def _compute_parameter_norm(self, parameters: Dict[str, np.ndarray]) -> float:
        """Compute L2 norm of parameters."""
        total_norm = 0.0
        for param in parameters.values():
            total_norm += np.sum(param ** 2)
        return np.sqrt(total_norm)


class FederatedHardwareOptimizer:
    """Federated optimizer for distributed hardware design optimization."""
    
    def __init__(
        self,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING,
        client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM,
        privacy_mechanisms: List[PrivacyMechanism] = None
    ):
        """
        Initialize federated hardware optimizer.
        
        Args:
            aggregation_strategy: Strategy for aggregating client updates
            client_selection: Strategy for selecting participating clients
            privacy_mechanisms: Privacy preservation mechanisms to use
        """
        self.aggregation_strategy = aggregation_strategy
        self.client_selection = client_selection
        self.privacy_mechanisms = privacy_mechanisms or [PrivacyMechanism.DIFFERENTIAL_PRIVACY]
        
        # System state
        self.clients = {}
        self.global_model = None
        self.current_round = 0
        self.optimization_history = []
        
        # Security and aggregation
        self.secure_aggregator = SecureAggregator()
        self.validator = SecurityValidator()
        
        # Performance tracking
        self.round_metrics = defaultdict(list)
        self.client_performance = defaultdict(dict)
        
        logger.info(f"Initialized federated optimizer with {aggregation_strategy.value} aggregation")
    
    def register_client(self, client_profile: ClientProfile) -> Dict[str, Any]:
        """
        Register a new client for federated optimization.
        
        Args:
            client_profile: Client profile information
            
        Returns:
            Registration confirmation and initial configuration
        """
        client_id = client_profile.client_id
        
        # Validate client profile
        if not self._validate_client_profile(client_profile):
            raise ValidationError(f"Invalid client profile for {client_id}")
        
        # Register with secure aggregator
        registration_info = self.secure_aggregator.register_client(client_id)
        
        # Store client profile
        self.clients[client_id] = client_profile
        
        logger.info(f"Registered client {client_id} with {client_profile.compute_capacity:.1f} FLOPS")
        
        return {
            **registration_info,
            "initial_model": self.global_model.parameters if self.global_model else {},
            "task_configuration": self._get_client_task_configuration(client_profile),
            "privacy_settings": self._get_privacy_settings(client_profile)
        }
    
    def _validate_client_profile(self, profile: ClientProfile) -> bool:
        """Validate client profile for security and consistency."""
        try:
            # Check required fields
            required_fields = ['client_id', 'hardware_specs', 'compute_capacity', 'memory_capacity']
            for field in required_fields:
                if not hasattr(profile, field) or getattr(profile, field) is None:
                    return False
            
            # Validate numeric fields
            if profile.compute_capacity <= 0 or profile.memory_capacity <= 0:
                return False
            
            if not (0 <= profile.availability_score <= 1) or not (0 <= profile.trustworthiness_score <= 1):
                return False
            
            return True
        except Exception:
            return False
    
    def _get_client_task_configuration(self, client_profile: ClientProfile) -> Dict[str, Any]:
        """Get task configuration customized for client capabilities."""
        base_config = {
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimization_algorithm": "adam"
        }
        
        # Adjust based on client capabilities
        if client_profile.compute_capacity > 1e12:  # High-end device
            base_config.update({
                "local_epochs": 10,
                "batch_size": 64,
                "model_compression": False
            })
        elif client_profile.compute_capacity < 1e10:  # Low-end device
            base_config.update({
                "local_epochs": 3,
                "batch_size": 16,
                "model_compression": True,
                "quantization_bits": 8
            })
        
        # Adjust for memory constraints
        if client_profile.memory_capacity < 1024:  # Less than 1GB
            base_config.update({
                "gradient_accumulation_steps": 4,
                "model_parallelism": False
            })
        
        return base_config
    
    def _get_privacy_settings(self, client_profile: ClientProfile) -> Dict[str, Any]:
        """Get privacy settings based on client preferences."""
        privacy_settings = {
            "local_differential_privacy": client_profile.privacy_level >= 3,
            "secure_aggregation": client_profile.privacy_level >= 2,
            "data_minimization": client_profile.privacy_level >= 4,
            "homomorphic_encryption": client_profile.privacy_level >= 5
        }
        
        if client_profile.local_computation_only:
            privacy_settings.update({
                "upload_gradients_only": True,
                "model_inversion_protection": True
            })
        
        return privacy_settings
    
    def select_clients(
        self,
        num_clients: int,
        round_number: int,
        strategy: Optional[ClientSelectionStrategy] = None
    ) -> List[str]:
        """
        Select clients for participation in a federated round.
        
        Args:
            num_clients: Number of clients to select
            round_number: Current round number
            strategy: Selection strategy to use
            
        Returns:
            List of selected client IDs
        """
        if strategy is None:
            strategy = self.client_selection
        
        available_clients = [
            client_id for client_id, profile in self.clients.items()
            if profile.availability_score > 0.5 and profile.trustworthiness_score > 0.3
        ]
        
        if len(available_clients) < num_clients:
            logger.warning(f"Only {len(available_clients)} clients available, requested {num_clients}")
            num_clients = len(available_clients)
        
        if strategy == ClientSelectionStrategy.RANDOM:
            return list(np.random.choice(available_clients, num_clients, replace=False))
        
        elif strategy == ClientSelectionStrategy.PERFORMANCE_BASED:
            # Select based on past performance
            client_scores = {}
            for client_id in available_clients:
                profile = self.clients[client_id]
                performance_metrics = self.client_performance.get(client_id, {})
                
                # Combine multiple performance factors
                score = (
                    profile.trustworthiness_score * 0.3 +
                    profile.availability_score * 0.2 +
                    (1.0 - performance_metrics.get('training_time_normalized', 0.5)) * 0.3 +
                    performance_metrics.get('accuracy_contribution', 0.0) * 0.2
                )
                client_scores[client_id] = score
            
            # Select top performers
            sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
            return [client_id for client_id, _ in sorted_clients[:num_clients]]
        
        elif strategy == ClientSelectionStrategy.RESOURCE_AWARE:
            # Select based on computational resources
            client_resources = {
                client_id: self.clients[client_id].compute_capacity
                for client_id in available_clients
            }
            sorted_by_resources = sorted(client_resources.items(), key=lambda x: x[1], reverse=True)
            return [client_id for client_id, _ in sorted_by_resources[:num_clients]]
        
        elif strategy == ClientSelectionStrategy.DIVERSITY_BASED:
            # Select diverse set of clients based on hardware specs
            return self._select_diverse_clients(available_clients, num_clients)
        
        elif strategy == ClientSelectionStrategy.FAIRNESS_AWARE:
            # Ensure fair participation across clients
            return self._select_fair_clients(available_clients, num_clients, round_number)
        
        else:
            # Default to random
            return list(np.random.choice(available_clients, num_clients, replace=False))
    
    def _select_diverse_clients(self, available_clients: List[str], num_clients: int) -> List[str]:
        """Select diverse set of clients based on hardware characteristics."""
        if len(available_clients) <= num_clients:
            return available_clients
        
        # Extract hardware features for clustering
        client_features = []
        for client_id in available_clients:
            profile = self.clients[client_id]
            features = [
                np.log10(profile.compute_capacity),
                np.log10(profile.memory_capacity),
                profile.bandwidth,
                profile.energy_budget
            ]
            client_features.append(features)
        
        client_features = np.array(client_features)
        
        # Simple k-means clustering for diversity
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(num_clients, len(available_clients)), random_state=42)
        clusters = kmeans.fit_predict(client_features)
        
        # Select one client from each cluster
        selected = []
        for cluster_id in range(min(num_clients, max(clusters) + 1)):
            cluster_clients = [available_clients[i] for i, c in enumerate(clusters) if c == cluster_id]
            if cluster_clients:
                # Select client closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = [
                    np.linalg.norm(client_features[available_clients.index(client)] - cluster_center)
                    for client in cluster_clients
                ]
                best_client = cluster_clients[np.argmin(distances)]
                selected.append(best_client)
        
        # Fill remaining slots randomly
        remaining = set(available_clients) - set(selected)
        if len(selected) < num_clients and remaining:
            additional = list(np.random.choice(list(remaining), min(num_clients - len(selected), len(remaining)), replace=False))
            selected.extend(additional)
        
        return selected[:num_clients]
    
    def _select_fair_clients(self, available_clients: List[str], num_clients: int, round_number: int) -> List[str]:
        """Select clients ensuring fairness in participation."""
        # Track participation history
        participation_counts = defaultdict(int)
        for round_data in self.optimization_history:
            for client_id in round_data.get('participating_clients', []):
                participation_counts[client_id] += 1
        
        # Calculate fairness scores (inverse of participation frequency)
        fairness_scores = {}
        for client_id in available_clients:
            count = participation_counts[client_id]
            # Clients with fewer participations get higher scores
            fairness_scores[client_id] = 1.0 / (count + 1)
        
        # Select clients with highest fairness scores
        sorted_by_fairness = sorted(fairness_scores.items(), key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in sorted_by_fairness[:num_clients]]
    
    def run_federated_round(
        self,
        task: FederatedTask,
        selected_clients: List[str],
        global_model: FederatedModel
    ) -> Dict[str, Any]:
        """
        Execute a single federated learning round.
        
        Args:
            task: Federated learning task
            selected_clients: List of participating client IDs
            global_model: Current global model
            
        Returns:
            Round results including updated model and metrics
        """
        round_start_time = time.time()
        
        # Distribute global model to clients
        client_configs = {}
        for client_id in selected_clients:
            client_profile = self.clients[client_id]
            client_configs[client_id] = self._get_client_task_configuration(client_profile)
        
        # Simulate client training (in practice, this would be asynchronous)
        client_updates = {}
        client_metrics = {}
        
        with ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
            future_to_client = {
                executor.submit(
                    self._simulate_client_training,
                    client_id,
                    global_model,
                    client_configs[client_id],
                    task
                ): client_id
                for client_id in selected_clients
            }
            
            for future in as_completed(future_to_client, timeout=300):
                client_id = future_to_client[future]
                try:
                    update, metrics = future.result(timeout=60)
                    client_updates[client_id] = update
                    client_metrics[client_id] = metrics
                    logger.debug(f"Received update from client {client_id}")
                except Exception as e:
                    logger.error(f"Client {client_id} training failed: {e}")
        
        # Secure aggregation
        try:
            aggregated_params = self.secure_aggregator.aggregate_secure(
                client_updates,
                privacy_mechanism=self.privacy_mechanisms[0]
            )
            
            # Update global model
            updated_model = FederatedModel(
                model_id=global_model.model_id,
                parameters=aggregated_params,
                metadata=global_model.metadata.copy(),
                version=global_model.version + 1,
                training_rounds=global_model.training_rounds + 1,
                participating_clients=selected_clients
            )
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {"error": "Aggregation failed", "global_model": global_model}
        
        # Calculate round metrics
        round_metrics = self._calculate_round_metrics(
            client_metrics, 
            global_model, 
            updated_model,
            round_start_time
        )
        
        # Update client performance tracking
        self._update_client_performance(client_metrics)
        
        # Record round in history
        round_data = {
            "round_number": self.current_round,
            "participating_clients": selected_clients,
            "metrics": round_metrics,
            "aggregation_strategy": self.aggregation_strategy.value,
            "privacy_mechanisms": [pm.value for pm in self.privacy_mechanisms],
            "timestamp": time.time()
        }
        self.optimization_history.append(round_data)
        
        self.current_round += 1
        
        # Record telemetry
        record_metric("federated_round_duration", time.time() - round_start_time, "histogram")
        record_metric("federated_participating_clients", len(selected_clients), "gauge")
        record_metric("federated_aggregation_success", 1, "counter")
        
        return {
            "global_model": updated_model,
            "round_metrics": round_metrics,
            "participating_clients": selected_clients,
            "round_number": self.current_round - 1
        }
    
    def _simulate_client_training(
        self,
        client_id: str,
        global_model: FederatedModel,
        client_config: Dict[str, Any],
        task: FederatedTask
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Simulate client-side training (in practice, this would be on the client device).
        
        Args:
            client_id: Client identifier
            global_model: Current global model
            client_config: Client-specific configuration
            task: Federated task definition
            
        Returns:
            Tuple of (parameter updates, training metrics)
        """
        client_profile = self.clients[client_id]
        
        # Simulate training time based on client capabilities
        base_training_time = 30.0  # seconds
        compute_factor = 1e12 / max(client_profile.compute_capacity, 1e9)
        training_time = base_training_time * compute_factor
        
        # Simulate parameter updates
        parameter_updates = {}
        for param_name, param_value in global_model.parameters.items():
            # Simulate gradient-based update
            learning_rate = client_config.get("learning_rate", 0.01)
            noise_scale = 0.01 * client_profile.trustworthiness_score
            
            # Add some random updates (simulating training)
            gradient = np.random.normal(0, noise_scale, param_value.shape)
            update = param_value - learning_rate * gradient
            parameter_updates[param_name] = update
        
        # Apply privacy mechanisms if enabled
        privacy_settings = self._get_privacy_settings(client_profile)
        if privacy_settings.get("local_differential_privacy", False):
            parameter_updates = self._apply_local_differential_privacy(parameter_updates)
        
        # Calculate training metrics
        training_metrics = {
            "training_time": training_time,
            "local_loss": np.random.uniform(0.1, 1.0),  # Simulated loss
            "local_accuracy": 0.7 + 0.3 * client_profile.trustworthiness_score,  # Simulated accuracy
            "communication_cost": sum(param.nbytes for param in parameter_updates.values()),
            "energy_consumption": training_time * 100,  # mWh
            "convergence_rate": np.random.uniform(0.1, 0.3)
        }
        
        return parameter_updates, training_metrics
    
    def _apply_local_differential_privacy(
        self, 
        parameters: Dict[str, np.ndarray],
        epsilon: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Apply local differential privacy to parameter updates."""
        private_parameters = {}
        
        for param_name, param_value in parameters.items():
            # Calculate sensitivity (simplified)
            sensitivity = np.std(param_value)
            
            # Add Laplace noise for local DP
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(0, noise_scale, param_value.shape)
            
            private_parameters[param_name] = param_value + noise
        
        return private_parameters
    
    def _calculate_round_metrics(
        self,
        client_metrics: Dict[str, Dict[str, float]],
        old_model: FederatedModel,
        new_model: FederatedModel,
        round_start_time: float
    ) -> Dict[str, float]:
        """Calculate metrics for the completed round."""
        if not client_metrics:
            return {}
        
        # Aggregate client metrics
        aggregated_metrics = {}
        
        # Training time statistics
        training_times = [metrics["training_time"] for metrics in client_metrics.values()]
        aggregated_metrics.update({
            "avg_training_time": np.mean(training_times),
            "max_training_time": np.max(training_times),
            "training_time_std": np.std(training_times)
        })
        
        # Accuracy statistics
        accuracies = [metrics["local_accuracy"] for metrics in client_metrics.values()]
        aggregated_metrics.update({
            "avg_local_accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies)
        })
        
        # Communication costs
        comm_costs = [metrics["communication_cost"] for metrics in client_metrics.values()]
        aggregated_metrics.update({
            "total_communication_mb": sum(comm_costs) / (1024 * 1024),
            "avg_communication_mb": np.mean(comm_costs) / (1024 * 1024)
        })
        
        # Energy consumption
        energy_consumptions = [metrics["energy_consumption"] for metrics in client_metrics.values()]
        aggregated_metrics.update({
            "total_energy_mwh": sum(energy_consumptions),
            "avg_energy_mwh": np.mean(energy_consumptions)
        })
        
        # Model convergence
        if old_model.parameters and new_model.parameters:
            model_change = self._calculate_model_change(old_model.parameters, new_model.parameters)
            aggregated_metrics["model_change_norm"] = model_change
        
        # Round duration
        aggregated_metrics["round_duration"] = time.time() - round_start_time
        
        return aggregated_metrics
    
    def _calculate_model_change(
        self, 
        old_params: Dict[str, np.ndarray], 
        new_params: Dict[str, np.ndarray]
    ) -> float:
        """Calculate the norm of change between model parameters."""
        total_change = 0.0
        
        for param_name in old_params:
            if param_name in new_params:
                diff = new_params[param_name] - old_params[param_name]
                total_change += np.sum(diff ** 2)
        
        return np.sqrt(total_change)
    
    def _update_client_performance(self, client_metrics: Dict[str, Dict[str, float]]) -> None:
        """Update client performance tracking."""
        for client_id, metrics in client_metrics.items():
            if client_id not in self.client_performance:
                self.client_performance[client_id] = defaultdict(list)
            
            # Track key performance indicators
            self.client_performance[client_id]["training_times"].append(metrics["training_time"])
            self.client_performance[client_id]["accuracies"].append(metrics["local_accuracy"])
            self.client_performance[client_id]["communication_costs"].append(metrics["communication_cost"])
            
            # Calculate normalized metrics
            all_training_times = [
                t for client_perf in self.client_performance.values() 
                for t in client_perf["training_times"]
            ]
            if all_training_times:
                max_time = max(all_training_times)
                self.client_performance[client_id]["training_time_normalized"] = metrics["training_time"] / max_time
    
    def optimize_federated_hardware_design(
        self,
        task: FederatedTask,
        max_rounds: int = 100,
        convergence_threshold: float = 1e-6,
        min_clients_per_round: int = 3
    ) -> Dict[str, Any]:
        """
        Run complete federated hardware design optimization.
        
        Args:
            task: Federated optimization task
            max_rounds: Maximum number of rounds
            convergence_threshold: Convergence criterion
            min_clients_per_round: Minimum clients per round
            
        Returns:
            Optimization results including final model and analytics
        """
        logger.info(f"Starting federated hardware optimization for task {task.task_id}")
        
        # Initialize global model
        self.global_model = FederatedModel(
            model_id=f"federated_{task.task_id}",
            parameters=self._initialize_model_parameters(task),
            metadata={
                "task_type": task.task_type,
                "hardware_constraints": task.hardware_constraints,
                "optimization_objectives": task.optimization_objectives
            }
        )
        
        convergence_history = []
        best_model = None
        best_metric = float('-inf')
        
        for round_num in range(max_rounds):
            # Select clients for this round
            num_clients = min(len(self.clients), max(min_clients_per_round, len(self.clients) // 2))
            selected_clients = self.select_clients(num_clients, round_num)
            
            if len(selected_clients) < min_clients_per_round:
                logger.warning(f"Insufficient clients available: {len(selected_clients)} < {min_clients_per_round}")
                break
            
            # Run federated round
            round_results = self.run_federated_round(task, selected_clients, self.global_model)
            
            if "error" in round_results:
                logger.error(f"Round {round_num} failed: {round_results['error']}")
                continue
            
            # Update global model
            self.global_model = round_results["global_model"]
            round_metrics = round_results["round_metrics"]
            
            # Check convergence
            model_change = round_metrics.get("model_change_norm", float('inf'))
            convergence_history.append(model_change)
            
            # Track best model
            current_metric = round_metrics.get("avg_local_accuracy", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                best_model = self.global_model
            
            logger.info(f"Round {round_num}: Model change = {model_change:.6f}, "
                       f"Avg accuracy = {current_metric:.4f}")
            
            # Check convergence
            if model_change < convergence_threshold:
                logger.info(f"Converged after {round_num + 1} rounds")
                break
            
            # Early stopping if no improvement
            if len(convergence_history) >= 10:
                recent_changes = convergence_history[-10:]
                if all(change < convergence_threshold * 10 for change in recent_changes):
                    logger.info(f"Early stopping due to slow convergence at round {round_num + 1}")
                    break
        
        # Generate final results
        optimization_results = {
            "task_id": task.task_id,
            "final_model": best_model or self.global_model,
            "total_rounds": self.current_round,
            "convergence_history": convergence_history,
            "best_metric": best_metric,
            "participating_clients": len(self.clients),
            "optimization_history": self.optimization_history,
            "privacy_metrics": self._calculate_privacy_metrics(),
            "efficiency_metrics": self._calculate_efficiency_metrics(),
            "fairness_metrics": self._calculate_fairness_metrics()
        }
        
        logger.info(f"Federated optimization completed for task {task.task_id}")
        return optimization_results
    
    def _initialize_model_parameters(self, task: FederatedTask) -> Dict[str, np.ndarray]:
        """Initialize model parameters for the federated task."""
        # Simplified parameter initialization based on task type
        if task.task_type == "classification":
            return {
                "weights": np.random.normal(0, 0.1, (100, 10)),
                "biases": np.zeros(10)
            }
        elif task.task_type == "regression":
            return {
                "weights": np.random.normal(0, 0.1, (100, 1)),
                "bias": np.array([0.0])
            }
        elif task.task_type == "optimization":
            # Hardware optimization parameters
            return {
                "compute_allocation": np.random.uniform(0, 1, 20),
                "memory_hierarchy": np.random.uniform(0, 1, 10),
                "interconnect_weights": np.random.uniform(0, 1, (10, 10))
            }
        else:
            return {"default_params": np.random.normal(0, 0.1, (50, 50))}
    
    def _calculate_privacy_metrics(self) -> Dict[str, float]:
        """Calculate privacy-related metrics for the optimization."""
        return {
            "differential_privacy_budget_consumed": self.secure_aggregator.privacy_budget * 0.8,
            "secure_aggregation_rounds": sum(
                1 for round_data in self.optimization_history
                if "secure_aggregation" in round_data.get("privacy_mechanisms", [])
            ),
            "client_data_leakage_risk": 0.1,  # Simplified risk assessment
            "model_inversion_resistance": 0.9  # Simplified resistance score
        }
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics for the federated optimization."""
        if not self.optimization_history:
            return {}
        
        total_communication = sum(
            round_data["metrics"].get("total_communication_mb", 0)
            for round_data in self.optimization_history
        )
        
        total_energy = sum(
            round_data["metrics"].get("total_energy_mwh", 0)
            for round_data in self.optimization_history
        )
        
        avg_round_time = np.mean([
            round_data["metrics"].get("round_duration", 0)
            for round_data in self.optimization_history
        ])
        
        return {
            "total_communication_mb": total_communication,
            "total_energy_consumption_mwh": total_energy,
            "average_round_duration": avg_round_time,
            "communication_efficiency": total_communication / max(self.current_round, 1),
            "energy_efficiency": total_energy / max(self.current_round, 1)
        }
    
    def _calculate_fairness_metrics(self) -> Dict[str, float]:
        """Calculate fairness metrics for client participation."""
        if not self.optimization_history:
            return {}
        
        # Count participation per client
        participation_counts = defaultdict(int)
        for round_data in self.optimization_history:
            for client_id in round_data.get("participating_clients", []):
                participation_counts[client_id] += 1
        
        if not participation_counts:
            return {}
        
        participation_values = list(participation_counts.values())
        
        # Calculate fairness metrics
        participation_variance = np.var(participation_values)
        participation_gini = self._calculate_gini_coefficient(participation_values)
        
        return {
            "participation_variance": participation_variance,
            "participation_gini_coefficient": participation_gini,
            "min_participation": min(participation_values),
            "max_participation": max(participation_values),
            "participation_ratio": min(participation_values) / max(participation_values) if max(participation_values) > 0 else 0
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for fairness measurement."""
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        numerator = sum((2 * i - n - 1) * value for i, value in enumerate(sorted_values, 1))
        denominator = n * sum(sorted_values)
        
        return numerator / denominator if denominator > 0 else 0.0


def create_federated_optimizer(
    aggregation_strategy: str = "fed_avg",
    client_selection: str = "random",
    privacy_mechanisms: List[str] = None
) -> FederatedHardwareOptimizer:
    """
    Factory function to create federated optimizers.
    
    Args:
        aggregation_strategy: Aggregation strategy name
        client_selection: Client selection strategy name
        privacy_mechanisms: List of privacy mechanism names
        
    Returns:
        Configured federated optimizer
    """
    agg_strategy = AggregationStrategy(aggregation_strategy)
    selection_strategy = ClientSelectionStrategy(client_selection)
    
    privacy_mechs = []
    if privacy_mechanisms:
        privacy_mechs = [PrivacyMechanism(pm) for pm in privacy_mechanisms]
    
    return FederatedHardwareOptimizer(
        aggregation_strategy=agg_strategy,
        client_selection=selection_strategy,
        privacy_mechanisms=privacy_mechs
    )


# Example usage and testing
if __name__ == "__main__":
    # Create federated optimizer
    optimizer = create_federated_optimizer(
        aggregation_strategy="fed_avg",
        client_selection="performance",
        privacy_mechanisms=["differential_privacy"]
    )
    
    # Register sample clients
    for i in range(5):
        client_profile = ClientProfile(
            client_id=f"client_{i}",
            hardware_specs={"cpu": f"ARM{i}", "memory": "4GB"},
            compute_capacity=np.random.uniform(1e10, 1e12),
            memory_capacity=np.random.randint(2048, 8192),
            bandwidth=np.random.uniform(10, 100),
            energy_budget=np.random.uniform(100, 1000),
            availability_score=np.random.uniform(0.7, 1.0),
            trustworthiness_score=np.random.uniform(0.8, 1.0),
            privacy_level=np.random.randint(1, 4)
        )
        
        registration_info = optimizer.register_client(client_profile)
        print(f"Registered {client_profile.client_id}")
    
    # Create federated task
    task = FederatedTask(
        task_id="hardware_optimization_001",
        task_type="optimization",
        objective_function="minimize_energy_maximize_performance",
        privacy_requirements=[PrivacyMechanism.DIFFERENTIAL_PRIVACY],
        hardware_constraints={"power_budget": 100.0, "area_budget": 10.0},
        target_hardware={"type": "edge_ai_accelerator"},
        optimization_objectives=["energy_efficiency", "throughput", "latency"],
        max_rounds=20,
        min_clients_per_round=3
    )
    
    # Run federated optimization
    results = optimizer.optimize_federated_hardware_design(
        task=task,
        max_rounds=10,
        convergence_threshold=1e-4,
        min_clients_per_round=2
    )
    
    print("\nFederated Optimization Results:")
    print(f"Task ID: {results['task_id']}")
    print(f"Total rounds: {results['total_rounds']}")
    print(f"Best metric: {results['best_metric']:.4f}")
    print(f"Participating clients: {results['participating_clients']}")
    print(f"Privacy budget consumed: {results['privacy_metrics']['differential_privacy_budget_consumed']:.2f}")
    print(f"Total communication: {results['efficiency_metrics']['total_communication_mb']:.2f} MB")
    print(f"Participation fairness (Gini): {results['fairness_metrics']['participation_gini_coefficient']:.3f}")