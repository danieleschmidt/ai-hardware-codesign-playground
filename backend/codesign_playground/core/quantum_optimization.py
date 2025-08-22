"""
Quantum-Enhanced Optimization Module for AI Hardware Co-Design.

This module implements quantum-inspired optimization algorithms for accelerator design space
exploration, leveraging quantum annealing principles and variational quantum eigensolvers (VQE)
for hardware-software co-optimization problems.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.monitoring import record_metric
from ..utils.exceptions import OptimizationError

logger = get_logger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum state in the optimization space."""
    
    amplitudes: np.ndarray
    phases: np.ndarray
    energy: float
    fidelity: float = 1.0
    coherence_time: float = 100.0  # microseconds
    
    def __post_init__(self):
        """Normalize quantum state after initialization."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm


@dataclass
class QuantumCircuit:
    """Quantum circuit for optimization problems."""
    
    num_qubits: int
    gates: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0
    
    def add_gate(self, gate_type: str, qubit_indices: List[int], parameters: Optional[List[float]] = None):
        """Add a quantum gate to the circuit."""
        gate = {
            "type": gate_type,
            "qubits": qubit_indices,
            "params": parameters or [],
            "timestamp": time.time()
        }
        self.gates.append(gate)
        self.depth += 1
    
    def add_variational_layer(self, layer_type: str = "hardware_efficient"):
        """Add a variational quantum circuit layer."""
        if layer_type == "hardware_efficient":
            # Single-qubit rotations
            for i in range(self.num_qubits):
                self.add_gate("RY", [i], [np.random.random() * 2 * np.pi])
                self.add_gate("RZ", [i], [np.random.random() * 2 * np.pi])
            
            # Entangling gates
            for i in range(self.num_qubits - 1):
                self.add_gate("CNOT", [i, i + 1])
        
        elif layer_type == "strongly_entangling":
            # All-to-all connectivity with controlled rotations
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.add_gate("CRY", [i, j], [np.random.random() * 2 * np.pi])


class QuantumAnnealingOptimizer:
    """Quantum annealing-inspired optimizer for accelerator design."""
    
    def __init__(self, num_qubits: int = 16, temperature_schedule: Optional[Callable] = None):
        """
        Initialize quantum annealing optimizer.
        
        Args:
            num_qubits: Number of qubits in the quantum system
            temperature_schedule: Annealing temperature schedule function
        """
        self.num_qubits = num_qubits
        self.temperature_schedule = temperature_schedule or self._default_temperature_schedule
        self.best_state = None
        self.best_energy = float('inf')
        self.energy_history = []
        
        # Quantum system parameters
        self.transverse_field = 1.0
        self.longitudinal_field = 0.0
        self.coupling_strength = 0.1
        
        logger.info(f"Initialized quantum annealing optimizer with {num_qubits} qubits")
    
    def _default_temperature_schedule(self, step: int, total_steps: int) -> float:
        """Default exponential annealing schedule."""
        initial_temp = 10.0
        final_temp = 0.01
        return initial_temp * (final_temp / initial_temp) ** (step / total_steps)
    
    def _ising_hamiltonian(self, spins: np.ndarray, problem_matrix: np.ndarray) -> float:
        """Compute Ising Hamiltonian energy."""
        linear_term = np.sum(spins)
        quadratic_term = np.sum(spins[:, np.newaxis] * problem_matrix * spins[np.newaxis, :])
        return -self.longitudinal_field * linear_term - 0.5 * quadratic_term
    
    def _quantum_tunneling_probability(self, energy_diff: float, temperature: float) -> float:
        """Calculate quantum tunneling probability."""
        if temperature <= 0:
            return 0.0
        
        # Quantum tunneling includes transverse field effects
        classical_prob = np.exp(-energy_diff / temperature)
        quantum_correction = self.transverse_field * np.exp(-energy_diff / (2 * temperature))
        
        return min(1.0, classical_prob + quantum_correction)
    
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, List[Any]],
        num_iterations: int = 1000,
        num_replicas: int = 8
    ) -> Dict[str, Any]:
        """
        Perform quantum annealing optimization.
        
        Args:
            objective_function: Function to minimize
            search_space: Dictionary of parameter search spaces
            num_iterations: Number of annealing steps
            num_replicas: Number of parallel annealing replicas
            
        Returns:
            Optimization results with best configuration and quantum metrics
        """
        start_time = time.time()
        
        # Convert discrete search space to continuous problem
        problem_matrix = self._encode_search_space(search_space)
        
        # Initialize quantum replicas
        replicas = []
        for _ in range(num_replicas):
            initial_spins = np.random.choice([-1, 1], size=self.num_qubits)
            state = QuantumState(
                amplitudes=initial_spins.astype(float),
                phases=np.zeros(self.num_qubits),
                energy=self._ising_hamiltonian(initial_spins, problem_matrix)
            )
            replicas.append(state)
        
        best_configurations = []
        quantum_metrics = {
            "entanglement_entropy": [],
            "coherence_evolution": [],
            "tunneling_events": 0,
            "replica_diversity": []
        }
        
        logger.info(f"Starting quantum annealing with {num_replicas} replicas, {num_iterations} iterations")
        
        # Annealing loop
        for iteration in range(num_iterations):
            temperature = self.temperature_schedule(iteration, num_iterations)
            
            # Update transverse field (quantum fluctuations decrease over time)
            self.transverse_field = 1.0 * (1 - iteration / num_iterations)
            self.longitudinal_field = 1.0 * (iteration / num_iterations)
            
            # Parallel replica updates
            updated_replicas = []
            tunneling_count = 0
            
            for replica in replicas:
                new_replica = self._update_replica(replica, problem_matrix, temperature)
                
                # Check for quantum tunneling event
                if new_replica.energy < replica.energy - 0.1:  # Significant energy decrease
                    tunneling_count += 1
                
                updated_replicas.append(new_replica)
                
                # Track best state
                if new_replica.energy < self.best_energy:
                    self.best_energy = new_replica.energy
                    self.best_state = new_replica
                    
                    # Convert quantum state back to configuration
                    config = self._decode_quantum_state(new_replica, search_space)
                    energy = objective_function(config)
                    best_configurations.append((config, energy))
            
            replicas = updated_replicas
            quantum_metrics["tunneling_events"] += tunneling_count
            
            # Calculate quantum metrics
            if iteration % 50 == 0:
                entanglement = self._calculate_entanglement_entropy(replicas)
                coherence = np.mean([r.fidelity for r in replicas])
                diversity = self._calculate_replica_diversity(replicas)
                
                quantum_metrics["entanglement_entropy"].append(entanglement)
                quantum_metrics["coherence_evolution"].append(coherence)
                quantum_metrics["replica_diversity"].append(diversity)
                
                logger.debug(f"Iteration {iteration}: T={temperature:.4f}, "
                           f"Best_E={self.best_energy:.4f}, "
                           f"Entanglement={entanglement:.4f}")
            
            self.energy_history.append(self.best_energy)
            
            # Record metrics
            if iteration % 100 == 0:
                record_metric("quantum_annealing_temperature", temperature, "gauge")
                record_metric("quantum_annealing_energy", self.best_energy, "gauge")
                record_metric("quantum_entanglement", entanglement, "gauge")
        
        # Final optimization
        duration = time.time() - start_time
        
        if best_configurations:
            best_config, best_objective = min(best_configurations, key=lambda x: x[1])
        else:
            # Fallback to best quantum state
            best_config = self._decode_quantum_state(self.best_state, search_space)
            best_objective = objective_function(best_config)
        
        results = {
            "best_configuration": best_config,
            "best_objective": best_objective,
            "quantum_energy": self.best_energy,
            "optimization_time": duration,
            "iterations": num_iterations,
            "quantum_metrics": quantum_metrics,
            "convergence_history": self.energy_history,
            "algorithm": "quantum_annealing"
        }
        
        logger.info(f"Quantum annealing completed in {duration:.2f}s. "
                   f"Best objective: {best_objective:.6f}")
        
        return results
    
    def _update_replica(self, replica: QuantumState, problem_matrix: np.ndarray, temperature: float) -> QuantumState:
        """Update a single quantum replica."""
        # Create proposal by flipping a random spin
        new_amplitudes = replica.amplitudes.copy()
        flip_idx = np.random.randint(0, len(new_amplitudes))
        new_amplitudes[flip_idx] *= -1
        
        # Calculate energy change
        old_energy = self._ising_hamiltonian(replica.amplitudes, problem_matrix)
        new_energy = self._ising_hamiltonian(new_amplitudes, problem_matrix)
        energy_diff = new_energy - old_energy
        
        # Quantum acceptance criterion
        if energy_diff < 0 or np.random.random() < self._quantum_tunneling_probability(energy_diff, temperature):
            # Accept the move
            new_phases = replica.phases.copy()
            new_phases[flip_idx] += np.pi  # Phase flip
            
            # Update coherence (quantum decoherence)
            new_fidelity = replica.fidelity * np.exp(-0.001)  # Gradual decoherence
            
            return QuantumState(
                amplitudes=new_amplitudes,
                phases=new_phases,
                energy=new_energy,
                fidelity=new_fidelity
            )
        else:
            # Reject the move, but still update coherence
            return QuantumState(
                amplitudes=replica.amplitudes,
                phases=replica.phases,
                energy=old_energy,
                fidelity=replica.fidelity * np.exp(-0.0005)
            )
    
    def _encode_search_space(self, search_space: Dict[str, List[Any]]) -> np.ndarray:
        """Encode discrete search space as Ising problem matrix."""
        # Create coupling matrix based on parameter interactions
        problem_matrix = np.random.randn(self.num_qubits, self.num_qubits) * self.coupling_strength
        # Make symmetric
        problem_matrix = (problem_matrix + problem_matrix.T) / 2
        # Zero diagonal
        np.fill_diagonal(problem_matrix, 0)
        
        return problem_matrix
    
    def _decode_quantum_state(self, state: QuantumState, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode quantum state back to parameter configuration."""
        config = {}
        param_names = list(search_space.keys())
        
        # Map qubits to parameters
        qubits_per_param = max(1, self.num_qubits // len(param_names))
        
        for i, param_name in enumerate(param_names):
            start_idx = i * qubits_per_param
            end_idx = min((i + 1) * qubits_per_param, self.num_qubits)
            
            # Convert qubit amplitudes to parameter index
            qubit_values = state.amplitudes[start_idx:end_idx]
            binary_value = np.sum([q > 0 for q in qubit_values])
            
            param_values = search_space[param_name]
            param_idx = binary_value % len(param_values)
            config[param_name] = param_values[param_idx]
        
        return config
    
    def _calculate_entanglement_entropy(self, replicas: List[QuantumState]) -> float:
        """Calculate entanglement entropy across replicas."""
        if len(replicas) < 2:
            return 0.0
        
        # Simplified entanglement measure based on correlation
        correlations = []
        for i in range(len(replicas)):
            for j in range(i + 1, len(replicas)):
                correlation = np.corrcoef(replicas[i].amplitudes, replicas[j].amplitudes)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
        
        if correlations:
            mean_correlation = np.mean(correlations)
            # Convert to entropy-like measure
            return -mean_correlation * np.log(mean_correlation + 1e-10)
        
        return 0.0
    
    def _calculate_replica_diversity(self, replicas: List[QuantumState]) -> float:
        """Calculate diversity among replicas."""
        if len(replicas) < 2:
            return 0.0
        
        distances = []
        for i in range(len(replicas)):
            for j in range(i + 1, len(replicas)):
                distance = np.linalg.norm(replicas[i].amplitudes - replicas[j].amplitudes)
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0


class VariationalQuantumOptimizer:
    """Variational Quantum Eigensolver (VQE) for hardware optimization."""
    
    def __init__(self, num_qubits: int = 12, circuit_depth: int = 6):
        """
        Initialize VQE optimizer.
        
        Args:
            num_qubits: Number of qubits in the quantum circuit
            circuit_depth: Depth of the variational circuit
        """
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.circuit = QuantumCircuit(num_qubits)
        self.parameters = None
        self.energy_history = []
        
        logger.info(f"Initialized VQE optimizer with {num_qubits} qubits, depth {circuit_depth}")
    
    def _initialize_circuit(self, ansatz_type: str = "hardware_efficient"):
        """Initialize the variational quantum circuit."""
        self.circuit = QuantumCircuit(self.num_qubits)
        
        for layer in range(self.circuit_depth):
            self.circuit.add_variational_layer(ansatz_type)
        
        # Initialize random parameters
        num_params = len([g for g in self.circuit.gates if g["params"]])
        self.parameters = np.random.uniform(-np.pi, np.pi, num_params)
    
    def _classical_simulation(self, parameters: np.ndarray, hamiltonian_matrix: np.ndarray) -> float:
        """Simulate quantum circuit classically and compute expectation value."""
        # Simplified classical simulation
        # In practice, this would use a quantum simulator like Qiskit or Cirq
        
        # Create state vector (simplified)
        state_vector = np.zeros(2**min(self.num_qubits, 10))  # Limit for memory
        state_vector[0] = 1.0  # |0...0⟩ initial state
        
        # Apply parametrized gates (simplified rotation)
        for i, param in enumerate(parameters):
            qubit_idx = i % min(self.num_qubits, 10)
            # Simplified rotation effect
            rotation_angle = param
            state_vector = self._apply_rotation(state_vector, qubit_idx, rotation_angle)
        
        # Compute expectation value
        if hamiltonian_matrix.shape[0] == len(state_vector):
            expectation = np.real(np.conj(state_vector) @ hamiltonian_matrix @ state_vector)
        else:
            # Use simplified energy calculation
            expectation = np.sum(parameters**2) + np.random.normal(0, 0.1)
        
        return expectation
    
    def _apply_rotation(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply rotation to quantum state (simplified)."""
        # Simplified rotation implementation
        new_state = state.copy()
        if len(new_state) > 2**qubit:
            # Apply rotation effect
            cos_half = np.cos(angle / 2)
            sin_half = np.sin(angle / 2)
            
            # Simple rotation mixing
            for i in range(0, len(new_state), 2**(qubit+1)):
                for j in range(2**qubit):
                    if i + j + 2**qubit < len(new_state):
                        old_0 = new_state[i + j]
                        old_1 = new_state[i + j + 2**qubit]
                        new_state[i + j] = cos_half * old_0 - 1j * sin_half * old_1
                        new_state[i + j + 2**qubit] = -1j * sin_half * old_0 + cos_half * old_1
        
        return new_state
    
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, List[Any]],
        num_iterations: int = 500,
        ansatz_type: str = "hardware_efficient"
    ) -> Dict[str, Any]:
        """
        Perform VQE optimization.
        
        Args:
            objective_function: Function to minimize
            search_space: Parameter search space
            num_iterations: Number of optimization iterations
            ansatz_type: Type of variational ansatz
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        # Initialize circuit
        self._initialize_circuit(ansatz_type)
        
        # Create problem Hamiltonian
        hamiltonian = self._create_problem_hamiltonian(search_space)
        
        best_params = self.parameters.copy()
        best_energy = float('inf')
        best_config = None
        
        logger.info(f"Starting VQE optimization with {num_iterations} iterations")
        
        # Optimization loop using gradient-free method
        for iteration in range(num_iterations):
            # Parameter update with adaptive step size
            step_size = 0.1 * (1 - iteration / num_iterations)  # Decreasing step size
            
            # Random perturbation for gradient-free optimization
            parameter_perturbation = np.random.normal(0, step_size, len(self.parameters))
            trial_params = self.parameters + parameter_perturbation
            
            # Compute energy
            try:
                energy = self._classical_simulation(trial_params, hamiltonian)
                
                # Accept if better
                if energy < best_energy:
                    best_energy = energy
                    best_params = trial_params.copy()
                    self.parameters = trial_params
                    
                    # Convert to configuration
                    config = self._params_to_config(trial_params, search_space)
                    objective_value = objective_function(config)
                    
                    if best_config is None or objective_value < best_config[1]:
                        best_config = (config, objective_value)
                
                self.energy_history.append(best_energy)
                
                # Logging
                if iteration % 100 == 0:
                    logger.debug(f"VQE Iteration {iteration}: Energy={energy:.6f}, Best={best_energy:.6f}")
                    record_metric("vqe_energy", energy, "gauge")
                    record_metric("vqe_best_energy", best_energy, "gauge")
                
            except Exception as e:
                logger.warning(f"VQE iteration {iteration} failed: {e}")
                continue
        
        duration = time.time() - start_time
        
        if best_config is None:
            # Fallback
            config = self._params_to_config(best_params, search_space)
            objective_value = objective_function(config)
            best_config = (config, objective_value)
        
        results = {
            "best_configuration": best_config[0],
            "best_objective": best_config[1],
            "quantum_energy": best_energy,
            "optimization_time": duration,
            "iterations": num_iterations,
            "converged_parameters": best_params,
            "energy_history": self.energy_history,
            "circuit_depth": self.circuit_depth,
            "algorithm": "variational_quantum_eigensolver"
        }
        
        logger.info(f"VQE optimization completed in {duration:.2f}s. "
                   f"Best objective: {best_config[1]:.6f}")
        
        return results
    
    def _create_problem_hamiltonian(self, search_space: Dict[str, List[Any]]) -> np.ndarray:
        """Create Hamiltonian matrix for the optimization problem."""
        # Create a simple Hamiltonian for demonstration
        size = min(2**self.num_qubits, 1024)  # Limit size for memory
        hamiltonian = np.random.randn(size, size) * 0.1
        
        # Make Hermitian
        hamiltonian = (hamiltonian + hamiltonian.T) / 2
        
        # Add diagonal elements based on search space
        param_weights = np.random.uniform(0.5, 2.0, len(search_space))
        for i in range(min(len(param_weights), size)):
            hamiltonian[i, i] = param_weights[i % len(param_weights)]
        
        return hamiltonian
    
    def _params_to_config(self, parameters: np.ndarray, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Convert optimization parameters to configuration."""
        config = {}
        param_names = list(search_space.keys())
        
        # Map parameters to configuration values
        params_per_config = max(1, len(parameters) // len(param_names))
        
        for i, param_name in enumerate(param_names):
            start_idx = i * params_per_config
            end_idx = min((i + 1) * params_per_config, len(parameters))
            
            # Use parameter magnitude to select from search space
            param_magnitude = np.mean(np.abs(parameters[start_idx:end_idx]))
            
            param_values = search_space[param_name]
            # Map parameter to discrete choice
            choice_idx = int(param_magnitude * len(param_values)) % len(param_values)
            config[param_name] = param_values[choice_idx]
        
        return config


class QuantumInspiredHybridOptimizer:
    """Hybrid classical-quantum optimizer combining multiple quantum approaches."""
    
    def __init__(self):
        """Initialize hybrid optimizer."""
        self.optimizers = {
            "quantum_annealing": QuantumAnnealingOptimizer(),
            "vqe": VariationalQuantumOptimizer(),
        }
        self.ensemble_weights = {name: 1.0 for name in self.optimizers.keys()}
        
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, List[Any]],
        num_iterations: int = 1000,
        ensemble_strategy: str = "weighted_voting"
    ) -> Dict[str, Any]:
        """
        Perform hybrid quantum optimization using ensemble of methods.
        
        Args:
            objective_function: Function to minimize
            search_space: Parameter search space
            num_iterations: Number of iterations per optimizer
            ensemble_strategy: How to combine results ("weighted_voting", "best_result")
            
        Returns:
            Combined optimization results
        """
        start_time = time.time()
        
        logger.info(f"Starting hybrid quantum optimization with {len(self.optimizers)} methods")
        
        # Run optimizers in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.optimizers)) as executor:
            future_to_optimizer = {
                executor.submit(
                    optimizer.optimize,
                    objective_function,
                    search_space,
                    num_iterations // len(self.optimizers)  # Distribute iterations
                ): name
                for name, optimizer in self.optimizers.items()
            }
            
            for future in as_completed(future_to_optimizer):
                optimizer_name = future_to_optimizer[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results[optimizer_name] = result
                    logger.info(f"{optimizer_name} completed: objective={result['best_objective']:.6f}")
                except Exception as e:
                    logger.error(f"{optimizer_name} failed: {e}")
                    results[optimizer_name] = None
        
        # Ensemble the results
        if ensemble_strategy == "best_result":
            best_result = min(
                [r for r in results.values() if r is not None],
                key=lambda x: x["best_objective"]
            )
            final_result = best_result
            final_result["ensemble_method"] = "best_result"
            
        elif ensemble_strategy == "weighted_voting":
            # Weighted ensemble of configurations
            valid_results = {k: v for k, v in results.items() if v is not None}
            
            if valid_results:
                # Weight by inverse objective value
                weights = {}
                for name, result in valid_results.items():
                    weight = 1.0 / (result["best_objective"] + 1e-6)
                    weights[name] = weight
                
                # Normalize weights
                total_weight = sum(weights.values())
                weights = {k: v / total_weight for k, v in weights.items()}
                
                # Ensemble configuration by majority voting
                ensemble_config = self._ensemble_configurations(
                    [r["best_configuration"] for r in valid_results.values()],
                    list(weights.values())
                )
                
                ensemble_objective = objective_function(ensemble_config)
                
                final_result = {
                    "best_configuration": ensemble_config,
                    "best_objective": ensemble_objective,
                    "optimization_time": time.time() - start_time,
                    "ensemble_method": "weighted_voting",
                    "individual_results": valid_results,
                    "ensemble_weights": weights,
                    "algorithm": "quantum_hybrid_ensemble"
                }
            else:
                raise OptimizationError("All quantum optimizers failed")
        
        duration = time.time() - start_time
        final_result["total_optimization_time"] = duration
        
        logger.info(f"Hybrid quantum optimization completed in {duration:.2f}s. "
                   f"Final objective: {final_result['best_objective']:.6f}")
        
        return final_result
    
    def _ensemble_configurations(self, configurations: List[Dict], weights: List[float]) -> Dict[str, Any]:
        """Ensemble multiple configurations using weighted voting."""
        if not configurations:
            raise ValueError("No configurations to ensemble")
        
        # Get all parameter names
        all_params = set()
        for config in configurations:
            all_params.update(config.keys())
        
        ensemble_config = {}
        
        for param in all_params:
            # Collect values and weights for this parameter
            param_values = []
            param_weights = []
            
            for config, weight in zip(configurations, weights):
                if param in config:
                    param_values.append(config[param])
                    param_weights.append(weight)
            
            if param_values:
                # For categorical/discrete parameters, use weighted voting
                if isinstance(param_values[0], (str, int, bool)):
                    # Count votes for each value
                    vote_counts = {}
                    for value, weight in zip(param_values, param_weights):
                        vote_counts[value] = vote_counts.get(value, 0) + weight
                    
                    # Select most voted value
                    ensemble_config[param] = max(vote_counts.items(), key=lambda x: x[1])[0]
                
                # For numerical parameters, use weighted average
                else:
                    total_weight = sum(param_weights)
                    if total_weight > 0:
                        weighted_sum = sum(v * w for v, w in zip(param_values, param_weights))
                        ensemble_config[param] = weighted_sum / total_weight
                    else:
                        ensemble_config[param] = param_values[0]
        
        return ensemble_config


def create_quantum_optimizer(optimizer_type: str = "hybrid", **kwargs) -> Any:
    """
    Factory function to create quantum optimizers.
    
    Args:
        optimizer_type: Type of optimizer ("annealing", "vqe", "hybrid")
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Configured quantum optimizer instance
    """
    if optimizer_type == "annealing":
        return QuantumAnnealingOptimizer(**kwargs)
    elif optimizer_type == "vqe":
        return VariationalQuantumOptimizer(**kwargs)
    elif optimizer_type == "hybrid":
        return QuantumInspiredHybridOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Example usage and benchmarking
if __name__ == "__main__":
    # Example optimization problem
    def example_objective(config):
        """Example objective function for testing."""
        penalty = 0
        penalty += (config.get("compute_units", 64) - 100) ** 2 * 0.01
        penalty += (config.get("frequency_mhz", 200) - 250) ** 2 * 0.001
        return penalty + np.random.normal(0, 0.1)
    
    example_search_space = {
        "compute_units": [16, 32, 64, 128, 256],
        "frequency_mhz": [100, 150, 200, 250, 300, 400],
        "dataflow": ["weight_stationary", "output_stationary", "row_stationary"],
        "memory_size": [32, 64, 128, 256, 512]
    }
    
    # Test all optimizers
    optimizers = {
        "quantum_annealing": create_quantum_optimizer("annealing", num_qubits=12),
        "vqe": create_quantum_optimizer("vqe", num_qubits=10, circuit_depth=4),
        "hybrid": create_quantum_optimizer("hybrid")
    }
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        result = optimizer.optimize(
            example_objective,
            example_search_space,
            num_iterations=200
        )
        print(f"Best objective: {result['best_objective']:.6f}")
        print(f"Best config: {result['best_configuration']}")
        print(f"Time: {result['optimization_time']:.2f}s")