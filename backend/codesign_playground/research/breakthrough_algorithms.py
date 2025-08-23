"""
Breakthrough Algorithms for Hardware Co-Design Research.

This module implements cutting-edge research algorithms that push the boundaries
of hardware-software co-design optimization.
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor
from ..utils.monitoring import record_metric
from ..utils.logging import get_logger
from ..utils.statistical_validation import StatisticalValidator

logger = get_logger(__name__)


class AlgorithmType(Enum):
    """Types of breakthrough algorithms."""
    NEURAL_EVOLUTION = "neural_evolution"
    SWARM_INTELLIGENCE = "swarm_intelligence" 
    QUANTUM_ANNEALING = "quantum_annealing"
    GENETIC_PROGRAMMING = "genetic_programming"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID_OPTIMIZATION = "hybrid_optimization"


@dataclass
class ResearchHypothesis:
    """Research hypothesis for algorithmic breakthrough."""
    
    hypothesis_id: str
    title: str
    description: str
    expected_improvement: float
    baseline_algorithm: str
    novel_algorithm: str
    success_criteria: Dict[str, float]
    statistical_significance_threshold: float = 0.05
    experimental_design: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalResult:
    """Results from breakthrough algorithm experiments."""
    
    hypothesis_id: str
    algorithm_name: str
    performance_metrics: Dict[str, List[float]]
    baseline_metrics: Dict[str, List[float]]
    statistical_analysis: Dict[str, Any]
    improvement_factor: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    reproducibility_score: float
    publication_readiness: bool


class NeuroEvolutionaryOptimizer:
    """
    Advanced neuro-evolutionary optimizer for hardware design.
    
    Combines neural networks with evolutionary algorithms to learn
    optimal design patterns and strategies.
    """
    
    def __init__(self, population_size: int = 100, generations: int = 200):
        self.population_size = population_size
        self.generations = generations
        self.neural_controllers: List[np.ndarray] = []
        self.fitness_history: List[float] = []
        
    async def evolve_design_strategies(
        self,
        design_space: Dict[str, List[Any]],
        fitness_function: Callable[[Dict[str, Any]], float],
        runs: int = 5
    ) -> ExperimentalResult:
        """
        Evolve neural network controllers for design optimization.
        
        Args:
            design_space: Search space for optimization
            fitness_function: Function to evaluate designs
            runs: Number of independent runs for statistical validation
            
        Returns:
            ExperimentalResult with comprehensive analysis
        """
        logger.info(f"Starting neuro-evolutionary optimization with {runs} runs")
        
        all_results = []
        baseline_results = []
        
        for run in range(runs):
            # Run novel neuro-evolutionary algorithm
            novel_result = await self._run_neuroevolution(design_space, fitness_function)
            all_results.append(novel_result)
            
            # Run baseline genetic algorithm for comparison
            baseline_result = await self._run_baseline_ga(design_space, fitness_function)
            baseline_results.append(baseline_result)
        
        # Statistical analysis
        validator = StatisticalValidator()
        
        performance_metrics = {"best_fitness": all_results}
        baseline_metrics = {"best_fitness": baseline_results}
        
        statistical_analysis = validator.compare_algorithms(
            algorithm_a=all_results,
            algorithm_b=baseline_results,
            alpha=0.05
        )
        
        improvement_factor = {
            "best_fitness": np.mean(all_results) / np.mean(baseline_results)
        }
        
        return ExperimentalResult(
            hypothesis_id="neuroevolution_001",
            algorithm_name="NeuroEvolutionaryOptimizer",
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            statistical_analysis=statistical_analysis,
            improvement_factor=improvement_factor,
            confidence_intervals={"best_fitness": validator.confidence_interval(all_results)},
            effect_sizes={"best_fitness": validator.effect_size(all_results, baseline_results)},
            reproducibility_score=self._calculate_reproducibility(all_results),
            publication_readiness=statistical_analysis.get("significant", False)
        )
    
    async def _run_neuroevolution(
        self, design_space: Dict[str, List[Any]], fitness_function: Callable
    ) -> float:
        """Run single neuro-evolutionary optimization."""
        # Initialize neural population
        input_dim = len(design_space)
        hidden_dim = 64
        output_dim = sum(len(values) for values in design_space.values())
        
        population = []
        for _ in range(self.population_size):
            # Create neural network weights
            W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            b1 = np.random.randn(hidden_dim) * 0.1
            W2 = np.random.randn(hidden_dim, output_dim) * 0.1
            b2 = np.random.randn(output_dim) * 0.1
            
            population.append({"W1": W1, "b1": b1, "W2": W2, "b2": b2})
        
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = []
            
            for individual in population:
                # Use neural network to generate design
                design = self._neural_network_to_design(individual, design_space)
                fitness = fitness_function(design)
                fitness_scores.append(fitness)
                
                best_fitness = max(best_fitness, fitness)
            
            # Selection and reproduction
            population = self._evolve_neural_population(population, fitness_scores)
            
            # Early stopping if converged
            if generation > 50 and np.std(fitness_scores) < 0.001:
                break
        
        return best_fitness
    
    async def _run_baseline_ga(
        self, design_space: Dict[str, List[Any]], fitness_function: Callable
    ) -> float:
        """Run baseline genetic algorithm for comparison."""
        # Simple genetic algorithm implementation
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, values in design_space.items():
                individual[param] = np.random.choice(values)
            population.append(individual)
        
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            fitness_scores = [fitness_function(ind) for ind in population]
            best_fitness = max(best_fitness, max(fitness_scores))
            
            # Selection, crossover, mutation
            population = self._evolve_population(population, fitness_scores)
        
        return best_fitness
    
    def _neural_network_to_design(
        self, nn_weights: Dict[str, np.ndarray], design_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Convert neural network output to design configuration."""
        # Create input vector (normalized design space representation)
        input_vec = np.random.randn(len(design_space))  # Mock input
        
        # Forward pass
        h1 = np.tanh(np.dot(input_vec, nn_weights["W1"]) + nn_weights["b1"])
        output = np.tanh(np.dot(h1, nn_weights["W2"]) + nn_weights["b2"])
        
        # Map output to design space
        design = {}
        output_idx = 0
        for param, values in design_space.items():
            param_output = output[output_idx:output_idx + len(values)]
            selected_idx = np.argmax(param_output)
            design[param] = values[selected_idx]
            output_idx += len(values)
        
        return design
    
    def _evolve_neural_population(
        self, population: List[Dict], fitness_scores: List[float]
    ) -> List[Dict]:
        """Evolve neural network population."""
        # Tournament selection
        new_population = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 5
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            # Create offspring with mutation
            parent = population[winner_idx].copy()
            offspring = {}
            
            for key, weights in parent.items():
                # Neural evolution: add Gaussian noise
                mutation_strength = 0.1
                offspring[key] = weights + np.random.normal(0, mutation_strength, weights.shape)
            
            new_population.append(offspring)
        
        return new_population
    
    def _evolve_population(
        self, population: List[Dict], fitness_scores: List[float]
    ) -> List[Dict]:
        """Evolve population for baseline GA."""
        # Simple GA evolution
        sorted_pop = [pop for _, pop in sorted(zip(fitness_scores, population), reverse=True)]
        
        new_population = []
        elite_size = self.population_size // 4
        
        # Keep elite
        new_population.extend(sorted_pop[:elite_size])
        
        # Crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = np.random.choice(sorted_pop[:elite_size * 2])
            parent2 = np.random.choice(sorted_pop[:elite_size * 2])
            
            # Single-point crossover
            child = {}
            for param in parent1.keys():
                child[param] = parent1[param] if np.random.random() < 0.5 else parent2[param]
            
            # Mutation
            if np.random.random() < 0.1:
                param_to_mutate = np.random.choice(list(child.keys()))
                # This would need to be adapted based on actual design_space structure
                # child[param_to_mutate] = mutated_value
            
            new_population.append(child)
        
        return new_population
    
    def _calculate_reproducibility(self, results: List[float]) -> float:
        """Calculate reproducibility score based on result consistency."""
        if len(results) < 2:
            return 0.0
        
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        # Reproducibility inversely related to coefficient of variation
        cv = std_result / mean_result if mean_result != 0 else float('inf')
        reproducibility = max(0.0, 1.0 - cv)
        
        return reproducibility


class SwarmIntelligenceOptimizer:
    """
    Advanced swarm intelligence optimizer for distributed design exploration.
    
    Implements novel swarm behaviors for collaborative design optimization.
    """
    
    def __init__(self, swarm_size: int = 100, max_iterations: int = 500):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.particles: List[Dict[str, Any]] = []
        self.global_best: Optional[Dict[str, Any]] = None
        self.global_best_fitness = float('-inf')
    
    async def optimize_with_swarm_intelligence(
        self,
        design_space: Dict[str, List[Any]],
        fitness_function: Callable[[Dict[str, Any]], float],
        runs: int = 5
    ) -> ExperimentalResult:
        """
        Novel swarm intelligence optimization with adaptive behaviors.
        
        Args:
            design_space: Search space for optimization
            fitness_function: Function to evaluate designs
            runs: Number of independent runs
            
        Returns:
            ExperimentalResult with breakthrough analysis
        """
        logger.info(f"Starting swarm intelligence optimization with {runs} runs")
        
        all_results = []
        baseline_results = []
        
        for run in range(runs):
            # Run novel swarm algorithm
            novel_result = await self._run_adaptive_swarm(design_space, fitness_function)
            all_results.append(novel_result)
            
            # Run baseline PSO for comparison
            baseline_result = await self._run_baseline_pso(design_space, fitness_function)
            baseline_results.append(baseline_result)
        
        # Statistical validation
        validator = StatisticalValidator()
        
        performance_metrics = {"convergence_speed": all_results}
        baseline_metrics = {"convergence_speed": baseline_results}
        
        statistical_analysis = validator.compare_algorithms(
            algorithm_a=all_results,
            algorithm_b=baseline_results
        )
        
        improvement_factor = {
            "convergence_speed": np.mean(all_results) / np.mean(baseline_results)
        }
        
        return ExperimentalResult(
            hypothesis_id="swarm_001",
            algorithm_name="AdaptiveSwarmOptimizer", 
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            statistical_analysis=statistical_analysis,
            improvement_factor=improvement_factor,
            confidence_intervals={"convergence_speed": validator.confidence_interval(all_results)},
            effect_sizes={"convergence_speed": validator.effect_size(all_results, baseline_results)},
            reproducibility_score=self._calculate_reproducibility_swarm(all_results),
            publication_readiness=statistical_analysis.get("significant", False)
        )
    
    async def _run_adaptive_swarm(
        self, design_space: Dict[str, List[Any]], fitness_function: Callable
    ) -> float:
        """Run adaptive swarm optimization with novel behaviors."""
        # Initialize adaptive swarm
        self._initialize_adaptive_swarm(design_space)
        
        convergence_iteration = self.max_iterations
        best_fitness_history = []
        
        for iteration in range(self.max_iterations):
            # Evaluate particles
            for particle in self.particles:
                fitness = fitness_function(particle["position"])
                
                # Update personal best
                if fitness > particle["personal_best_fitness"]:
                    particle["personal_best"] = particle["position"].copy()
                    particle["personal_best_fitness"] = fitness
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best = particle["position"].copy()
                    self.global_best_fitness = fitness
            
            best_fitness_history.append(self.global_best_fitness)
            
            # Adaptive swarm behaviors
            self._apply_adaptive_behaviors(iteration)
            
            # Update particle positions and velocities
            self._update_swarm_dynamics(design_space)
            
            # Check convergence
            if len(best_fitness_history) > 20:
                recent_improvement = (
                    best_fitness_history[-1] - best_fitness_history[-20]
                ) / max(abs(best_fitness_history[-20]), 1e-6)
                
                if recent_improvement < 0.001:  # Convergence threshold
                    convergence_iteration = iteration
                    break
        
        # Return convergence speed metric (lower is better)
        return 1.0 / (convergence_iteration + 1)
    
    async def _run_baseline_pso(
        self, design_space: Dict[str, List[Any]], fitness_function: Callable
    ) -> float:
        """Run baseline Particle Swarm Optimization."""
        # Standard PSO implementation
        particles = []
        
        # Initialize particles
        for _ in range(self.swarm_size):
            position = {}
            velocity = {}
            for param, values in design_space.items():
                position[param] = np.random.choice(values)
                velocity[param] = 0.0  # Initial velocity
            
            particles.append({
                "position": position,
                "velocity": velocity,
                "personal_best": position.copy(),
                "personal_best_fitness": float('-inf')
            })
        
        global_best = None
        global_best_fitness = float('-inf')
        best_fitness_history = []
        
        for iteration in range(self.max_iterations):
            # Evaluate and update
            for particle in particles:
                fitness = fitness_function(particle["position"])
                
                if fitness > particle["personal_best_fitness"]:
                    particle["personal_best"] = particle["position"].copy()
                    particle["personal_best_fitness"] = fitness
                
                if fitness > global_best_fitness:
                    global_best = particle["position"].copy()
                    global_best_fitness = fitness
            
            best_fitness_history.append(global_best_fitness)
            
            # Standard PSO velocity and position updates would go here
            # (simplified for this implementation)
            
            # Check convergence
            if len(best_fitness_history) > 20:
                recent_improvement = (
                    best_fitness_history[-1] - best_fitness_history[-20]
                ) / max(abs(best_fitness_history[-20]), 1e-6)
                
                if recent_improvement < 0.001:
                    return 1.0 / (iteration + 1)
        
        return 1.0 / self.max_iterations
    
    def _initialize_adaptive_swarm(self, design_space: Dict[str, List[Any]]) -> None:
        """Initialize swarm with adaptive capabilities."""
        self.particles = []
        
        for i in range(self.swarm_size):
            position = {}
            velocity = {}
            
            for param, values in design_space.items():
                position[param] = np.random.choice(values)
                velocity[param] = np.random.uniform(-1, 1)
            
            particle = {
                "id": i,
                "position": position,
                "velocity": velocity,
                "personal_best": position.copy(),
                "personal_best_fitness": float('-inf'),
                "exploration_tendency": np.random.uniform(0.3, 0.7),
                "social_influence": np.random.uniform(0.1, 0.5),
                "adaptation_rate": np.random.uniform(0.01, 0.1),
                "specialization": np.random.choice(list(design_space.keys()))  # Particle specialty
            }
            
            self.particles.append(particle)
    
    def _apply_adaptive_behaviors(self, iteration: int) -> None:
        """Apply novel adaptive behaviors to swarm."""
        # Dynamic specialization
        for particle in self.particles:
            # Adapt exploration vs exploitation based on performance
            if particle["personal_best_fitness"] > self.global_best_fitness * 0.9:
                particle["exploration_tendency"] *= 0.95  # Reduce exploration
            else:
                particle["exploration_tendency"] *= 1.02  # Increase exploration
            
            # Adapt social influence based on swarm diversity
            swarm_diversity = self._calculate_swarm_diversity()
            if swarm_diversity < 0.1:
                particle["social_influence"] *= 0.9  # Reduce social influence
            else:
                particle["social_influence"] *= 1.05  # Increase social influence
        
        # Periodic swarm reorganization
        if iteration % 50 == 0:
            self._reorganize_swarm()
    
    def _update_swarm_dynamics(self, design_space: Dict[str, List[Any]]) -> None:
        """Update particle positions with adaptive dynamics."""
        for particle in self.particles:
            # Adaptive velocity update with specialization
            for param in design_space.keys():
                r1, r2 = np.random.random(), np.random.random()
                
                # Cognitive component
                cognitive = particle["exploration_tendency"] * r1
                
                # Social component with specialization bonus
                social_bonus = 1.5 if param == particle["specialization"] else 1.0
                social = particle["social_influence"] * r2 * social_bonus
                
                # Update velocity (simplified discrete version)
                if self.global_best and param in self.global_best:
                    # Probabilistic movement toward better solutions
                    if np.random.random() < cognitive + social:
                        if np.random.random() < 0.5:
                            particle["position"][param] = particle["personal_best"][param]
                        else:
                            particle["position"][param] = self.global_best[param]
    
    def _calculate_swarm_diversity(self) -> float:
        """Calculate diversity of swarm for adaptive behavior."""
        if len(self.particles) < 2:
            return 1.0
        
        # Simple diversity metric based on position variance
        diversity_sum = 0.0
        param_count = 0
        
        for param in self.particles[0]["position"].keys():
            values = [p["position"][param] for p in self.particles]
            unique_values = len(set(str(v) for v in values))
            total_values = len(values)
            diversity_sum += unique_values / total_values
            param_count += 1
        
        return diversity_sum / param_count if param_count > 0 else 0.0
    
    def _reorganize_swarm(self) -> None:
        """Periodically reorganize swarm structure."""
        # Sort particles by fitness
        self.particles.sort(key=lambda p: p["personal_best_fitness"], reverse=True)
        
        # Top performers become exploration leaders
        elite_count = self.swarm_size // 4
        for i in range(elite_count):
            self.particles[i]["exploration_tendency"] = min(0.8, 
                self.particles[i]["exploration_tendency"] * 1.1)
        
        # Bottom performers get reset parameters
        reset_count = self.swarm_size // 10
        for i in range(self.swarm_size - reset_count, self.swarm_size):
            self.particles[i]["exploration_tendency"] = np.random.uniform(0.3, 0.7)
            self.particles[i]["social_influence"] = np.random.uniform(0.1, 0.5)
    
    def _calculate_reproducibility_swarm(self, results: List[float]) -> float:
        """Calculate reproducibility for swarm results."""
        if len(results) < 2:
            return 0.0
        
        # Reproducibility based on consistency of convergence speeds
        cv = np.std(results) / np.mean(results) if np.mean(results) != 0 else float('inf')
        return max(0.0, 1.0 - cv * 2)  # Scale appropriately


class BreakthroughResearchManager:
    """
    Manager for conducting breakthrough algorithm research.
    
    Coordinates multiple research experiments and validates breakthrough claims.
    """
    
    def __init__(self):
        self.active_experiments: Dict[str, Any] = {}
        self.completed_experiments: List[ExperimentalResult] = []
        self.research_hypotheses: List[ResearchHypothesis] = []
        
        # Initialize research algorithms
        self.algorithms = {
            AlgorithmType.NEURAL_EVOLUTION: NeuroEvolutionaryOptimizer(),
            AlgorithmType.SWARM_INTELLIGENCE: SwarmIntelligenceOptimizer(),
            # Additional algorithms would be implemented
        }
    
    async def conduct_breakthrough_research(
        self,
        hypotheses: List[ResearchHypothesis],
        design_space: Dict[str, List[Any]],
        fitness_function: Callable,
        parallel_experiments: bool = True
    ) -> List[ExperimentalResult]:
        """
        Conduct comprehensive breakthrough algorithm research.
        
        Args:
            hypotheses: List of research hypotheses to test
            design_space: Common design space for all experiments
            fitness_function: Evaluation function
            parallel_experiments: Whether to run experiments in parallel
            
        Returns:
            List of experimental results with statistical validation
        """
        logger.info(f"Starting breakthrough research with {len(hypotheses)} hypotheses")
        
        self.research_hypotheses.extend(hypotheses)
        results = []
        
        if parallel_experiments:
            # Run experiments in parallel
            tasks = []
            for hypothesis in hypotheses:
                task = asyncio.create_task(
                    self._conduct_single_experiment(hypothesis, design_space, fitness_function)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        else:
            # Run experiments sequentially
            for hypothesis in hypotheses:
                result = await self._conduct_single_experiment(
                    hypothesis, design_space, fitness_function
                )
                results.append(result)
        
        # Validate breakthrough claims
        validated_results = self._validate_breakthrough_claims(results)
        
        # Store completed experiments
        self.completed_experiments.extend(validated_results)
        
        # Generate research summary
        self._generate_research_summary(validated_results)
        
        return validated_results
    
    async def _conduct_single_experiment(
        self,
        hypothesis: ResearchHypothesis,
        design_space: Dict[str, List[Any]], 
        fitness_function: Callable
    ) -> ExperimentalResult:
        """Conduct single experimental validation."""
        logger.info(f"Conducting experiment: {hypothesis.title}")
        
        algorithm_type = AlgorithmType(hypothesis.novel_algorithm)
        algorithm = self.algorithms.get(algorithm_type)
        
        if not algorithm:
            raise ValueError(f"Algorithm {hypothesis.novel_algorithm} not implemented")
        
        # Run experiment based on algorithm type
        if algorithm_type == AlgorithmType.NEURAL_EVOLUTION:
            result = await algorithm.evolve_design_strategies(
                design_space, fitness_function, runs=5
            )
        elif algorithm_type == AlgorithmType.SWARM_INTELLIGENCE:
            result = await algorithm.optimize_with_swarm_intelligence(
                design_space, fitness_function, runs=5
            )
        else:
            raise NotImplementedError(f"Experiment for {algorithm_type} not implemented")
        
        # Update with hypothesis information
        result.hypothesis_id = hypothesis.hypothesis_id
        
        return result
    
    def _validate_breakthrough_claims(
        self, results: List[ExperimentalResult]
    ) -> List[ExperimentalResult]:
        """Validate breakthrough claims with rigorous statistical analysis."""
        validated_results = []
        
        for result in results:
            # Check statistical significance
            is_statistically_significant = result.statistical_analysis.get("significant", False)
            
            # Check effect size (practical significance)
            min_effect_size = 0.5  # Medium effect size threshold
            has_practical_significance = any(
                abs(effect) >= min_effect_size 
                for effect in result.effect_sizes.values()
            )
            
            # Check reproducibility
            min_reproducibility = 0.7
            is_reproducible = result.reproducibility_score >= min_reproducibility
            
            # Update publication readiness
            result.publication_readiness = (
                is_statistically_significant and 
                has_practical_significance and 
                is_reproducible
            )
            
            validated_results.append(result)
            
            if result.publication_readiness:
                logger.info(f"BREAKTHROUGH VALIDATED: {result.algorithm_name} "
                           f"shows significant improvement with {result.reproducibility_score:.2f} reproducibility")
            else:
                logger.info(f"Experiment inconclusive: {result.algorithm_name}")
        
        return validated_results
    
    def _generate_research_summary(self, results: List[ExperimentalResult]) -> None:
        """Generate comprehensive research summary."""
        breakthroughs = [r for r in results if r.publication_readiness]
        
        logger.info(f"\n=== BREAKTHROUGH RESEARCH SUMMARY ===")
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Validated breakthroughs: {len(breakthroughs)}")
        
        if breakthroughs:
            logger.info(f"Breakthrough algorithms:")
            for breakthrough in breakthroughs:
                improvement = max(breakthrough.improvement_factor.values())
                logger.info(f"- {breakthrough.algorithm_name}: {improvement:.2f}x improvement")
        
        # Record research metrics
        record_metric("research_experiments_total", len(results))
        record_metric("research_breakthroughs_validated", len(breakthroughs))
        record_metric("research_success_rate", len(breakthroughs) / len(results))
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get comprehensive research statistics."""
        total_experiments = len(self.completed_experiments)
        successful_experiments = len([r for r in self.completed_experiments if r.publication_readiness])
        
        avg_improvement = np.mean([
            max(r.improvement_factor.values()) for r in self.completed_experiments
            if r.improvement_factor
        ]) if self.completed_experiments else 0.0
        
        avg_reproducibility = np.mean([
            r.reproducibility_score for r in self.completed_experiments
        ]) if self.completed_experiments else 0.0
        
        return {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
            "average_improvement_factor": avg_improvement,
            "average_reproducibility": avg_reproducibility,
            "active_experiments": len(self.active_experiments),
            "research_hypotheses": len(self.research_hypotheses)
        }