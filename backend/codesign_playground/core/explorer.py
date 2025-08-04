"""
Design space exploration for hardware-software co-design.

This module provides tools for exploring the design space of neural network models
and hardware accelerators to find optimal configurations.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .accelerator import AcceleratorDesigner, Accelerator, ModelProfile
from .optimizer import ModelOptimizer


@dataclass
class DesignPoint:
    """Single point in the design space."""
    
    config: Dict[str, Any]
    metrics: Dict[str, float]
    accelerator: Optional[Accelerator] = None
    model: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert design point to dictionary."""
        result = {
            "config": self.config,
            "metrics": self.metrics,
        }
        if self.accelerator:
            result["accelerator"] = self.accelerator.to_dict()
        return result


@dataclass
class DesignSpaceResult:
    """Results from design space exploration."""
    
    design_points: List[DesignPoint]
    pareto_frontier: List[DesignPoint]
    best_designs: Dict[str, DesignPoint]
    exploration_time: float
    total_evaluations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "design_points": [dp.to_dict() for dp in self.design_points],
            "pareto_frontier": [dp.to_dict() for dp in self.pareto_frontier],
            "best_designs": {k: v.to_dict() for k, v in self.best_designs.items()},
            "exploration_time": self.exploration_time,
            "total_evaluations": self.total_evaluations,
        }
    
    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DesignSpaceExplorer:
    """Design space exploration for hardware-software co-design."""
    
    def __init__(self, parallel_workers: int = 4):
        """
        Initialize design space explorer.
        
        Args:
            parallel_workers: Number of parallel evaluation workers
        """
        self.parallel_workers = parallel_workers
        self.designer = AcceleratorDesigner()
        self.evaluation_cache = {}
    
    def explore(
        self,
        model: Any,
        design_space: Dict[str, List[Any]],
        objectives: List[str] = ["latency", "power", "area"],
        num_samples: int = 100,
        strategy: str = "random",
        constraints: Optional[Dict[str, float]] = None
    ) -> DesignSpaceResult:
        """
        Explore the design space for optimal configurations.
        
        Args:
            model: Neural network model to optimize for
            design_space: Dictionary defining design space parameters
            objectives: Optimization objectives
            num_samples: Number of design points to evaluate
            strategy: Exploration strategy ("random", "grid", "evolutionary")
            constraints: Hard constraints on design parameters
            
        Returns:
            DesignSpaceResult with exploration results
        """
        import time
        start_time = time.time()
        
        # Generate design points based on strategy
        if strategy == "random":
            design_configs = self._generate_random_samples(design_space, num_samples)
        elif strategy == "grid":
            design_configs = self._generate_grid_samples(design_space, num_samples)
        elif strategy == "evolutionary":
            design_configs = self._generate_evolutionary_samples(design_space, num_samples, model)
        else:
            raise ValueError(f"Unknown exploration strategy: {strategy}")
        
        # Filter by constraints
        if constraints:
            design_configs = self._apply_constraints(design_configs, constraints)
        
        # Evaluate design points in parallel
        design_points = self._evaluate_designs_parallel(model, design_configs, objectives)
        
        # Compute Pareto frontier
        pareto_frontier = self._compute_pareto_frontier(design_points, objectives)
        
        # Find best designs for each objective
        best_designs = self._find_best_designs(design_points, objectives)
        
        exploration_time = time.time() - start_time
        
        return DesignSpaceResult(
            design_points=design_points,
            pareto_frontier=pareto_frontier,
            best_designs=best_designs,
            exploration_time=exploration_time,
            total_evaluations=len(design_points),
        )
    
    def plot_pareto(
        self, 
        results: DesignSpaceResult, 
        objectives: Optional[List[str]] = None,
        save_to: Optional[str] = None
    ) -> None:
        """
        Plot Pareto frontier visualization.
        
        Args:
            results: Design space exploration results
            objectives: Objectives to plot (defaults to first 2)
            save_to: Optional file path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Visualization libraries not available. Install matplotlib and plotly.")
            return
        
        if not objectives:
            objectives = list(results.design_points[0].metrics.keys())[:2]
        
        if len(objectives) < 2:
            print("Need at least 2 objectives for Pareto plot")
            return
        
        # Extract data for plotting
        all_points_x = [dp.metrics[objectives[0]] for dp in results.design_points]
        all_points_y = [dp.metrics[objectives[1]] for dp in results.design_points]
        
        pareto_x = [dp.metrics[objectives[0]] for dp in results.pareto_frontier]
        pareto_y = [dp.metrics[objectives[1]] for dp in results.pareto_frontier]
        
        # Create interactive Plotly plot
        fig = go.Figure()
        
        # All design points
        fig.add_trace(go.Scatter(
            x=all_points_x,
            y=all_points_y,
            mode='markers',
            name='All Designs',
            marker=dict(size=6, opacity=0.6, color='lightblue'),
            hovertemplate=f"{objectives[0]}: %{{x}}<br>{objectives[1]}: %{{y}}<extra></extra>"
        ))
        
        # Pareto frontier
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode='markers+lines',
            name='Pareto Frontier',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2),
            hovertemplate=f"Pareto: {objectives[0]}: %{{x}}<br>{objectives[1]}: %{{y}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Design Space Exploration: {objectives[0]} vs {objectives[1]}",
            xaxis_title=objectives[0].title(),
            yaxis_title=objectives[1].title(),
            hovermode='closest'
        )
        
        if save_to:
            fig.write_html(save_to)
            print(f"Pareto plot saved to {save_to}")
        else:
            fig.show()
    
    def _generate_random_samples(
        self, 
        design_space: Dict[str, List[Any]], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate random samples from design space."""
        samples = []
        for _ in range(num_samples):
            sample = {}
            for param, values in design_space.items():
                sample[param] = random.choice(values)
            samples.append(sample)
        return samples
    
    def _generate_grid_samples(
        self,
        design_space: Dict[str, List[Any]],
        max_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate grid samples from design space."""
        from itertools import product
        
        # Calculate grid resolution to stay under max_samples
        param_names = list(design_space.keys())
        param_values = list(design_space.values())
        
        total_combinations = np.prod([len(values) for values in param_values])
        
        if total_combinations <= max_samples:
            # Full grid
            samples = []
            for combination in product(*param_values):
                sample = dict(zip(param_names, combination))
                samples.append(sample)
        else:
            # Subsample grid
            samples = []
            step_sizes = [max(1, len(values) // int(max_samples**(1/len(param_values))))
                         for values in param_values]
            
            subsampled_values = [
                values[::step] for values, step in zip(param_values, step_sizes)
            ]
            
            for combination in product(*subsampled_values):
                sample = dict(zip(param_names, combination))
                samples.append(sample)
        
        return samples[:max_samples]
    
    def _generate_evolutionary_samples(
        self,
        design_space: Dict[str, List[Any]],
        num_samples: int,
        model: Any
    ) -> List[Dict[str, Any]]:
        """Generate samples using evolutionary algorithm."""
        population_size = min(20, num_samples // 5)
        generations = num_samples // population_size
        
        # Initialize random population
        population = self._generate_random_samples(design_space, population_size)
        
        all_samples = population.copy()
        
        for generation in range(generations):
            # Evaluate population (simplified)
            scores = []
            for config in population:
                accelerator = self._config_to_accelerator(config)
                performance = accelerator.estimate_performance()
                # Simple fitness: balance performance and efficiency
                score = performance["efficiency_ops_w"]
                scores.append(score)
            
            # Selection: keep top 50%
            sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            survivors = [config for config, _ in sorted_pop[:population_size//2]]
            
            # Crossover and mutation
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = self._crossover(parent1, parent2, design_space)
                child = self._mutate(child, design_space)
                new_population.append(child)
            
            population = new_population
            all_samples.extend(population)
        
        return all_samples[:num_samples]
    
    def _apply_constraints(
        self, 
        configs: List[Dict[str, Any]], 
        constraints: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Apply hard constraints to filter configurations."""
        filtered = []
        for config in configs:
            valid = True
            for param, limit in constraints.items():
                if param in config:
                    if isinstance(limit, tuple):  # Range constraint
                        if not (limit[0] <= config[param] <= limit[1]):
                            valid = False
                            break
                    else:  # Upper bound constraint
                        if config[param] > limit:
                            valid = False
                            break
            if valid:
                filtered.append(config)
        return filtered
    
    def _evaluate_designs_parallel(
        self,
        model: Any,
        configs: List[Dict[str, Any]],
        objectives: List[str]
    ) -> List[DesignPoint]:
        """Evaluate design configurations in parallel."""
        design_points = []
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit evaluation tasks
            future_to_config = {
                executor.submit(self._evaluate_single_design, model, config, objectives): config
                for config in configs
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    design_point = future.result()
                    design_points.append(design_point)
                except Exception as exc:
                    print(f"Design evaluation failed for {config}: {exc}")
        
        return design_points
    
    def _evaluate_single_design(
        self, 
        model: Any, 
        config: Dict[str, Any], 
        objectives: List[str]
    ) -> DesignPoint:
        """Evaluate a single design configuration."""
        # Create accelerator from config
        accelerator = self._config_to_accelerator(config)
        
        # Get performance metrics
        performance = accelerator.estimate_performance()
        
        # Create model optimizer for this configuration
        optimizer = ModelOptimizer(model, accelerator)
        
        # Extract metrics for objectives
        metrics = {}
        for objective in objectives:
            if objective == "latency":
                metrics["latency"] = performance["latency_ms"]
            elif objective == "power":
                metrics["power"] = performance["power_w"]
            elif objective == "area":
                metrics["area"] = performance["area_mm2"]
            elif objective == "throughput":
                metrics["throughput"] = performance["throughput_ops_s"]
            elif objective == "efficiency":
                metrics["efficiency"] = performance["efficiency_ops_w"]
            else:
                # Default: try to get from performance dict
                metrics[objective] = performance.get(objective, 0.0)
        
        return DesignPoint(
            config=config,
            metrics=metrics,
            accelerator=accelerator,
            model=model
        )
    
    def _config_to_accelerator(self, config: Dict[str, Any]) -> Accelerator:
        """Convert configuration dictionary to Accelerator object."""
        return self.designer.design(
            compute_units=config.get("compute_units", 64),
            memory_hierarchy=config.get("memory_hierarchy", ["sram_64kb", "dram"]),
            dataflow=config.get("dataflow", "weight_stationary"),
            frequency_mhz=config.get("frequency_mhz", 200.0),
            data_width=config.get("data_width", 8),
            precision=config.get("precision", "int8"),
            power_budget_w=config.get("power_budget_w", 5.0),
        )
    
    def _compute_pareto_frontier(
        self, 
        design_points: List[DesignPoint], 
        objectives: List[str]
    ) -> List[DesignPoint]:
        """Compute Pareto frontier from design points."""
        pareto_points = []
        
        for i, point1 in enumerate(design_points):
            is_pareto = True
            
            for j, point2 in enumerate(design_points):
                if i == j:
                    continue
                
                # Check if point2 dominates point1
                dominates = True
                for obj in objectives:
                    # Assume lower is better for all objectives
                    if point2.metrics[obj] >= point1.metrics[obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_points.append(point1)
        
        # Sort Pareto points by first objective
        pareto_points.sort(key=lambda p: p.metrics[objectives[0]])
        return pareto_points
    
    def _find_best_designs(
        self, 
        design_points: List[DesignPoint], 
        objectives: List[str]
    ) -> Dict[str, DesignPoint]:
        """Find best design for each individual objective."""
        best_designs = {}
        
        for objective in objectives:
            best_point = min(design_points, key=lambda p: p.metrics[objective])
            best_designs[f"best_{objective}"] = best_point
        
        return best_designs
    
    def _crossover(
        self, 
        parent1: Dict[str, Any], 
        parent2: Dict[str, Any],
        design_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Crossover operation for evolutionary algorithm."""
        child = {}
        for param in design_space.keys():
            # Random choice between parents
            child[param] = random.choice([parent1[param], parent2[param]])
        return child
    
    def _mutate(
        self, 
        config: Dict[str, Any], 
        design_space: Dict[str, List[Any]],
        mutation_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Mutation operation for evolutionary algorithm."""
        mutated = config.copy()
        
        for param, values in design_space.items():
            if random.random() < mutation_rate:
                mutated[param] = random.choice(values)
        
        return mutated