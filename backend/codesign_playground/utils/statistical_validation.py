"""
Statistical validation tools for AI Hardware Co-Design Playground.

This module provides comprehensive statistical validation and significance testing
for optimization results, algorithm comparisons, and experimental validation.
"""

import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    summary: str
    tests: List[StatisticalTest]
    descriptive_stats: Dict[str, Any]
    recommendations: List[str]
    confidence_level: float = 0.95
    significance_level: float = 0.05


class StatisticalValidator:
    """Framework for statistical validation of optimization results."""
    
    def __init__(self, significance_level: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical validator.
        
        Args:
            significance_level: Alpha level for hypothesis testing
            confidence_level: Confidence level for intervals
        """
        self.significance_level = significance_level
        self.confidence_level = confidence_level
        self.available_tests = {
            "t_test": self._t_test,
            "wilcoxon": self._wilcoxon_test,
            "mann_whitney": self._mann_whitney_test,
            "kruskal_wallis": self._kruskal_wallis_test,
            "friedman": self._friedman_test,
            "bootstrap": self._bootstrap_test,
            "permutation": self._permutation_test,
        }
    
    def validate_optimization_results(
        self,
        results_1: List[float],
        results_2: List[float],
        test_type: str = "auto",
        paired: bool = False
    ) -> StatisticalTest:
        """
        Validate differences between two sets of optimization results.
        
        Args:
            results_1: First set of results
            results_2: Second set of results
            test_type: Type of test ("auto", "t_test", "wilcoxon", "mann_whitney")
            paired: Whether samples are paired
            
        Returns:
            Statistical test result
        """
        if not results_1 or not results_2:
            raise ValueError("Both result sets must contain at least one value")
        
        # Auto-select test if needed
        if test_type == "auto":
            test_type = self._select_appropriate_test(results_1, results_2, paired)
        
        if test_type not in self.available_tests:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return self.available_tests[test_type](results_1, results_2, paired)
    
    def compare_multiple_algorithms(
        self,
        algorithm_results: Dict[str, List[float]],
        control_algorithm: Optional[str] = None
    ) -> ValidationReport:
        """
        Compare multiple algorithms with statistical validation.
        
        Args:
            algorithm_results: Dictionary mapping algorithm names to result lists
            control_algorithm: Name of control algorithm for pairwise comparisons
            
        Returns:
            Comprehensive validation report
        """
        if len(algorithm_results) < 2:
            raise ValueError("Need at least 2 algorithms for comparison")
        
        tests = []
        descriptive_stats = {}
        
        # Calculate descriptive statistics for each algorithm
        for alg_name, results in algorithm_results.items():
            descriptive_stats[alg_name] = self._calculate_descriptive_stats(results)
        
        # Overall comparison test (Kruskal-Wallis)
        all_results = list(algorithm_results.values())
        overall_test = self._kruskal_wallis_test(all_results, paired=False)
        overall_test.test_name = "Overall Algorithm Comparison (Kruskal-Wallis)"
        tests.append(overall_test)
        
        # Pairwise comparisons
        if control_algorithm and control_algorithm in algorithm_results:
            control_results = algorithm_results[control_algorithm]
            for alg_name, results in algorithm_results.items():
                if alg_name != control_algorithm:
                    pairwise_test = self._mann_whitney_test(control_results, results, False)
                    pairwise_test.test_name = f"{control_algorithm} vs {alg_name}"
                    tests.append(pairwise_test)
        else:
            # All pairwise comparisons
            alg_names = list(algorithm_results.keys())
            for i in range(len(alg_names)):
                for j in range(i + 1, len(alg_names)):
                    alg1, alg2 = alg_names[i], alg_names[j]
                    pairwise_test = self._mann_whitney_test(
                        algorithm_results[alg1], algorithm_results[alg2], False
                    )
                    pairwise_test.test_name = f"{alg1} vs {alg2}"
                    tests.append(pairwise_test)
        
        # Generate report
        summary = self._generate_comparison_summary(descriptive_stats, tests)
        recommendations = self._generate_recommendations(descriptive_stats, tests)
        
        return ValidationReport(
            summary=summary,
            tests=tests,
            descriptive_stats=descriptive_stats,
            recommendations=recommendations,
            confidence_level=self.confidence_level,
            significance_level=self.significance_level
        )
    
    def validate_convergence(
        self,
        convergence_history: List[float],
        window_size: int = 10,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Validate algorithm convergence with statistical tests.
        
        Args:
            convergence_history: List of objective values over iterations
            window_size: Window size for convergence testing
            tolerance: Convergence tolerance
            
        Returns:
            Convergence validation results
        """
        if len(convergence_history) < window_size * 2:
            return {
                "converged": False,
                "reason": "Insufficient data for convergence testing",
                "convergence_iteration": -1
            }
        
        convergence_iteration = -1
        convergence_tests = []
        
        # Sliding window convergence test
        for i in range(window_size, len(convergence_history) - window_size):
            window1 = convergence_history[i-window_size:i]
            window2 = convergence_history[i:i+window_size]
            
            # Test for significant difference between windows
            test_result = self._t_test(window1, window2, paired=False)
            
            # Check variance within windows
            var1 = statistics.variance(window1) if len(window1) > 1 else 0
            var2 = statistics.variance(window2) if len(window2) > 1 else 0
            
            # Convergence criteria
            converged = (
                not test_result.significant and  # No significant difference
                abs(statistics.mean(window2) - statistics.mean(window1)) < tolerance and  # Small change
                max(var1, var2) < tolerance  # Low variance
            )
            
            if converged and convergence_iteration == -1:
                convergence_iteration = i
            
            convergence_tests.append({
                "iteration": i,
                "converged": converged,
                "p_value": test_result.p_value,
                "mean_diff": abs(statistics.mean(window2) - statistics.mean(window1)),
                "max_variance": max(var1, var2)
            })
        
        # Calculate convergence rate
        if convergence_iteration > 0:
            initial_value = convergence_history[0]
            converged_value = statistics.mean(convergence_history[convergence_iteration:convergence_iteration + window_size])
            convergence_rate = abs(converged_value - initial_value) / convergence_iteration
        else:
            convergence_rate = 0
        
        return {
            "converged": convergence_iteration > 0,
            "convergence_iteration": convergence_iteration,
            "convergence_rate": convergence_rate,
            "final_value": convergence_history[-1],
            "convergence_tests": convergence_tests,
            "improvement": convergence_history[-1] - convergence_history[0]
        }
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_func: callable = statistics.mean,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Confidence interval (lower, upper)
        """
        import random
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            bootstrap_sample = [random.choice(data) for _ in range(len(data))]
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        bootstrap_stats.sort()
        n = len(bootstrap_stats)
        
        lower_idx = int(lower_percentile / 100 * n)
        upper_idx = int(upper_percentile / 100 * n)
        
        return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]
    
    def effect_size_analysis(
        self,
        group1: List[float],
        group2: List[float]
    ) -> Dict[str, float]:
        """
        Calculate effect size measures.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Dictionary of effect size measures
        """
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        std1 = statistics.stdev(group1) if len(group1) > 1 else 0
        std2 = statistics.stdev(group2) if len(group2) > 1 else 0
        
        # Cohen's d
        pooled_std = math.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                              (len(group1) + len(group2) - 2))
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        # Glass's delta
        glass_delta = (mean2 - mean1) / std1 if std1 > 0 else 0
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(group1) + len(group2)) - 9))
        hedges_g = cohens_d * correction_factor
        
        # Cliff's delta (non-parametric effect size)
        cliffs_delta = self._calculate_cliffs_delta(group1, group2)
        
        return {
            "cohens_d": cohens_d,
            "glass_delta": glass_delta,
            "hedges_g": hedges_g,
            "cliffs_delta": cliffs_delta,
            "mean_difference": mean2 - mean1,
            "percent_improvement": ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
        }
    
    def _select_appropriate_test(
        self,
        data1: List[float],
        data2: List[float],
        paired: bool
    ) -> str:
        """Select appropriate statistical test based on data characteristics."""
        # Check normality (simplified)
        n1, n2 = len(data1), len(data2)
        
        # For small samples or non-parametric situations, use non-parametric tests
        if n1 < 30 or n2 < 30:
            return "wilcoxon" if paired else "mann_whitney"
        
        # For larger samples, could use t-test if assumptions are met
        # For simplicity, we'll use non-parametric tests as they're more robust
        return "wilcoxon" if paired else "mann_whitney"
    
    def _t_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Perform t-test (requires scipy)."""
        try:
            from scipy import stats
            
            if paired:
                statistic, p_value = stats.ttest_rel(data1, data2)
                test_name = "Paired t-test"
            else:
                statistic, p_value = stats.ttest_ind(data1, data2)
                test_name = "Independent t-test"
            
            effect_size_results = self.effect_size_analysis(data1, data2)
            
            return StatisticalTest(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                effect_size=effect_size_results["cohens_d"],
                interpretation=self._interpret_p_value(p_value)
            )
            
        except ImportError:
            # Fallback implementation
            return self._manual_t_test(data1, data2, paired)
    
    def _wilcoxon_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Perform Wilcoxon test (requires scipy)."""
        try:
            from scipy import stats
            
            if paired:
                statistic, p_value = stats.wilcoxon(data1, data2)
                test_name = "Wilcoxon signed-rank test"
            else:
                # Use Mann-Whitney for unpaired data
                return self._mann_whitney_test(data1, data2, paired)
            
            effect_size_results = self.effect_size_analysis(data1, data2)
            
            return StatisticalTest(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                effect_size=effect_size_results["cliffs_delta"],
                interpretation=self._interpret_p_value(p_value)
            )
            
        except ImportError:
            return self._manual_wilcoxon_test(data1, data2)
    
    def _mann_whitney_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Perform Mann-Whitney U test (requires scipy)."""
        try:
            from scipy import stats
            
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            effect_size_results = self.effect_size_analysis(data1, data2)
            
            return StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                effect_size=effect_size_results["cliffs_delta"],
                interpretation=self._interpret_p_value(p_value)
            )
            
        except ImportError:
            return self._manual_mann_whitney_test(data1, data2)
    
    def _kruskal_wallis_test(self, data_groups: List[List[float]], paired: bool) -> StatisticalTest:
        """Perform Kruskal-Wallis test (requires scipy)."""
        try:
            from scipy import stats
            
            statistic, p_value = stats.kruskal(*data_groups)
            
            return StatisticalTest(
                test_name="Kruskal-Wallis test",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                interpretation=self._interpret_p_value(p_value)
            )
            
        except ImportError:
            return self._manual_kruskal_wallis_test(data_groups)
    
    def _friedman_test(self, data_groups: List[List[float]], paired: bool) -> StatisticalTest:
        """Perform Friedman test (requires scipy)."""
        try:
            from scipy import stats
            
            # Friedman test requires matched groups
            statistic, p_value = stats.friedmanchisquare(*data_groups)
            
            return StatisticalTest(
                test_name="Friedman test",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                interpretation=self._interpret_p_value(p_value)
            )
            
        except ImportError:
            logger.warning("Scipy not available, using Kruskal-Wallis instead")
            return self._kruskal_wallis_test(data_groups, paired)
    
    def _bootstrap_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Perform bootstrap test."""
        import random
        
        n_bootstrap = 1000
        observed_diff = statistics.mean(data2) - statistics.mean(data1)
        
        # Combine data for null hypothesis
        combined_data = data1 + data2
        n1, n2 = len(data1), len(data2)
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            random.shuffle(combined_data)
            bootstrap_group1 = combined_data[:n1]
            bootstrap_group2 = combined_data[n1:n1+n2]
            bootstrap_diff = statistics.mean(bootstrap_group2) - statistics.mean(bootstrap_group1)
            bootstrap_diffs.append(bootstrap_diff)
        
        # Calculate p-value
        extreme_count = sum(1 for diff in bootstrap_diffs if abs(diff) >= abs(observed_diff))
        p_value = extreme_count / n_bootstrap
        
        return StatisticalTest(
            test_name="Bootstrap test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _permutation_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Perform permutation test."""
        import random
        from itertools import combinations
        
        n_permutations = min(1000, math.factorial(len(data1) + len(data2)) // (math.factorial(len(data1)) * math.factorial(len(data2))))
        observed_diff = statistics.mean(data2) - statistics.mean(data1)
        
        combined_data = data1 + data2
        n1 = len(data1)
        
        permutation_diffs = []
        for _ in range(n_permutations):
            random.shuffle(combined_data)
            perm_group1 = combined_data[:n1]
            perm_group2 = combined_data[n1:]
            perm_diff = statistics.mean(perm_group2) - statistics.mean(perm_group1)
            permutation_diffs.append(perm_diff)
        
        # Calculate p-value
        extreme_count = sum(1 for diff in permutation_diffs if abs(diff) >= abs(observed_diff))
        p_value = extreme_count / n_permutations
        
        return StatisticalTest(
            test_name="Permutation test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _manual_t_test(self, data1: List[float], data2: List[float], paired: bool) -> StatisticalTest:
        """Manual implementation of t-test."""
        if paired and len(data1) != len(data2):
            raise ValueError("Paired test requires equal sample sizes")
        
        if paired:
            differences = [d2 - d1 for d1, d2 in zip(data1, data2)]
            mean_diff = statistics.mean(differences)
            std_diff = statistics.stdev(differences) if len(differences) > 1 else 0
            t_stat = mean_diff / (std_diff / math.sqrt(len(differences))) if std_diff > 0 else 0
            df = len(differences) - 1
        else:
            mean1, mean2 = statistics.mean(data1), statistics.mean(data2)
            var1 = statistics.variance(data1) if len(data1) > 1 else 0
            var2 = statistics.variance(data2) if len(data2) > 1 else 0
            n1, n2 = len(data1), len(data2)
            
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = math.sqrt(pooled_var * (1/n1 + 1/n2))
            t_stat = (mean2 - mean1) / se if se > 0 else 0
            df = n1 + n2 - 2
        
        # Approximate p-value using normal distribution for large samples
        if df > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            # For small samples, approximate using normal (not ideal but functional)
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return StatisticalTest(
            test_name="Manual t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        # Using approximation for standard normal CDF
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(group1), len(group2)
        
        greater_count = 0
        equal_count = 0
        
        for x1 in group1:
            for x2 in group2:
                if x2 > x1:
                    greater_count += 1
                elif x2 == x1:
                    equal_count += 1
        
        return (greater_count - (n1 * n2 - greater_count - equal_count)) / (n1 * n2)
    
    def _calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics."""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        n = len(data)
        
        stats = {
            "count": n,
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "std": statistics.stdev(data) if n > 1 else 0,
            "var": statistics.variance(data) if n > 1 else 0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
        }
        
        # Quantiles
        if n >= 4:
            quantiles = statistics.quantiles(data, n=4)
            stats["q1"] = quantiles[0]
            stats["q3"] = quantiles[2]
            stats["iqr"] = quantiles[2] - quantiles[0]
        else:
            stats["q1"] = stats["min"]
            stats["q3"] = stats["max"]
            stats["iqr"] = stats["range"]
        
        return stats
    
    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value."""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.1:
            return "Marginally significant (p < 0.1)"
        else:
            return "Not significant (p >= 0.1)"
    
    def _generate_comparison_summary(
        self, 
        descriptive_stats: Dict[str, Dict[str, float]], 
        tests: List[StatisticalTest]
    ) -> str:
        """Generate summary of algorithm comparison."""
        # Find best performing algorithm
        best_alg = max(descriptive_stats.keys(), key=lambda k: descriptive_stats[k]["mean"])
        best_mean = descriptive_stats[best_alg]["mean"]
        
        # Count significant differences
        significant_tests = sum(1 for test in tests if test.significant)
        total_tests = len(tests)
        
        summary = f"Best performing algorithm: {best_alg} (mean = {best_mean:.4f}). "
        summary += f"Found {significant_tests}/{total_tests} statistically significant differences. "
        
        if significant_tests > 0:
            summary += "Results suggest meaningful performance differences between algorithms."
        else:
            summary += "No significant performance differences detected."
        
        return summary
    
    def _generate_recommendations(
        self, 
        descriptive_stats: Dict[str, Dict[str, float]], 
        tests: List[StatisticalTest]
    ) -> List[str]:
        """Generate recommendations based on statistical analysis."""
        recommendations = []
        
        # Performance ranking
        ranking = sorted(descriptive_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
        best_alg = ranking[0][0]
        
        recommendations.append(f"Recommended algorithm: {best_alg} based on mean performance")
        
        # Significance recommendations
        significant_tests = [test for test in tests if test.significant]
        if len(significant_tests) > 0:
            recommendations.append("Significant differences detected - algorithm choice matters")
        else:
            recommendations.append("No significant differences - consider implementation complexity")
        
        # Variance analysis
        variances = {alg: stats["std"] for alg, stats in descriptive_stats.items()}
        most_stable = min(variances.keys(), key=lambda k: variances[k])
        recommendations.append(f"Most stable algorithm: {most_stable} (lowest std deviation)")
        
        # Sample size recommendations
        min_samples = min(stats["count"] for stats in descriptive_stats.values())
        if min_samples < 30:
            recommendations.append("Consider increasing sample size for more reliable statistical conclusions")
        
        return recommendations
    
    def _manual_mann_whitney_test(self, data1: List[float], data2: List[float]) -> StatisticalTest:
        """Manual implementation of Mann-Whitney U test."""
        # Simplified implementation
        n1, n2 = len(data1), len(data2)
        combined = [(val, 1) for val in data1] + [(val, 2) for val in data2]
        combined.sort()
        
        # Assign ranks
        ranks1 = []
        for i, (val, group) in enumerate(combined):
            if group == 1:
                ranks1.append(i + 1)
        
        U1 = sum(ranks1) - n1 * (n1 + 1) / 2
        U2 = n1 * n2 - U1
        U = min(U1, U2)
        
        # Approximate p-value for large samples
        mean_U = n1 * n2 / 2
        std_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mean_U) / std_U if std_U > 0 else 0
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        
        return StatisticalTest(
            test_name="Manual Mann-Whitney U test",
            statistic=U,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _manual_wilcoxon_test(self, data1: List[float], data2: List[float]) -> StatisticalTest:
        """Manual implementation of Wilcoxon test."""
        if len(data1) != len(data2):
            raise ValueError("Wilcoxon test requires paired data of equal length")
        
        differences = [d2 - d1 for d1, d2 in zip(data1, data2)]
        abs_diffs = [(abs(diff), diff > 0) for diff in differences if diff != 0]
        abs_diffs.sort()
        
        # Assign ranks
        signed_ranks = []
        for i, (abs_diff, is_positive) in enumerate(abs_diffs):
            rank = i + 1
            signed_ranks.append(rank if is_positive else -rank)
        
        W = sum(rank for rank in signed_ranks if rank > 0)
        n = len(signed_ranks)
        
        if n > 0:
            mean_W = n * (n + 1) / 4
            std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (W - mean_W) / std_W if std_W > 0 else 0
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            p_value = 1.0
            W = 0
        
        return StatisticalTest(
            test_name="Manual Wilcoxon test",
            statistic=W,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _manual_kruskal_wallis_test(self, data_groups: List[List[float]]) -> StatisticalTest:
        """Manual implementation of Kruskal-Wallis test."""
        # Combine all data with group labels
        combined = []
        for group_idx, group_data in enumerate(data_groups):
            for value in group_data:
                combined.append((value, group_idx))
        
        combined.sort()
        n = len(combined)
        
        # Assign ranks
        ranks = [i + 1 for i in range(n)]
        
        # Calculate rank sums for each group
        rank_sums = [0] * len(data_groups)
        for i, (value, group_idx) in enumerate(combined):
            rank_sums[group_idx] += ranks[i]
        
        # Kruskal-Wallis H statistic
        H = 0
        for i, group_data in enumerate(data_groups):
            ni = len(group_data)
            if ni > 0:
                H += (rank_sums[i] ** 2) / ni
        
        H = (12 / (n * (n + 1))) * H - 3 * (n + 1)
        
        # Approximate p-value using chi-square distribution
        df = len(data_groups) - 1
        # Simplified p-value approximation
        p_value = 1 - self._chi_square_cdf(H, df) if H > 0 else 1.0
        
        return StatisticalTest(
            test_name="Manual Kruskal-Wallis test",
            statistic=H,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _chi_square_cdf(self, x: float, df: int) -> float:
        """Approximate chi-square CDF (simplified)."""
        if x <= 0:
            return 0
        
        # Very rough approximation for demonstration
        # In practice, use scipy.stats.chi2.cdf
        if df == 1:
            return 2 * self._normal_cdf(math.sqrt(x)) - 1
        elif df == 2:
            return 1 - math.exp(-x / 2)
        else:
            # Rough normal approximation for large df
            z = (x - df) / math.sqrt(2 * df)
            return self._normal_cdf(z)


def validate_algorithm_performance(
    algorithm_results: Dict[str, List[float]],
    significance_level: float = 0.05
) -> ValidationReport:
    """
    High-level function to validate algorithm performance.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to performance results
        significance_level: Statistical significance level
        
    Returns:
        Comprehensive validation report
    """
    validator = StatisticalValidator(significance_level=significance_level)
    return validator.compare_multiple_algorithms(algorithm_results)


def test_convergence(convergence_data: List[float], tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Test algorithm convergence.
    
    Args:
        convergence_data: List of objective values over iterations
        tolerance: Convergence tolerance
        
    Returns:
        Convergence test results
    """
    validator = StatisticalValidator()
    return validator.validate_convergence(convergence_data, tolerance=tolerance)