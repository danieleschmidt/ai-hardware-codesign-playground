"""
Adaptive learning and self-improving patterns for AI Hardware Co-Design Playground.

This module provides machine learning-based optimization, pattern recognition,
and adaptive behavior that improves system performance over time.
"""

import json
import time
import random
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

from ..utils.monitoring import record_metric, monitor_function

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for adaptive systems."""
    CONSERVATIVE = "conservative"  # Gradual adaptation
    BALANCED = "balanced"         # Moderate adaptation
    AGGRESSIVE = "aggressive"     # Rapid adaptation
    EXPLORATION = "exploration"   # Focus on discovering new patterns


@dataclass
class LearningOutcome:
    """Result of a learning episode."""
    
    timestamp: float
    context: Dict[str, Any]
    action_taken: str
    reward: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp,
            "context": self.context,
            "action_taken": self.action_taken,
            "reward": self.reward,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class AdaptationRule:
    """Rule for adaptive behavior."""
    
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    priority: int = 5
    success_count: int = 0
    failure_count: int = 0
    last_applied: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of this rule."""
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)
    
    @property
    def confidence(self) -> float:
        """Calculate confidence in this rule."""
        total = self.success_count + self.failure_count
        if total < 5:
            return 0.5  # Low confidence with insufficient data
        return min(0.95, self.success_rate + (total / 100) * 0.1)


class PatternRecognizer:
    """Recognizes patterns in system behavior and usage."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize pattern recognizer.
        
        Args:
            window_size: Size of sliding window for pattern analysis
        """
        self.window_size = window_size
        self._event_history = deque(maxlen=window_size)
        self._patterns = defaultdict(list)
        self._pattern_weights = defaultdict(float)
        
        logger.info(f"Initialized PatternRecognizer with window size {window_size}")
    
    def record_event(self, event_type: str, context: Dict[str, Any], outcome: float) -> None:
        """
        Record an event for pattern analysis.
        
        Args:
            event_type: Type of event (e.g., 'model_optimization', 'design_exploration')
            context: Context information (parameters, environment, etc.)
            outcome: Outcome score (performance, success rate, etc.)
        """
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "context": context,
            "outcome": outcome
        }
        
        self._event_history.append(event)
        self._analyze_patterns()
        
        record_metric("pattern_event_recorded", 1, "counter", {"event_type": event_type})
    
    def get_pattern_prediction(self, event_type: str, context: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict outcome based on recognized patterns.
        
        Args:
            event_type: Type of event
            context: Current context
            
        Returns:
            Tuple of (predicted_outcome, confidence)
        """
        pattern_key = self._create_pattern_key(event_type, context)
        
        if pattern_key in self._patterns:
            outcomes = self._patterns[pattern_key]
            weight = self._pattern_weights[pattern_key]
            
            if len(outcomes) >= 3:
                predicted_outcome = statistics.mean(outcomes[-5:])  # Last 5 outcomes
                confidence = min(0.95, weight / 10.0)  # Confidence based on pattern strength
                return predicted_outcome, confidence
        
        # No pattern found, return neutral prediction
        return 0.5, 0.1
    
    def get_optimization_suggestions(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions based on learned patterns.
        
        Args:
            event_type: Type of event to optimize
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze patterns for this event type
        event_patterns = {}
        for event in self._event_history:
            if event["type"] == event_type:
                key = self._create_pattern_key(event_type, event["context"])
                if key not in event_patterns:
                    event_patterns[key] = []
                event_patterns[key].append(event["outcome"])
        
        # Find best performing patterns
        best_patterns = sorted(
            event_patterns.items(),
            key=lambda x: statistics.mean(x[1]) if x[1] else 0,
            reverse=True
        )[:3]
        
        for pattern_key, outcomes in best_patterns:
            if len(outcomes) >= 2:
                avg_outcome = statistics.mean(outcomes)
                if avg_outcome > 0.7:  # Only suggest high-performing patterns
                    # Extract context from pattern key
                    context = self._decode_pattern_key(pattern_key)
                    suggestions.append({
                        "type": "pattern_optimization",
                        "context": context,
                        "expected_outcome": avg_outcome,
                        "confidence": min(0.9, len(outcomes) / 10.0),
                        "sample_size": len(outcomes)
                    })
        
        return suggestions
    
    def _analyze_patterns(self) -> None:
        """Analyze event history to identify patterns."""
        if len(self._event_history) < 10:
            return
        
        # Group events by type and context patterns
        for event in list(self._event_history)[-50:]:  # Analyze recent events
            pattern_key = self._create_pattern_key(event["type"], event["context"])
            
            self._patterns[pattern_key].append(event["outcome"])
            self._pattern_weights[pattern_key] += 1.0
            
            # Keep pattern history manageable
            if len(self._patterns[pattern_key]) > 20:
                self._patterns[pattern_key] = self._patterns[pattern_key][-15:]
    
    def _create_pattern_key(self, event_type: str, context: Dict[str, Any]) -> str:
        """Create a pattern key from event type and context."""
        # Simplify context to key parameters for pattern matching
        key_params = []
        
        for key, value in sorted(context.items()):
            if isinstance(value, (int, float)):
                # Discretize numeric values for pattern matching
                if isinstance(value, float):
                    discretized = round(value, 1)
                else:
                    discretized = value
                key_params.append(f"{key}:{discretized}")
            elif isinstance(value, str) and len(value) < 20:
                key_params.append(f"{key}:{value}")
        
        return f"{event_type}|{'|'.join(key_params[:5])}"  # Limit to 5 key params
    
    def _decode_pattern_key(self, pattern_key: str) -> Dict[str, Any]:
        """Decode pattern key back to context dictionary."""
        parts = pattern_key.split("|")
        context = {}
        
        for part in parts[1:]:  # Skip event type
            if ":" in part:
                key, value = part.split(":", 1)
                # Try to convert back to original type
                try:
                    if "." in value:
                        context[key] = float(value)
                    else:
                        context[key] = int(value)
                except ValueError:
                    context[key] = value
        
        return context


class AdaptiveBehaviorEngine:
    """Engine for adaptive behavior based on learned patterns."""
    
    def __init__(self, learning_mode: LearningMode = LearningMode.BALANCED):
        """
        Initialize adaptive behavior engine.
        
        Args:
            learning_mode: Mode for learning and adaptation
        """
        self.learning_mode = learning_mode
        self._adaptation_rules: List[AdaptationRule] = []
        self._learning_history = deque(maxlen=1000)
        self._context_memory = defaultdict(lambda: deque(maxlen=100))
        
        # Learning parameters
        self._exploration_rate = self._get_exploration_rate()
        self._learning_rate = self._get_learning_rate()
        
        # Load default adaptation rules
        self._load_default_rules()
        
        logger.info(f"Initialized AdaptiveBehaviorEngine with mode: {learning_mode.value}")
    
    def adapt_behavior(self, context: Dict[str, Any], available_actions: List[str]) -> Tuple[str, float]:
        """
        Select adaptive behavior based on context and learned patterns.
        
        Args:
            context: Current context/state
            available_actions: List of possible actions
            
        Returns:
            Tuple of (selected_action, confidence)
        """
        # Check if any adaptation rules apply
        applicable_rules = [
            rule for rule in self._adaptation_rules
            if rule.condition(context)
        ]
        
        if applicable_rules:
            # Sort by priority and confidence
            applicable_rules.sort(key=lambda r: (r.priority, r.confidence), reverse=True)
            best_rule = applicable_rules[0]
            
            try:
                action = best_rule.action(context)
                if action in available_actions:
                    return action, best_rule.confidence
            except Exception as e:
                logger.warning(f"Rule {best_rule.name} failed: {e}")
        
        # Fallback: use pattern-based selection or exploration
        if random.random() < self._exploration_rate:
            # Exploration: try less-used actions
            action = self._select_exploratory_action(context, available_actions)
            confidence = 0.3
        else:
            # Exploitation: use best known action
            action = self._select_best_action(context, available_actions)
            confidence = 0.7
        
        return action, confidence
    
    def learn_from_outcome(
        self,
        context: Dict[str, Any],
        action: str,
        outcome: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Learn from the outcome of an action.
        
        Args:
            context: Context when action was taken
            action: Action that was taken
            outcome: Outcome/reward (0.0 to 1.0)
            metadata: Additional metadata
        """
        learning_outcome = LearningOutcome(
            timestamp=time.time(),
            context=context.copy(),
            action_taken=action,
            reward=outcome,
            confidence=self._calculate_learning_confidence(context, action, outcome),
            metadata=metadata or {}
        )
        
        self._learning_history.append(learning_outcome)
        
        # Update adaptation rules based on outcome
        self._update_adaptation_rules(learning_outcome)
        
        # Store context-action outcome for future reference
        context_key = self._create_context_key(context)
        self._context_memory[context_key].append((action, outcome, time.time()))
        
        record_metric("adaptive_learning_outcome", outcome, "gauge", {"action": action})
        
        logger.debug(
            f"Learned from outcome: action={action}, reward={outcome:.3f}",
            extra={"context": context, "metadata": metadata}
        )
    
    def add_adaptation_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: Callable[[Dict[str, Any]], str],
        priority: int = 5
    ) -> None:
        """
        Add a new adaptation rule.
        
        Args:
            name: Name of the rule
            condition: Function that checks if rule applies
            action: Function that determines action to take
            priority: Priority of the rule (higher = more important)
        """
        rule = AdaptationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority
        )
        
        self._adaptation_rules.append(rule)
        logger.info(f"Added adaptation rule: {name}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning and adaptation statistics."""
        if not self._learning_history:
            return {"error": "No learning data available"}
        
        recent_outcomes = [outcome.reward for outcome in list(self._learning_history)[-50:]]
        
        # Rule performance
        rule_stats = {}
        for rule in self._adaptation_rules:
            rule_stats[rule.name] = {
                "success_rate": rule.success_rate,
                "confidence": rule.confidence,
                "applications": rule.success_count + rule.failure_count,
                "last_applied": rule.last_applied
            }
        
        # Learning trends
        learning_trend = self._calculate_learning_trend()
        
        return {
            "total_learning_episodes": len(self._learning_history),
            "avg_recent_reward": statistics.mean(recent_outcomes) if recent_outcomes else 0.0,
            "learning_mode": self.learning_mode.value,
            "exploration_rate": self._exploration_rate,
            "learning_rate": self._learning_rate,
            "learning_trend": learning_trend,
            "adaptation_rules": rule_stats,
            "context_patterns": len(self._context_memory)
        }
    
    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge for persistence or transfer."""
        return {
            "learning_mode": self.learning_mode.value,
            "adaptation_rules": [
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "success_count": rule.success_count,
                    "failure_count": rule.failure_count,
                    "last_applied": rule.last_applied
                }
                for rule in self._adaptation_rules
            ],
            "learning_history": [outcome.to_dict() for outcome in list(self._learning_history)[-100:]],
            "context_patterns": {
                key: [(action, reward, timestamp) for action, reward, timestamp in outcomes]
                for key, outcomes in list(self._context_memory.items())
            }
        }
    
    def _select_exploratory_action(self, context: Dict[str, Any], available_actions: List[str]) -> str:
        """Select action for exploration (trying less-used options)."""
        context_key = self._create_context_key(context)
        
        if context_key in self._context_memory:
            # Count how often each action has been tried
            action_counts = defaultdict(int)
            for action, _, _ in self._context_memory[context_key]:
                action_counts[action] += 1
            
            # Prefer less-tried actions
            least_tried_actions = [
                action for action in available_actions
                if action_counts[action] == min(action_counts.get(a, 0) for a in available_actions)
            ]
            
            return random.choice(least_tried_actions)
        
        # No history, random choice
        return random.choice(available_actions)
    
    def _select_best_action(self, context: Dict[str, Any], available_actions: List[str]) -> str:
        """Select best action based on learned patterns."""
        context_key = self._create_context_key(context)
        
        if context_key in self._context_memory:
            # Calculate average reward for each action
            action_rewards = defaultdict(list)
            for action, reward, timestamp in self._context_memory[context_key]:
                if action in available_actions:
                    # Weight recent outcomes more heavily
                    weight = math.exp(-(time.time() - timestamp) / 86400)  # Decay over days
                    action_rewards[action].append(reward * weight)
            
            if action_rewards:
                # Select action with highest average reward
                best_action = max(
                    action_rewards.keys(),
                    key=lambda a: statistics.mean(action_rewards[a])
                )
                return best_action
        
        # No learned preference, random choice
        return random.choice(available_actions)
    
    def _update_adaptation_rules(self, outcome: LearningOutcome) -> None:
        """Update adaptation rules based on learning outcome."""
        for rule in self._adaptation_rules:
            if rule.condition(outcome.context):
                try:
                    expected_action = rule.action(outcome.context)
                    if expected_action == outcome.action_taken:
                        if outcome.reward > 0.6:
                            rule.success_count += 1
                        else:
                            rule.failure_count += 1
                        rule.last_applied = outcome.timestamp
                except Exception:
                    # Rule failed to execute
                    rule.failure_count += 1
    
    def _calculate_learning_confidence(self, context: Dict[str, Any], action: str, outcome: float) -> float:
        """Calculate confidence in learning outcome."""
        context_key = self._create_context_key(context)
        
        if context_key in self._context_memory:
            similar_outcomes = [
                reward for past_action, reward, _ in self._context_memory[context_key]
                if past_action == action
            ]
            
            if len(similar_outcomes) >= 3:
                # High confidence if consistent with past outcomes
                past_avg = statistics.mean(similar_outcomes)
                deviation = abs(outcome - past_avg)
                return max(0.1, 1.0 - deviation)
        
        return 0.5  # Medium confidence for new situations
    
    def _calculate_learning_trend(self) -> str:
        """Calculate overall learning trend."""
        if len(self._learning_history) < 20:
            return "insufficient_data"
        
        recent_rewards = [outcome.reward for outcome in list(self._learning_history)[-20:]]
        earlier_rewards = [outcome.reward for outcome in list(self._learning_history)[-40:-20]]
        
        if not earlier_rewards:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_rewards)
        earlier_avg = statistics.mean(earlier_rewards)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a simplified key from context for pattern matching."""
        key_parts = []
        for key, value in sorted(context.items())[:5]:  # Limit to 5 key features
            if isinstance(value, (int, float)):
                # Discretize numeric values
                if isinstance(value, float):
                    discretized = round(value / 0.1) * 0.1  # Round to 0.1
                else:
                    discretized = value
                key_parts.append(f"{key}:{discretized}")
            elif isinstance(value, str) and len(value) < 15:
                key_parts.append(f"{key}:{value}")
        
        return "|".join(key_parts)
    
    def _get_exploration_rate(self) -> float:
        """Get exploration rate based on learning mode."""
        rates = {
            LearningMode.CONSERVATIVE: 0.05,
            LearningMode.BALANCED: 0.15,
            LearningMode.AGGRESSIVE: 0.25,
            LearningMode.EXPLORATION: 0.4
        }
        return rates.get(self.learning_mode, 0.15)
    
    def _get_learning_rate(self) -> float:
        """Get learning rate based on learning mode."""
        rates = {
            LearningMode.CONSERVATIVE: 0.01,
            LearningMode.BALANCED: 0.05,
            LearningMode.AGGRESSIVE: 0.1,
            LearningMode.EXPLORATION: 0.15
        }
        return rates.get(self.learning_mode, 0.05)
    
    def _load_default_rules(self) -> None:
        """Load default adaptation rules."""
        
        # Rule 1: High CPU usage suggests reducing parallel workers
        self.add_adaptation_rule(
            name="reduce_workers_high_cpu",
            condition=lambda ctx: ctx.get("cpu_percent", 0) > 80,
            action=lambda ctx: "reduce_parallelism",
            priority=8
        )
        
        # Rule 2: Low accuracy suggests trying different optimization
        self.add_adaptation_rule(
            name="try_different_optimization",
            condition=lambda ctx: ctx.get("accuracy", 1.0) < 0.8,
            action=lambda ctx: "alternative_optimization",
            priority=7
        )
        
        # Rule 3: Fast execution suggests increasing batch size
        self.add_adaptation_rule(
            name="increase_batch_fast_execution",
            condition=lambda ctx: ctx.get("execution_time", 1.0) < 0.1,
            action=lambda ctx: "increase_batch_size",
            priority=6
        )
        
        # Rule 4: Memory pressure suggests enabling compression
        self.add_adaptation_rule(
            name="enable_compression_memory_pressure",
            condition=lambda ctx: ctx.get("memory_percent", 0) > 85,
            action=lambda ctx: "enable_compression",
            priority=8
        )
        
        # Rule 5: Successful design patterns should be reused
        self.add_adaptation_rule(
            name="reuse_successful_patterns",
            condition=lambda ctx: ctx.get("previous_success_rate", 0) > 0.9,
            action=lambda ctx: "reuse_pattern",
            priority=9
        )


# Global adaptive learning instances
_pattern_recognizer: Optional[PatternRecognizer] = None
_adaptive_engine: Optional[AdaptiveBehaviorEngine] = None


def get_pattern_recognizer() -> PatternRecognizer:
    """Get global pattern recognizer instance."""
    global _pattern_recognizer
    
    if _pattern_recognizer is None:
        _pattern_recognizer = PatternRecognizer()
    
    return _pattern_recognizer


def get_adaptive_engine(learning_mode: LearningMode = LearningMode.BALANCED) -> AdaptiveBehaviorEngine:
    """Get global adaptive behavior engine."""
    global _adaptive_engine
    
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveBehaviorEngine(learning_mode)
    
    return _adaptive_engine


def record_pattern(event_type: str, context: Dict[str, Any], outcome: float) -> None:
    """
    Convenience function to record a pattern.
    
    Args:
        event_type: Type of event
        context: Context information
        outcome: Outcome score (0.0 to 1.0)
    """
    get_pattern_recognizer().record_event(event_type, context, outcome)


def adapt_behavior(context: Dict[str, Any], available_actions: List[str]) -> Tuple[str, float]:
    """
    Convenience function for adaptive behavior selection.
    
    Args:
        context: Current context
        available_actions: Available actions
        
    Returns:
        Tuple of (selected_action, confidence)
    """
    return get_adaptive_engine().adapt_behavior(context, available_actions)


def learn_from_outcome(context: Dict[str, Any], action: str, outcome: float) -> None:
    """
    Convenience function to learn from outcome.
    
    Args:
        context: Context when action was taken
        action: Action that was taken
        outcome: Outcome score (0.0 to 1.0)
    """
    get_adaptive_engine().learn_from_outcome(context, action, outcome)