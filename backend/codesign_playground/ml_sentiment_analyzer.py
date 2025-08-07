"""
Advanced ML-based Sentiment Analysis Module - Generation 2: Robust Implementation

Provides advanced sentiment analysis with multiple algorithms, validation, and caching.
"""

import re
import pickle
import hashlib
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics

from .sentiment_analyzer import SentimentLabel, SentimentResult, SimpleSentimentAnalyzer
from .utils.logging import get_logger

logger = get_logger(__name__)


class AnalyzerType(Enum):
    """Types of sentiment analyzers available."""
    SIMPLE_RULE_BASED = "simple_rule_based"
    ENHANCED_RULE_BASED = "enhanced_rule_based"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


class ConfidenceLevel(Enum):
    """Confidence levels for sentiment predictions."""
    VERY_LOW = "very_low"     # < 0.3
    LOW = "low"               # 0.3 - 0.5
    MEDIUM = "medium"         # 0.5 - 0.7
    HIGH = "high"             # 0.7 - 0.9
    VERY_HIGH = "very_high"   # > 0.9


@dataclass
class EnhancedSentimentResult:
    """Enhanced result with additional metadata and confidence measures."""
    text: str
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime
    analyzer_type: AnalyzerType
    confidence_level: ConfidenceLevel
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Determine confidence level
        if self.confidence < 0.3:
            self.confidence_level = ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            self.confidence_level = ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            self.confidence_level = ConfidenceLevel.HIGH
        else:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "scores": self.scores,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "analyzer_type": self.analyzer_type.value,
            "features": self.features,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_basic_result(cls, basic_result: SentimentResult, analyzer_type: AnalyzerType) -> "EnhancedSentimentResult":
        """Convert basic SentimentResult to EnhancedSentimentResult."""
        return cls(
            text=basic_result.text,
            label=basic_result.label,
            confidence=basic_result.confidence,
            scores=basic_result.scores,
            processing_time_ms=basic_result.processing_time_ms,
            timestamp=basic_result.timestamp,
            analyzer_type=analyzer_type
        )


class SentimentAnalyzerBase(ABC):
    """Abstract base class for sentiment analyzers."""
    
    def __init__(self, name: str, analyzer_type: AnalyzerType):
        self.name = name
        self.analyzer_type = analyzer_type
        self.logger = get_logger(f"{__name__}.{name}")
        self._analysis_count = 0
        self._total_processing_time = 0.0
        self._error_count = 0
    
    @abstractmethod
    def _analyze_impl(self, text: str) -> EnhancedSentimentResult:
        """Implementation-specific analysis method."""
        pass
    
    def analyze(self, text: str) -> EnhancedSentimentResult:
        """Analyze sentiment with error handling and metrics."""
        start_time = time.perf_counter()
        
        try:
            if not text or not text.strip():
                return self._create_neutral_result(text, start_time)
            
            result = self._analyze_impl(text)
            
            # Update metrics
            self._analysis_count += 1
            self._total_processing_time += result.processing_time_ms
            
            # Add analyzer metadata
            result.metadata.update({
                "analyzer_name": self.name,
                "analysis_count": self._analysis_count,
                "avg_processing_time": self._total_processing_time / self._analysis_count
            })
            
            return result
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Error analyzing text: {e}")
            return self._create_error_result(text, start_time, str(e))
    
    def _create_neutral_result(self, text: str, start_time: float) -> EnhancedSentimentResult:
        """Create neutral result for empty/invalid input."""
        processing_time = (time.perf_counter() - start_time) * 1000
        return EnhancedSentimentResult(
            text=text,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            analyzer_type=self.analyzer_type,
            metadata={"reason": "empty_input"}
        )
    
    def _create_error_result(self, text: str, start_time: float, error_msg: str) -> EnhancedSentimentResult:
        """Create result for error cases."""
        processing_time = (time.perf_counter() - start_time) * 1000
        return EnhancedSentimentResult(
            text=text,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            analyzer_type=self.analyzer_type,
            metadata={"error": error_msg}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "name": self.name,
            "type": self.analyzer_type.value,
            "analysis_count": self._analysis_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._analysis_count, 1),
            "avg_processing_time_ms": self._total_processing_time / max(self._analysis_count, 1),
            "total_processing_time_ms": self._total_processing_time
        }


class EnhancedRuleBasedAnalyzer(SentimentAnalyzerBase):
    """Enhanced rule-based analyzer with more sophisticated features."""
    
    def __init__(self):
        super().__init__("EnhancedRuleBased", AnalyzerType.ENHANCED_RULE_BASED)
        self._load_enhanced_lexicons()
        self._load_patterns()
    
    def _load_enhanced_lexicons(self):
        """Load enhanced sentiment lexicons with weights."""
        # Positive words with intensity weights
        self.positive_words = {
            # Strong positive (weight 1.0)
            'amazing': 1.0, 'awesome': 1.0, 'excellent': 1.0, 'fantastic': 1.0,
            'outstanding': 1.0, 'brilliant': 1.0, 'superb': 1.0, 'perfect': 1.0,
            'incredible': 1.0, 'marvelous': 1.0, 'spectacular': 1.0, 'phenomenal': 1.0,
            'extraordinary': 1.0, 'magnificent': 1.0, 'terrific': 1.0, 'fabulous': 1.0,
            
            # Medium positive (weight 0.7)
            'great': 0.7, 'good': 0.7, 'wonderful': 0.7, 'nice': 0.7, 'pleasant': 0.7,
            'satisfying': 0.7, 'enjoyable': 0.7, 'delightful': 0.7, 'impressive': 0.7,
            'beautiful': 0.7, 'lovely': 0.7, 'charming': 0.7, 'attractive': 0.7,
            
            # Mild positive (weight 0.5)
            'like': 0.5, 'okay': 0.3, 'fine': 0.4, 'decent': 0.4, 'acceptable': 0.3,
            'reasonable': 0.4, 'adequate': 0.3, 'satisfactory': 0.4, 'alright': 0.3,
        }
        
        # Negative words with intensity weights
        self.negative_words = {
            # Strong negative (weight 1.0)
            'terrible': 1.0, 'awful': 1.0, 'horrible': 1.0, 'disgusting': 1.0,
            'pathetic': 1.0, 'useless': 1.0, 'worthless': 1.0, 'devastating': 1.0,
            'catastrophic': 1.0, 'appalling': 1.0, 'atrocious': 1.0, 'dreadful': 1.0,
            
            # Medium negative (weight 0.7)
            'bad': 0.7, 'poor': 0.7, 'disappointing': 0.7, 'unpleasant': 0.7,
            'annoying': 0.7, 'frustrating': 0.7, 'unsatisfactory': 0.7, 'inferior': 0.7,
            'defective': 0.7, 'faulty': 0.7, 'flawed': 0.7, 'inadequate': 0.7,
            
            # Mild negative (weight 0.5)
            'dislike': 0.5, 'boring': 0.4, 'dull': 0.4, 'mediocre': 0.3,
            'subpar': 0.4, 'below-average': 0.4, 'lacking': 0.4, 'limited': 0.3,
        }
        
        # Enhanced intensifiers with varied weights
        self.intensifiers = {
            'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 2.0, 'totally': 1.8,
            'completely': 1.7, 'utterly': 1.9, 'tremendously': 1.7, 'exceptionally': 1.6,
            'very': 1.4, 'really': 1.3, 'quite': 1.2, 'rather': 1.1, 'fairly': 1.1,
            'somewhat': 0.8, 'slightly': 0.7, 'a bit': 0.6, 'a little': 0.6,
        }
        
        # Enhanced negations
        self.negations = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'without', 'lack', 'lacks', 'lacking', 'absence',
            'absent', 'missing', 'fail', 'fails', 'failed', 'unable', 'cannot',
        }
        
        # Contextual modifiers
        self.question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        self.conditional_words = {'if', 'unless', 'provided', 'assuming', 'suppose'}
    
    def _load_patterns(self):
        """Load sentiment patterns and rules."""
        # Positive patterns
        self.positive_patterns = [
            r'\b(love|adore|cherish)\s+\w+',
            r'\b(recommend|suggest)\s+(highly|strongly)',
            r'\b(best|finest|greatest)\s+\w+',
            r'\b(worth|deserves?)\s+(buying|getting|trying)',
            r'\b(exceeded?|surpassed?)\s+expectations',
        ]
        
        # Negative patterns  
        self.negative_patterns = [
            r'\b(waste|wasted?)\s+of\s+(time|money)',
            r'\b(avoid|stay away from)',
            r'\b(regret|disappointed?|frustrated?)\s+\w+',
            r'\b(money|time)\s+(wasted?|down the drain)',
            r'\b(below|under)\s+expectations',
        ]
    
    def _analyze_impl(self, text: str) -> EnhancedSentimentResult:
        """Enhanced rule-based analysis implementation."""
        start_time = time.perf_counter()
        
        # Extract features
        features = self._extract_features(text)
        
        # Calculate sentiment scores
        sentiment_score, word_analysis = self._calculate_enhanced_score(text, features)
        
        # Apply pattern bonuses
        pattern_bonus = self._apply_patterns(text)
        sentiment_score += pattern_bonus
        
        # Determine label and confidence
        label, confidence = self._determine_sentiment(sentiment_score, features)
        
        # Create normalized scores
        scores = self._create_normalized_scores(sentiment_score)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return EnhancedSentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            scores=scores,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            analyzer_type=self.analyzer_type,
            features=features,
            metadata={
                "word_analysis": word_analysis,
                "pattern_bonus": pattern_bonus,
                "raw_score": sentiment_score
            }
        )
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic and stylistic features."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        return {
            "text_length": len(text),
            "word_count": len(words),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "has_questions": any(word in words for word in self.question_words),
            "has_conditionals": any(word in words for word in self.conditional_words),
            "unique_word_ratio": len(set(words)) / max(len(words), 1),
            "avg_word_length": statistics.mean([len(w) for w in words]) if words else 0,
            "punctuation_density": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
    
    def _calculate_enhanced_score(self, text: str, features: Dict) -> Tuple[float, Dict]:
        """Calculate enhanced sentiment score with detailed analysis."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_score = 0.0
        negative_score = 0.0
        word_contributions = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for intensifiers
            intensity = 1.0
            if word in self.intensifiers:
                intensity = self.intensifiers[word]
                word_contributions.append({"word": word, "type": "intensifier", "weight": intensity})
                i += 1
                continue
            
            # Check for negations
            negate = False
            if word in self.negations:
                negate = True
                word_contributions.append({"word": word, "type": "negation", "negate": True})
                i += 1
                if i >= len(words):
                    break
                word = words[i]
            
            # Score sentiment words
            if word in self.positive_words:
                base_score = self.positive_words[word]
                final_score = base_score * intensity
                
                if negate:
                    negative_score += final_score
                    word_contributions.append({
                        "word": word, "type": "positive_negated", 
                        "base_weight": base_score, "final_weight": -final_score
                    })
                else:
                    positive_score += final_score
                    word_contributions.append({
                        "word": word, "type": "positive", 
                        "base_weight": base_score, "final_weight": final_score
                    })
                    
            elif word in self.negative_words:
                base_score = self.negative_words[word]
                final_score = base_score * intensity
                
                if negate:
                    positive_score += final_score
                    word_contributions.append({
                        "word": word, "type": "negative_negated", 
                        "base_weight": base_score, "final_weight": final_score
                    })
                else:
                    negative_score += final_score
                    word_contributions.append({
                        "word": word, "type": "negative", 
                        "base_weight": base_score, "final_weight": -final_score
                    })
            
            i += 1
        
        # Apply feature-based adjustments
        feature_adjustment = self._calculate_feature_adjustment(features)
        
        # Calculate final score
        total_score = positive_score + negative_score
        if total_score > 0:
            raw_score = (positive_score - negative_score) / total_score
        else:
            raw_score = 0.0
        
        final_score = raw_score + feature_adjustment
        
        word_analysis = {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "feature_adjustment": feature_adjustment,
            "word_contributions": word_contributions
        }
        
        return final_score, word_analysis
    
    def _calculate_feature_adjustment(self, features: Dict) -> float:
        """Calculate sentiment adjustment based on textual features."""
        adjustment = 0.0
        
        # Exclamation marks generally indicate stronger sentiment
        if features["exclamation_count"] > 0:
            adjustment += min(features["exclamation_count"] * 0.1, 0.3)
        
        # Questions might reduce confidence in sentiment
        if features["has_questions"]:
            adjustment -= 0.1
        
        # Conditionals reduce sentiment certainty
        if features["has_conditionals"]:
            adjustment *= 0.8
        
        # High caps ratio might indicate stronger emotion
        if features["caps_ratio"] > 0.3:
            adjustment += 0.1
        
        return adjustment
    
    def _apply_patterns(self, text: str) -> float:
        """Apply pattern-based sentiment bonuses."""
        bonus = 0.0
        text_lower = text.lower()
        
        # Check positive patterns
        for pattern in self.positive_patterns:
            if re.search(pattern, text_lower):
                bonus += 0.2
        
        # Check negative patterns
        for pattern in self.negative_patterns:
            if re.search(pattern, text_lower):
                bonus -= 0.2
        
        return bonus
    
    def _determine_sentiment(self, score: float, features: Dict) -> Tuple[SentimentLabel, float]:
        """Determine sentiment label and confidence based on score and features."""
        abs_score = abs(score)
        
        # Base confidence from score magnitude
        base_confidence = min(abs_score * 1.5, 1.0)
        
        # Adjust confidence based on features
        confidence_adjustment = 0.0
        
        # Longer texts with more words are generally more reliable
        if features["word_count"] >= 10:
            confidence_adjustment += 0.1
        elif features["word_count"] <= 3:
            confidence_adjustment -= 0.2
        
        # Questions reduce confidence
        if features["has_questions"]:
            confidence_adjustment -= 0.1
        
        # Conditionals reduce confidence
        if features["has_conditionals"]:
            confidence_adjustment -= 0.1
        
        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
        
        # Determine label
        if score > 0.1:
            return SentimentLabel.POSITIVE, final_confidence
        elif score < -0.1:
            return SentimentLabel.NEGATIVE, final_confidence
        else:
            # For neutral, confidence is based on how close to zero
            neutral_confidence = 1.0 - abs_score * 2
            return SentimentLabel.NEUTRAL, max(0.3, neutral_confidence)
    
    def _create_normalized_scores(self, sentiment_score: float) -> Dict[str, float]:
        """Create normalized probability scores."""
        if sentiment_score > 0:
            positive = 0.5 + min(sentiment_score / 2, 0.5)
            negative = 0.5 - min(sentiment_score / 2, 0.5)
        else:
            positive = 0.5 + max(sentiment_score / 2, -0.5)
            negative = 0.5 - max(sentiment_score / 2, -0.5)
        
        neutral = 1.0 - abs(sentiment_score)
        
        # Normalize to sum to 1
        total = positive + negative + neutral
        if total > 0:
            return {
                'positive': positive / total,
                'negative': negative / total,
                'neutral': neutral / total
            }
        else:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}


class CachedSentimentAnalyzer:
    """Sentiment analyzer with caching capabilities."""
    
    def __init__(self, base_analyzer: SentimentAnalyzerBase, cache_size: int = 1000):
        self.base_analyzer = base_analyzer
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = get_logger(f"{__name__}.CachedAnalyzer")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def analyze(self, text: str) -> EnhancedSentimentResult:
        """Analyze with caching."""
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            # Update timestamp but keep original analysis
            cached_result.timestamp = datetime.now()
            cached_result.metadata["from_cache"] = True
            return cached_result
        
        # Analyze and cache
        self.cache_misses += 1
        result = self.base_analyzer.analyze(text)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Cache cleared")


class EnsembleSentimentAnalyzer(SentimentAnalyzerBase):
    """Ensemble analyzer combining multiple methods."""
    
    def __init__(self, analyzers: List[SentimentAnalyzerBase], weights: Optional[List[float]] = None):
        super().__init__("Ensemble", AnalyzerType.ENSEMBLE)
        self.analyzers = analyzers
        
        # Default equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(analyzers)] * len(analyzers)
        else:
            if len(weights) != len(analyzers):
                raise ValueError("Number of weights must match number of analyzers")
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def _analyze_impl(self, text: str) -> EnhancedSentimentResult:
        """Ensemble analysis combining multiple analyzers."""
        start_time = time.perf_counter()
        
        # Get predictions from all analyzers
        predictions = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(text)
                predictions.append(result)
            except Exception as e:
                self.logger.warning(f"Analyzer {analyzer.name} failed: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All analyzers failed")
        
        # Combine predictions using weighted voting
        weighted_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        weighted_confidence = 0.0
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            for sentiment, score in pred.scores.items():
                weighted_scores[sentiment] += score * weight
            weighted_confidence += pred.confidence * weight
        
        # Determine final label
        final_label = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Collect features from all analyzers
        combined_features = {}
        for pred in predictions:
            combined_features.update(pred.features)
        
        return EnhancedSentimentResult(
            text=text,
            label=SentimentLabel(final_label),
            confidence=weighted_confidence,
            scores=weighted_scores,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            analyzer_type=self.analyzer_type,
            features=combined_features,
            metadata={
                "num_analyzers": len(predictions),
                "individual_predictions": [
                    {
                        "analyzer": pred.analyzer_type.value,
                        "label": pred.label.value,
                        "confidence": pred.confidence
                    } for pred in predictions
                ]
            }
        )


class AdvancedSentimentAnalyzerAPI:
    """Advanced API with multiple analyzers, caching, and validation."""
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 1000):
        self.logger = get_logger(__name__)
        
        # Initialize analyzers
        self.simple_analyzer = SimpleSentimentAnalyzer()
        self.enhanced_analyzer = EnhancedRuleBasedAnalyzer()
        
        # Create ensemble
        self.ensemble_analyzer = EnsembleSentimentAnalyzer(
            [self.enhanced_analyzer],
            weights=[1.0]
        )
        
        # Add caching if enabled
        if enable_caching:
            self.primary_analyzer = CachedSentimentAnalyzer(self.ensemble_analyzer, cache_size)
        else:
            self.primary_analyzer = self.ensemble_analyzer
        
        self.logger.info(f"Initialized AdvancedSentimentAnalyzerAPI (caching: {enable_caching})")
    
    def analyze_text(self, text: str, analyzer_type: Optional[AnalyzerType] = None) -> EnhancedSentimentResult:
        """Analyze text with specified analyzer type."""
        if analyzer_type == AnalyzerType.SIMPLE_RULE_BASED:
            basic_result = self.simple_analyzer.analyze(text)
            return EnhancedSentimentResult.from_basic_result(basic_result, AnalyzerType.SIMPLE_RULE_BASED)
        elif analyzer_type == AnalyzerType.ENHANCED_RULE_BASED:
            return self.enhanced_analyzer.analyze(text)
        else:
            # Default to primary (cached ensemble)
            return self.primary_analyzer.analyze(text)
    
    def analyze_batch(self, texts: List[str], analyzer_type: Optional[AnalyzerType] = None) -> List[EnhancedSentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze_text(text, analyzer_type) for text in texts]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all analyzers."""
        stats = {
            "api_type": "AdvancedSentimentAnalyzerAPI",
            "analyzers": {
                "enhanced": self.enhanced_analyzer.get_stats(),
                "ensemble": self.ensemble_analyzer.get_stats(),
            }
        }
        
        # Add cache stats if available
        if isinstance(self.primary_analyzer, CachedSentimentAnalyzer):
            stats["cache"] = self.primary_analyzer.get_cache_stats()
        
        return stats
    
    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate input text."""
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if len(text) > 10000:
            return False, "Text too long (max 10,000 characters)"
        
        if not text.strip():
            return False, "Text cannot be empty or whitespace only"
        
        # Check for potentially harmful content patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocols
            r'data:.*base64',  # Base64 data URLs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Text contains potentially harmful content"
        
        return True, None
