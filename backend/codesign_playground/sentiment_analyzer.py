"""
Sentiment Analysis Module - Generation 1: Simple Implementation

Provides core sentiment analysis functionality with basic classification.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
            "scores": self.scores,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


class SimpleSentimentAnalyzer:
    """Basic rule-based sentiment analyzer using word lists."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_word_lists()
        
    def _load_word_lists(self) -> None:
        """Load positive and negative word lists."""
        # Basic sentiment word lists
        self.positive_words = {
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'good', 'wonderful',
            'outstanding', 'brilliant', 'superb', 'perfect', 'love', 'like', 'enjoy',
            'happy', 'pleased', 'satisfied', 'delighted', 'impressed', 'recommend',
            'beautiful', 'incredible', 'marvelous', 'terrific', 'fabulous', 'remarkable',
            'extraordinary', 'phenomenal', 'magnificent', 'spectacular', 'best', 'better'
        }
        
        self.negative_words = {
            'awful', 'terrible', 'horrible', 'bad', 'worst', 'hate', 'dislike', 'ugly',
            'disgusting', 'disappointing', 'frustrated', 'angry', 'annoyed', 'upset',
            'sad', 'depressed', 'pathetic', 'useless', 'worthless', 'ridiculous',
            'stupid', 'dumb', 'boring', 'waste', 'poor', 'worse', 'failed', 'broken',
            'problem', 'issue', 'error', 'wrong', 'difficult', 'complicated', 'confusing'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'quite': 1.2, 'totally': 1.8,
            'absolutely': 2.0, 'incredibly': 1.7, 'highly': 1.4, 'super': 1.6
        }
        
        self.negations = {'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none'}
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        # Split into words
        words = text.split()
        return [word.strip() for word in words if word.strip()]
    
    def _calculate_sentiment_score(self, words: List[str]) -> Tuple[float, Dict[str, int]]:
        """Calculate sentiment score based on word lists."""
        positive_score = 0.0
        negative_score = 0.0
        word_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': len(words)}
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for intensifiers
            intensity = 1.0
            if word in self.intensifiers:
                intensity = self.intensifiers[word]
                i += 1
                if i >= len(words):
                    break
                word = words[i]
            
            # Check for negations (flip sentiment of next word)
            negate = False
            if word in self.negations:
                negate = True
                i += 1
                if i >= len(words):
                    break
                word = words[i]
            
            # Score the word
            if word in self.positive_words:
                score = 1.0 * intensity
                if negate:
                    negative_score += score
                    word_counts['negative'] += 1
                else:
                    positive_score += score
                    word_counts['positive'] += 1
            elif word in self.negative_words:
                score = 1.0 * intensity
                if negate:
                    positive_score += score
                    word_counts['positive'] += 1
                else:
                    negative_score += score
                    word_counts['negative'] += 1
            else:
                word_counts['neutral'] += 1
                
            i += 1
        
        # Calculate final score (-1 to 1 range)
        total_score = positive_score + negative_score
        if total_score > 0:
            sentiment_score = (positive_score - negative_score) / total_score
        else:
            sentiment_score = 0.0
            
        return sentiment_score, word_counts
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text."""
        start_time = datetime.now()
        
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                processing_time_ms=0.0,
                timestamp=start_time
            )
        
        # Preprocess text
        words = self._preprocess_text(text)
        
        # Calculate sentiment score
        sentiment_score, word_counts = self._calculate_sentiment_score(words)
        
        # Determine label and confidence
        if sentiment_score > 0.1:
            label = SentimentLabel.POSITIVE
            confidence = min(sentiment_score * 2, 1.0)  # Scale to 0-1
        elif sentiment_score < -0.1:
            label = SentimentLabel.NEGATIVE
            confidence = min(abs(sentiment_score) * 2, 1.0)  # Scale to 0-1
        else:
            label = SentimentLabel.NEUTRAL
            confidence = 1.0 - abs(sentiment_score) * 2  # Higher confidence for neutral when score is near 0
        
        # Create normalized scores
        scores = {
            'positive': max(0, sentiment_score),
            'negative': max(0, -sentiment_score),
            'neutral': 1.0 - abs(sentiment_score)
        }
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return SentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            scores=scores,
            processing_time_ms=processing_time,
            timestamp=start_time
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze(text) for text in texts]


class SentimentAnalyzerAPI:
    """Main API class for sentiment analysis."""
    
    def __init__(self):
        self.analyzer = SimpleSentimentAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def analyze_text(self, text: str) -> SentimentResult:
        """Public API method for single text analysis."""
        try:
            return self.analyzer.analyze(text)
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            raise
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Public API method for batch text analysis."""
        try:
            return self.analyzer.analyze_batch(texts)
        except Exception as e:
            self.logger.error(f"Error analyzing batch: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get analyzer statistics."""
        return {
            "analyzer_type": "SimpleSentimentAnalyzer",
            "positive_words_count": len(self.analyzer.positive_words),
            "negative_words_count": len(self.analyzer.negative_words),
            "intensifiers_count": len(self.analyzer.intensifiers),
            "negations_count": len(self.analyzer.negations)
        }
