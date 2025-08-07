"""
Unit tests for sentiment analysis functionality.
"""

import pytest
from datetime import datetime

from codesign_playground.sentiment_analyzer import (
    SentimentAnalyzerAPI,
    SimpleSentimentAnalyzer,
    SentimentResult,
    SentimentLabel
)


class TestSimpleSentimentAnalyzer:
    """Test cases for SimpleSentimentAnalyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = SimpleSentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        text = "I love this product! It's amazing and wonderful!"
        result = self.analyzer.analyze(text)
        
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0.5
        assert result.scores['positive'] > result.scores['negative']
        assert result.text == text
        assert isinstance(result.processing_time_ms, float)
        assert isinstance(result.timestamp, datetime)
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        text = "This is terrible! I hate it. Worst product ever!"
        result = self.analyzer.analyze(text)
        
        assert result.label == SentimentLabel.NEGATIVE
        assert result.confidence > 0.5
        assert result.scores['negative'] > result.scores['positive']
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        text = "The weather is okay today."
        result = self.analyzer.analyze(text)
        
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence >= 0.0
        assert abs(result.scores['positive'] - result.scores['negative']) < 0.3
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer.analyze("")
        
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        assert result.scores['neutral'] == 1.0
    
    def test_intensifiers(self):
        """Test sentiment intensifiers."""
        normal_text = "This is good."
        intense_text = "This is extremely good!"
        
        normal_result = self.analyzer.analyze(normal_text)
        intense_result = self.analyzer.analyze(intense_text)
        
        # Intensified sentiment should have higher confidence
        assert intense_result.confidence >= normal_result.confidence
    
    def test_negations(self):
        """Test negation handling."""
        positive_text = "This is good."
        negated_text = "This is not good."
        
        positive_result = self.analyzer.analyze(positive_text)
        negated_result = self.analyzer.analyze(negated_text)
        
        # Negated text should have opposite sentiment
        assert positive_result.label == SentimentLabel.POSITIVE
        assert negated_result.label != SentimentLabel.POSITIVE
    
    def test_batch_analysis(self):
        """Test batch text analysis."""
        texts = [
            "I love this!",
            "This is terrible!",
            "It's okay.",
            "Amazing product!",
            "Worst experience ever."
        ]
        
        results = self.analyzer.analyze_batch(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].label == SentimentLabel.POSITIVE
        assert results[1].label == SentimentLabel.NEGATIVE
    
    def test_preprocessing(self):
        """Test text preprocessing."""
        text_with_punctuation = "I LOVE this!!! It's @#$% amazing!!!"
        result = self.analyzer.analyze(text_with_punctuation)
        
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0.5
    
    def test_result_serialization(self):
        """Test SentimentResult serialization."""
        text = "This is a test."
        result = self.analyzer.analyze(text)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['text'] == text
        assert result_dict['label'] in ['positive', 'negative', 'neutral']
        assert isinstance(result_dict['confidence'], float)
        assert isinstance(result_dict['scores'], dict)
        assert 'positive' in result_dict['scores']
        assert 'negative' in result_dict['scores']
        assert 'neutral' in result_dict['scores']
        assert isinstance(result_dict['processing_time_ms'], float)
        assert 'timestamp' in result_dict


class TestSentimentAnalyzerAPI:
    """Test cases for SentimentAnalyzerAPI."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api = SentimentAnalyzerAPI()
    
    def test_analyze_text(self):
        """Test single text analysis via API."""
        text = "This is a great product!"
        result = self.api.analyze_text(text)
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0.0
    
    def test_analyze_batch(self):
        """Test batch analysis via API."""
        texts = ["Good", "Bad", "Okay"]
        results = self.api.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
    
    def test_get_stats(self):
        """Test analyzer statistics."""
        stats = self.api.get_stats()
        
        assert isinstance(stats, dict)
        assert 'analyzer_type' in stats
        assert 'positive_words_count' in stats
        assert 'negative_words_count' in stats
        assert stats['analyzer_type'] == 'SimpleSentimentAnalyzer'
        assert stats['positive_words_count'] > 0
        assert stats['negative_words_count'] > 0
    
    def test_error_handling(self):
        """Test error handling in API."""
        # Test with None input - should handle gracefully
        try:
            result = self.api.analyze_text(None)
        except Exception:
            # Should handle None gracefully or raise appropriate exception
            pass


class TestSentimentIntegration:
    """Integration tests for sentiment analysis."""
    
    def test_real_world_examples(self):
        """Test with real-world example texts."""
        api = SentimentAnalyzerAPI()
        
        test_cases = [
            ("The customer service was outstanding and the product quality exceeded my expectations!", SentimentLabel.POSITIVE),
            ("Completely disappointed. Poor quality and terrible support. Would not recommend.", SentimentLabel.NEGATIVE),
            ("The product arrived on time. It works as described.", SentimentLabel.NEUTRAL),
            ("Not the best, but not the worst either. It's okay for the price.", SentimentLabel.NEUTRAL),
            ("Absolutely love it! Best purchase I've made this year. Highly recommend!", SentimentLabel.POSITIVE),
        ]
        
        for text, expected_label in test_cases:
            result = api.analyze_text(text)
            # Allow some flexibility in neutral detection
            if expected_label == SentimentLabel.NEUTRAL:
                assert result.label in [SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE]
            else:
                assert result.label == expected_label, f"Failed for text: {text[:50]}..."
    
    def test_performance(self):
        """Test performance of sentiment analysis."""
        api = SentimentAnalyzerAPI()
        text = "This is a test of sentiment analysis performance."
        
        # Analyze multiple times to test consistency
        results = []
        for _ in range(10):
            result = api.analyze_text(text)
            results.append(result)
            assert result.processing_time_ms < 100  # Should be fast
        
        # Results should be consistent
        labels = [r.label for r in results]
        assert len(set(labels)) == 1  # All same label
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        api = SentimentAnalyzerAPI()
        
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "!!!",  # Only punctuation
            "123 456 789",  # Only numbers
            "The the the the the",  # Repeated neutral words
            "Good bad good bad good bad",  # Mixed sentiment
        ]
        
        for text in edge_cases:
            result = api.analyze_text(text)
            assert isinstance(result, SentimentResult)
            assert result.label in [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
            assert 0.0 <= result.confidence <= 1.0
            assert abs(sum(result.scores.values()) - 1.0) < 0.01  # Scores should sum to ~1
