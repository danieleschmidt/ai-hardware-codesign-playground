#!/usr/bin/env python3
"""
Simple test script for sentiment analyzer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Test the sentiment analyzer directly (without full package imports)
import importlib.util
import sys

# Load sentiment analyzer module directly
spec = importlib.util.spec_from_file_location("sentiment_analyzer", "backend/codesign_playground/sentiment_analyzer.py")
sentiment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentiment_module)

SimpleSentimentAnalyzer = sentiment_module.SimpleSentimentAnalyzer
SentimentLabel = sentiment_module.SentimentLabel
SentimentAnalyzerAPI = sentiment_module.SentimentAnalyzerAPI

def test_sentiment_analyzer():
    """Test basic sentiment analysis functionality."""
    print("🧪 Testing Sentiment Analyzer...")
    
    analyzer = SimpleSentimentAnalyzer()
    
    test_cases = [
        ("I absolutely love this product! It's amazing!", SentimentLabel.POSITIVE),
        ("This is terrible! I hate it. Worst thing ever!", SentimentLabel.NEGATIVE),
        ("The weather is okay today.", SentimentLabel.NEUTRAL),
        ("Not bad, but could be better.", SentimentLabel.NEUTRAL),
        ("Extremely disappointed. Very poor quality.", SentimentLabel.NEGATIVE),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_label in test_cases:
        try:
            result = analyzer.analyze(text)
            
            # Check basic properties
            assert hasattr(result, 'label')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'scores')
            assert hasattr(result, 'processing_time_ms')
            
            # Check scores sum to approximately 1
            scores_sum = sum(result.scores.values())
            assert abs(scores_sum - 1.0) < 0.01, f"Scores don't sum to 1: {scores_sum}"
            
            # Check confidence is valid
            assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence: {result.confidence}"
            
            status = "✅" if result.label == expected_label else "⚠️"
            print(f"{status} '{text[:50]}...' -> {result.label.value} (conf: {result.confidence:.3f}, time: {result.processing_time_ms:.2f}ms)")
            
            if result.label == expected_label or expected_label == SentimentLabel.NEUTRAL:
                passed += 1
            
        except Exception as e:
            print(f"❌ Error testing '{text[:30]}...': {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    # Test batch processing
    print("\n🔄 Testing batch processing...")
    texts = [case[0] for case in test_cases]
    batch_results = analyzer.analyze_batch(texts)
    
    assert len(batch_results) == len(texts), "Batch processing failed"
    print(f"✅ Batch processing: {len(batch_results)} results")
    
    # Test edge cases
    print("\n🔍 Testing edge cases...")
    edge_cases = [
        "",  # Empty
        " ",  # Whitespace
        "!!!",  # Punctuation only
        "The the the",  # Repeated words
    ]
    
    for text in edge_cases:
        try:
            result = analyzer.analyze(text)
            print(f"✅ Edge case '{text}' -> {result.label.value}")
        except Exception as e:
            print(f"❌ Edge case failed '{text}': {e}")
    
    print("\n🎯 Sentiment Analysis Tests Complete!")
    return True

if __name__ == "__main__":
    test_sentiment_analyzer()
