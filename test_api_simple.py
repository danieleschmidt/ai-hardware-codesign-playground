#!/usr/bin/env python3
"""
Simple test script for sentiment API functionality.
"""

import importlib.util
import json

# Load sentiment analyzer API directly
spec = importlib.util.spec_from_file_location("api_sentiment", "backend/codesign_playground/api_sentiment.py")

def test_api():
    """Test API sentiment analysis functionality."""
    print("🌐 Testing API Interface...")
    
    try:
        # Test the analyzer dependency function
        spec_analyzer = importlib.util.spec_from_file_location("sentiment_analyzer", "backend/codesign_playground/sentiment_analyzer.py")
        analyzer_module = importlib.util.module_from_spec(spec_analyzer)
        spec_analyzer.loader.exec_module(analyzer_module)
        
        # Test SentimentAnalyzerAPI instantiation
        api = analyzer_module.SentimentAnalyzerAPI()
        print("✅ SentimentAnalyzerAPI created successfully")
        
        # Test get_stats functionality
        stats = api.get_stats()
        print(f"✅ Stats retrieved: {stats['analyzer_type']}")
        print(f"   - Positive words: {stats['positive_words_count']}")
        print(f"   - Negative words: {stats['negative_words_count']}")
        
        # Test single analysis
        result = api.analyze_text("This is an amazing product!")
        print(f"✅ Single analysis: {result.label.value} (confidence: {result.confidence:.3f})")
        
        # Test batch analysis
        texts = ["Great!", "Terrible!", "Okay."]
        batch_results = api.analyze_batch(texts)
        print(f"✅ Batch analysis: {len(batch_results)} results")
        for i, r in enumerate(batch_results):
            print(f"   {i+1}. '{texts[i]}' -> {r.label.value}")
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'label' in result_dict
        assert 'confidence' in result_dict
        print(f"✅ Serialization works: {len(result_dict)} fields")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎯 API Tests Complete!")
    return True

if __name__ == "__main__":
    test_api()
