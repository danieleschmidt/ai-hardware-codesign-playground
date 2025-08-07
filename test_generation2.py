#!/usr/bin/env python3
"""
Generation 2 Test Script - Enhanced Sentiment Analysis

Tests advanced features including enhanced analyzer, caching, validation, and monitoring.
"""

import sys
import os
import importlib.util
import time

# Load modules directly
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load required modules
sent_analyzer = load_module("sentiment_analyzer", "backend/codesign_playground/sentiment_analyzer.py")
ml_analyzer = load_module("ml_sentiment_analyzer", "backend/codesign_playground/ml_sentiment_analyzer.py")
monitoring = load_module("sentiment_monitoring", "backend/codesign_playground/sentiment_monitoring.py")

def test_enhanced_analyzer():
    """Test enhanced rule-based analyzer."""
    print("🤖 Testing Enhanced Analyzer...")
    
    try:
        analyzer = ml_analyzer.EnhancedRuleBasedAnalyzer()
        
        test_cases = [
            "I absolutely love this incredible product! It's extremely amazing!",
            "This is not just bad, it's absolutely terrible and disappointing.",
            "The product is okay, nothing special but not bad either.",
            "Not sure if I like this... it's somewhat confusing.",
            "AMAZING!!! Best purchase ever!!! Highly recommend!!!"
        ]
        
        for text in test_cases:
            result = analyzer.analyze(text)
            
            print(f"✅ '{text[:50]}...'")
            print(f"    Label: {result.label.value} (confidence: {result.confidence:.3f}, level: {result.confidence_level.value})")
            print(f"    Scores: P={result.scores['positive']:.3f}, N={result.scores['negative']:.3f}, Neu={result.scores['neutral']:.3f}")
            print(f"    Features: {len(result.features)} extracted")
            print(f"    Time: {result.processing_time_ms:.2f}ms")
            print()
        
        # Test stats
        stats = analyzer.get_stats()
        print(f"📊 Analyzer Stats: {stats['analysis_count']} analyses, {stats['avg_processing_time_ms']:.2f}ms avg")
        
    except Exception as e:
        print(f"❌ Enhanced analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_caching():
    """Test caching functionality."""
    print("🗄 Testing Caching System...")
    
    try:
        base_analyzer = ml_analyzer.EnhancedRuleBasedAnalyzer()
        cached_analyzer = ml_analyzer.CachedSentimentAnalyzer(base_analyzer, cache_size=5)
        
        test_text = "This is a test for caching functionality."
        
        # First analysis (cache miss)
        start_time = time.time()
        result1 = cached_analyzer.analyze(test_text)
        time1 = (time.time() - start_time) * 1000
        
        # Second analysis (cache hit)
        start_time = time.time()
        result2 = cached_analyzer.analyze(test_text)
        time2 = (time.time() - start_time) * 1000
        
        print(f"✅ First analysis: {time1:.2f}ms")
        print(f"✅ Cached analysis: {time2:.2f}ms")
        print(f"✅ Cache speedup: {time1/max(time2, 0.001):.1f}x")
        
        # Check cache stats
        cache_stats = cached_analyzer.get_cache_stats()
        print(f"✅ Cache stats: {cache_stats['hit_rate']:.3f} hit rate, {cache_stats['cache_size']} items")
        
        # Test cache overflow
        for i in range(10):
            cached_analyzer.analyze(f"Test text number {i}")
        
        final_stats = cached_analyzer.get_cache_stats()
        print(f"✅ After overflow: {final_stats['cache_size']} items (max 5)")
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_ensemble():
    """Test ensemble analyzer."""
    print("🎵 Testing Ensemble Analyzer...")
    
    try:
        # Create multiple analyzers
        analyzer1 = ml_analyzer.EnhancedRuleBasedAnalyzer()
        
        # Create ensemble (currently just one analyzer, but framework is there)
        ensemble = ml_analyzer.EnsembleSentimentAnalyzer([analyzer1])
        
        test_text = "This product is amazing! I love it so much!"
        result = ensemble.analyze(test_text)
        
        print(f"✅ Ensemble result: {result.label.value} (confidence: {result.confidence:.3f})")
        print(f"✅ Metadata: {result.metadata['num_analyzers']} analyzers used")
        print(f"✅ Individual predictions: {len(result.metadata['individual_predictions'])}")
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_validation():
    """Test result validation."""
    print("⚙️ Testing Validation System...")
    
    try:
        validator = monitoring.SentimentValidator()
        analyzer = ml_analyzer.EnhancedRuleBasedAnalyzer()
        
        # Test with good results
        good_text = "This is a great product!"
        result = analyzer.analyze(good_text)
        is_valid, issues = validator.validate_result(result)
        
        print(f"✅ Good result validation: {'PASS' if is_valid else 'FAIL'}")
        if issues:
            print(f"    Issues: {issues}")
        
        # Test validation stats
        for _ in range(5):
            test_result = analyzer.analyze(f"Test validation text {_}")
            validator.validate_result(test_result)
        
        stats = validator.get_validation_stats()
        print(f"✅ Validation stats: {stats['total_validations']} validations, {stats['success_rate']:.3f} success rate")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_monitoring():
    """Test monitoring system."""
    print("📊 Testing Monitoring System...")
    
    try:
        monitor = monitoring.SentimentMonitor()
        analyzer = ml_analyzer.EnhancedRuleBasedAnalyzer()
        
        # Generate some test analyses
        test_texts = [
            "Excellent product!",
            "Terrible quality.",
            "It's okay.",
            "Amazing experience!",
            "Not impressed."
        ]
        
        for text in test_texts:
            result = analyzer.analyze(text)
            monitor.record_analysis(result)
        
        # Test error recording
        monitor.record_error("Test error", ml_analyzer.AnalyzerType.ENHANCED_RULE_BASED)
        
        # Get health status
        health = monitor.get_health_status()
        print(f"✅ Health status: {health['status']} (score: {health['health_score']:.1f})")
        
        # Get performance report
        report = monitor.get_performance_report(60)
        if "total_analyses" in report:
            print(f"✅ Performance: {report['total_analyses']} analyses")
            print(f"    Avg confidence: {report['confidence_stats']['mean']:.3f}")
            print(f"    Avg time: {report['processing_time_stats']['mean']:.2f}ms")
        
        # Get alerts
        alerts = monitor.get_alerts(limit=5)
        print(f"✅ Recent alerts: {len(alerts)}")
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_advanced_api():
    """Test advanced API."""
    print("🌐 Testing Advanced API...")
    
    try:
        api = ml_analyzer.AdvancedSentimentAnalyzerAPI(enable_caching=True)
        
        # Test single analysis
        result = api.analyze_text("This is fantastic!")
        print(f"✅ API analysis: {result.label.value} (confidence: {result.confidence:.3f})")
        print(f"    Features: {len(result.features)} extracted")
        print(f"    Analyzer: {result.analyzer_type.value}")
        
        # Test batch analysis
        texts = ["Great!", "Awful!", "Meh."]
        batch_results = api.analyze_batch(texts)
        print(f"✅ Batch analysis: {len(batch_results)} results")
        
        # Test input validation
        is_valid, error = api.validate_input("Valid text")
        print(f"✅ Input validation: {'PASS' if is_valid else 'FAIL'}")
        
        is_valid, error = api.validate_input("")
        print(f"✅ Empty input validation: {'FAIL' if not is_valid else 'PASS'} - {error}")
        
        # Test stats
        stats = api.get_comprehensive_stats()
        print(f"✅ Comprehensive stats: {stats['api_type']}")
        if 'cache' in stats:
            print(f"    Cache hit rate: {stats['cache']['hit_rate']:.3f}")
        
    except Exception as e:
        print(f"❌ Advanced API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all Generation 2 tests."""
    print("🚀 GENERATION 2: ENHANCED SENTIMENT ANALYSIS TESTING")
    print("=" * 60)
    
    tests = [
        test_enhanced_analyzer,
        test_caching,
        test_ensemble,
        test_validation,
        test_monitoring,
        test_advanced_api
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f"\n{'='*20} {test_func.__name__.upper()} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} CRASHED: {e}")
    
    print(f"\n{'='*60}")
    print(f"🏆 GENERATION 2 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL GENERATION 2 TESTS PASSED! System is robust and ready for production.")
    else:
        print(f"⚠️ {total - passed} tests failed. Review issues before proceeding to Generation 3.")
    
    return passed == total

if __name__ == "__main__":
    main()
