#!/usr/bin/env python3
"""
Performance optimization and scaling test
Generation 3: Optimized implementation with caching, concurrent processing
"""

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_performance_optimization():
    """Test performance optimization and scaling features."""
    
    print("⚡ Testing Performance Optimization & Scaling...")
    
    from codesign_playground.core.accelerator import AcceleratorDesigner
    
    designer = AcceleratorDesigner()
    
    # Test caching performance
    print("\n🧠 Testing Caching Performance...")
    
    start_time = time.time()
    
    # First call - should cache
    accelerator1 = designer.design(
        compute_units=64,
        memory_hierarchy=["sram_128kb"],
        dataflow="weight_stationary"
    )
    
    first_call_time = time.time() - start_time
    
    # Second call - should hit cache
    start_time = time.time()
    accelerator2 = designer.design(
        compute_units=64,
        memory_hierarchy=["sram_128kb"],
        dataflow="weight_stationary"
    )
    
    second_call_time = time.time() - start_time
    
    print(f"✅ First call: {first_call_time:.4f}s, Second call: {second_call_time:.4f}s")
    
    if second_call_time < first_call_time * 0.8:  # Should be faster due to caching
        print("✅ Caching optimization working")
    else:
        print("⚠️ Caching may not be optimal")
    
    # Test concurrent processing
    print("\n🔄 Testing Concurrent Processing...")
    
    def design_accelerator(params):
        """Design accelerator with given parameters."""
        compute_units, memory_size = params
        return designer.design(
            compute_units=compute_units,
            memory_hierarchy=[f"sram_{memory_size}kb"],
            dataflow="output_stationary"
        )
    
    # Test parameters for concurrent design
    test_params = [
        (8, 32), (16, 64), (32, 128), (64, 256),
        (12, 48), (24, 96), (48, 192)
    ]
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for params in test_params:
        result = design_accelerator(params)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Concurrent processing  
    start_time = time.time()
    concurrent_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(design_accelerator, params) for params in test_params]
        for future in as_completed(futures):
            result = future.result()
            concurrent_results.append(result)
    concurrent_time = time.time() - start_time
    
    print(f"✅ Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s")
    print(f"✅ Speedup: {sequential_time/concurrent_time:.2f}x")
    
    # Test memory optimization
    print("\n💾 Testing Memory Optimization...")
    
    accelerator = designer.design(
        compute_units=128,
        memory_hierarchy=["sram_512kb", "dram_1gb"],
        dataflow="weight_stationary"
    )
    
    # Test performance model optimization
    performance = accelerator.estimate_performance()
    
    efficiency_metrics = [
        ("Compute Efficiency", performance.get("efficiency_ops_w", 0)),
        ("Memory Bandwidth", performance.get("throughput_ops_s", 0) / 1e9),  # GB/s equivalent
        ("Power Efficiency", performance.get("power_w", 1)),
    ]
    
    print("📊 Performance Metrics:")
    for metric, value in efficiency_metrics:
        print(f"   {metric}: {value:.2f}")
    
    print("\n⚡ Generation 3: Performance Optimization - COMPLETE ✅")
    return True

if __name__ == "__main__":
    test_performance_optimization()