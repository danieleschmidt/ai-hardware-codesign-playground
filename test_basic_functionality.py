#!/usr/bin/env python3
"""
Basic functionality test for AI Hardware Co-Design Playground
Generation 1: Simple implementation test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from codesign_playground import AcceleratorDesigner, ModelOptimizer, DesignSpaceExplorer

def test_basic_functionality():
    """Test basic functionality of core components."""
    
    print("🧠 Testing AcceleratorDesigner...")
    designer = AcceleratorDesigner()
    
    # Create a simple accelerator design
    accelerator = designer.design(
        compute_units=32,
        memory_hierarchy=["sram_64kb"],
        dataflow="weight_stationary"
    )
    print(f"✅ Created accelerator with {accelerator.compute_units} compute units")
    
    print("\n🎯 Testing ModelOptimizer...")
    # Create dummy model for testing
    dummy_model = {"type": "test_model", "layers": 3}
    optimizer = ModelOptimizer(dummy_model, accelerator)
    print("✅ ModelOptimizer instantiated successfully")
    
    print("\n🔍 Testing DesignSpaceExplorer...")
    explorer = DesignSpaceExplorer()
    
    # Simple design space
    design_space = {
        "compute_units": [16, 32],
        "memory_size_kb": [32, 64],
    }
    print(f"✅ Design space defined with {len(design_space)} parameters")
    
    print("\n🚀 Generation 1 Basic Functionality: COMPLETE")
    return True

if __name__ == "__main__":
    test_basic_functionality()