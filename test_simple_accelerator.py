#!/usr/bin/env python3
"""
Simple accelerator test without complex dependencies
Generation 1: Basic functionality verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_simple_accelerator():
    """Test basic accelerator design without complex dependencies."""
    
    print("🧠 Testing Basic AcceleratorDesigner...")
    
    # Test core imports without complex dependencies
    from codesign_playground.core.accelerator import AcceleratorDesigner, Accelerator
    
    designer = AcceleratorDesigner()
    print("✅ AcceleratorDesigner created successfully")
    
    # Create basic accelerator
    accelerator = designer.design(
        compute_units=32,
        memory_hierarchy=["sram_64kb"],
        dataflow="weight_stationary"
    )
    
    print(f"✅ Accelerator created: {accelerator.compute_units} compute units")
    print(f"✅ Memory hierarchy: {accelerator.memory_hierarchy}")
    print(f"✅ Dataflow: {accelerator.dataflow}")
    
    # Test basic properties
    accelerator_dict = accelerator.to_dict()
    print(f"✅ Accelerator serialization: {len(accelerator_dict)} properties")
    
    print("\n🎯 Generation 1: Basic Functionality - COMPLETE ✅")
    print("🚀 Ready for Generation 2: Robust Implementation")
    
    return True

if __name__ == "__main__":
    test_simple_accelerator()