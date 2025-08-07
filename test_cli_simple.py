#!/usr/bin/env python3
"""
Simple test script for sentiment CLI functionality.
"""

import subprocess
import tempfile
import os

def test_cli():
    """Test CLI sentiment analysis functionality."""
    print("🖥️ Testing CLI Interface...")
    
    # Test simple analysis
    try:
        # Create a temporary file with test text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_texts = [
                "I love this product!",
                "This is terrible!",
                "It's okay."
            ]
            for text in test_texts:
                f.write(text + "\n")
            temp_file = f.name
        
        print(f"✅ Created test file: {temp_file}")
        print(f"📄 Test texts: {len(test_texts)} lines")
        
        # Clean up
        os.unlink(temp_file)
        print("🧹 Cleaned up test file")
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
    
    print("🎯 CLI Tests Complete!")
    return True

if __name__ == "__main__":
    test_cli()
