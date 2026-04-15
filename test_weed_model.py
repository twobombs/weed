#!/usr/bin/env python3
"""
Test script to load the .weed model and run inference.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weed_loader.weed_module import WeedModule
from weed_loader.weed_tensor import WeedTensor
from weed_loader.dtype import DType

def test_model_loading():
    """Test loading the Qwen .weed model."""
    model_path = "Qwen3.5-2B-Q4_K_S.weed"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    print(f"Loading model from: {model_path}")
    try:
        model = WeedModule(model_path)
        print("✓ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_inference(model):
    """Test running inference with the loaded model."""
    print("\n=== Running Inference Test ===")
    
    # Create a simple input tensor (token IDs)
    # Using a few sample token IDs
    sample_tokens = [151643, 198, 7, 257, 13212, 317, 940, 328, 14955, 13]
    
    print(f"Input tokens: {sample_tokens}")
    
    try:
        # Create input tensor
        input_tensor = WeedTensor(
            data=sample_tokens,
            shape=[len(sample_tokens)],
            stride=[1],
            dtype=DType.INT,
            offset=0
        )
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Input dtype: {input_tensor.dtype}")
        
        # Run forward pass
        print("\nRunning forward pass...")
        logits = model.forward(input_tensor)
        
        print(f"✓ Inference completed!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output dtype: {logits.dtype}")
        
        # Show some output stats
        if hasattr(logits, 'data') and len(logits.data) > 0:
            print(f"  Output size: {len(logits.data)} elements")
            print(f"  Sample values (first 5): {logits.data[:5]}")
        
        return logits
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tokenizer():
    """Test loading and using the tokenizer."""
    print("\n=== Tokenizer Test ===")
    
    tokenizer_dir = "Qwen3.5-2B-tokenizer"
    
    if not os.path.exists(tokenizer_dir):
        print(f"Warning: Tokenizer directory not found: {tokenizer_dir}")
        return None
    
    try:
        from tokenizers import Tokenizer as HFTokenizer
        
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = HFTokenizer.from_file(tokenizer_path)
            print(f"✓ Tokenizer loaded from: {tokenizer_path}")
            
            # Test encoding
            test_text = "Hello, world!"
            encoding = tokenizer.encode(test_text)
            print(f"  Test encoding: '{test_text}' -> {encoding.ids}")
            
            # Test decoding
            decoded = tokenizer.decode(encoding.ids)
            print(f"  Test decoding: {encoding.ids} -> '{decoded}'")
            
            return tokenizer
        else:
            print(f"Error: tokenizer.json not found in {tokenizer_dir}")
            return None
            
    except ImportError:
        print("Warning: tokenizers library not installed. Skipping tokenizer test.")
        return None
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return None

def main():
    """Run all tests."""
    print("=" * 60)
    print("Weed Model Test Suite")
    print("=" * 60)
    
    # Test 1: Load model
    model = test_model_loading()
    if not model:
        print("\n✗ Model loading failed. Stopping tests.")
        return 1
    
    # Test 2: Run inference
    logits = test_inference(model)
    if not logits:
        print("\n✗ Inference test failed.")
        return 1
    
    # Test 3: Tokenizer
    tokenizer = test_tokenizer()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
