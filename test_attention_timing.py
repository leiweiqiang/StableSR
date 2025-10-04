#!/usr/bin/env python3
"""
Test script to measure attention initialization timing
"""

import time
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_attention_initialization():
    """Test different attention initialization times"""
    
    print("=" * 60)
    print("Testing Attention Initialization Timing")
    print("=" * 60)
    
    # Test 1: Standard AttnBlock
    print("\n1. Testing AttnBlock initialization...")
    start_time = time.time()
    
    try:
        from ldm.modules.diffusionmodules.model import AttnBlock
        attn_block = AttnBlock(in_channels=512)
        end_time = time.time()
        print(f"   AttnBlock(512) initialization time: {end_time - start_time:.4f} seconds")
        print(f"   ✓ AttnBlock initialized successfully")
    except Exception as e:
        print(f"   ❌ AttnBlock failed: {e}")
    
    # Test 2: MemoryEfficientAttnBlock
    print("\n2. Testing MemoryEfficientAttnBlock initialization...")
    start_time = time.time()
    
    try:
        from ldm.modules.diffusionmodules.model import MemoryEfficientAttnBlock
        mem_attn_block = MemoryEfficientAttnBlock(in_channels=512)
        end_time = time.time()
        print(f"   MemoryEfficientAttnBlock(512) initialization time: {end_time - start_time:.4f} seconds")
        print(f"   ✓ MemoryEfficientAttnBlock initialized successfully")
    except Exception as e:
        print(f"   ❌ MemoryEfficientAttnBlock failed: {e}")
    
    # Test 3: make_attn function
    print("\n3. Testing make_attn function...")
    start_time = time.time()
    
    try:
        from ldm.modules.diffusionmodules.model import make_attn
        attn = make_attn(in_channels=512, attn_type="vanilla")
        end_time = time.time()
        print(f"   make_attn(512, 'vanilla') initialization time: {end_time - start_time:.4f} seconds")
        print(f"   ✓ make_attn function completed successfully")
        print(f"   Attention type created: {type(attn).__name__}")
    except Exception as e:
        print(f"   ❌ make_attn failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check xformers availability
    print("\n4. Checking xformers availability...")
    try:
        from ldm.modules.diffusionmodules.model import XFORMERS_IS_AVAILBLE
        print(f"   XFORMERS_IS_AVAILBLE: {XFORMERS_IS_AVAILBLE}")
        
        if XFORMERS_IS_AVAILBLE:
            try:
                import xformers
                print(f"   xformers version: {xformers.__version__}")
            except:
                print("   xformers import failed")
        else:
            print("   xformers not available")
    except Exception as e:
        print(f"   ❌ xformers check failed: {e}")
    
    # Test 5: Test with different channel sizes
    print("\n5. Testing with different channel sizes...")
    channel_sizes = [64, 128, 256, 512, 1024]
    
    for channels in channel_sizes:
        start_time = time.time()
        try:
            attn = make_attn(in_channels=channels, attn_type="vanilla")
            end_time = time.time()
            print(f"   make_attn({channels}, 'vanilla'): {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"   ❌ make_attn({channels}) failed: {e}")
    
    print("\n" + "=" * 60)
    print("Attention initialization timing test completed!")
    print("=" * 60)


def test_model_initialization():
    """Test full model initialization timing"""
    print("\n" + "=" * 60)
    print("Testing Full Model Initialization Timing")
    print("=" * 60)
    
    # Test UNetModelDualcondV2WithEdge initialization
    print("\nTesting UNetModelDualcondV2WithEdge initialization...")
    start_time = time.time()
    
    try:
        from ldm.modules.diffusionmodules.unet_with_edge import UNetModelDualcondV2WithEdge
        
        # Create model with edge processing enabled
        model = UNetModelDualcondV2WithEdge(
            image_size=32,
            in_channels=4,
            model_channels=320,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=[4, 2, 1],
            channel_mult=[1, 2, 4, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            context_dim=1024,
            semb_channels=256,
            use_edge_processing=True,
            edge_input_channels=3,
        )
        
        end_time = time.time()
        print(f"   UNetModelDualcondV2WithEdge initialization time: {end_time - start_time:.4f} seconds")
        print(f"   ✓ Model initialized successfully")
        
        # Count attention modules
        attn_count = 0
        for name, module in model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'in_channels'):
                attn_count += 1
        
        print(f"   Total attention modules: {attn_count}")
        
    except Exception as e:
        print(f"   ❌ UNetModelDualcondV2WithEdge initialization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_attention_initialization()
    test_model_initialization()
