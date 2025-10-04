#!/usr/bin/env python3
"""
Debug script to analyze AttnBlock initialization bottleneck
"""

import time
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_attn_block_initialization():
    """Debug AttnBlock initialization step by step"""
    
    print("=" * 60)
    print("Debugging AttnBlock Initialization")
    print("=" * 60)
    
    in_channels = 512
    
    # Step 1: Test individual components
    print(f"\nTesting individual components for {in_channels} channels:")
    
    # Test Conv2d creation
    start_time = time.time()
    conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    end_time = time.time()
    print(f"   Conv2d({in_channels}, {in_channels}, 1x1) creation: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    end_time = time.time()
    print(f"   Conv2d({in_channels}, {in_channels}, 1x1) creation: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    end_time = time.time()
    print(f"   Conv2d({in_channels}, {in_channels}, 1x1) creation: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    conv_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    end_time = time.time()
    print(f"   Conv2d({in_channels}, {in_channels}, 1x1) creation: {end_time - start_time:.4f} seconds")
    
    # Test Normalize
    start_time = time.time()
    try:
        from ldm.modules.diffusionmodules.model import Normalize
        norm = Normalize(in_channels)
        end_time = time.time()
        print(f"   Normalize({in_channels}) creation: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"   Normalize creation failed: {e}")
    
    # Step 2: Test with different channel sizes
    print(f"\nTesting Conv2d creation with different channel sizes:")
    channel_sizes = [64, 128, 256, 512, 1024, 2048]
    
    for channels in channel_sizes:
        start_time = time.time()
        conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        end_time = time.time()
        print(f"   Conv2d({channels:4d}, {channels:4d}, 1x1): {end_time - start_time:.4f} seconds")
    
    # Step 3: Test parameter initialization
    print(f"\nTesting parameter initialization for {in_channels} channels:")
    
    # Create conv layer
    conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    # Test weight initialization
    start_time = time.time()
    with torch.no_grad():
        nn.init.kaiming_normal_(conv.weight)
    end_time = time.time()
    print(f"   Kaiming normal weight init: {end_time - start_time:.4f} seconds")
    
    # Test bias initialization
    start_time = time.time()
    with torch.no_grad():
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
    end_time = time.time()
    print(f"   Bias zero init: {end_time - start_time:.4f} seconds")
    
    # Step 4: Test memory allocation
    print(f"\nTesting memory allocation patterns:")
    
    # Test CUDA memory allocation (if available)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   CUDA available: {torch.cuda.get_device_name()}")
        
        start_time = time.time()
        conv_cuda = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0).to(device)
        end_time = time.time()
        print(f"   Conv2d creation + CUDA transfer: {end_time - start_time:.4f} seconds")
        
        # Test forward pass to initialize lazy operations
        start_time = time.time()
        x = torch.randn(1, in_channels, 64, 64).to(device)
        with torch.no_grad():
            _ = conv_cuda(x)
        end_time = time.time()
        print(f"   First forward pass: {end_time - start_time:.4f} seconds")
    else:
        print("   CUDA not available")
    
    # Step 5: Test full AttnBlock creation with timing
    print(f"\nTesting full AttnBlock creation with detailed timing:")
    
    try:
        from ldm.modules.diffusionmodules.model import AttnBlock
        
        # Create with detailed timing
        start_time = time.time()
        
        # Simulate the __init__ process step by step
        print("   Creating AttnBlock components...")
        
        # Normalize
        step_start = time.time()
        norm = Normalize(in_channels)
        step_end = time.time()
        print(f"     Normalize: {step_end - step_start:.4f} seconds")
        
        # Q, K, V convolutions
        step_start = time.time()
        q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        step_end = time.time()
        print(f"     Q conv: {step_end - step_start:.4f} seconds")
        
        step_start = time.time()
        k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        step_end = time.time()
        print(f"     K conv: {step_end - step_start:.4f} seconds")
        
        step_start = time.time()
        v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        step_end = time.time()
        print(f"     V conv: {step_end - step_start:.4f} seconds")
        
        # Projection
        step_start = time.time()
        proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        step_end = time.time()
        print(f"     Proj out: {step_end - step_start:.4f} seconds")
        
        end_time = time.time()
        print(f"   Total component creation: {end_time - start_time:.4f} seconds")
        
        # Now create actual AttnBlock
        start_time = time.time()
        attn_block = AttnBlock(in_channels)
        end_time = time.time()
        print(f"   Actual AttnBlock creation: {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"   ❌ AttnBlock creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_attn_block_initialization()
