#!/usr/bin/env python3
"""
Test script to analyze GroupNorm initialization timing
"""

import time
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_groupnorm_timing():
    """Test GroupNorm initialization timing"""
    
    print("=" * 60)
    print("Testing GroupNorm Initialization Timing")
    print("=" * 60)
    
    # Test different channel sizes
    channel_sizes = [64, 128, 256, 512, 1024, 2048]
    
    for channels in channel_sizes:
        print(f"\nTesting GroupNorm with {channels} channels:")
        
        # Test GroupNorm creation
        start_time = time.time()
        groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        end_time = time.time()
        print(f"   GroupNorm(32, {channels}) creation: {end_time - start_time:.4f} seconds")
        
        # Test parameter initialization
        start_time = time.time()
        with torch.no_grad():
            nn.init.ones_(groupnorm.weight)
            nn.init.zeros_(groupnorm.bias)
        end_time = time.time()
        print(f"   Parameter initialization: {end_time - start_time:.4f} seconds")
        
        # Test forward pass
        start_time = time.time()
        x = torch.randn(1, channels, 64, 64)
        with torch.no_grad():
            _ = groupnorm(x)
        end_time = time.time()
        print(f"   First forward pass: {end_time - start_time:.4f} seconds")
    
    # Test different num_groups
    print(f"\nTesting GroupNorm with different num_groups for 512 channels:")
    num_groups_list = [1, 8, 16, 32, 64, 128, 256, 512]
    
    for num_groups in num_groups_list:
        if 512 % num_groups == 0:  # num_groups must divide num_channels
            start_time = time.time()
            groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=512, eps=1e-6, affine=True)
            end_time = time.time()
            print(f"   GroupNorm({num_groups:3d}, 512) creation: {end_time - start_time:.4f} seconds")
    
    # Test LayerNorm vs GroupNorm
    print(f"\nComparing LayerNorm vs GroupNorm for 512 channels:")
    
    start_time = time.time()
    layernorm = nn.LayerNorm(512)
    end_time = time.time()
    print(f"   LayerNorm(512) creation: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    groupnorm = nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6, affine=True)
    end_time = time.time()
    print(f"   GroupNorm(32, 512) creation: {end_time - start_time:.4f} seconds")
    
    # Test BatchNorm vs GroupNorm
    print(f"\nComparing BatchNorm vs GroupNorm for 512 channels:")
    
    start_time = time.time()
    batchnorm = nn.BatchNorm2d(512)
    end_time = time.time()
    print(f"   BatchNorm2d(512) creation: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    groupnorm = nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6, affine=True)
    end_time = time.time()
    print(f"   GroupNorm(32, 512) creation: {end_time - start_time:.4f} seconds")
    
    # Test with CUDA
    if torch.cuda.is_available():
        print(f"\nTesting with CUDA:")
        device = torch.device('cuda')
        
        start_time = time.time()
        groupnorm_cuda = nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6, affine=True).to(device)
        end_time = time.time()
        print(f"   GroupNorm(32, 512) creation + CUDA transfer: {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        x = torch.randn(1, 512, 64, 64).to(device)
        with torch.no_grad():
            _ = groupnorm_cuda(x)
        end_time = time.time()
        print(f"   First forward pass on CUDA: {end_time - start_time:.4f} seconds")


def test_normalize_function():
    """Test the Normalize function specifically"""
    
    print("\n" + "=" * 60)
    print("Testing Normalize Function")
    print("=" * 60)
    
    try:
        from ldm.modules.diffusionmodules.model import Normalize
        
        channel_sizes = [64, 128, 256, 512, 1024]
        
        for channels in channel_sizes:
            print(f"\nTesting Normalize({channels}):")
            
            start_time = time.time()
            norm = Normalize(channels)
            end_time = time.time()
            print(f"   Normalize({channels}) creation: {end_time - start_time:.4f} seconds")
            
            # Test forward pass
            start_time = time.time()
            x = torch.randn(1, channels, 64, 64)
            with torch.no_grad():
                _ = norm(x)
            end_time = time.time()
            print(f"   Forward pass: {end_time - start_time:.4f} seconds")
            
    except Exception as e:
        print(f"   ❌ Normalize function test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_groupnorm_timing()
    test_normalize_function()
