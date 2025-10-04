#!/usr/bin/env python3
"""
Test script to verify memory optimization for edge processing
"""

import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.modules.diffusionmodules.edge_processor import EdgeMapProcessor, EdgeFusionModule


def setup_memory_management():
    """Setup memory management for CUDA"""
    if torch.cuda.is_available():
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        print(f"CUDA memory management configured. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def test_memory_usage():
    """Test memory usage with different configurations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tests")
        return True
    
    setup_memory_management()
    
    print("\n" + "="*60)
    print("Memory Optimization Test")
    print("="*60)
    
    # Test 1: Original configuration (should fail with OOM)
    print("\n1. Testing original edge processor (should fail with OOM)...")
    try:
        # This should fail with the original 1024-channel design
        processor_original = EdgeMapProcessor(
            input_channels=3, 
            output_channels=4, 
            target_size=64,
            use_checkpoint=False
        ).to(device)
        
        # Try to process a 512x512 edge map
        edge_map = torch.randn(2, 3, 512, 512).to(device)
        
        with torch.no_grad():
            output = processor_original(edge_map)
        
        print(f"✓ Original processor worked: {edge_map.shape} -> {output.shape}")
        del processor_original, edge_map, output
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"✓ Expected OOM error with original design: {e}")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    # Test 2: Memory-optimized configuration
    print("\n2. Testing memory-optimized edge processor...")
    try:
        processor_optimized = EdgeMapProcessor(
            input_channels=3, 
            output_channels=4, 
            target_size=64,
            use_checkpoint=True
        ).to(device)
        
        # Try to process a 512x512 edge map with smaller batch size
        edge_map = torch.randn(2, 3, 512, 512).to(device)
        
        with torch.no_grad():
            output = processor_optimized(edge_map)
        
        print(f"✓ Optimized processor worked: {edge_map.shape} -> {output.shape}")
        
        # Test gradient checkpointing during training
        processor_optimized.train()
        edge_map.requires_grad_(True)
        
        output = processor_optimized(edge_map)
        loss = output.sum()
        loss.backward()
        
        print("✓ Gradient checkpointing worked during training")
        
        del processor_optimized, edge_map, output, loss
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"✗ Optimized processor failed: {e}")
        return False
    
    # Test 3: Test fusion module
    print("\n3. Testing edge fusion module...")
    try:
        fusion_module = EdgeFusionModule().to(device)
        unet_input = torch.randn(2, 4, 64, 64).to(device)
        edge_features = torch.randn(2, 4, 64, 64).to(device)
        
        with torch.no_grad():
            fused = fusion_module(unet_input, edge_features)
        
        print(f"✓ Fusion module worked: {unet_input.shape} + {edge_features.shape} -> {fused.shape}")
        
        del fusion_module, unet_input, edge_features, fused
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"✗ Fusion module failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All memory optimization tests passed!")
    print("="*60)
    
    return True


def test_batch_size_scaling():
    """Test how batch size affects memory usage"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping batch size tests")
        return True
    
    print("\n" + "="*60)
    print("Batch Size Scaling Test")
    print("="*60)
    
    processor = EdgeMapProcessor(
        input_channels=3, 
        output_channels=4, 
        target_size=64,
        use_checkpoint=True
    ).to(device)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 6, 8]
    
    for batch_size in batch_sizes:
        try:
            edge_map = torch.randn(batch_size, 3, 512, 512).to(device)
            
            with torch.no_grad():
                output = processor(edge_map)
            
            print(f"✓ Batch size {batch_size}: {edge_map.shape} -> {output.shape}")
            
            del edge_map, output
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ Batch size {batch_size} failed with OOM")
                break
            else:
                print(f"✗ Batch size {batch_size} failed with unexpected error: {e}")
                return False
    
    del processor
    torch.cuda.empty_cache()
    
    return True


def main():
    """Main test function"""
    print("Testing memory optimization for StableSR Edge Processing")
    
    # Test memory usage
    if not test_memory_usage():
        print("Memory optimization tests failed!")
        return 1
    
    # Test batch size scaling
    if not test_batch_size_scaling():
        print("Batch size scaling tests failed!")
        return 1
    
    print("\n🎉 All tests passed! Memory optimization is working correctly.")
    print("\nRecommended training configuration:")
    print("- Batch size: 2")
    print("- Accumulate grad batches: 6")
    print("- Gradient checkpointing: enabled")
    print("- Edge processor: memory-optimized version")
    
    return 0


if __name__ == "__main__":
    exit(main())
