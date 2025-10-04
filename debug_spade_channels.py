#!/usr/bin/env python3
"""
Debug script for SPADE channel mismatch issue in StableSR Edge v2

The error occurs because:
1. SPADE expects semb_channels=256 input channels (from config)
2. But it's receiving only 4 channels (likely from edge processing)
3. This suggests struct_cond is not being passed correctly or has wrong dimensions

This script will help identify where the channel mismatch occurs.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to Python path
sys.path.append('/home/tra/pd/StableSR_Edge_v2')

from ldm.modules.diffusionmodules.unet_with_edge import UNetModelDualcondV2WithEdge
from ldm.modules.spade import SPADE

def debug_spade_channels():
    """Debug the SPADE channel mismatch issue"""
    
    print("=== Debugging SPADE Channel Mismatch ===")
    print()
    
    # 1. Check SPADE initialization parameters
    print("1. SPADE Module Configuration:")
    norm_nc = 128  # This should match the ResBlockDual out_channels
    label_nc = 256  # This should match semb_channels from config
    spade = SPADE(norm_nc, label_nc, 'spadeinstance3x3')
    print(f"   SPADE expects input with {label_nc} channels")
    print(f"   SPADE mlp_shared first layer: {spade.mlp_shared[0]}")
    print()
    
    # 2. Test SPADE with correct input
    print("2. Testing SPADE with correct input dimensions:")
    batch_size = 2
    correct_input = torch.randn(batch_size, 256, 64, 64)  # 256 channels as expected
    x_dic = torch.randn(batch_size, 128, 64, 64)  # Feature tensor
    
    try:
        output = spade(x_dic, correct_input)
        print(f"   ✓ SPADE works with correct input: {correct_input.shape}")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ SPADE failed even with correct input: {e}")
    print()
    
    # 3. Test SPADE with wrong input (4 channels)
    print("3. Testing SPADE with wrong input (4 channels - this should fail):")
    wrong_input = torch.randn(batch_size, 4, 64, 64)  # 4 channels - this is the problem
    
    try:
        output = spade(x_dic, wrong_input)
        print(f"   ✗ SPADE should have failed but didn't!")
    except Exception as e:
        print(f"   ✓ SPADE correctly failed with wrong input: {e}")
    print()
    
    # 4. Check UNet model configuration
    print("4. UNet Model Configuration:")
    try:
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
            semb_channels=256,  # This should be 256
            use_edge_processing=True,
            edge_input_channels=3,
        )
        print("   ✓ UNet model created successfully")
        print(f"   semb_channels: 256")
        print(f"   use_edge_processing: {model.use_edge_processing}")
    except Exception as e:
        print(f"   ✗ UNet model creation failed: {e}")
        return
    print()
    
    # 5. Test forward pass with edge processing
    print("5. Testing forward pass with edge processing:")
    try:
        # Create test inputs
        unet_input = torch.randn(batch_size, 4, 64, 64)
        edge_map = torch.randn(batch_size, 3, 512, 512)
        timesteps = torch.randint(0, 1000, (batch_size,))
        context = torch.randn(batch_size, 77, 1024)
        struct_cond = torch.randn(batch_size, 256, 96, 96)  # This should be 256 channels!
        
        print(f"   Input shapes:")
        print(f"     unet_input: {unet_input.shape}")
        print(f"     edge_map: {edge_map.shape}")
        print(f"     struct_cond: {struct_cond.shape} (should be 256 channels)")
        print(f"     context: {context.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(
                x=unet_input,
                timesteps=timesteps,
                context=context,
                struct_cond=struct_cond,
                edge_map=edge_map
            )
        
        print(f"   ✓ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        print(f"   This is likely the same error you're seeing in training")
    print()
    
    # 6. Debug edge fusion
    print("6. Debugging edge fusion:")
    try:
        edge_features = model.edge_processor(edge_map)
        print(f"   Edge features shape: {edge_features.shape}")
        
        fused_input = model.edge_fusion(unet_input, edge_features)
        print(f"   Fused input shape: {fused_input.shape}")
        print(f"   Expected: [batch_size, 8, 64, 64] (4 + 4 = 8 channels)")
        
    except Exception as e:
        print(f"   ✗ Edge fusion failed: {e}")
    print()
    
    # 7. Recommendations
    print("7. Recommendations to fix the issue:")
    print("   a) Check that struct_cond has 256 channels (not 4)")
    print("   b) Verify edge processing is not interfering with struct_cond")
    print("   c) Check if edge_map is being passed as struct_cond by mistake")
    print("   d) Look at the training data loading - struct_cond might be wrong shape")
    print()

def check_tensor_shapes_in_training():
    """Check what tensor shapes are being passed during training"""
    print("=== Checking Training Tensor Shapes ===")
    print()
    
    # This function would be called from your training script
    # to debug the actual tensor shapes during training
    
    print("To debug during training, add this to your training script:")
    print()
    print("```python")
    print("# In your training_step method, before calling apply_model:")
    print("print(f'struct_cond shape: {struct_cond.shape if struct_cond is not None else None}')")
    print("print(f'edge_map shape: {edge_map.shape if edge_map is not None else None}')")
    print("print(f'x_noisy shape: {x_noisy.shape}')")
    print()
    print("# Check if edge_map is being passed as struct_cond:")
    print("if edge_map is not None and struct_cond is not None:")
    print("    print(f'Are they the same tensor? {torch.equal(edge_map, struct_cond)}')")
    print("    print(f'edge_map channels: {edge_map.shape[1]}')")
    print("    print(f'struct_cond channels: {struct_cond.shape[1]}')")
    print("```")

if __name__ == "__main__":
    debug_spade_channels()
    check_tensor_shapes_in_training()
