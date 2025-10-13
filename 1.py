#!/usr/bin/env python
"""
Script to check what keys are in a checkpoint and their shapes
"""

import torch
import sys

def check_checkpoint(ckpt_path):
    """Check checkpoint contents"""
    print(f"\nLoading checkpoint: {ckpt_path}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Get state dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"✓ Checkpoint contains 'state_dict' key")
        else:
            state_dict = ckpt
            print(f"✓ Checkpoint is a state_dict directly")
        
        print(f"\nTotal keys in checkpoint: {len(state_dict)}")
        
        # Check for first conv layer
        first_conv_key = 'model.diffusion_model.input_blocks.0.0.weight'
        if first_conv_key in state_dict:
            shape = state_dict[first_conv_key].shape
            print(f"\n✓ First conv layer found:")
            print(f"  Key: {first_conv_key}")
            print(f"  Shape: {shape}")
            print(f"  Input channels: {shape[1]}")
            if shape[1] == 8:
                print(f"  → This is an EDGE-ENABLED checkpoint (8 input channels)")
            elif shape[1] == 4:
                print(f"  → This is a STANDARD checkpoint (4 input channels)")
        else:
            print(f"\n✗ First conv layer not found with key: {first_conv_key}")
        
        # Check for EdgeMapProcessor
        edge_keys = [k for k in state_dict.keys() if 'edge_processor' in k]
        if edge_keys:
            print(f"\n✓ EdgeMapProcessor found ({len(edge_keys)} keys):")
            for key in edge_keys[:10]:  # Show first 10
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"  - {key}: {shape}")
            if len(edge_keys) > 10:
                print(f"  ... and {len(edge_keys) - 10} more keys")
        else:
            print(f"\n✗ No EdgeMapProcessor parameters found")
        
        # Count total parameters
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        print(f"\nTotal parameters: {total_params:,}")
        
        # Show some example keys
        print(f"\nExample keys (first 20):")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {i+1}. {key}: {shape}")
        
        # Check for other components
        print(f"\nOther components:")
        vae_keys = [k for k in state_dict.keys() if 'first_stage_model' in k]
        print(f"  - VAE (first_stage_model): {len(vae_keys)} keys")
        
        text_keys = [k for k in state_dict.keys() if 'cond_stage_model' in k]
        print(f"  - Text Encoder (cond_stage_model): {len(text_keys)} keys")
        
        struct_keys = [k for k in state_dict.keys() if 'structcond_stage_model' in k]
        print(f"  - Structural Encoder (structcond_stage_model): {len(struct_keys)} keys")
        
        unet_keys = [k for k in state_dict.keys() if 'diffusion_model' in k and 'edge_processor' not in k]
        print(f"  - UNet (diffusion_model, excluding edge): {len(unet_keys)} keys")
        
        # Metadata
        if 'epoch' in ckpt:
            print(f"\nTraining metadata:")
            print(f"  - Epoch: {ckpt['epoch']}")
        if 'global_step' in ckpt:
            print(f"  - Global step: {ckpt['global_step']}")
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint_keys.py <checkpoint_path>")
        print("\nExample:")
        print("  python check_checkpoint_keys.py checkpoints/model_epoch_10.ckpt")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    check_checkpoint(ckpt_path)
