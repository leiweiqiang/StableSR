#!/usr/bin/env python
"""
Quick diagnostic script to test edge model loading and inference
"""
import torch
import sys
sys.path.insert(0, '/root/dp/StableSR_Edge_v2')

from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
import numpy as np

def test_model_loading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    config = OmegaConf.load(config_path)
    
    print("="*80)
    print("Creating model...")
    print("="*80)
    
    # Create model
    model = LatentDiffusionSRTextWTWithEdge(
        first_stage_config=config.model.params.first_stage_config,
        cond_stage_config=config.model.params.cond_stage_config,
        structcond_stage_config=config.model.params.structcond_stage_config,
        num_timesteps_cond=config.model.params.get('num_timesteps_cond', 1),
        cond_stage_key=config.model.params.get('cond_stage_key', 'image'),
        cond_stage_trainable=config.model.params.get('cond_stage_trainable', False),
        concat_mode=config.model.params.get('concat_mode', True),
        conditioning_key=config.model.params.get('conditioning_key', 'crossattn'),
        scale_factor=config.model.params.get('scale_factor', 0.18215),
        scale_by_std=config.model.params.get('scale_by_std', False),
        unfrozen_diff=config.model.params.get('unfrozen_diff', False),
        random_size=config.model.params.get('random_size', False),
        test_gt=config.model.params.get('test_gt', False),
        p2_gamma=config.model.params.get('p2_gamma', None),
        p2_k=config.model.params.get('p2_k', None),
        time_replace=config.model.params.get('time_replace', 1000),
        use_usm=config.model.params.get('use_usm', True),
        mix_ratio=config.model.params.get('mix_ratio', 0.0),
        use_edge_processing=True,
        edge_input_channels=config.model.params.get('edge_input_channels', 3),
        linear_start=config.model.params.get('linear_start', 0.00085),
        linear_end=config.model.params.get('linear_end', 0.0120),
        timesteps=config.model.params.get('timesteps', 1000),
        first_stage_key=config.model.params.get('first_stage_key', 'image'),
        image_size=config.model.params.get('image_size', 512),
        channels=config.model.params.get('channels', 4),
        unet_config=config.model.params.get('unet_config', None),
        use_ema=config.model.params.get('use_ema', False)
    ).to(device)
    
    print(f"✓ Model created successfully")
    print(f"  use_edge_processing: {model.use_edge_processing}")
    print(f"  UNet first conv in_channels: {model.model.diffusion_model.input_blocks[0][0].weight.shape[1]}")
    
    # Load checkpoint
    ckpt_path = "logs/2025-10-07T02-28-22_stablesr_edge_8_channels/checkpoints/epoch=000030.ckpt"
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"{'='*80}")
    
    model.init_from_ckpt(ckpt_path)
    
    print(f"\n✓ Checkpoint loaded")
    
    # Check if edge_processor has weights
    if hasattr(model.model.diffusion_model, 'edge_processor'):
        edge_proc = model.model.diffusion_model.edge_processor
        print(f"  edge_processor exists: True")
        print(f"  edge_processor parameters: {sum(p.numel() for p in edge_proc.parameters())}")
        
        # Check if weights are not all zeros
        first_weight = edge_proc.initial_conv[0].weight
        is_initialized = torch.abs(first_weight).sum() > 0
        print(f"  edge_processor initialized: {is_initialized}")
        print(f"  First conv weight mean: {first_weight.mean():.6f}, std: {first_weight.std():.6f}")
    else:
        print(f"  edge_processor exists: False ❌")
    
    # Test forward pass
    print(f"\n{'='*80}")
    print("Testing forward pass with edge map...")
    print(f"{'='*80}")
    
    model.eval()
    with torch.no_grad():
        # Create dummy inputs
        batch_size = 1
        dummy_latent = torch.randn(batch_size, 4, 64, 64).to(device)
        dummy_edge = torch.randn(batch_size, 3, 512, 512).to(device)
        dummy_cond = ['']
        dummy_t = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # Process through cond_stage_model
        c = model.cond_stage_model(dummy_cond)
        
        # Create struct_cond
        t_ori = torch.tensor([999]).repeat(batch_size).long().to(device)
        struct_cond = model.structcond_stage_model(dummy_latent, t_ori)
        
        # Test apply_model with edge_map
        try:
            output = model.apply_model(dummy_latent, t_ori, c, struct_cond, edge_map=dummy_edge)
            print(f"✓ Forward pass successful!")
            print(f"  Input shape: {dummy_latent.shape}")
            print(f"  Edge map shape: {dummy_edge.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            
            # Check if output looks reasonable (not all zeros or inf)
            if torch.isnan(output).any():
                print(f"  ⚠️  WARNING: Output contains NaN values!")
            elif torch.isinf(output).any():
                print(f"  ⚠️  WARNING: Output contains Inf values!")
            elif torch.abs(output).max() > 100:
                print(f"  ⚠️  WARNING: Output values very large (max: {output.abs().max():.2f})")
            else:
                print(f"  ✓ Output looks normal")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Diagnostic complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_model_loading()




