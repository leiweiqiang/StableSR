#!/usr/bin/env python3
"""
Test script to verify the edge processing channel mismatch fix
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/root/dp/StableSR_Edge_v2')

def test_edge_model_loading():
    """Test that the edge model can be loaded without channel mismatch errors"""
    print("Testing edge model loading...")
    
    try:
        from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
        from omegaconf import OmegaConf
        
        # Load config
        config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        config = OmegaConf.load(config_path)
        
        # Create model with edge processing
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
        )
        
        print("✓ Model created successfully")
        
        # Test checkpoint loading (this should not raise channel mismatch error)
        checkpoint_path = "models/ldm/stable-diffusion-v1/model.ckpt"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            model.init_from_ckpt(checkpoint_path)
            print("✓ Checkpoint loaded successfully without channel mismatch error")
        else:
            print(f"⚠ Checkpoint not found at {checkpoint_path}, skipping checkpoint test")
        
        # Test forward pass with edge map
        print("Testing forward pass with edge map...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy inputs
        batch_size = 1
        unet_input = torch.randn(batch_size, 4, 64, 64).to(device)
        edge_map = torch.randn(batch_size, 3, 512, 512).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.randn(batch_size, 77, 1024).to(device)
        struct_cond = torch.randn(batch_size, 256, 96, 96).to(device)
        
        # Test forward pass
        with torch.no_grad():
            output = model.model.diffusion_model(
                x=unet_input,
                timesteps=timesteps,
                context=context,
                struct_cond=struct_cond,
                edge_map=edge_map
            )
        
        print(f"✓ Forward pass successful: {unet_input.shape} -> {output.shape}")
        print("✓ All tests passed! Channel mismatch issue is fixed.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_edge_model_loading()
    sys.exit(0 if success else 1)






