#!/usr/bin/env python3
"""
Test script for edge processing functionality in StableSR
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.modules.diffusionmodules.edge_processor import EdgeMapProcessor, EdgeFusionModule
from ldm.modules.diffusionmodules.unet_with_edge import UNetModelDualcondV2WithEdge
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge


def create_test_edge_map(size=(512, 512)):
    """Create a test edge map using Canny edge detection"""
    # Create a simple test image
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Draw some geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
    cv2.line(img, (100, 300), (400, 100), (200, 200, 200), 3)
    
    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    # Convert to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_3ch


def test_edge_processor():
    """Test EdgeMapProcessor"""
    print("Testing EdgeMapProcessor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = EdgeMapProcessor(input_channels=3, output_channels=4, target_size=64).to(device)
    
    # Test with different input sizes
    test_sizes = [(512, 512), (256, 256), (128, 128)]
    
    for h, w in test_sizes:
        # Create test edge map
        edge_map_np = create_test_edge_map((h, w))
        edge_map = torch.from_numpy(edge_map_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        edge_map = edge_map.to(device)
        
        # Process edge map
        with torch.no_grad():
            output = processor(edge_map)
        
        print(f"Input: {edge_map.shape} -> Output: {output.shape}")
        assert output.shape == (1, 4, 64, 64), f"Expected (1, 4, 64, 64), got {output.shape}"
    
    print("✓ EdgeMapProcessor test passed!")


def test_edge_fusion():
    """Test EdgeFusionModule"""
    print("Testing EdgeFusionModule...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_module = EdgeFusionModule().to(device)
    
    # Test inputs
    unet_input = torch.randn(2, 4, 64, 64).to(device)
    edge_features = torch.randn(2, 4, 64, 64).to(device)
    
    with torch.no_grad():
        fused = fusion_module(unet_input, edge_features)
    
    print(f"Fusion: {unet_input.shape} + {edge_features.shape} -> {fused.shape}")
    assert fused.shape == (2, 8, 64, 64), f"Expected (2, 8, 64, 64), got {fused.shape}"
    
    print("✓ EdgeFusionModule test passed!")


def test_unet_with_edge():
    """Test UNetModelDualcondV2WithEdge"""
    print("Testing UNetModelDualcondV2WithEdge...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
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
    ).to(device)
    
    # Test inputs
    batch_size = 2
    unet_input = torch.randn(batch_size, 4, 64, 64).to(device)
    edge_map = torch.randn(batch_size, 3, 512, 512).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 77, 1024).to(device)
    struct_cond = torch.randn(batch_size, 256, 96, 96).to(device)
    
    # Test forward pass with edge map
    with torch.no_grad():
        output = model(
            x=unet_input,
            timesteps=timesteps,
            context=context,
            struct_cond=struct_cond,
            edge_map=edge_map
        )
    
    print(f"Input: {unet_input.shape}")
    print(f"Edge map: {edge_map.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == unet_input.shape, f"Expected {unet_input.shape}, got {output.shape}"
    
    print("✓ UNetModelDualcondV2WithEdge test passed!")


def test_latent_diffusion_with_edge():
    """Test LatentDiffusionSRTextWTWithEdge"""
    print("Testing LatentDiffusionSRTextWTWithEdge...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple config for testing
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'unet_config': {
            'target': 'ldm.modules.diffusionmodules.unet_with_edge.UNetModelDualcondV2WithEdge',
            'params': {
                'image_size': 32,
                'in_channels': 4,
                'model_channels': 320,
                'out_channels': 4,
                'num_res_blocks': 2,
                'attention_resolutions': [4, 2, 1],
                'channel_mult': [1, 2, 4, 4],
                'num_head_channels': 64,
                'use_spatial_transformer': True,
                'context_dim': 1024,
                'semb_channels': 256,
                'use_edge_processing': True,
                'edge_input_channels': 3,
            }
        },
        'first_stage_config': {
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 4,
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 4,
                    'resolution': 512,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 4, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0,
                },
                'lossconfig': {
                    'target': 'torch.nn.Identity'
                }
            }
        },
        'cond_stage_config': {
            'target': 'ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder',
            'params': {
                'freeze': True,
                'layer': 'penultimate'
            }
        },
        'structcond_stage_config': {
            'target': 'ldm.modules.diffusionmodules.openaimodel.EncoderUNetModelWT',
            'params': {
                'image_size': 96,
                'in_channels': 4,
                'model_channels': 256,
                'out_channels': 256,
                'num_res_blocks': 2,
                'attention_resolutions': [4, 2, 1],
                'dropout': 0,
                'channel_mult': [1, 1, 2, 2],
                'conv_resample': True,
                'dims': 2,
                'use_checkpoint': False,
                'use_fp16': False,
                'num_heads': 4,
                'num_head_channels': -1,
                'num_heads_upsample': -1,
                'use_scale_shift_norm': False,
                'resblock_updown': False,
                'use_new_attention_order': False,
            }
        },
        'degradation': {
            'resize_prob': [0.2, 0.7, 0.1],
            'resize_range': [0.15, 1.5],
            'gaussian_noise_prob': 0.5,
            'noise_range': [1, 30],
            'poisson_noise_prob': 0.5,
            'poisson_scale_range': [0.05, 3.0],
            'gray_noise_prob': 0.4,
            'jpeg_range': [30, 95],
            'second_blur_prob': 0.8,
            'resize_prob2': [0.3, 0.4, 0.3],
            'resize_range2': [0.3, 1.2],
            'gaussian_noise_prob2': 0.5,
            'noise_range2': [1, 25],
            'poisson_noise_prob2': 0.5,
            'poisson_scale_range2': [0.05, 2.5],
            'gray_noise_prob2': 0.4,
            'jpeg_range2': [30, 95],
            'gt_size': 512,
            'no_degradation_prob': 0.0
        },
        'sf': 4,
        'data': {
            'params': {
                'batch_size': 2,
                'train': {
                    'params': {
                        'queue_size': 100
                    }
                }
            }
        }
    })
    
    # Create model
    model = LatentDiffusionSRTextWTWithEdge(
        unet_config=config.unet_config,
        first_stage_config=config.first_stage_config,
        cond_stage_config=config.cond_stage_config,
        structcond_stage_config=config.structcond_stage_config,
        use_edge_processing=True,
        edge_input_channels=3,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        first_stage_key="image",
        cond_stage_key="caption",
        image_size=512,
        channels=4,
        cond_stage_trainable=False,
        conditioning_key="crossattn",
        scale_factor=0.18215,
        use_ema=False,
        unfrozen_diff=False,
        random_size=False,
        time_replace=1000,
        use_usm=True,
    ).to(device)
    
    # Set configs attribute (required by parent class)
    model.configs = config
    
    # Test with dummy batch
    batch_size = 2
    batch = {
        'gt': torch.randn(batch_size, 3, 512, 512).to(device),
        'img_edge': torch.randn(batch_size, 3, 512, 512).to(device),
        'kernel1': torch.randn(batch_size, 1, 21, 21).to(device),
        'kernel2': torch.randn(batch_size, 1, 11, 11).to(device),
        'sinc_kernel': torch.randn(batch_size, 1, 21, 21).to(device),
    }
    
    # Test get_input
    with torch.no_grad():
        result = model.get_input(batch, return_first_stage_outputs=True)
        print(f"get_input result length: {len(result)}")
        print(f"z shape: {result[0].shape}")
        print(f"edge_map shape: {result[-1].shape}")
    
    print("✓ LatentDiffusionSRTextWTWithEdge test passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Edge Processing Functionality in StableSR")
    print("=" * 60)
    
    try:
        test_edge_processor()
        test_edge_fusion()
        test_unet_with_edge()
        test_latent_diffusion_with_edge()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
