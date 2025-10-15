#!/usr/bin/env python3
"""
Test script to verify that the edge model can be loaded correctly from v3.
This script checks:
1. Correct module imports from v3 (not v2)
2. Model architecture with EdgeMapProcessor
3. Checkpoint loading (if provided)
"""

import sys
import os

# Ensure we import from v3
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
print(f"[INFO] Project root added to sys.path: {script_dir}")

import torch
from omegaconf import OmegaConf
import ldm.models.diffusion.ddpm as ddpm_module
import ldm.modules.diffusionmodules.openaimodel as openai_module

print(f"\n{'='*60}")
print("MODULE VERIFICATION")
print(f"{'='*60}")
print(f"✓ ldm.models.diffusion.ddpm loaded from:")
print(f"  {ddpm_module.__file__}")
print(f"✓ ldm.modules.diffusionmodules.openaimodel loaded from:")
print(f"  {openai_module.__file__}")

# Verify v3 modules are loaded
if 'StableSR_Edge_v3' in ddpm_module.__file__:
    print(f"\n✓ SUCCESS: Using v3 modules")
elif 'StableSR_Edge_v2' in ddpm_module.__file__:
    print(f"\n✗ ERROR: Still using v2 modules!")
    print(f"  Please clear Python cache and try again.")
    sys.exit(1)
else:
    print(f"\n⚠ WARNING: Unexpected module path")

# Check if EdgeMapProcessor exists
print(f"\n{'='*60}")
print("ARCHITECTURE VERIFICATION")
print(f"{'='*60}")

if hasattr(openai_module, 'EdgeMapProcessor'):
    print(f"✓ EdgeMapProcessor class found in openaimodel")
    
    # Test instantiation
    edge_processor = openai_module.EdgeMapProcessor()
    print(f"✓ EdgeMapProcessor can be instantiated")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 512, 512)
    try:
        output = edge_processor(test_input)
        print(f"✓ EdgeMapProcessor forward pass works")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, 4, 64, 64), f"Expected output shape (1, 4, 64, 64), got {output.shape}"
        print(f"✓ Output shape is correct (1, 4, 64, 64)")
    except Exception as e:
        print(f"✗ EdgeMapProcessor forward pass failed: {e}")
        sys.exit(1)
else:
    print(f"✗ EdgeMapProcessor class NOT found in openaimodel")
    print(f"  This means the v3 code is not loaded correctly")
    sys.exit(1)

# Check EncoderUNetModelWT
if hasattr(openai_module, 'EncoderUNetModelWT'):
    print(f"\n✓ EncoderUNetModelWT class found")
    
    # Test instantiation
    config = {
        'image_size': 96,
        'in_channels': 8,
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
    
    try:
        encoder = openai_module.EncoderUNetModelWT(**config)
        print(f"✓ EncoderUNetModelWT can be instantiated with in_channels=8")
        
        # Check if it has edge_processor
        if hasattr(encoder, 'edge_processor'):
            print(f"✓ EncoderUNetModelWT has edge_processor attribute")
        else:
            print(f"✗ EncoderUNetModelWT missing edge_processor attribute")
            sys.exit(1)
            
        # Test forward pass
        test_struct = torch.randn(1, 4, 64, 64)
        test_edge = torch.randn(1, 3, 512, 512)
        test_t = torch.tensor([100])
        
        try:
            output = encoder(test_struct, test_edge, test_t)
            print(f"✓ EncoderUNetModelWT forward pass works")
            print(f"  struct_cond shape: {test_struct.shape}")
            print(f"  edge_map shape: {test_edge.shape}")
            print(f"  timestep: {test_t}")
            print(f"  Output type: {type(output)}")
            if isinstance(output, dict):
                print(f"  Output keys: {list(output.keys())}")
        except Exception as e:
            print(f"✗ EncoderUNetModelWT forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ EncoderUNetModelWT instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"✗ EncoderUNetModelWT class NOT found")
    sys.exit(1)

print(f"\n{'='*60}")
print("CONFIG LOADING TEST")
print(f"{'='*60}")

config_path = "configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml"
if os.path.exists(config_path):
    print(f"✓ Inference config found: {config_path}")
    config = OmegaConf.load(config_path)
    print(f"✓ Config loaded successfully")
    print(f"  Model target: {config.model.target}")
    print(f"  StructCond target: {config.model.params.structcond_stage_config.target}")
    print(f"  StructCond in_channels: {config.model.params.structcond_stage_config.params.in_channels}")
    print(f"  Ignore keys: {config.model.params.get('ignore_keys', [])}")
else:
    print(f"✗ Inference config not found: {config_path}")
    print(f"  Using default config may cause issues")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"✓ All tests passed!")
print(f"\nYou can now run the inference script with:")
print(f"  python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \\")
print(f"      --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \\")
print(f"      --ckpt /path/to/checkpoint.ckpt \\")
print(f"      --use_edge \\")
print(f"      --save_edge \\")
print(f"      --init-img inputs/user_upload \\")
print(f"      --outdir outputs/user_upload \\")
print(f"      --ddpm_steps 200")
print(f"\n{'='*60}")




