#!/usr/bin/env python3
"""
Inspect a checkpoint to see what keys it contains.
This helps determine if the checkpoint has edge support components.

Usage:
    python inspect_checkpoint.py /path/to/checkpoint.ckpt
"""

import sys
import torch
from collections import defaultdict

def inspect_checkpoint(ckpt_path):
    print(f"\n{'='*80}")
    print(f"CHECKPOINT INSPECTION: {ckpt_path}")
    print(f"{'='*80}\n")
    
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return
    
    print(f"📦 Checkpoint Contents:")
    print(f"   Top-level keys: {list(pl_sd.keys())}")
    
    if "global_step" in pl_sd:
        print(f"   Global Step: {pl_sd['global_step']}")
    
    if "epoch" in pl_sd:
        print(f"   Epoch: {pl_sd['epoch']}")
    
    if "state_dict" not in pl_sd:
        print(f"\n✗ No 'state_dict' found in checkpoint!")
        return
    
    sd = pl_sd["state_dict"]
    print(f"\n📊 State Dict Statistics:")
    print(f"   Total keys: {len(sd.keys())}")
    
    # Organize keys by component
    components = defaultdict(list)
    for key in sd.keys():
        if '.' in key:
            component = key.split('.')[0]
            components[component].append(key)
        else:
            components['root'].append(key)
    
    print(f"\n🔍 Components Found:")
    for comp, keys in sorted(components.items()):
        print(f"   {comp}: {len(keys)} keys")
    
    # Check for edge-related components
    print(f"\n🎯 Edge Support Check:")
    
    edge_processor_keys = [k for k in sd.keys() if 'edge_processor' in k]
    if edge_processor_keys:
        print(f"   ✓ EdgeMapProcessor found ({len(edge_processor_keys)} keys)")
        print(f"     Sample keys:")
        for key in edge_processor_keys[:5]:
            print(f"       - {key}")
        if len(edge_processor_keys) > 5:
            print(f"       ... and {len(edge_processor_keys) - 5} more")
    else:
        print(f"   ✗ No EdgeMapProcessor keys found")
        print(f"     → This checkpoint may not support edge features")
        print(f"     → EdgeMapProcessor will be randomly initialized if used")
    
    structcond_keys = [k for k in sd.keys() if 'structcond_stage_model' in k]
    if structcond_keys:
        print(f"\n   ✓ structcond_stage_model found ({len(structcond_keys)} keys)")
        
        # Check in_channels
        first_conv_key = None
        for key in structcond_keys:
            if 'input_blocks.0.0.weight' in key:
                first_conv_key = key
                break
        
        if first_conv_key:
            weight = sd[first_conv_key]
            in_channels = weight.shape[1]
            print(f"     First conv in_channels: {in_channels}")
            if in_channels == 8:
                print(f"     ✓ Expects 8 channels (4 latent + 4 edge)")
            elif in_channels == 4:
                print(f"     ⚠ Expects only 4 channels (no edge support)")
            else:
                print(f"     ⚠ Unexpected in_channels: {in_channels}")
    else:
        print(f"\n   ✗ No structcond_stage_model keys found")
        print(f"     → structcond_stage_model will be randomly initialized")
    
    # Check other important components
    print(f"\n🧩 Other Important Components:")
    
    components_to_check = {
        'model.diffusion_model': 'Main UNet',
        'first_stage_model': 'VAE Encoder/Decoder',
        'cond_stage_model': 'Text Encoder',
    }
    
    for comp_prefix, comp_name in components_to_check.items():
        comp_keys = [k for k in sd.keys() if k.startswith(comp_prefix)]
        if comp_keys:
            print(f"   ✓ {comp_name}: {len(comp_keys)} keys")
        else:
            print(f"   ✗ {comp_name}: Not found")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if edge_processor_keys and len(structcond_keys) > 0:
        print(f"   ✓ This checkpoint appears to have FULL edge support")
        print(f"   → Use: configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml")
        print(f"   → Run with: --use_edge --save_edge")
    elif len(structcond_keys) > 0 and not edge_processor_keys:
        first_conv_key = None
        for key in structcond_keys:
            if 'input_blocks.0.0.weight' in key:
                first_conv_key = key
                break
        if first_conv_key and sd[first_conv_key].shape[1] == 8:
            print(f"   ⚠ Checkpoint has structcond with 8 channels but NO edge_processor")
            print(f"   → The edge_processor will be randomly initialized")
            print(f"   → You can try inference but quality may be suboptimal")
            print(f"   → Consider fine-tuning or using a different checkpoint")
        else:
            print(f"   ✗ Checkpoint has structcond but not configured for edges")
            print(f"   → This checkpoint is NOT for edge-based inference")
            print(f"   → Run without --use_edge flag")
    else:
        print(f"   ✗ Checkpoint does NOT have structcond_stage_model")
        print(f"   → Use training config: v2-finetune_text_T_512_edge_800_32x32.yaml")
        print(f"   → This will randomly initialize structcond (not ideal for inference)")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py /path/to/checkpoint.ckpt")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    inspect_checkpoint(ckpt_path)




