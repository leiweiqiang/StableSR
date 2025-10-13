#!/usr/bin/env python3
"""
Inspect parameters in a checkpoint file.
Given a checkpoint and a key pattern, print out the first 20 matching parameters.
"""

import torch
import argparse
import sys
from pathlib import Path


def load_checkpoint(ckpt_path):
    """Load a checkpoint file"""
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return ckpt
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


def get_state_dict(ckpt):
    """Extract state_dict from checkpoint (handles different formats)"""
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        elif 'model' in ckpt:
            return ckpt['model']
        else:
            # Assume the whole thing is a state dict
            return ckpt
    return ckpt


def inspect_parameters(ckpt_path, key_pattern=None, limit=20, show_values=False):
    """
    Inspect parameters in a checkpoint
    
    Args:
        ckpt_path: Path to checkpoint
        key_pattern: Optional string pattern to filter parameter keys
        limit: Number of parameters to show (default: 20)
        show_values: If True, print actual parameter values
    """
    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path)
    
    # Extract state dict
    state_dict = get_state_dict(ckpt)
    
    print(f"\nCheckpoint: {Path(ckpt_path).name}")
    print(f"Total parameters in state_dict: {len(state_dict)}")
    
    # Get all keys
    all_keys = sorted(state_dict.keys())
    
    # Filter keys if pattern provided
    if key_pattern:
        matching_keys = [k for k in all_keys if key_pattern in k]
        print(f"Keys matching pattern '{key_pattern}': {len(matching_keys)}")
        keys_to_show = matching_keys[:limit]
    else:
        print(f"Showing first {limit} parameters (no filter applied)")
        keys_to_show = all_keys[:limit]
    
    print(f"\n{'='*80}")
    print(f"PARAMETER DETAILS (showing first {len(keys_to_show)}):")
    print(f"{'='*80}\n")
    
    for idx, key in enumerate(keys_to_show, 1):
        param = state_dict[key]
        
        print(f"{idx}. {key}")
        
        if isinstance(param, torch.Tensor):
            print(f"   Type: Tensor")
            print(f"   Shape: {tuple(param.shape)}")
            print(f"   Dtype: {param.dtype}")
            print(f"   Device: {param.device}")
            print(f"   Requires grad: {param.requires_grad}")
            
            # Statistics
            if param.numel() > 0:
                print(f"   Min: {param.min().item()}")
                print(f"   Max: {param.max().item()}")
                
                # Mean and std only work on floating point or complex dtypes
                try:
                    mean_val = param.float().mean().item()
                    std_val = param.float().std().item()
                    print(f"   Mean: {mean_val:.6e}")
                    print(f"   Std: {std_val:.6e}")
                except Exception:
                    # For dtypes that can't be converted to float
                    print(f"   Mean: N/A (dtype {param.dtype})")
                    print(f"   Std: N/A (dtype {param.dtype})")
                
                # Show sample values if requested
                if show_values:
                    flat = param.flatten()
                    n_show = min(10, len(flat))
                    print(f"   First {n_show} values: {flat[:n_show].tolist()}")
        else:
            print(f"   Type: {type(param).__name__}")
            print(f"   Value: {param}")
        
        print()  # Empty line between parameters
    
    # Summary
    if key_pattern and len(matching_keys) > limit:
        print(f"{'='*80}")
        print(f"... and {len(matching_keys) - limit} more parameters matching '{key_pattern}'")
        print(f"{'='*80}")
    elif not key_pattern and len(all_keys) > limit:
        print(f"{'='*80}")
        print(f"... and {len(all_keys) - limit} more parameters in total")
        print(f"{'='*80}")
    
    # Print all matching keys if there aren't too many
    if key_pattern and len(matching_keys) <= 100:
        print(f"\nAll {len(matching_keys)} keys matching '{key_pattern}':")
        for key in matching_keys:
            shape_info = f" {tuple(state_dict[key].shape)}" if isinstance(state_dict[key], torch.Tensor) else ""
            print(f"  - {key}{shape_info}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect parameters in a checkpoint file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show first 20 parameters
  python inspect_checkpoint.py checkpoint.ckpt
  
  # Show parameters with "encoder" in the name
  python inspect_checkpoint.py checkpoint.ckpt --key encoder
  
  # Show first 50 parameters matching "diffusion"
  python inspect_checkpoint.py checkpoint.ckpt --key diffusion --limit 50
  
  # Show parameters with actual values
  python inspect_checkpoint.py checkpoint.ckpt --key "model.encoder" --show-values
        """
    )
    
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--key', '-k', type=str, default=None,
                        help='Filter parameters by key pattern (substring match)')
    parser.add_argument('--limit', '-l', type=int, default=20,
                        help='Number of parameters to show (default: 20)')
    parser.add_argument('--show-values', '-v', action='store_true',
                        help='Show sample parameter values')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Inspect checkpoint
    inspect_parameters(
        args.checkpoint,
        key_pattern=args.key,
        limit=args.limit,
        show_values=args.show_values
    )


if __name__ == "__main__":
    main()
