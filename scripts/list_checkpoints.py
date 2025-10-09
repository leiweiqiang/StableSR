#!/usr/bin/env python3
"""
Quick utility to list all available checkpoints
"""

import os
import sys
from pathlib import Path
import re

def list_checkpoints(logs_dir="logs"):
    """List all checkpoints in the logs directory"""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"Error: {logs_dir} does not exist")
        return
    
    print(f"Scanning: {logs_path.absolute()}")
    print("=" * 80)
    
    all_checkpoints = []
    
    for exp_dir in sorted(logs_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        ckpt_dir = exp_dir / "checkpoints"
        
        if not ckpt_dir.exists():
            continue
        
        checkpoints = list(ckpt_dir.glob("*.ckpt"))
        
        if checkpoints:
            print(f"\n📁 Experiment: {exp_name}")
            print(f"   Path: {ckpt_dir}")
            print(f"   Checkpoints:")
            
            for ckpt in sorted(checkpoints):
                size = ckpt.stat().st_size / (1024**3)  # Size in GB
                is_last = ckpt.name == "last.ckpt"
                marker = "✅"  # All checkpoints will be processed by default
                status = "(will be processed)"
                
                # Extract epoch number
                epoch_info = ""
                match = re.search(r'epoch=(\d+)', ckpt.name)
                if match:
                    epoch_num = match.group(1)
                    epoch_info = f" [Epoch {int(epoch_num)}]"
                
                print(f"     {marker} {ckpt.name}{epoch_info} - {size:.2f} GB {status}")
                
                # All checkpoints will be processed by default now
                all_checkpoints.append((exp_name, ckpt, epoch_info))
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total checkpoints found: {len(all_checkpoints)}")
    print(f"  These will be processed by auto_inference.py (default behavior)")
    print(f"  Note: All checkpoints including last.ckpt are processed by default")
    
    if len(all_checkpoints) == 0:
        print("\n⚠️  No checkpoints found!")
        print("   Checkpoints are typically saved during training.")
        print("   Epoch checkpoints: epoch=000100.ckpt, epoch=000200.ckpt, etc.")
        print("   Latest checkpoint: last.ckpt")
    
    print("=" * 80)

if __name__ == "__main__":
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    list_checkpoints(logs_dir)

