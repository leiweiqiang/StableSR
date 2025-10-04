#!/usr/bin/env python3
"""
Training script for StableSR with edge processing support
"""

import argparse
import os
import sys
from omegaconf import OmegaConf
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main as train_main


def setup_memory_management():
    """Setup memory management for CUDA"""
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        print(f"CUDA memory management configured. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def get_parser():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description="Train StableSR with edge processing")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use (comma-separated)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="stablesr_edge",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate"
    )
    
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run tests, don't train"
    )
    
    return parser


def test_edge_processing():
    """Test edge processing functionality"""
    print("Testing edge processing functionality...")
    
    # Run the test script
    import subprocess
    result = subprocess.run([sys.executable, "test_edge_processing.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Edge processing tests passed!")
        return True
    else:
        print("❌ Edge processing tests failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def main():
    """Main function"""
    parser = get_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("StableSR Training with Edge Processing")
    print("=" * 60)
    
    # Setup memory management
    setup_memory_management()
    
    # Test edge processing functionality first (skip if dependencies missing)
    try:
        if not test_edge_processing():
            print("Edge processing tests failed. Please fix the issues before training.")
            return 1
    except Exception as e:
        print(f"Edge processing tests skipped due to dependency issues: {e}")
        print("Proceeding with training...")
    
    if args.test_only:
        print("Tests completed successfully!")
        return 0
    
    # Prepare training arguments
    train_args = [
        "--base", args.config,
        "--gpus", args.gpus,
        "--name", args.name,
        "--train",
        "--scale_lr" if args.scale_lr else "--no-scale_lr"
    ]
    
    if args.resume:
        train_args.extend(["--resume", args.resume])
    
    print(f"Starting training with config: {args.config}")
    print(f"Experiment name: {args.name}")
    print(f"GPUs: {args.gpus}")
    print(f"Resume: {args.resume if args.resume else 'No'}")
    print(f"Scale LR: {args.scale_lr}")
    
    # Override sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["main.py"] + train_args
    
    try:
        # Run training
        train_main()
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    print("Training completed!")
    return 0


if __name__ == "__main__":
    exit(main())
