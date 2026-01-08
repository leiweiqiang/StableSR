#!/usr/bin/env python3
"""
Example Usage of Multi-GPU Inference for StableSR Blended Input
==============================================================

This script demonstrates how to use the multi-GPU inference functionality
with different configurations and scenarios.

Usage:
    python example_multi_gpu_usage.py
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.inference_blended_input_tile_multi_gpu import (
    MultiGPUImageProcessor,
    load_model_from_config,
    setup_timesteps
)
from omegaconf import OmegaConf


def check_gpu_availability():
    """Check available GPUs and their memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Multi-GPU processing requires CUDA.")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
    
    return True


def example_single_image_multi_gpu():
    """Example: Process a single large image with multi-GPU"""
    print("\n" + "="*60)
    print("Example 1: Single Large Image Multi-GPU Processing")
    print("="*60)
    
    # Configuration
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
    ckpt_path = "checkpoints/model.ckpt"
    vqgan_ckpt_path = "checkpoints/vqgan_model.ckpt"
    
    lr_image_path = "inputs/lr_images/large_image.png"
    edge_image_path = "inputs/edges_512/large_edge.png"
    output_dir = "outputs/example_multi_gpu/"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Model checkpoint not found: {ckpt_path}")
        return
    
    if not os.path.exists(vqgan_ckpt_path):
        print(f"❌ VQGAN checkpoint not found: {vqgan_ckpt_path}")
        return
    
    if not os.path.exists(lr_image_path):
        print(f"❌ LR image not found: {lr_image_path}")
        return
    
    if not os.path.exists(edge_image_path):
        print(f"❌ Edge image not found: {edge_image_path}")
        return
    
    # Load models
    print("Loading models...")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, verbose=False)
    model.configs = config
    
    # Load VQGAN model
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, vqgan_ckpt_path)
    vq_model.decoder.fusion_w = 0.5
    
    # Setup timesteps
    setup_timesteps(model, 200)
    
    # Setup multi-GPU processor
    gpu_ids = [0, 1] if torch.cuda.device_count() >= 2 else [0]
    print(f"Setting up multi-GPU processor with GPUs: {gpu_ids}")
    
    multi_gpu_processor = MultiGPUImageProcessor(
        model, vq_model, gpu_ids, 
        tile_size=1280, 
        tile_overlap=64
    )
    
    # Process image
    print("Processing image...")
    start_time = time.time()
    
    try:
        from scripts.inference_blended_input_tile_multi_gpu import inference_single_image_tiled_multi_gpu
        
        output_image, blended_pil, lr_upscaled_pil, edge_pil = inference_single_image_tiled_multi_gpu(
            multi_gpu_processor,
            lr_image_path,
            edge_image_path,
            blend_alpha=0.5,
            blend_beta=0.5,
            lr_downscale_factor=16,
            ddpm_steps=200,
            seed=42,
            text_prompt="",
            colorfix_type="none",
            tile_size=1280,
            tile_overlap=64,
            vqgan_tile_size=1280,
            vqgan_tile_stride=320,
            dec_w=0.5,
            start_from_noise=False
        )
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.png")
        output_image.save(output_path)
        print(f"✅ Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


def example_batch_processing():
    """Example: Batch processing with multi-GPU"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing with Multi-GPU")
    print("="*60)
    
    # Configuration
    lr_dir = "inputs/lr_images/"
    edge_dir = "inputs/edges_512/"
    output_dir = "outputs/batch_multi_gpu/"
    
    # Check if directories exist
    if not os.path.exists(lr_dir):
        print(f"❌ LR images directory not found: {lr_dir}")
        return
    
    if not os.path.exists(edge_dir):
        print(f"❌ Edge images directory not found: {edge_dir}")
        return
    
    # Count images
    lr_images = [f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    edge_images = [f for f in os.listdir(edge_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(lr_images)} LR images and {len(edge_images)} edge images")
    
    if len(lr_images) == 0:
        print("❌ No LR images found")
        return
    
    if len(edge_images) == 0:
        print("❌ No edge images found")
        return
    
    print("✅ Batch processing setup complete")
    print("To run batch processing, use:")
    print(f"python scripts/inference_blended_input_tile_multi_gpu.py \\")
    print(f"    --lr-img {lr_dir} \\")
    print(f"    --edge-img {edge_dir} \\")
    print(f"    --outdir {output_dir} \\")
    print(f"    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \\")
    print(f"    --ckpt checkpoints/model.ckpt \\")
    print(f"    --vqgan-ckpt checkpoints/vqgan_model.ckpt \\")
    print(f"    --multi-gpu \\")
    print(f"    --gpu-ids 0,1,2,3 \\")
    print(f"    --batch-mode \\")
    print(f"    --batch-size 4")


def example_performance_comparison():
    """Example: Performance comparison setup"""
    print("\n" + "="*60)
    print("Example 3: Performance Comparison Setup")
    print("="*60)
    
    print("To run performance comparison, use:")
    print("python scripts/performance_comparison.py \\")
    print("    --lr-img inputs/lr_images/test_image.png \\")
    print("    --edge-img inputs/edges_512/test_edge.png \\")
    print("    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \\")
    print("    --ckpt checkpoints/model.ckpt \\")
    print("    --vqgan-ckpt checkpoints/vqgan_model.ckpt \\")
    print("    --gpu-ids 0,1,2,3 \\")
    print("    --iterations 3")
    
    print("\nExpected results:")
    print("- Single-GPU: ~45 seconds per image")
    print("- Multi-GPU (4x): ~18 seconds per image")
    print("- Speedup: ~2.4x")


def main():
    print("Multi-GPU Inference Examples for StableSR Blended Input")
    print("="*60)
    
    # Check GPU availability
    if not check_gpu_availability():
        return
    
    # Example 1: Single image processing
    example_single_image_multi_gpu()
    
    # Example 2: Batch processing
    example_batch_processing()
    
    # Example 3: Performance comparison
    example_performance_comparison()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure you have the required model checkpoints")
    print("2. Prepare your input images")
    print("3. Run the examples or use the provided scripts")
    print("4. Monitor GPU usage with 'nvidia-smi'")
    print("5. Adjust parameters based on your hardware")


if __name__ == "__main__":
    main()

