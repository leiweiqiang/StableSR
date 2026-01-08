#!/usr/bin/env python3
"""
Performance Comparison Script for Single-GPU vs Multi-GPU Inference
==================================================================

This script compares the performance of single-GPU and multi-GPU inference
to demonstrate the speedup achieved with multi-GPU processing.

Usage:
    python scripts/performance_comparison.py \
        --lr-img inputs/lr_images/test_image.png \
        --edge-img inputs/edges_512/test_edge.png \
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
        --ckpt checkpoints/model.ckpt \
        --vqgan-ckpt checkpoints/vqgan_model.ckpt \
        --gpu-ids 0,1,2,3 \
        --iterations 3
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.inference_blended_input_tile import inference_single_image_tiled
from scripts.inference_blended_input_tile_multi_gpu import (
    inference_single_image_tiled_multi_gpu, 
    MultiGPUImageProcessor,
    load_model_from_config,
    setup_timesteps
)
from omegaconf import OmegaConf


def benchmark_single_gpu(model, vq_model, lr_image_path, edge_map_path, iterations=3, **kwargs):
    """
    Benchmark single-GPU inference
    """
    print("🔥 Benchmarking Single-GPU Inference...")
    
    times = []
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")
        
        start_time = time.time()
        try:
            output_image, blended_pil, lr_upscaled_pil, edge_pil = inference_single_image_tiled(
                model, vq_model, lr_image_path, edge_map_path, **kwargs
            )
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"    Time: {iteration_time:.2f} seconds")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
        return avg_time, std_time
    else:
        print("  No successful iterations")
        return None, None


def benchmark_multi_gpu(multi_gpu_processor, lr_image_path, edge_map_path, iterations=3, **kwargs):
    """
    Benchmark multi-GPU inference
    """
    print("🚀 Benchmarking Multi-GPU Inference...")
    
    times = []
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")
        
        start_time = time.time()
        try:
            output_image, blended_pil, lr_upscaled_pil, edge_pil = inference_single_image_tiled_multi_gpu(
                multi_gpu_processor, lr_image_path, edge_map_path, **kwargs
            )
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"    Time: {iteration_time:.2f} seconds")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
        return avg_time, std_time
    else:
        print("  No successful iterations")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Performance Comparison: Single-GPU vs Multi-GPU")
    
    # Input arguments
    parser.add_argument("--lr-img", type=str, required=True,
                       help="Path to LR image file")
    parser.add_argument("--edge-img", type=str, required=True,
                       help="Path to edge map file")
    
    # Model arguments
    parser.add_argument("--config", type=str,
                       default="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml",
                       help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--vqgan-ckpt", type=str, required=True,
                       help="Path to VQGAN model checkpoint")
    
    # Processing arguments
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                       help="Blending weight for LR upscaled image")
    parser.add_argument("--blend-beta", type=float, default=0.5,
                       help="Blending weight for edge map")
    parser.add_argument("--lr-downscale-factor", type=int, default=16,
                       help="Downscale factor for LR")
    parser.add_argument("--ddpm-steps", type=int, default=200,
                       help="Number of DDPM sampling steps")
    parser.add_argument("--tile-size", type=int, default=1280,
                       help="Size for tiling operation")
    parser.add_argument("--tile-overlap", type=int, default=64,
                       help="Overlap size for tiling")
    parser.add_argument("--vqgan-tile-size", type=int, default=1280,
                       help="Size for VQGAN tile operation")
    parser.add_argument("--vqgan-tile-stride", type=int, default=320,
                       help="Stride for VQGAN tile operation")
    
    # Multi-GPU arguments
    parser.add_argument("--gpu-ids", type=str, default="0",
                       help="Comma-separated list of GPU IDs to use")
    
    # Benchmark arguments
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations for benchmarking")
    parser.add_argument("--skip-single-gpu", action="store_true",
                       help="Skip single-GPU benchmark")
    parser.add_argument("--skip-multi-gpu", action="store_true",
                       help="Skip multi-GPU benchmark")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    print("="*80)
    print("Performance Comparison: Single-GPU vs Multi-GPU Inference")
    print("="*80)
    print(f"LR Image: {args.lr_img}")
    print(f"Edge Image: {args.edge_img}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Iterations: {args.iterations}")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, verbose=False)
    model.configs = config
    
    # Load VQGAN model
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, args.vqgan_ckpt)
    vq_model.decoder.fusion_w = 0.5
    
    # Setup timesteps
    setup_timesteps(model, args.ddpm_steps)
    
    # Prepare inference parameters
    inference_kwargs = {
        'blend_alpha': args.blend_alpha,
        'blend_beta': args.blend_beta,
        'lr_downscale_factor': args.lr_downscale_factor,
        'ddpm_steps': args.ddpm_steps,
        'seed': 42,
        'text_prompt': "",
        'colorfix_type': "none",
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'vqgan_tile_size': args.vqgan_tile_size,
        'vqgan_tile_stride': args.vqgan_tile_stride,
        'dec_w': 0.5,
        'start_from_noise': False
    }
    
    # Benchmark single-GPU
    single_gpu_time = None
    single_gpu_std = None
    
    if not args.skip_single_gpu:
        print(f"\n{'='*60}")
        single_gpu_time, single_gpu_std = benchmark_single_gpu(
            model, vq_model, args.lr_img, args.edge_img, 
            args.iterations, **inference_kwargs
        )
    
    # Setup multi-GPU processor
    multi_gpu_processor = None
    if not args.skip_multi_gpu:
        print(f"\n{'='*60}")
        print("Setting up multi-GPU processor...")
        multi_gpu_processor = MultiGPUImageProcessor(
            model, vq_model, gpu_ids, 
            tile_size=args.tile_size, 
            tile_overlap=args.tile_overlap
        )
    
    # Benchmark multi-GPU
    multi_gpu_time = None
    multi_gpu_std = None
    
    if not args.skip_multi_gpu and multi_gpu_processor:
        print(f"\n{'='*60}")
        multi_gpu_time, multi_gpu_std = benchmark_multi_gpu(
            multi_gpu_processor, args.lr_img, args.edge_img, 
            args.iterations, **inference_kwargs
        )
    
    # Print results
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    if single_gpu_time is not None:
        print(f"Single-GPU (GPU {gpu_ids[0]}):")
        print(f"  Average time: {single_gpu_time:.2f} ± {single_gpu_std:.2f} seconds")
        print(f"  Throughput: {1.0/single_gpu_time:.3f} images/second")
    
    if multi_gpu_time is not None:
        print(f"Multi-GPU (GPUs {gpu_ids}):")
        print(f"  Average time: {multi_gpu_time:.2f} ± {multi_gpu_std:.2f} seconds")
        print(f"  Throughput: {1.0/multi_gpu_time:.3f} images/second")
    
    if single_gpu_time is not None and multi_gpu_time is not None:
        speedup = single_gpu_time / multi_gpu_time
        efficiency = speedup / len(gpu_ids) * 100
        
        print(f"\nSpeedup Analysis:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Time saved: {single_gpu_time - multi_gpu_time:.2f} seconds per image")
        
        if speedup > 1.5:
            print(f"  ✅ Significant speedup achieved!")
        elif speedup > 1.1:
            print(f"  ⚡ Moderate speedup achieved")
        else:
            print(f"  ⚠️  Limited speedup - consider larger images or different tile sizes")
    
    print(f"{'='*80}")
    
    # Memory usage summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        for i, gpu_id in enumerate(gpu_ids):
            if i < torch.cuda.device_count():
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"  GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")


if __name__ == "__main__":
    main()

