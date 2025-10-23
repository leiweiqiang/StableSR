#!/usr/bin/env python3
"""
Inference Script for Blended Input with Tiling Support
=======================================================

This script combines the blended input approach from inference_blended_input.py
with the tiling functionality from sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py.

Features:
1. Blended Input: LR image (downscaled) + Edge map blended with two parameters
2. Tiling Support: Process large images by splitting into tiles
3. VQGAN Integration: Uses VQGAN decoder for final output
4. Batch Processing: Support for batch processing multiple images

Usage:
    # Single large image with tiling
    python scripts/inference_blended_input_tile.py \
        --lr-img inputs/lr_images/large_image.png \
        --edge-img inputs/edges_512/large_edge.png \
        --outdir outputs/blended_tile/ \
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
        --ckpt checkpoints/model.ckpt \
        --vqgan-ckpt checkpoints/vqgan_model.ckpt \
        --blend-alpha 0.5 \
        --blend-beta 0.5 \
        --lr-downscale-factor 16 \
        --ddpm-steps 200 \
        --tile-size 1280 \
        --tile-overlap 32

    # Batch processing with tiling
    python scripts/inference_blended_input_tile.py \
        --lr-img inputs/lr_images/ \
        --edge-img inputs/edges_512/ \
        --outdir outputs/batch_tile/ \
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
        --ckpt checkpoints/model.ckpt \
        --vqgan-ckpt checkpoints/vqgan_model.ckpt \
        --batch-mode \
        --tile-size 1280

Author: StableSR Blended Input Tiling
Date: 2025-01-19
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from tqdm import tqdm, trange
import time
import copy
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from util_image import ImageSpliterTh


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {desired_count} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    
    return set(all_steps)


def load_lr_image(path, size=32):
    """
    Load LR image and prepare for model input
    
    Args:
        path: Path to LR image file
        size: Target size for LR (default 32x32)
    
    Returns:
        Tensor of shape [1, 3, size, size] in range [-1, 1]
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    
    # Convert to numpy and normalize
    image = np.array(image).astype(np.float32) / 255.0
    
    # To tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize to [-1, 1]
    image = 2.0 * image - 1.0
    
    return image


def load_lr_image_with_dimensions(path, height, width):
    """
    Load LR image with specific dimensions (no downscaling)
    
    Args:
        path: Path to LR image file
        height: Target height
        width: Target width
    
    Returns:
        Tensor of shape [1, 3, height, width] in range [-1, 1]
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.BICUBIC)
    
    # Convert to numpy and normalize
    image = np.array(image).astype(np.float32) / 255.0
    
    # To tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize to [-1, 1]
    image = 2.0 * image - 1.0
    
    return image


def load_edge_map(path, height=512, width=512):
    """
    Load edge map and prepare for model input
    
    Args:
        path: Path to edge map file (can be grayscale or RGB)
        height: Target height for edge map (default 512)
        width: Target width for edge map (default 512)
    
    Returns:
        Tensor of shape [1, 3, height, width] in range [-1, 1]
    """
    edge = Image.open(path).convert("RGB")  # Convert to RGB
    edge = edge.resize((width, height), Image.BICUBIC)
    
    # Convert to numpy and normalize
    edge = np.array(edge).astype(np.float32) / 255.0
    
    # To tensor [C, H, W]
    edge = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize to [-1, 1]
    edge = 2.0 * edge - 1.0
    
    return edge


def create_blended_input_tiled(lr_image, edge_map, blend_alpha=0.5, blend_beta=0.5, output_h=512, output_w=512):
    """
    Create blended image from LR and edge map for tiled processing
    
    Args:
        lr_image: Tensor [1, 3, 32, 32] in range [-1, 1]
        edge_map: Tensor [1, 3, output_h, output_w] in range [-1, 1]
        blend_alpha: Blending weight for LR upscaled image
        blend_beta: Blending weight for edge map
        output_h: Target output height
        output_w: Target output width
    
    Returns:
        blended_image: Tensor [1, 3, output_h, output_w] in range [-1, 1]
        lr_upscaled: Tensor [1, 3, output_h, output_w] in range [-1, 1]
    """
    # Step 1: Upscale LR to output size using bicubic interpolation
    lr_upscaled = F.interpolate(
        lr_image,
        size=(output_h, output_w),
        mode='bicubic',
        align_corners=False
    )
    
    # Step 2: Convert from [-1, 1] to [0, 1] for blending
    lr_upscaled_01 = (lr_upscaled + 1.0) / 2.0
    edge_01 = (edge_map + 1.0) / 2.0
    
    # Step 3: Two-parameter blending
    blended = blend_alpha * lr_upscaled_01 + blend_beta * edge_01
    blended = torch.clamp(blended, 0.0, 1.0)  # Ensure [0, 1] range
    
    # Step 4: Convert back to [-1, 1] range
    blended_image = blended * 2.0 - 1.0
    blended_image = torch.clamp(blended_image, -1.0, 1.0)
    
    return blended_image, lr_upscaled


def load_model_from_config(config, ckpt, verbose=False):
    """
    Load model from checkpoint
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("\nMissing keys:")
        for key in m:
            print(f"  - {key}")
    
    if len(u) > 0 and verbose:
        print("\nUnexpected keys:")
        for key in u:
            print(f"  - {key}")
    
    model.cuda()
    model.eval()
    
    return model


def setup_timesteps(model, ddpm_steps):
    """
    Setup timestep schedule for DDPM sampling
    """
    # Initial full schedule
    model.register_schedule(
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3
    )
    model.num_timesteps = 1000
    
    # Create shortened schedule
    use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    
    # Ensure schedule buffers are on correct device
    device = next(model.parameters()).device
    if hasattr(model, 'sqrt_alphas_cumprod'):
        model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
    if hasattr(model, 'sqrt_one_minus_alphas_cumprod'):
        model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
    if hasattr(model, 'alphas_cumprod'):
        model.alphas_cumprod = model.alphas_cumprod.to(device)
    
    # Ensure entire model is on device
    model = model.to(device)
    if hasattr(model, 'structcond_stage_model'):
        model.structcond_stage_model = model.structcond_stage_model.to(device)


def inference_single_image_tiled(
    model,
    vq_model,
    lr_image_path,
    edge_map_path,
    blend_alpha=0.5,
    blend_beta=0.5,
    lr_downscale_factor=16,
    output_size=None,  # Will be calculated from LR image size and downscale factor
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
):
    """
    Run tiled inference on a single image pair with blended input
    
    Args:
        model: StableSR model
        vq_model: VQGAN model
        lr_image_path: Path to LR image (will be resized based on downscale factor)
        edge_map_path: Path to edge map (will be resized to output_size)
        blend_alpha: Blending weight for LR upscaled image
        blend_beta: Blending weight for edge map
        lr_downscale_factor: Downscale factor for LR (e.g., 16 means 512/16=32x32)
        output_size: Size of output
        ddpm_steps: Number of sampling steps
        seed: Random seed
        text_prompt: Optional text prompt for conditioning
        colorfix_type: Color correction method ('none', 'adain', 'wavelet')
        tile_size: Size for tiling operation (in pixels)
        tile_overlap: Overlap size for tiling (in latent space)
        vqgan_tile_size: Size for VQGAN tile operation (in pixels)
        vqgan_tile_stride: Stride for VQGAN tile operation (in pixels)
        dec_w: Weight for combining VQGAN and Diffusion
        start_from_noise: If True, start from pure noise; else start from noisy z_blend
    
    Returns:
        output_image: PIL Image
        blended_pil: PIL Image of blended input
        lr_upscaled_pil: PIL Image of LR upscaled
        edge_pil: PIL Image of edge map
    """
    seed_everything(seed)
    device = next(model.parameters()).device
    
    # Load LR image to get original size
    lr_image_orig = Image.open(lr_image_path).convert("RGB")
    orig_h, orig_w = lr_image_orig.size[1], lr_image_orig.size[0]  # PIL uses (width, height)
    
    # Calculate output size from LR image size and downscale factor
    if output_size is None:
        output_h = orig_h * lr_downscale_factor
        output_w = orig_w * lr_downscale_factor
    else:
        output_h = output_size
        output_w = output_size
    
    # Use original LR image dimensions directly (no downscaling)
    lr_height = orig_h
    lr_width = orig_w
    
    print(f"\n{'='*60}")
    print("Tiled Inference with Blended Input")
    print(f"{'='*60}")
    print(f"LR image: {os.path.basename(lr_image_path)}")
    print(f"Edge map: {os.path.basename(edge_map_path)}")
    print(f"Blend alpha: {blend_alpha}")
    print(f"Blend beta: {blend_beta}")
    print(f"Original LR size: {orig_w}x{orig_h}")
    print(f"LR downscale factor: {lr_downscale_factor}")
    print(f"LR size: {lr_width}x{lr_height}")
    print(f"Output size: {output_w}x{output_h}")
    print(f"Sampling steps: {ddpm_steps}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Tile overlap: {tile_overlap}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"{'='*60}\n")
    
    # 1. Load inputs
    print("Step 1/8: Loading LR image and edge map...")
    lr_image = load_lr_image_with_dimensions(lr_image_path, height=lr_height, width=lr_width).to(device)
    edge_map = load_edge_map(edge_map_path, height=output_h, width=output_w).to(device)
    print(f"  LR image shape: {lr_image.shape}")
    print(f"  Edge map shape: {edge_map.shape}")
    
    # 2. Create blended image
    print(f"\nStep 2/8: Creating blended image (alpha={blend_alpha}, beta={blend_beta})...")
    blended_image, lr_upscaled = create_blended_input_tiled(lr_image, edge_map, blend_alpha, blend_beta, output_h, output_w)
    print(f"  Blended image shape: {blended_image.shape}")
    print(f"  Blended image range: [{blended_image.min():.3f}, {blended_image.max():.3f}]")
    print(f"  LR upscaled shape: {lr_upscaled.shape}")
    
    # 3. Setup timesteps for sampling
    print(f"\nStep 3/8: Setting up timestep schedule ({ddpm_steps} steps)...")
    setup_timesteps(model, ddpm_steps)
    print(f"  Using timesteps: {len(model.ori_timesteps)} steps")
    
    # 4. Prepare structural conditioning input
    print("\nStep 4/8: Preparing structural conditioning input...")
    print(f"  Using blended image as structural conditioning input")
    
    # 5. Text conditioning
    print(f"\nStep 5/8: Preparing text conditioning...")
    text_list = [text_prompt] if text_prompt else [""]
    print(f"  Text prompt: '{text_list[0]}'")
    with torch.no_grad():
        semantic_c = model.cond_stage_model(text_list)
    print(f"  Text embedding shape: {semantic_c.shape}")
    
    # 6. Create initial noise
    print("\nStep 6/8: Preparing initial state...")
    noise = torch.randn_like(blended_image)
    
    if start_from_noise:
        x_T = noise
        print("  Starting from: Pure Gaussian noise")
    else:
        # Use the last timestep from the shortened schedule
        t_start_idx = len(model.ori_timesteps) - 1
        t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
        t_start = repeat(t_start, '1 -> b', b=1)
        
        # Use q_sample with the current (shortened) schedule
        x_T = model.q_sample(x_start=blended_image, t=t_start, noise=noise)
        print(f"  Starting from: Noisy blended image at t={t_start_idx}")
    
    print(f"  Initial noise shape: {x_T.shape}")
    
    # 7. Check if tiling is needed
    print(f"\nStep 7/8: Checking tiling requirements...")
    blended_h, blended_w = output_h, output_w
    print(f"  Image size: {blended_h}x{blended_w}")
    print(f"  Tile size: {tile_size}x{tile_size}")
    
    # 8. Run tiled diffusion sampling
    print(f"\nStep 8/8: Running tiled diffusion sampling...")
    
    # Pad if necessary
    if not (blended_h % 32 == 0 and blended_w % 32 == 0):
        flag_pad = True
        pad_h = ((blended_h // 32) + 1) * 32 - blended_h
        pad_w = ((blended_w // 32) + 1) * 32 - blended_w
        blended_image_padded = F.pad(blended_image, pad=(0, pad_w, 0, pad_h), mode='reflect')
        print(f"  Padded to: {blended_image_padded.shape[2]}x{blended_image_padded.shape[3]}")
    else:
        flag_pad = False
        blended_image_padded = blended_image
    
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                if blended_image_padded.shape[2] > vqgan_tile_size or blended_image_padded.shape[3] > vqgan_tile_size:
                    print("  Using tiled processing...")
                    # Tiled processing with increased overlap for better blending
                    im_spliter = ImageSpliterTh(blended_image_padded, vqgan_tile_size, vqgan_tile_stride, sf=1)
                    
                    # Initialize Gaussian weights for better tile blending
                    im_spliter.weight = im_spliter._gaussian_weights(
                        vqgan_tile_size, vqgan_tile_size, 
                        blended_image_padded.size(0), 
                        blended_image_padded.device
                    )
                    
                    for im_blend_pch, index_infos in im_spliter:
                        seed_everything(seed)
                        
                        # Encode blended patch to latent space
                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_blend_pch))
                        
                        # Prepare noise for this patch
                        noise_patch = torch.randn_like(init_latent)
                        
                        if start_from_noise:
                            x_T_patch = noise_patch
                        else:
                            t_start_idx = len(model.ori_timesteps) - 1
                            t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
                            t_start = repeat(t_start, '1 -> b', b=im_blend_pch.size(0))
                            x_T_patch = model.q_sample_respace(
                                x_start=init_latent, 
                                t=t_start, 
                                sqrt_alphas_cumprod=model.sqrt_alphas_cumprod,
                                sqrt_one_minus_alphas_cumprod=model.sqrt_one_minus_alphas_cumprod,
                                noise=noise_patch
                            )
                        
                        # Sample using canvas method for tiling
                        samples, _ = model.sample_canvas(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            batch_size=im_blend_pch.size(0),
                            timesteps=ddpm_steps,
                            time_replace=ddpm_steps,
                            x_T=x_T_patch,
                            return_intermediates=True,
                            tile_size=int(tile_size/8),
                            tile_overlap=tile_overlap,
                            batch_size_sample=1
                        )
                        
                        # Decode with VQGAN
                        _, enc_fea_lq = vq_model.encode(im_blend_pch)
                        x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                        
                        # Color correction
                        if colorfix_type == 'adain':
                            x_samples = adaptive_instance_normalization(x_samples, im_blend_pch)
                        elif colorfix_type == 'wavelet':
                            x_samples = wavelet_reconstruction(x_samples, im_blend_pch)
                        
                        # Update spliter with Gaussian weights for seamless blending
                        im_spliter.update_gaussian(x_samples, index_infos)
                    
                    # Gather results
                    im_sr = im_spliter.gather()
                    im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
                    
                    # Crop to exact output size if needed
                    if im_sr.shape[2] != output_h or im_sr.shape[3] != output_w:
                        print(f"  Cropping from {im_sr.shape[2]}x{im_sr.shape[3]} to {output_h}x{output_w}")
                        im_sr = im_sr[:, :, :output_h, :output_w]
                    
                else:
                    print("  Using direct processing...")
                    # Direct processing (no tiling needed)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(blended_image_padded))
                    
                    noise = torch.randn_like(init_latent)
                    
                    if start_from_noise:
                        x_T = noise
                    else:
                        t_start_idx = len(model.ori_timesteps) - 1
                        t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
                        t_start = repeat(t_start, '1 -> b', b=blended_image_padded.size(0))
                        x_T = model.q_sample_respace(
                            x_start=init_latent, 
                            t=t_start, 
                            sqrt_alphas_cumprod=model.sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod=model.sqrt_one_minus_alphas_cumprod,
                            noise=noise
                        )
                    
                    # Sample using canvas method
                    samples, _ = model.sample_canvas(
                        cond=semantic_c,
                        struct_cond=init_latent,
                        batch_size=blended_image_padded.size(0),
                        timesteps=ddpm_steps,
                        time_replace=ddpm_steps,
                        x_T=x_T,
                        return_intermediates=True,
                        tile_size=int(tile_size/8),
                        tile_overlap=tile_overlap,
                        batch_size_sample=1
                    )
                    
                    # Decode with VQGAN
                    _, enc_fea_lq = vq_model.encode(blended_image_padded)
                    x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                    
                    # Color correction
                    if colorfix_type == 'adain':
                        x_samples = adaptive_instance_normalization(x_samples, blended_image_padded)
                    elif colorfix_type == 'wavelet':
                        x_samples = wavelet_reconstruction(x_samples, blended_image_padded)
                    
                    im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
                    
                    # Crop to exact output size if needed
                    if im_sr.shape[2] != output_h or im_sr.shape[3] != output_w:
                        print(f"  Cropping from {im_sr.shape[2]}x{im_sr.shape[3]} to {output_h}x{output_w}")
                        im_sr = im_sr[:, :, :output_h, :output_w]
    
    # Remove padding if applied
    if flag_pad:
        im_sr = im_sr[:, :blended_h, :blended_w, :] if len(im_sr.shape) == 4 else im_sr[:, :blended_h, :blended_w]
    
    # Convert to numpy
    im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c
    
    # Convert to PIL Image
    output_image = Image.fromarray(im_sr[0].astype(np.uint8))
    
    # Convert blended image to PIL for saving
    blended_vis = torch.clamp((blended_image + 1.0) / 2.0, min=0.0, max=1.0)
    blended_np = 255.0 * rearrange(blended_vis[0].cpu().numpy(), 'c h w -> h w c')
    blended_pil = Image.fromarray(blended_np.astype(np.uint8))
    
    # Convert LR upscaled to PIL for saving
    lr_upscaled_vis = torch.clamp((lr_upscaled + 1.0) / 2.0, min=0.0, max=1.0)
    lr_upscaled_np = 255.0 * rearrange(lr_upscaled_vis[0].cpu().numpy(), 'c h w -> h w c')
    lr_upscaled_pil = Image.fromarray(lr_upscaled_np.astype(np.uint8))
    
    # Convert edge map to PIL for saving
    edge_vis = torch.clamp((edge_map + 1.0) / 2.0, min=0.0, max=1.0)
    edge_np = 255.0 * rearrange(edge_vis[0].cpu().numpy(), 'c h w -> h w c')
    edge_pil = Image.fromarray(edge_np.astype(np.uint8))
    
    return output_image, blended_pil, lr_upscaled_pil, edge_pil


def batch_inference_tiled(
    model,
    vq_model,
    lr_dir,
    edge_dir,
    output_dir,
    blend_alpha=0.5,
    blend_beta=0.5,
    **kwargs
):
    """
    Process multiple image pairs in batch with tiling support
    """
    # Find all LR images
    lr_images = sorted(glob.glob(os.path.join(lr_dir, "*.png")) + 
                      glob.glob(os.path.join(lr_dir, "*.jpg")))
    
    # Find corresponding edge maps
    edge_images = []
    for lr_path in lr_images:
        basename = os.path.basename(lr_path)
        edge_path = os.path.join(edge_dir, basename)
        if not os.path.exists(edge_path):
            # Try with different extension
            name_no_ext = os.path.splitext(basename)[0]
            edge_path = os.path.join(edge_dir, name_no_ext + ".png")
            if not os.path.exists(edge_path):
                print(f"Warning: No edge map found for {basename}, skipping...")
                continue
        edge_images.append(edge_path)
    
    print(f"\nFound {len(lr_images)} LR images and {len(edge_images)} edge maps")
    
    if len(lr_images) != len(edge_images):
        print("Warning: Mismatch in number of LR images and edge maps!")
        lr_images = lr_images[:len(edge_images)]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each pair
    for i, (lr_path, edge_path) in enumerate(tqdm(zip(lr_images, edge_images), 
                                                   total=len(lr_images),
                                                   desc="Processing images")):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(lr_images)}: {os.path.basename(lr_path)}")
        print(f"{'='*60}")
        
        try:
            output_image, blended_pil, lr_upscaled_pil, edge_pil = inference_single_image_tiled(
                model,
                vq_model,
                lr_path,
                edge_path,
                blend_alpha=blend_alpha,
                blend_beta=blend_beta,
                **kwargs
            )
            
            # Save output
            basename = os.path.basename(lr_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            output_path = os.path.join(output_dir, basename)
            output_image.save(output_path)
            print(f"\n✓ Saved output: {output_path}")
            
            # Save blended input
            blended_path = os.path.join(output_dir, f"{name_no_ext}_blended.png")
            blended_pil.save(blended_path)
            print(f"✓ Saved blended: {blended_path}")
            
            # Save LR upscaled
            lr_upscaled_path = os.path.join(output_dir, f"{name_no_ext}_lr_upscaled.png")
            lr_upscaled_pil.save(lr_upscaled_path)
            print(f"✓ Saved LR upscaled: {lr_upscaled_path}")
            
            # Save edge map
            edge_path_out = os.path.join(output_dir, f"{name_no_ext}_edge.png")
            edge_pil.save(edge_path_out)
            print(f"✓ Saved edge map: {edge_path_out}")
            
        except Exception as e:
            print(f"\n✗ Error processing {basename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(lr_images)} images")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Blended Input Tiled Inference for StableSR")
    
    # Input arguments
    parser.add_argument("--lr-img", type=str, required=True,
                       help="Path to LR image file or directory")
    parser.add_argument("--edge-img", type=str, required=True,
                       help="Path to edge map file or directory")
    parser.add_argument("--outdir", type=str, default="outputs/blended_tile/",
                       help="Output directory for results")
    
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
                       help="Blending weight for LR upscaled image (default: 0.5)")
    parser.add_argument("--blend-beta", type=float, default=0.5,
                       help="Blending weight for edge map (default: 0.5)")
    parser.add_argument("--lr-downscale-factor", type=int, default=16,
                       help="Downscale factor for LR (e.g., 16 means 512/16=32x32) (default: 16)")
    parser.add_argument("--output-size", type=int, default=None,
                       help="Size of output image (default: None, calculated from LR size * downscale factor)")
    parser.add_argument("--ddpm-steps", type=int, default=200,
                       help="Number of DDPM sampling steps (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--text-prompt", type=str, default="",
                       help="Optional text prompt for semantic conditioning")
    
    # Tiling arguments
    parser.add_argument("--tile-size", type=int, default=512,
                       help="Size for tiling operation (in pixels) (default: 512)")
    parser.add_argument("--tile-overlap", type=int, default=64,
                       help="Overlap size for tiling (in latent space) (default: 64)")
    parser.add_argument("--vqgan-tile-size", type=int, default=512,
                       help="Size for VQGAN tile operation (in pixels) (default: 512)")
    parser.add_argument("--vqgan-tile-stride", type=int, default=320,
                       help="Stride for VQGAN tile operation (in pixels) (default: 320)")
    parser.add_argument("--dec-w", type=float, default=0.5,
                       help="Weight for combining VQGAN and Diffusion (default: 0.5)")
    
    # Sampling options
    parser.add_argument("--start-from-noise", action="store_true",
                       help="Start sampling from pure noise (default: start from noisy blended image)")
    parser.add_argument("--colorfix", type=str, default="none",
                       choices=["none", "adain", "wavelet"],
                       help="Color correction method (default: none)")
    
    # Output options
    parser.add_argument("--batch-mode", action="store_true",
                       help="Process all images in directories (batch mode)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed loading information")
    
    args = parser.parse_args()
    
    # Load config and model
    print("\n" + "="*60)
    print("Blended Input Tiled Inference for StableSR")
    print("="*60 + "\n")
    
    print("Loading configuration and model...")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, verbose=args.verbose)
    model.configs = config
    
    # Load VQGAN model
    print("Loading VQGAN model...")
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, args.vqgan_ckpt)
    vq_model.decoder.fusion_w = args.dec_w
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Auto-detect batch mode if directories are provided
    if os.path.isdir(args.lr_img) and os.path.isdir(args.edge_img):
        if not args.batch_mode:
            print("📁 Detected directory inputs - automatically enabling batch mode")
            args.batch_mode = True
    
    # Validate inputs
    if args.batch_mode and not os.path.isdir(args.lr_img):
        print(f"❌ Error: --lr-img must be a directory in batch mode, got: {args.lr_img}")
        return
    
    if args.batch_mode and not os.path.isdir(args.edge_img):
        print(f"❌ Error: --edge-img must be a directory in batch mode, got: {args.edge_img}")
        return
    
    # Run inference
    if args.batch_mode:
        # Batch processing
        batch_inference_tiled(
            model,
            vq_model,
            args.lr_img,
            args.edge_img,
            args.outdir,
            blend_alpha=args.blend_alpha,
            blend_beta=args.blend_beta,
            lr_downscale_factor=args.lr_downscale_factor,
            output_size=args.output_size,
            ddpm_steps=args.ddpm_steps,
            seed=args.seed,
            text_prompt=args.text_prompt,
            colorfix_type=args.colorfix,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            vqgan_tile_size=args.vqgan_tile_size,
            vqgan_tile_stride=args.vqgan_tile_stride,
            dec_w=args.dec_w,
            start_from_noise=args.start_from_noise
        )
    else:
        # Single image processing
        output_image, blended_pil, lr_upscaled_pil, edge_pil = inference_single_image_tiled(
            model,
            vq_model,
            args.lr_img,
            args.edge_img,
            blend_alpha=args.blend_alpha,
            blend_beta=args.blend_beta,
            lr_downscale_factor=args.lr_downscale_factor,
            output_size=args.output_size,
            ddpm_steps=args.ddpm_steps,
            seed=args.seed,
            text_prompt=args.text_prompt,
            colorfix_type=args.colorfix,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            vqgan_tile_size=args.vqgan_tile_size,
            vqgan_tile_stride=args.vqgan_tile_stride,
            dec_w=args.dec_w,
            start_from_noise=args.start_from_noise
        )
        
        # Save main output
        output_path = os.path.join(args.outdir, "output.png")
        output_image.save(output_path)
        print(f"\n✓ Output saved to: {output_path}")
        
        # Save blended input image
        blended_path = os.path.join(args.outdir, "blended_input.png")
        blended_pil.save(blended_path)
        print(f"✓ Blended input saved to: {blended_path}")
        
        # Save LR upscaled image
        lr_upscaled_path = os.path.join(args.outdir, "lr_upscaled.png")
        lr_upscaled_pil.save(lr_upscaled_path)
        print(f"✓ LR upscaled saved to: {lr_upscaled_path}")
        
        # Save edge map image
        edge_path_out = os.path.join(args.outdir, "edge_input.png")
        edge_pil.save(edge_path_out)
        print(f"✓ Edge input saved to: {edge_path_out}")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
