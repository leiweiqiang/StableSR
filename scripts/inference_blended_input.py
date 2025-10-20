#!/usr/bin/env python3
"""
Inference Script for Blended Input (LR Downscaled + Edge 512x512)
===================================================================

This script performs super-resolution inference using a blended input approach:
1. Input: LR image (downscaled by factor, e.g., 512/16=32x32) + Edge map (512x512)
2. Blend: Upscale LR to 512x512 and blend with edge map using two parameters
   Formula: blended = blend_alpha * lr_upscaled + blend_beta * edge
3. Output: High-quality SR image (512x512)

The blended image is encoded to latent space and passed through the time-aware
encoder to generate structural conditioning for the diffusion model.

Usage:
    # Single image inference
    python scripts/inference_blended_input.py \
        --lr-img inputs/lr_images/image.png \
        --edge-img inputs/edges_512/edge.png \
        --outdir outputs/blended_inference/ \
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
        --ckpt checkpoints/model.ckpt \
        --blend-alpha 0.5 \
        --blend-beta 0.5 \
        --lr-downscale-factor 16 \
        --ddpm-steps 200

    # Batch processing
    python scripts/inference_blended_input.py \
        --lr-img inputs/lr_images/ \
        --edge-img inputs/edges_512/ \
        --outdir outputs/batch/ \
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
        --ckpt checkpoints/model.ckpt \
        --blend-alpha 0.5 \
        --blend-beta 0.5 \
        --lr-downscale-factor 16 \
        --batch-mode

Author: StableSR Blended Input
Date: 2025-10-19
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
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


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


def load_edge_map(path, size=512):
    """
    Load edge map and prepare for model input
    
    Args:
        path: Path to edge map file (can be grayscale or RGB)
        size: Target size for edge map (default 512x512)
    
    Returns:
        Tensor of shape [1, 3, size, size] in range [-1, 1]
    """
    edge = Image.open(path).convert("RGB")  # Convert to RGB
    edge = edge.resize((size, size), Image.BICUBIC)
    
    # Convert to numpy and normalize
    edge = np.array(edge).astype(np.float32) / 255.0
    
    # To tensor [C, H, W]
    edge = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize to [-1, 1]
    edge = 2.0 * edge - 1.0
    
    return edge


def create_blended_input(lr_image, edge_map, blend_alpha=0.5, blend_beta=0.5, output_size=512):
    """
    Create blended image from LR and edge map using two-parameter blending
    
    Formula: blended = blend_alpha * lr_upscaled + blend_beta * edge
    
    Args:
        lr_image: Tensor [1, 3, 32, 32] in range [-1, 1]
        edge_map: Tensor [1, 3, 512, 512] in range [-1, 1]
        blend_alpha: Blending weight for LR upscaled image
        blend_beta: Blending weight for edge map
        output_size: Target output size (default 512)
    
    Returns:
        blended_image: Tensor [1, 3, 512, 512] in range [-1, 1]
        lr_upscaled: Tensor [1, 3, 512, 512] in range [-1, 1] (upscaled LR)
    """
    # Step 1: Upscale LR to output size using bicubic interpolation
    lr_upscaled = F.interpolate(
        lr_image,
        size=(output_size, output_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Step 2: Convert from [-1, 1] to [0, 1] for blending
    lr_upscaled_01 = (lr_upscaled + 1.0) / 2.0
    edge_01 = (edge_map + 1.0) / 2.0
    
    # Step 3: Two-parameter blending
    # blend_alpha controls LR weight, blend_beta controls edge weight
    blended = blend_alpha * lr_upscaled_01 + blend_beta * edge_01
    blended = torch.clamp(blended, 0.0, 1.0)  # Ensure [0, 1] range
    
    # Step 4: Convert back to [-1, 1] range
    blended_image = blended * 2.0 - 1.0
    blended_image = torch.clamp(blended_image, -1.0, 1.0)
    
    return blended_image, lr_upscaled


def setup_timesteps(model, ddpm_steps):
    """
    Setup timestep schedule for DDPM sampling
    
    Args:
        model: StableSR model
        ddpm_steps: Number of sampling steps (e.g., 200)
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
    
    # Ensure schedule buffers are on correct device after registration
    # register_schedule creates new tensors that default to CPU
    device = next(model.parameters()).device
    if hasattr(model, 'sqrt_alphas_cumprod'):
        model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
    if hasattr(model, 'sqrt_one_minus_alphas_cumprod'):
        model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
    if hasattr(model, 'alphas_cumprod'):
        model.alphas_cumprod = model.alphas_cumprod.to(device)
    
    # Ensure entire model (including sub-modules) is on device
    # This is critical for structcond_stage_model's time_embed module
    model = model.to(device)
    if hasattr(model, 'structcond_stage_model'):
        model.structcond_stage_model = model.structcond_stage_model.to(device)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process
    
    Args:
        num_timesteps: Total number of timesteps (1000)
        section_counts: List with desired step count (e.g., [200])
    
    Returns:
        Set of timestep indices to use
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


def inference_single_image(
    model,
    lr_image_path,
    edge_map_path,
    blend_alpha=0.5,
    blend_beta=0.5,
    lr_downscale_factor=16,
    output_size=512,
    ddpm_steps=200,
    seed=42,
    text_prompt="",
    start_from_noise=False,
    colorfix_type="none",
    save_intermediates=False
):
    """
    Run inference on a single image pair
    
    Args:
        model: StableSR model
        lr_image_path: Path to LR image (will be resized based on downscale factor)
        edge_map_path: Path to edge map (will be resized to output_size)
        blend_alpha: Blending weight for LR upscaled image, default 0.5
        blend_beta: Blending weight for edge map, default 0.5
        lr_downscale_factor: Downscale factor for LR (e.g., 16 means 512/16=32x32), default 16
        output_size: Size of output, default 512
        ddpm_steps: Number of sampling steps, default 200
        seed: Random seed
        text_prompt: Optional text prompt for conditioning
        start_from_noise: If True, start from pure noise; else start from noisy z_blend
        colorfix_type: Color correction method ('none', 'adain', 'wavelet')
        save_intermediates: If True, save intermediate images
    
    Returns:
        output_image: PIL Image (512x512)
        intermediates: Dict with intermediate tensors (if save_intermediates=True)
    """
    seed_everything(seed)
    device = next(model.parameters()).device
    
    # Compute LR size from downscale factor
    lr_size = output_size // lr_downscale_factor
    
    print(f"\n{'='*60}")
    print("Inference with Blended Input")
    print(f"{'='*60}")
    print(f"LR image: {os.path.basename(lr_image_path)}")
    print(f"Edge map: {os.path.basename(edge_map_path)}")
    print(f"Blend alpha: {blend_alpha}")
    print(f"Blend beta: {blend_beta}")
    print(f"LR downscale factor: {lr_downscale_factor}")
    print(f"LR size: {lr_size}x{lr_size}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Sampling steps: {ddpm_steps}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"{'='*60}\n")
    
    intermediates = {}
    
    # 1. Load inputs
    print("Step 1/8: Loading LR image and edge map...")
    lr_image = load_lr_image(lr_image_path, size=lr_size).to(device)
    edge_map = load_edge_map(edge_map_path, size=output_size).to(device)
    print(f"  LR image shape: {lr_image.shape}")
    print(f"  Edge map shape: {edge_map.shape}")
    
    if save_intermediates:
        intermediates['lr_input'] = lr_image
        intermediates['edge_input'] = edge_map
    
    # 2. Create blended image
    print(f"\nStep 2/8: Creating blended image (alpha={blend_alpha}, beta={blend_beta})...")
    blended_image, lr_upscaled = create_blended_input(lr_image, edge_map, blend_alpha, blend_beta, output_size)
    print(f"  Blended image shape: {blended_image.shape}")
    print(f"  Blended image range: [{blended_image.min():.3f}, {blended_image.max():.3f}]")
    print(f"  LR upscaled shape: {lr_upscaled.shape}")
    
    if save_intermediates:
        intermediates['blended_image'] = blended_image
        intermediates['lr_upscaled'] = lr_upscaled
    
    # 3. Encode blended image to latent space
    print("\nStep 3/8: Encoding blended image to latent space...")
    with torch.no_grad():
        encoder_posterior = model.encode_first_stage(blended_image)
        z_blend = model.get_first_stage_encoding(encoder_posterior)
    print(f"  Latent shape: {z_blend.shape}")
    print(f"  Latent range: [{z_blend.min():.3f}, {z_blend.max():.3f}]")
    
    if save_intermediates:
        intermediates['z_blend'] = z_blend
    
    # 4. Prepare structural conditioning input
    # Note: Pass z_blend as struct_cond - the sample() method will call
    # structcond_stage_model internally during the sampling loop
    print("\nStep 4/8: Preparing structural conditioning input...")
    print(f"  Using blended latent (z_blend) as structural conditioning input")
    print(f"  The time-aware encoder will be called during sampling loop")
    
    # 5. Text conditioning
    print(f"\nStep 5/8: Preparing text conditioning...")
    text_list = [text_prompt] if text_prompt else [""]
    print(f"  Text prompt: '{text_list[0]}'")
    with torch.no_grad():
        semantic_c = model.cond_stage_model(text_list)
    print(f"  Text embedding shape: {semantic_c.shape}")
    
    # 6. Setup timesteps for sampling
    print(f"\nStep 6/8: Setting up timestep schedule ({ddpm_steps} steps)...")
    setup_timesteps(model, ddpm_steps)
    print(f"  Using timesteps: {len(model.ori_timesteps)} steps")
    
    # 7. Create initial noise
    print("\nStep 7/8: Preparing initial state...")
    noise = torch.randn_like(z_blend)
    
    if start_from_noise:
        # Option A: Start from pure noise (more variation)
        x_T = noise
        print("  Starting from: Pure Gaussian noise")
    else:
        # Option B: Start from noisy z_blend (better structure preservation)
        # Use the last timestep from the shortened schedule
        t_start_idx = len(model.ori_timesteps) - 1  # Last index in shortened schedule
        t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
        t_start = repeat(t_start, '1 -> b', b=1)
        
        # Use q_sample with the current (shortened) schedule
        x_T = model.q_sample(x_start=z_blend, t=t_start, noise=noise)
        print(f"  Starting from: Noisy blended latent at t={t_start_idx} (better structure)")
    
    print(f"  Initial noise shape: {x_T.shape}")
    
    # 8. Run diffusion sampling
    print(f"\nStep 8/8: Running diffusion sampling ({ddpm_steps} steps)...")
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                samples, intermediates_sampling = model.sample(
                    cond=semantic_c,
                    struct_cond=z_blend,  # Pass z_blend, not pre-computed features!
                    batch_size=1,
                    timesteps=ddpm_steps,
                    time_replace=ddpm_steps,
                    x_T=x_T,
                    return_intermediates=True
                )
    
    print(f"  Sampled latent shape: {samples.shape}")
    
    # 9. Decode to image space
    print("\nDecoding to image space...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
    
    print(f"  Decoded image shape: {x_samples.shape}")
    print(f"  Decoded image range: [{x_samples.min():.3f}, {x_samples.max():.3f}]")
    
    # 10. Color correction (optional)
    if colorfix_type == "adain":
        print("Applying AdaIN color correction...")
        x_samples = adaptive_instance_normalization(x_samples, blended_image)
    elif colorfix_type == "wavelet":
        print("Applying wavelet color correction...")
        x_samples = wavelet_reconstruction(x_samples, blended_image)
    
    # 11. Post-process
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    # Convert to PIL Image
    x_sample = 255.0 * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    output_image = Image.fromarray(x_sample.astype(np.uint8))
    
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
    
    if save_intermediates:
        intermediates['samples'] = samples
        intermediates['decoded'] = x_samples
        intermediates['lr_input'] = lr_image
        intermediates['edge_input'] = edge_map
        intermediates['blended_image'] = blended_image
        intermediates['lr_upscaled'] = lr_upscaled
        return output_image, blended_pil, lr_upscaled_pil, edge_pil, intermediates
    
    return output_image, blended_pil, lr_upscaled_pil, edge_pil, None


def batch_inference(
    model,
    lr_dir,
    edge_dir,
    output_dir,
    blend_alpha=0.5,
    blend_beta=0.5,
    **kwargs
):
    """
    Process multiple image pairs in batch
    
    Args:
        model: StableSR model
        lr_dir: Directory containing LR images (will be downscaled as configured)
        edge_dir: Directory containing edge maps (512x512)
        output_dir: Output directory
        blend_alpha: Blending weight for LR upscaled image
        blend_beta: Blending weight for edge map
        **kwargs: Additional arguments for inference_single_image (including lr_downscale_factor)
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
            output_image, blended_pil, lr_upscaled_pil, edge_pil, _ = inference_single_image(
                model,
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


def load_model_from_config(config, ckpt, verbose=False):
    """
    Load model from checkpoint
    
    Args:
        config: OmegaConf config
        ckpt: Path to checkpoint file
        verbose: Print detailed loading info
    
    Returns:
        model: Loaded model in eval mode
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
    print("Model loaded successfully!\n")
    
    return model


def save_comparison_grid(lr_image, edge_map, blended_image, output_image, save_path):
    """
    Save a comparison grid showing all intermediate steps
    
    Args:
        lr_image: Tensor [1, 3, 32, 32]
        edge_map: Tensor [1, 3, 512, 512]
        blended_image: Tensor [1, 3, 512, 512]
        output_image: PIL Image (512x512)
        save_path: Where to save the grid
    """
    from torchvision.utils import make_grid
    
    # Upscale LR for visualization
    lr_vis = F.interpolate(lr_image, size=(512, 512), mode='nearest')
    
    # Convert all to [0, 1] range
    lr_vis = (lr_vis + 1.0) / 2.0
    edge_vis = (edge_map + 1.0) / 2.0
    blended_vis = (blended_image + 1.0) / 2.0
    output_tensor = torch.from_numpy(np.array(output_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    output_tensor = output_tensor.to(lr_vis.device)
    
    # Create grid
    grid = torch.cat([lr_vis, edge_vis, blended_vis, output_tensor], dim=0)
    grid = make_grid(grid, nrow=4, padding=10, pad_value=1.0)
    
    # Save
    grid_np = rearrange(grid.cpu().numpy(), 'c h w -> h w c')
    grid_np = (grid_np * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    print(f"Saved comparison grid to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Blended Input Inference for StableSR")
    
    # Input arguments
    parser.add_argument("--lr-img", type=str, required=True,
                       help="Path to LR image file or directory (will be resized based on downscale factor)")
    parser.add_argument("--edge-img", type=str, required=True,
                       help="Path to edge map file or directory (512x512 or will be resized)")
    parser.add_argument("--outdir", type=str, default="outputs/blended_inference/",
                       help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--config", type=str,
                       default="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml",
                       help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    
    # Processing arguments
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                       help="Blending weight for LR upscaled image (default: 0.5)")
    parser.add_argument("--blend-beta", type=float, default=0.5,
                       help="Blending weight for edge map (default: 0.5)")
    parser.add_argument("--lr-downscale-factor", type=int, default=16,
                       help="Downscale factor for LR (e.g., 16 means 512/16=32x32) (default: 16)")
    parser.add_argument("--output-size", type=int, default=512,
                       help="Size of output image (default: 512)")
    parser.add_argument("--ddpm-steps", type=int, default=200,
                       help="Number of DDPM sampling steps (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--text-prompt", type=str, default="",
                       help="Optional text prompt for semantic conditioning")
    
    # Sampling options
    parser.add_argument("--start-from-noise", action="store_true",
                       help="Start sampling from pure noise (default: start from noisy z_blend)")
    parser.add_argument("--colorfix", type=str, default="none",
                       choices=["none", "adain", "wavelet"],
                       help="Color correction method (default: none)")
    
    # Output options
    parser.add_argument("--batch-mode", action="store_true",
                       help="Process all images in directories (batch mode)")
    parser.add_argument("--save-intermediates", action="store_true",
                       help="Save intermediate images (blended, etc.)")
    parser.add_argument("--save-comparison", action="store_true",
                       help="Save comparison grid with all steps")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed loading information")
    
    args = parser.parse_args()
    
    # Load config and model
    print("\n" + "="*60)
    print("Blended Input Inference for StableSR")
    print("="*60 + "\n")
    
    print("Loading configuration and model...")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, verbose=args.verbose)
    model.configs = config
    
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
        print("Tip: Remove --batch-mode flag for single image processing")
        return
    
    if args.batch_mode and not os.path.isdir(args.edge_img):
        print(f"❌ Error: --edge-img must be a directory in batch mode, got: {args.edge_img}")
        print("Tip: Remove --batch-mode flag for single image processing")
        return
    
    if not args.batch_mode and os.path.isdir(args.lr_img):
        print(f"❌ Error: --lr-img is a directory but --batch-mode is not enabled")
        print(f"Please use --batch-mode flag to process directories")
        return
    
    if not args.batch_mode and os.path.isdir(args.edge_img):
        print(f"❌ Error: --edge-img is a directory but --batch-mode is not enabled")
        print(f"Please use --batch-mode flag to process directories")
        return
    
    # Run inference
    if args.batch_mode:
        # Batch processing
        if not os.path.isdir(args.lr_img):
            print(f"Error: --lr-img must be a directory in batch mode")
            return
        if not os.path.isdir(args.edge_img):
            print(f"Error: --edge-img must be a directory in batch mode")
            return
        
        batch_inference(
            model,
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
            start_from_noise=args.start_from_noise,
            colorfix_type=args.colorfix,
            save_intermediates=args.save_intermediates
        )
    else:
        # Single image processing
        output_image, blended_pil, lr_upscaled_pil, edge_pil, intermediates = inference_single_image(
            model,
            args.lr_img,
            args.edge_img,
            blend_alpha=args.blend_alpha,
            blend_beta=args.blend_beta,
            lr_downscale_factor=args.lr_downscale_factor,
            output_size=args.output_size,
            ddpm_steps=args.ddpm_steps,
            seed=args.seed,
            text_prompt=args.text_prompt,
            start_from_noise=args.start_from_noise,
            colorfix_type=args.colorfix,
            save_intermediates=args.save_intermediates
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
        
        # Save intermediates if requested
        if args.save_intermediates and intermediates:
            print("\nSaving intermediate images...")
            
            # Save LR input (upscaled for visualization)
            lr_vis = F.interpolate(intermediates['lr_input'], size=(512, 512), mode='nearest')
            lr_vis = torch.clamp((lr_vis + 1.0) / 2.0, 0, 1)
            lr_np = 255 * rearrange(lr_vis[0].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(lr_np.astype(np.uint8)).save(
                os.path.join(args.outdir, "lr_input.png")
            )
            
            # Save edge input
            edge_vis = torch.clamp((intermediates['edge_input'] + 1.0) / 2.0, 0, 1)
            edge_np = 255 * rearrange(edge_vis[0].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(edge_np.astype(np.uint8)).save(
                os.path.join(args.outdir, "edge_input.png")
            )
            
            # Save blended image
            blended_vis = torch.clamp((intermediates['blended_image'] + 1.0) / 2.0, 0, 1)
            blended_np = 255 * rearrange(blended_vis[0].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(blended_np.astype(np.uint8)).save(
                os.path.join(args.outdir, "blended_image.png")
            )
            
            print("✓ Intermediate images saved")
        
        # Save comparison grid if requested
        if args.save_comparison:
            print("\nCreating comparison grid...")
            lr_size_grid = args.output_size // args.lr_downscale_factor
            lr_image_grid = load_lr_image(args.lr_img, size=lr_size_grid).to(next(model.parameters()).device)
            edge_map_grid = load_edge_map(args.edge_img, size=args.output_size).to(next(model.parameters()).device)
            blended_image_grid, _ = create_blended_input(lr_image_grid, edge_map_grid, args.blend_alpha, args.blend_beta, args.output_size)
            
            grid_path = os.path.join(args.outdir, "comparison_grid.png")
            save_comparison_grid(lr_image_grid, edge_map_grid, blended_image_grid, output_image, grid_path)
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

