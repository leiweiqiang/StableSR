"""
Edge-to-Image Generation using Canny Edge Maps
================================================

This script generates images from canny edge maps using a trained StableSR model.

Key differences from super-resolution inference:
- Input: Canny edge maps (single-channel or 3-channel RGB)
- Output: Generated images at SAME size as input (no upscaling)
- Method: Edge map is used as struct_cond (primary structure guidance)
- No LR image input, no GT image needed

Pipeline:
1. Load canny edge map from file (supports grayscale or RGB)
2. Normalize to [-1, 1] range
3. Encode to latent space via VQGAN encoder
4. Use edge latent as struct_cond for diffusion sampling
5. Start from pure noise or noisy edge latent
6. Sample using DDPM/DDIM to generate image
7. Decode latent to image space
8. Save generated image

Usage:
    python scripts/inference_edge_to_image.py \\
        --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \\
        --ckpt path/to/checkpoint.ckpt \\
        --edge-img path/to/edge_maps/ \\
        --outdir outputs/edge_to_image/ \\
        --ddim_steps 50 \\
        --input_size 512 \\
        --start_from_edge
"""

import argparse, os, sys, glob
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

# Ensure we import from the current project directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ldm.util import instantiate_from_config
import copy


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process.
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


def load_model_from_config(config, ckpt, verbose=False):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_edge_map(path, target_size=512):
    """
    Load edge map from file and prepare for model input
    
    Args:
        path: Path to edge map image (can be grayscale or RGB)
        target_size: Target size for resizing (default 512)
        
    Returns:
        Tensor of shape [1, 3, H, W] normalized to [-1, 1]
    """
    # Load image
    edge_image = Image.open(path).convert("RGB")  # Convert to RGB (even if grayscale)
    
    # Resize to target size
    edge_image = edge_image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to numpy array and normalize to [0, 1]
    edge_np = np.array(edge_image).astype(np.float32) / 255.0
    
    # Convert to tensor [C, H, W]
    edge_np = np.transpose(edge_np, (2, 0, 1))
    edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).float()
    
    # Normalize to [-1, 1]
    edge_tensor = 2.0 * edge_tensor - 1.0
    
    return edge_tensor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--edge-img",
        type=str,
        nargs="?",
        required=True,
        help="Path to folder with canny edge map images"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/edge_to_image",
        help="Directory to write results to"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=50,
        help="Number of ddpm sampling steps"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="How many samples to produce for each edge map"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml",
        help="Path to config which constructs model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="Path to checkpoint of model"
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt",
        help="Path to checkpoint of VQGAN model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Input and output image size (must match training size)"
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.0,
        help="Weight for VQGAN decoder fusion (0.0 = no fusion, recommended for edge-to-image)"
    )
    parser.add_argument(
        "--start_from_edge",
        action="store_true",
        help="Start diffusion from noisy edge latent instead of pure noise"
    )
    parser.add_argument(
        "--start_timestep",
        type=int,
        default=999,
        help="Starting timestep if using --start_from_edge (0-999, higher = more noise)"
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="",
        help="Text prompt for conditional generation (empty for unconditional)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Maximum number of images to process (-1 for all)"
    )

    opt = parser.parse_args()
    
    # Expand paths
    opt.ckpt = os.path.expanduser(opt.ckpt)
    opt.vqgan_ckpt = os.path.expanduser(opt.vqgan_ckpt)
    opt.edge_img = os.path.expanduser(opt.edge_img)
    opt.outdir = os.path.expanduser(opt.outdir)
    
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("=" * 80)
    print("Edge-to-Image Generation")
    print("=" * 80)
    print(f"Input edge directory: {opt.edge_img}")
    print(f"Output directory: {opt.outdir}")
    print(f"Image size: {opt.input_size}×{opt.input_size}")
    print(f"DDPM steps: {opt.ddpm_steps}")
    print(f"Decoder fusion weight: {opt.dec_w}")
    print(f"Start from edge: {opt.start_from_edge}")
    if opt.start_from_edge:
        print(f"  Starting timestep: {opt.start_timestep}")
    print(f"Text prompt: '{opt.text_prompt}' (empty = unconditional)")
    print(f"Seed: {opt.seed}")
    print("=" * 80)

    # Load VQGAN model
    print("\nLoading VQGAN model...")
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w
    print(f"✓ VQGAN loaded (fusion weight: {opt.dec_w})")

    # Load diffusion model
    print("\nLoading diffusion model...")
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)
    model = model.to(device)
    
    # Setup timestep schedule
    model.register_schedule(
        given_betas=None, 
        beta_schedule="linear", 
        timesteps=1000,
        linear_start=0.00085, 
        linear_end=0.0120, 
        cosine_s=8e-3
    )
    model.num_timesteps = 1000
    
    # Store original schedule for respace
    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)
    
    # Create shortened timestep schedule
    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
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
    model = model.to(device)
    print(f"✓ Diffusion model loaded")
    print(f"  Timesteps: {len(model.ori_timesteps)} (from 1000)")

    # Create output directory
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    # Find all edge map images
    edge_path = opt.edge_img
    if os.path.isfile(edge_path):
        edge_list = [edge_path]
    else:
        edge_list = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            edge_list.extend(glob.glob(os.path.join(edge_path, ext)))
        edge_list.sort()
    
    if opt.max_images > 0:
        edge_list = edge_list[:opt.max_images]
    
    print(f"\nFound {len(edge_list)} edge map images")
    
    if len(edge_list) == 0:
        print("ERROR: No edge map images found!")
        return
    
    # Process images
    batch_size = opt.n_samples
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    
    results = []
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                
                for idx, edge_file in enumerate(tqdm(edge_list, desc="Generating images")):
                    # Load edge map
                    edge_input = load_edge_map(edge_file, opt.input_size).to(device)
                    
                    # Encode edge map to latent space
                    edge_latent_generator, enc_fea_edge = vq_model.encode(edge_input)
                    edge_latent = model.get_first_stage_encoding(edge_latent_generator)
                    
                    # Prepare text conditioning
                    text_prompt = [opt.text_prompt] * batch_size
                    semantic_c = model.cond_stage_model(text_prompt)
                    
                    # Initialize noise
                    if opt.start_from_edge:
                        # Start from noisy edge latent
                        noise = torch.randn_like(edge_latent)
                        t = repeat(
                            torch.tensor([opt.start_timestep]), 
                            '1 -> b', 
                            b=batch_size
                        ).to(device).long()
                        x_T = model.q_sample_respace(
                            x_start=edge_latent,
                            t=t,
                            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                            noise=noise
                        )
                    else:
                        # Start from pure noise
                        x_T = torch.randn_like(edge_latent)
                    
                    # Replicate for n_samples
                    if batch_size > 1:
                        edge_latent = edge_latent.repeat(batch_size, 1, 1, 1)
                        x_T = x_T.repeat(batch_size, 1, 1, 1)
                    
                    # Sample using edge latent as struct_cond
                    # Option B: Use edge as struct_cond only, no separate edge_map
                    samples, _ = model.sample(
                        cond=semantic_c,
                        struct_cond=edge_latent,  # Edge latent as primary guidance
                        batch_size=batch_size,
                        timesteps=opt.ddpm_steps,
                        time_replace=opt.ddpm_steps,
                        x_T=x_T,
                        return_intermediates=True,
                        # Note: Not passing edge_map parameter (Option B approach)
                    )
                    
                    # Decode to image space
                    if opt.dec_w > 0:
                        # Use edge features for fusion
                        x_samples = vq_model.decode(samples * 1.0 / model.scale_factor, enc_fea_edge)
                    else:
                        # No fusion (recommended)
                        x_samples = vq_model.decode(samples * 1.0 / model.scale_factor, None)
                    
                    # Clamp to [0, 1]
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # Save results
                    edge_basename = os.path.splitext(os.path.basename(edge_file))[0]
                    
                    for sample_idx in range(batch_size):
                        x_sample = 255.0 * rearrange(
                            x_samples[sample_idx].cpu().numpy(), 
                            'c h w -> h w c'
                        )
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        
                        if batch_size > 1:
                            save_path = os.path.join(
                                outpath, 
                                f"{edge_basename}_sample{sample_idx:02d}.png"
                            )
                        else:
                            save_path = os.path.join(
                                outpath, 
                                f"{edge_basename}_generated.png"
                            )
                        
                        img.save(save_path)
                        results.append(save_path)
                    
                    # Also save edge map for reference
                    edge_ref = torch.clamp((edge_input + 1.0) / 2.0, min=0.0, max=1.0)
                    edge_ref = 255.0 * rearrange(
                        edge_ref[0].cpu().numpy(), 
                        'c h w -> h w c'
                    )
                    edge_ref_path = os.path.join(outpath, f"{edge_basename}_edge_ref.png")
                    Image.fromarray(edge_ref.astype(np.uint8)).save(edge_ref_path)
                
                toc = time.time()
    
    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Processed: {len(edge_list)} edge maps")
    print(f"Generated: {len(results)} images")
    print(f"Time elapsed: {toc - tic:.2f} seconds")
    print(f"Average time per edge: {(toc - tic) / len(edge_list):.2f} seconds")
    print(f"Output directory: {outpath}")
    print("=" * 80)


if __name__ == "__main__":
    main()


