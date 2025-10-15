"""Edge-Enhanced Super-Resolution Inference Script

This script provides edge-enhanced super-resolution inference based on
sr_val_ddpm_text_T_vqganfin_old.py with integrated EdgeMapGenerator support.

Features:
- Edge map generation using EdgeMapGenerator (unified training/inference logic)
- Support for both GT-based and LR-based edge generation
- Multiple edge modes: standard, white/black edges, dummy edge maps
- Batch processing with configurable batch size
- Color correction options (adain, wavelet, nofix)
- Comprehensive logging and debugging output

Usage:
    # Basic usage with edge processing
    conda activate sr_infer
    python new_features/EdgeInference/sr_val_edge_inference.py \\
        --init-img inputs/lr_images \\
        --gt-img inputs/gt_images \\
        --outdir outputs/edge_inference \\
        --use_edge_processing \\
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \\
        --ckpt models/your_edge_model.ckpt

Author: StableSR_Edge_v3 Team
Date: 2025-10-15
"""

import argparse
import os
import sys
import glob
import shutil
import inspect
from datetime import datetime

# Ensure we import from the current project directory
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

# Edge-specific imports
import cv2
from basicsr.utils.edge_utils import EdgeMapGenerator

# Create global EdgeMapGenerator instance with more sensitive parameters
# 降低阈值因子以检测更多边缘 (原始: 0.7/1.3, 改进: 0.4/0.9)
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.4,  # 从0.7降到0.4，检测更多边缘
    canny_threshold_upper_factor=0.9   # 从1.3降到0.9，检测更多边缘
)


class Logger(object):
    """
    Logger class that writes to both console and file
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
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


def chunk(it, size):
    """Chunk an iterable into fixed-size chunks"""
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    """Load model from config and checkpoint"""
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


def load_img(path):
    """Load image and convert to tensor format"""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def generate_edge_map(image_tensor):
    """
    Generate edge map from input image tensor
    
    Args:
        image_tensor: Input image tensor [B, 3, H, W], values in [-1, 1]
        
    Returns:
        edge_map: Edge map tensor [B, 3, H, W], values in [-1, 1]
    
    Note:
        Uses EdgeMapGenerator for unified edge generation logic
        that matches training-time edge map generation.
    """
    return edge_generator.generate_from_tensor(
        image_tensor,
        input_format='RGB',
        normalize_range='[-1,1]'
    )


def generate_edge_map_from_gt(gt_image_path, device):
    """
    Generate edge map from ground truth image file
    
    Args:
        gt_image_path: Path to ground truth image (original resolution)
        device: Device to place the tensor on
        
    Returns:
        edge_map: Edge map tensor [1, 3, H_gt, W_gt], values in [-1, 1]
                 at ORIGINAL GT resolution
    
    Note:
        This matches the training setup where GT images are used for
        edge map generation. Edge maps maintain GT resolution.
    """
    # Load GT image at ORIGINAL resolution
    gt_image = Image.open(gt_image_path).convert("RGB")
    
    # Convert to numpy array [H, W, C] - keep original resolution
    img_np = np.array(gt_image).astype(np.float32) / 255.0
    
    # Convert to tensor format [C, H, W]
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
    
    # Normalize to [-1, 1]
    img_tensor = 2.0 * img_tensor - 1.0
    
    # Generate edge map from ORIGINAL resolution GT image
    edge_map = generate_edge_map(img_tensor)
    
    return edge_map.to(device)


def main():
    parser = argparse.ArgumentParser(description="Edge-Enhanced Super-Resolution Inference")

    # Input/Output paths
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input LR image directory",
        default="inputs/user_upload",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="directory to write results to",
        default="outputs/edge_inference",
    )
    parser.add_argument(
        "--gt-img",
        type=str,
        nargs="?",
        help="[RECOMMENDED] Path to ground truth images directory for edge map generation. "
             "Training uses GT images at ORIGINAL resolution for edge maps, so using GT images "
             "during inference ensures consistency and better results.",
        default=None,
    )
    
    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=200,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="batch size for processing images",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size (LR resolution after resize)",
    )
    
    # Edge processing options
    parser.add_argument(
        "--use_edge_processing",
        action="store_true",
        help="Enable edge processing for enhanced super-resolution",
    )
    parser.add_argument(
        "--use_white_edge",
        action="store_true",
        help="Use black (all negative ones) edge maps instead of generated edge maps",
    )
    parser.add_argument(
        "--use_dummy_edge",
        action="store_true",
        help="Use a fixed dummy edge map for all images",
    )
    parser.add_argument(
        "--dummy_edge_path",
        type=str,
        default="/stablesr_dataset/default_edge.png",
        help="Path to the dummy edge image to use when --use_dummy_edge is enabled",
    )
    
    # Color correction
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type: adain, wavelet, or nofix",
    )
    
    # Model parameters
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Maximum number of images to process (-1 for all images)",
    )
    parser.add_argument(
        "--specific_file",
        type=str,
        default="",
        help="Process only this specific file (filename only, not full path)",
    )

    opt = parser.parse_args()
    
    # Expand user paths (~ symbols)
    opt.ckpt = os.path.expanduser(opt.ckpt)
    opt.vqgan_ckpt = os.path.expanduser(opt.vqgan_ckpt)
    opt.init_img = os.path.expanduser(opt.init_img)
    opt.outdir = os.path.expanduser(opt.outdir)
    if opt.gt_img:
        opt.gt_img = os.path.expanduser(opt.gt_img)
    if opt.dummy_edge_path:
        opt.dummy_edge_path = os.path.expanduser(opt.dummy_edge_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Print configuration
    print('=' * 80)
    print('Edge-Enhanced Super-Resolution Inference')
    print('=' * 80)
    
    print('\n>>>>>>>>>>Color Correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('✓ Use AdaIN color correction')
    elif opt.colorfix_type == 'wavelet':
        print('✓ Use wavelet color correction')
    else:
        print('✗ No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    print('>>>>>>>>>>Edge Processing>>>>>>>>>>>')
    if opt.use_edge_processing:
        print('✓ Edge processing ENABLED - using edge-enhanced model')
        if opt.use_white_edge:
            print('  Mode: BLACK edge maps (no edges)')
        elif opt.use_dummy_edge:
            print(f'  Mode: DUMMY edge map from: {opt.dummy_edge_path}')
        elif opt.gt_img:
            print(f'  Mode: GT-based edge generation')
            print(f'  GT directory: {opt.gt_img}')
            print('  ✓ This matches the training setup')
        else:
            print('  Mode: LR-based edge generation')
            print('  ⚠ WARNING: Training uses GT images for edge maps!')
            print('  ⚠ Using LR images may cause domain mismatch.')
            print('  ⚠ Recommend using --gt-img for best results.')
    else:
        print('✗ Edge processing DISABLED - using standard model')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # Load VQGAN model
    print('Loading VQGAN model...')
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w
    print('✓ VQGAN model loaded\n')

    seed_everything(opt.seed)

    # Image transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])

    # Load diffusion model
    print('Loading diffusion model...')
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    print('✓ Diffusion model loaded\n')
    
    # Check if model supports edge processing
    model_supports_edge = hasattr(model, 'use_edge_processing') and model.use_edge_processing
    if opt.use_edge_processing:
        if model_supports_edge:
            print("✓ Model supports edge processing - edge inference will be used\n")
        else:
            print("⚠ Warning: --use_edge_processing flag is set, but model does not support edge processing")
            print("  Edge maps will be generated and saved, but may not affect results")
            print("  To use edge processing, ensure config file has use_edge_processing: true\n")

    # Create output directory
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(outpath, f"edge_inference_{timestamp}.log")
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"\n{'=' * 80}")
    print(f"Edge Inference Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 80}\n")

    batch_size = opt.n_samples

    # Get image list
    if opt.specific_file:
        # Process only specific file
        if not os.path.exists(os.path.join(opt.init_img, opt.specific_file)):
            print(f"ERROR: Specified file not found: {opt.specific_file}")
            return
        img_list_ori = [opt.specific_file]
        print(f"Processing specific file: {opt.specific_file}\n")
    else:
        # Process all or limited number of files
        img_list_ori = sorted([f for f in os.listdir(opt.init_img) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Limit number of images if max_images is specified
        if opt.max_images > 0:
            img_list_ori = img_list_ori[:opt.max_images]
            print(f"Processing {len(img_list_ori)} images (limited by max_images)\n")
        else:
            print(f"Processing all {len(img_list_ori)} images\n")
    
    # Load and prepare images
    img_list = copy.deepcopy(img_list_ori)
    init_image_list = []
    for item in img_list_ori:
        output_path = os.path.join(outpath, item)
        if os.path.exists(output_path):
            print(f"Skipping {item} (already exists)")
            img_list.remove(item)
            continue
        cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
        cur_image = transform(cur_image)
        cur_image = cur_image.clamp(-1, 1)
        init_image_list.append(cur_image)
    
    if len(init_image_list) == 0:
        print("\nNo images to process (all already exist in output directory)")
        sys.stdout = logger.terminal
        logger.close()
        return
    
    init_image_list = torch.cat(init_image_list, dim=0)
    niters = math.ceil(init_image_list.size(0) / batch_size)
    init_image_list = init_image_list.chunk(niters)

    # Register diffusion schedule
    print("\nRegistering diffusion schedule...")
    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

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
    print(f"✓ Schedule registered with {opt.ddpm_steps} steps\n")

    # Print model parameters info
    print("Model parameters:")
    param_list = []
    untrain_paramlist = []
    name_list = []
    for k, v in model.named_parameters():
        if 'spade' in k or 'structcond_stage_model' in k:
            param_list.append(v)
        else:
            name_list.append(k)
            untrain_paramlist.append(v)
    trainable_params = sum(p.numel() for p in param_list)
    untrainable_params = sum(p.numel() for p in untrain_paramlist)
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Untrainable params: {untrainable_params:,}\n")

    # Check if model.sample supports edge_map parameter
    model_sample_supports_edge = False
    if opt.use_edge_processing:
        sample_sig = inspect.signature(model.sample)
        model_sample_supports_edge = 'edge_map' in sample_sig.parameters
        if model_sample_supports_edge:
            print("✓ model.sample() supports edge_map parameter - will use edge-enhanced sampling\n")
        else:
            print("⚠ model.sample() does not support edge_map parameter")
            print("  Edge maps will be saved but not used in sampling\n")
    
    # Start inference
    print(f"{'=' * 80}")
    print("Starting inference...")
    print(f"{'=' * 80}\n")
    
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                
                for n in trange(niters, desc="Processing batches"):
                    init_image = init_image_list[n]
                    
                    # Encode LR image
                    init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                    init_latent = model.get_first_stage_encoding(init_latent_generator)
                    
                    # Get text conditioning (empty for SR)
                    text_init = [''] * init_image.size(0)
                    semantic_c = model.cond_stage_model(text_init)

                    # Prepare noise
                    noise = torch.randn_like(init_latent)
                    t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                    t = t.to(device).long()
                    x_T = model.q_sample_respace(
                        x_start=init_latent, t=t,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        noise=noise
                    )
                    x_T = None

                    # Edge processing
                    if opt.use_edge_processing:
                        edge_maps = []
                        
                        if opt.use_dummy_edge:
                            # Use fixed dummy edge map from file
                            if not os.path.exists(opt.dummy_edge_path):
                                print(f"ERROR: Dummy edge file not found: {opt.dummy_edge_path}")
                                print("Falling back to black edge maps")
                                for i in range(init_image.size(0)):
                                    black_edge_map = -torch.ones_like(init_image[i:i+1])
                                    edge_maps.append(black_edge_map)
                                edge_maps = torch.cat(edge_maps, dim=0)
                            else:
                                dummy_edge_img = Image.open(opt.dummy_edge_path).convert('RGB')
                                target_size = (init_image.size(3), init_image.size(2))
                                dummy_edge_img = dummy_edge_img.resize(target_size, Image.BICUBIC)
                                to_tensor = torchvision.transforms.ToTensor()
                                dummy_edge_tensor = to_tensor(dummy_edge_img)
                                dummy_edge_tensor = dummy_edge_tensor * 2.0 - 1.0
                                dummy_edge_tensor = dummy_edge_tensor.unsqueeze(0).to(device)
                                for i in range(init_image.size(0)):
                                    edge_maps.append(dummy_edge_tensor)
                                edge_maps = torch.cat(edge_maps, dim=0)
                        
                        elif opt.use_white_edge:
                            # Use black edge maps (no edges)
                            for i in range(init_image.size(0)):
                                black_edge_map = -torch.ones_like(init_image[i:i+1])
                                edge_maps.append(black_edge_map)
                            edge_maps = torch.cat(edge_maps, dim=0)
                        
                        else:
                            # Generate edge maps from GT or LR images
                            for i in range(init_image.size(0)):
                                batch_start_idx = n * batch_size
                                img_idx = batch_start_idx + i
                                
                                if img_idx < len(img_list_ori):
                                    img_name = img_list_ori[img_idx]
                                    img_basename = os.path.splitext(os.path.basename(img_name))[0]
                                    
                                    edge_map = None
                                    if opt.gt_img:
                                        # Use GT image for edge map generation
                                        gt_img_path = os.path.join(opt.gt_img, img_basename + '.png')
                                        if not os.path.exists(gt_img_path):
                                            for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                                                alt_path = os.path.join(opt.gt_img, img_basename + ext)
                                                if os.path.exists(alt_path):
                                                    gt_img_path = alt_path
                                                    break
                                        
                                        if os.path.exists(gt_img_path):
                                            edge_map = generate_edge_map_from_gt(gt_img_path, device)
                                        else:
                                            raise FileNotFoundError(
                                                f"GT image not found for {img_basename}. "
                                                f"Expected at: {gt_img_path}"
                                            )
                                    else:
                                        # Use LR image for edge map generation
                                        if n == 0 and i == 0:
                                            print("⚠ WARNING: Using LR images for edge maps")
                                        edge_map = generate_edge_map(init_image[i:i+1])
                                    
                                    edge_maps.append(edge_map)
                            
                            edge_maps = torch.cat(edge_maps, dim=0)
                        
                        # Print resolution info for first batch
                        if n == 0:
                            print(f"\n=== Resolution Info (Batch 0) ===")
                            print(f"LR input: {init_image.shape}")
                            print(f"Edge map: {edge_maps.shape}")
                            print(f"Init latent: {init_latent.shape}")
                            print(f"Edge range: [{edge_maps.min():.3f}, {edge_maps.max():.3f}]")
                            print(f"{'=' * 40}\n")
                        
                        # Save edge maps
                        edge_map_dir = os.path.join(outpath, "edge_maps")
                        os.makedirs(edge_map_dir, exist_ok=True)
                        
                        for i in range(edge_maps.size(0)):
                            batch_start_idx = n * batch_size
                            img_idx = batch_start_idx + i
                            if img_idx < len(img_list_ori):
                                img_name = img_list_ori[img_idx]
                                img_basename = os.path.splitext(os.path.basename(img_name))[0]
                                
                                edge_map_np = edge_maps[i].cpu().numpy()
                                edge_map_np = (edge_map_np + 1.0) / 2.0 * 255.0
                                edge_map_np = np.clip(edge_map_np, 0, 255)
                                edge_map_np = np.transpose(edge_map_np, (1, 2, 0))
                                
                                edge_map_save_path = os.path.join(edge_map_dir, f"{img_basename}_edge.png")
                                Image.fromarray(edge_map_np.astype(np.uint8)).save(edge_map_save_path)
                        
                        # Sample with edge maps
                        if model_sample_supports_edge:
                            samples, _ = model.sample(
                                cond=semantic_c,
                                struct_cond=init_latent,
                                batch_size=init_image.size(0),
                                timesteps=opt.ddpm_steps,
                                time_replace=opt.ddpm_steps,
                                x_T=x_T,
                                return_intermediates=True,
                                edge_map=edge_maps
                            )
                        else:
                            samples, _ = model.sample(
                                cond=semantic_c,
                                struct_cond=init_latent,
                                batch_size=init_image.size(0),
                                timesteps=opt.ddpm_steps,
                                time_replace=opt.ddpm_steps,
                                x_T=x_T,
                                return_intermediates=True
                            )
                    else:
                        # Standard sampling without edge processing
                        samples, _ = model.sample(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            batch_size=init_image.size(0),
                            timesteps=opt.ddpm_steps,
                            time_replace=opt.ddpm_steps,
                            x_T=x_T,
                            return_intermediates=True
                        )
                    
                    # Decode samples
                    x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                    
                    # Apply color correction
                    if opt.colorfix_type == 'adain':
                        x_samples = adaptive_instance_normalization(x_samples, init_image)
                    elif opt.colorfix_type == 'wavelet':
                        x_samples = wavelet_reconstruction(x_samples, init_image)
                    
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    # Save results
                    for i in range(init_image.size(0)):
                        img_name = img_list.pop(0)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        
                        suffix = "_edge" if opt.use_edge_processing else ""
                        output_path = os.path.join(outpath, basename + suffix + '.png')
                        Image.fromarray(x_sample.astype(np.uint8)).save(output_path)

                toc = time.time()
                total_time = toc - tic
                avg_time = total_time / len(img_list_ori)
                
                print(f"\n{'=' * 80}")
                print(f"Inference completed!")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average time per image: {avg_time:.2f}s")
                print(f"Results saved to: {outpath}")
                print(f"{'=' * 80}\n")

    # Close logger
    print(f"Log saved to: {log_file}")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()

