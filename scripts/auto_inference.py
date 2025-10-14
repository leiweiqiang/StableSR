#!/usr/bin/env python3
"""
Automatic inference script for StableSR Edge models
Finds all checkpoint files (excluding last.ckpt) and generates validation results
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
import re
import json
import csv
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import warnings
import builtins

# Add project root to Python path for basicsr imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import basicsr metrics
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# Suppress warnings for LPIPS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips.lpips")

# Load LPIPS model (using taming implementation for consistency with training)
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None  # Temporarily disable print
try:
    from taming.modules.losses.lpips import LPIPS
    lpips_model = LPIPS()  # Uses VGG16 with learned weights, same as training
    # Move to GPU if available and set to eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips_model.to(device).eval()
    lpips_available = True
except (ImportError, Exception) as e:
    lpips_available = False
    lpips_model = None
    device = torch.device('cpu')
builtins.print = _original_print  # Restore print

# LPIPS transform (taming LPIPS expects [0,1] RGB, has internal ScalingLayer)
lpips_transform = transforms.Compose([
    transforms.ToTensor()  # Convert PIL to tensor [0,1]
])


def find_checkpoints(logs_dir, include_last=False):
    """
    Find all checkpoint files in logs directory excluding last.ckpt
    
    Args:
        logs_dir: Path to logs directory
        include_last: If True, also include last.ckpt files
    
    Returns:
        list of tuples: (experiment_name, checkpoint_path, epoch_number)
    """
    checkpoints = []
    logs_path = Path(logs_dir)
    
    # Find all experiment directories in logs
    for exp_dir in sorted(logs_path.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        ckpt_dir = exp_dir / "checkpoints"
        
        if not ckpt_dir.exists():
            print(f"No checkpoints directory found for {exp_name}")
            continue
        
        # Find all .ckpt files
        for ckpt_file in sorted(ckpt_dir.glob("*.ckpt")):
            # Skip last.ckpt unless explicitly included
            if ckpt_file.name == "last.ckpt":
                if include_last:
                    checkpoints.append((exp_name, str(ckpt_file), "last"))
                continue
            
            # Extract epoch number from filename (e.g., epoch=000285.ckpt)
            match = re.search(r'epoch=(\d+)', ckpt_file.name)
            if match:
                epoch_num = match.group(1)
                checkpoints.append((exp_name, str(ckpt_file), epoch_num))
            else:
                print(f"Warning: Could not parse epoch number from {ckpt_file.name}")
    
    return checkpoints


def create_output_dir(base_output_dir, exp_name, epoch_num, sub_folder=None, create=True):
    """
    Create output directory structure: base_output_dir/exp_name/[sub_folder/]epochs_N
    
    Args:
        base_output_dir: Base directory for outputs
        exp_name: Experiment name
        epoch_num: Epoch number or "last"
        sub_folder: Optional subfolder name under exp_name (e.g., "step4", "step200")
        create: If True, create the directory. If False, just return the path.
    
    Returns:
        Path to output directory (with ~ expanded and resolved to absolute path)
    """
    # Expand ~ in base directory
    base_output_dir = os.path.expanduser(base_output_dir)
    
    # Resolve to absolute path (follows symlinks)
    base_output_dir = os.path.abspath(base_output_dir)
    
    # Handle "last" checkpoint specially
    if epoch_num == "last":
        epoch_dir = "epochs_last"
    else:
        # Try to convert to int, if fails use as string (e.g., "baseline")
        try:
            epoch_dir = f"epochs_{int(epoch_num):d}"
        except (ValueError, TypeError):
            epoch_dir = f"epochs_{epoch_num}"
    
    # Build path with optional subfolder
    if sub_folder:
        output_path = os.path.join(base_output_dir, exp_name, sub_folder, epoch_dir)
    else:
        output_path = os.path.join(base_output_dir, exp_name, epoch_dir)
    
    if create:
        # Use os.makedirs with exist_ok to handle existing directories and symlinks
        os.makedirs(output_path, exist_ok=True)
    
    return output_path


def calculate_metrics(output_dir, gt_img_dir, results_dict):
    """
    Calculate PSNR and SSIM between generated images and ground truth
    
    Args:
        output_dir: Directory containing generated images
        gt_img_dir: Directory containing ground truth images
        results_dict: Dictionary to store results
        
    Returns:
        Dictionary with metrics for each image and average metrics
    """
    # Expand paths
    output_dir = os.path.expanduser(output_dir)
    gt_img_dir = os.path.expanduser(gt_img_dir)
    
    output_path = Path(output_dir)
    gt_path = Path(gt_img_dir)
    
    if not output_path.exists():
        print(f"Output directory does not exist: {output_dir}")
        return None
        
    if not gt_path.exists():
        print(f"GT directory does not exist: {gt_img_dir}")
        return None
    
    metrics = {
        'images': [],
        'average_psnr': 0.0,
        'average_ssim': 0.0,
        'average_lpips': 0.0,
        'total_images': 0
    }
    
    # Find all generated images (exclude edge suffix for comparison)
    generated_images = sorted(output_path.glob('*.png'))
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    for gen_img_path in generated_images:
        # Get base name without edge suffix, but keep the extension
        img_base_name = gen_img_path.stem.replace('_edge', '')
        img_extension = gen_img_path.suffix  # e.g., '.png'
        img_name = img_base_name + img_extension  # e.g., '0801.png'
        
        # Find corresponding GT image
        gt_img_path = gt_path / f"{img_base_name}.png"
        if not gt_img_path.exists():
            # Try with jpg extension
            gt_img_path = gt_path / f"{img_base_name}.jpg"
            if not gt_img_path.exists():
                print(f"Warning: GT image not found for {img_base_name}")
                continue
        
        # Read images
        gen_img = cv2.imread(str(gen_img_path))
        gt_img = cv2.imread(str(gt_img_path))
        
        if gen_img is None or gt_img is None:
            print(f"Warning: Could not read images for {img_base_name}")
            continue
        
        # Ensure same size
        if gen_img.shape != gt_img.shape:
            print(f"Warning: Size mismatch for {img_base_name}: {gen_img.shape} vs {gt_img.shape}")
            # Resize to match
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))
        
        # Calculate PSNR using basicsr
        # Images are already in [0, 255] range from cv2.imread
        img_psnr = calculate_psnr(gen_img, gt_img, crop_border=0, input_order='HWC', test_y_channel=False)
        
        # Calculate SSIM using basicsr
        img_ssim = calculate_ssim(gen_img, gt_img, crop_border=0, input_order='HWC', test_y_channel=False)
        
        # Calculate LPIPS if available
        img_lpips = 0.0
        if lpips_available and lpips_model is not None:
            try:
                # Convert BGR to RGB for LPIPS
                gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
                gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL and apply transform
                gen_pil = Image.fromarray(gen_rgb)
                gt_pil = Image.fromarray(gt_rgb)
                
                gen_tensor = lpips_transform(gen_pil).unsqueeze(0).to(device)
                gt_tensor = lpips_transform(gt_pil).unsqueeze(0).to(device)
                
                # Compute LPIPS
                with torch.no_grad():
                    lpips_value = lpips_model(gen_tensor, gt_tensor)
                    # taming LPIPS returns [B, 1, 1, 1], extract scalar
                    img_lpips = lpips_value.squeeze().item()
            except Exception as e:
                print(f"Warning: LPIPS calculation failed for {img_base_name}: {e}")
                img_lpips = -1.0  # Mark as failed
        
        metrics['images'].append({
            'image_name': img_name,
            'psnr': float(img_psnr),
            'ssim': float(img_ssim),
            'lpips': float(img_lpips) if lpips_available else None
        })
        
        total_psnr += img_psnr
        total_ssim += img_ssim
        if lpips_available and img_lpips >= 0:
            total_lpips += img_lpips
        count += 1
    
    if count > 0:
        metrics['average_psnr'] = total_psnr / count
        metrics['average_ssim'] = total_ssim / count
        if lpips_available:
            metrics['average_lpips'] = total_lpips / count
        else:
            metrics['average_lpips'] = None
        metrics['total_images'] = count
        
        print(f"\n{'='*60}")
        print(f"Metrics Summary:")
        print(f"  Total images: {count}")
        print(f"  Average PSNR: {metrics['average_psnr']:.4f} dB")
        print(f"  Average SSIM: {metrics['average_ssim']:.4f}")
        if lpips_available:
            print(f"  Average LPIPS: {metrics['average_lpips']:.4f}")
        print(f"{'='*60}\n")
        
        # Save to JSON
        json_path = output_path / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {json_path}")
        
        # Save to CSV
        csv_path = output_path / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if lpips_available:
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM', 'LPIPS'])
                for img_metrics in metrics['images']:
                    writer.writerow([
                        img_metrics['image_name'],
                        f"{img_metrics['psnr']:.4f}",
                        f"{img_metrics['ssim']:.4f}",
                        f"{img_metrics['lpips']:.4f}" if img_metrics['lpips'] is not None else 'N/A'
                    ])
                avg_lpips = f"{metrics['average_lpips']:.4f}" if metrics['average_lpips'] is not None else 'N/A'
                writer.writerow(['Average', f"{metrics['average_psnr']:.4f}", f"{metrics['average_ssim']:.4f}", avg_lpips])
            else:
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM'])
                for img_metrics in metrics['images']:
                    writer.writerow([
                        img_metrics['image_name'],
                        f"{img_metrics['psnr']:.4f}",
                        f"{img_metrics['ssim']:.4f}"
                    ])
                writer.writerow(['Average', f"{metrics['average_psnr']:.4f}", f"{metrics['average_ssim']:.4f}"])
        print(f"Metrics saved to: {csv_path}")
        
        return metrics
    else:
        print("No valid image pairs found for evaluation")
        return None


def run_inference(checkpoint_path, output_dir, config_file, init_img_dir, 
                  gt_img_dir, vqgan_ckpt, ddpm_steps=200, dec_w=0.5, 
                  seed=42, n_samples=1, colorfix_type="wavelet", 
                  input_size=512, use_edge_processing=True, use_white_edge=False, 
                  use_dummy_edge=False, dummy_edge_path=None,
                  calculate_metrics_flag=True, dry_run=False):
    """
    Run inference using the appropriate script based on edge processing setting
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for results
        config_file: Path to config yaml file
        init_img_dir: Directory with low-resolution input images
        gt_img_dir: Directory with ground truth high-resolution images
        vqgan_ckpt: Path to VQGAN checkpoint
        ddpm_steps: Number of DDPM steps
        dec_w: Decoder weight
        seed: Random seed
        n_samples: Number of samples to generate
        colorfix_type: Type of color correction
        use_edge_processing: Whether to use edge processing (selects edge vs non-edge script)
        use_white_edge: Whether to use black (all negative ones) edge maps for no-edge mode
        use_dummy_edge: Whether to use a fixed dummy edge map
        dummy_edge_path: Path to dummy edge image file
        dry_run: If True, only print the command without executing
    """
    # Select the appropriate script based on edge processing
    if use_edge_processing:
        script_name = "scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py"
        print(f"Using EDGE-enabled script: {script_name}")
    else:
        script_name = "scripts/sr_val_ddpm_text_T_vqganfin_old.py"
        print(f"Using STANDARD (non-edge) script: {script_name}")
    
    # Construct command
    cmd = [
        "python", script_name,
        "--config", config_file,
        "--ckpt", checkpoint_path,
        "--init-img", init_img_dir,
        "--outdir", output_dir,
        "--ddpm_steps", str(ddpm_steps),
        "--dec_w", str(dec_w),
        "--seed", str(seed),
        "--n_samples", str(n_samples),
        "--vqgan_ckpt", vqgan_ckpt,
        "--colorfix_type", colorfix_type,
        "--input_size", str(input_size),
    ]
    
    # Add --gt-img only if using edge processing (edge script uses it)
    # Non-edge script doesn't have --gt-img parameter
    if use_edge_processing:
        cmd.extend(["--gt-img", gt_img_dir])
    
    # Add edge-specific flags only when using edge processing
    if use_edge_processing:
        cmd.append("--use_edge_processing")
        
        if use_white_edge:
            cmd.append("--use_white_edge")
        
        if use_dummy_edge:
            cmd.append("--use_dummy_edge")
            if dummy_edge_path:
                cmd.extend(["--dummy_edge_path", dummy_edge_path])
    
    # Print command
    print("\n" + "="*80)
    print("Running inference:")
    cmd_str = " ".join(cmd)
    print(cmd_str)
    print("="*80 + "\n")
    
    # Save command to file in output directory
    os.makedirs(output_dir, exist_ok=True)
    cmd_file = os.path.join(output_dir, "inference_command.txt")
    try:
        with open(cmd_file, 'w') as f:
            f.write("# Inference Command\n")
            f.write("# Generated on: " + __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            f.write(cmd_str + "\n\n")
            f.write("# Command with parameters:\n")
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"config: {config_file}\n")
            f.write(f"init_img: {init_img_dir}\n")
            f.write(f"gt_img: {gt_img_dir}\n")
            f.write(f"output: {output_dir}\n")
            f.write(f"ddpm_steps: {ddpm_steps}\n")
            f.write(f"dec_w: {dec_w}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"n_samples: {n_samples}\n")
            f.write(f"vqgan_ckpt: {vqgan_ckpt}\n")
            f.write(f"colorfix_type: {colorfix_type}\n")
            f.write(f"use_edge_processing: {use_edge_processing}\n")
            f.write(f"use_white_edge: {use_white_edge}\n")
            f.write(f"use_dummy_edge: {use_dummy_edge}\n")
            f.write(f"dummy_edge_path: {dummy_edge_path}\n")
        print(f"✓ Command saved to: {cmd_file}")
    except Exception as e:
        print(f"⚠ Warning: Could not save command file: {e}")
    
    # Copy config yaml file to output directory
    try:
        import shutil
        config_basename = os.path.basename(config_file)
        config_dest = os.path.join(output_dir, f"config_{config_basename}")
        shutil.copy2(config_file, config_dest)
        print(f"✓ Config file saved to: {config_dest}")
    except Exception as e:
        print(f"⚠ Warning: Could not copy config file: {e}")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return True
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # After successful inference, calculate metrics if enabled
        if calculate_metrics_flag:
            print(f"\n{'='*80}")
            print("Calculating metrics...")
            print(f"{'='*80}")
            metrics = calculate_metrics(output_dir, gt_img_dir, {})
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Automatic inference for all checkpoints")
    
    # Directory paths
    parser.add_argument("--logs_dir", type=str, 
                       default="logs",
                       help="Directory containing experiment logs")
    parser.add_argument("--output_base", type=str, 
                       default=os.path.expanduser("~/validation_results"),
                       help="Base directory for validation results")
    parser.add_argument("--sub_folder", type=str, default=None,
                       help="Optional subfolder name under each experiment (e.g., 'step4', 'step200')")
    
    # Data paths
    parser.add_argument("--init_img", type=str,
                       default=os.path.expanduser("~/nas/test_dataset/128x128_valid_LR"),
                       help="Directory with low-resolution input images")
    parser.add_argument("--gt_img", type=str,
                       default=os.path.expanduser("~/nas/test_dataset/512x512_White_GT"),
                       help="Directory with ground truth images (use 512x512_White_GT)")
    
    # Model configuration
    parser.add_argument("--config", type=str,
                       default="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
                       help="Path to model config file. Use v2-finetune_text_T_512_edge.yaml for edge models, "
                            "v2-finetune_text_T_512.yaml for standard models")
    parser.add_argument("--ckpt", type=str, default=None,
                       help="Specific checkpoint to process (if set, auto-discovery is disabled)")
    parser.add_argument("--vqgan_ckpt", type=str,
                       default=os.path.expanduser("~/checkpoints/vqgan_cfw_00011.ckpt"),
                       help="Path to VQGAN checkpoint")
    
    # Inference parameters
    parser.add_argument("--ddpm_steps", type=int, default=200,
                       help="Number of DDPM steps")
    parser.add_argument("--dec_w", type=float, default=0.5,
                       help="Decoder weight")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--n_samples", type=int, default=1,
                       help="Number of samples per image")
    parser.add_argument("--colorfix_type", type=str, default="wavelet",
                       help="Color correction type")
    parser.add_argument("--input_size", type=int, default=512,
                       help="Input size for LR images (should be 512 to match training with resize_lq=True)")
    parser.add_argument("--use_edge_processing", action="store_true", default=False,
                       help="Use edge processing (requires edge-trained model). "
                            "Use with --config ending in _edge.yaml")
    parser.add_argument("--no_edge_processing", dest="use_edge_processing", 
                       action="store_false",
                       help="Disable edge processing (for standard models)")
    parser.add_argument("--use_white_edge", action="store_true", default=False,
                       help="Use black (all negative ones) edge maps instead of generated edge maps (no edge mode)")
    parser.add_argument("--use_dummy_edge", action="store_true", default=False,
                       help="Use a fixed dummy edge map for all images")
    parser.add_argument("--dummy_edge_path", type=str, default="/stablesr_dataset/default_edge.png",
                       help="Path to dummy edge image file")
    
    # Script options
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing them")
    parser.add_argument("--exp_filter", type=str, default=None,
                       help="Only process experiments matching this substring")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip inference if output directory already exists (default: True)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                       help="Force overwrite existing results")
    parser.add_argument("--include_last", action="store_true", default=True,
                       help="Also process last.ckpt files (default: True)")
    parser.add_argument("--exclude_last", dest="include_last", action="store_false",
                       help="Exclude last.ckpt files")
    parser.add_argument("--calculate_metrics", action="store_true", default=True,
                       help="Calculate PSNR/SSIM/LPIPS metrics (default: True)")
    parser.add_argument("--no_calculate_metrics", dest="calculate_metrics", action="store_false",
                       help="Skip all metric calculations")
    parser.add_argument("--epoch_override", type=str, default=None,
                       help="Override epoch number for output directory naming (useful for comparison models)")
    parser.add_argument("--exp_name_override", type=str, default=None,
                       help="Override experiment name for output directory naming (useful for comparison models)")
    
    args = parser.parse_args()
    
    # Check if specific checkpoint is provided
    if args.ckpt:
        # Single checkpoint mode
        print(f"Single checkpoint mode enabled")
        print(f"Checkpoint: {args.ckpt}")
        
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"Error: Checkpoint file not found: {args.ckpt}")
            return
        
        # Extract experiment name and epoch from checkpoint path
        # Example: logs/exp_name/checkpoints/epoch=000047.ckpt
        if args.epoch_override:
            # Use override epoch number (for comparison models)
            epoch_num = args.epoch_override
        elif "epoch=" in ckpt_path.name:
            epoch_num = re.search(r'epoch=(\d+)', ckpt_path.name).group(1)
        elif ckpt_path.name == "last.ckpt":
            epoch_num = "last"
        else:
            epoch_num = "unknown"
        
        # Try to extract experiment name from path
        if args.exp_name_override:
            # Use override experiment name (for comparison models)
            exp_name = args.exp_name_override
        elif "checkpoints" in str(ckpt_path):
            exp_name = ckpt_path.parent.parent.name
        else:
            exp_name = "custom_model"
        
        checkpoints = [(exp_name, str(ckpt_path), epoch_num)]
        print(f"Experiment: {exp_name}")
        print(f"Epoch: {epoch_num}")
    else:
        # Auto-discovery mode (original behavior)
        print(f"Searching for checkpoints in: {args.logs_dir}")
        if not args.include_last:
            print("Note: Excluding last.ckpt files")
        else:
            print("Note: Including last.ckpt files (default)")
        checkpoints = find_checkpoints(args.logs_dir, include_last=args.include_last)
        
        if not checkpoints:
            print("No checkpoints found!")
            return
        
        print(f"\nFound {len(checkpoints)} checkpoint(s):")
        for exp_name, ckpt_path, epoch_num in checkpoints:
            print(f"  - {exp_name}: epoch={epoch_num}")
        
        # Filter experiments if requested
        if args.exp_filter:
            checkpoints = [(e, c, ep) for e, c, ep in checkpoints 
                          if args.exp_filter in e]
            print(f"\nFiltered to {len(checkpoints)} checkpoint(s) matching '{args.exp_filter}'")
    
    # Process each checkpoint
    total = len(checkpoints)
    successful = 0
    skipped = 0
    
    for idx, (exp_name, ckpt_path, epoch_num) in enumerate(checkpoints, 1):
        print(f"\n{'='*80}")
        print(f"Processing checkpoint {idx}/{total}")
        print(f"Experiment: {exp_name}")
        print(f"Epoch: {epoch_num}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*80}")
        
        # Get output directory path (don't create yet)
        output_dir = create_output_dir(args.output_base, exp_name, epoch_num, 
                                       sub_folder=args.sub_folder, create=False)
        print(f"Output directory: {output_dir}")
        
        # Check if output already exists
        skip_inference = False
        if args.skip_existing:
            output_path = Path(output_dir)
            if output_path.exists():
                # Check if directory has any files (not just subdirectories)
                has_files = any(f.is_file() for f in output_path.rglob('*.png'))
                if has_files:
                    print(f"✓ Output directory already exists with files.")
                    # Check if metrics files exist
                    metrics_json = output_path / 'metrics.json'
                    metrics_csv = output_path / 'metrics.csv'
                    if metrics_json.exists() and metrics_csv.exists():
                        print(f"✓ Metrics files already exist. Skipping completely...")
                        skipped += 1
                        continue
                    elif args.calculate_metrics:
                        print(f"→ Metrics files not found. Will calculate metrics only...")
                        skip_inference = True
                    else:
                        print(f"✓ Skipping (metrics calculation disabled)...")
                        skipped += 1
                        continue
                else:
                    print(f"⚠ Output directory exists but is empty. Will process...")
            else:
                print(f"→ Output directory doesn't exist. Will process...")
        
        # Now create the output directory
        output_dir = create_output_dir(args.output_base, exp_name, epoch_num, 
                                       sub_folder=args.sub_folder, create=True)
        
        # Run inference or just calculate metrics
        if skip_inference:
            # Only calculate metrics for existing images
            print(f"\n{'='*80}")
            print("Calculating metrics for existing results...")
            print(f"{'='*80}")
            metrics = calculate_metrics(output_dir, args.gt_img, {})
            if metrics is not None:
                successful += 1
        else:
            # Run full inference
            success = run_inference(
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                config_file=args.config,
                init_img_dir=args.init_img,
                gt_img_dir=args.gt_img,
                vqgan_ckpt=args.vqgan_ckpt,
                ddpm_steps=args.ddpm_steps,
                dec_w=args.dec_w,
                seed=args.seed,
                n_samples=args.n_samples,
                colorfix_type=args.colorfix_type,
                input_size=args.input_size,
                use_edge_processing=args.use_edge_processing,
                use_white_edge=args.use_white_edge,
                use_dummy_edge=args.use_dummy_edge,
                dummy_edge_path=args.dummy_edge_path,
                calculate_metrics_flag=args.calculate_metrics,
                dry_run=args.dry_run
            )
            
            if success:
                successful += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total checkpoints: {total}")
    print(f"Successfully processed: {successful}")
    if skipped > 0:
        print(f"Skipped (existing): {skipped}")
    if total - successful - skipped > 0:
        print(f"Failed: {total - successful - skipped}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
