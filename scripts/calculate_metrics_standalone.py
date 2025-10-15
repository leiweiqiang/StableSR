#!/usr/bin/env python3
"""
Standalone metrics calculation script for StableSR Edge models
Calculates PSNR, SSIM, and LPIPS between generated images and ground truth
"""

import os
import sys
import argparse
from pathlib import Path
import json
import csv
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


def calculate_metrics(output_dir, gt_img_dir, crop_border=0):
    """
    Calculate PSNR, SSIM, and LPIPS between generated images and ground truth
    
    Args:
        output_dir: Directory containing generated images
        gt_img_dir: Directory containing ground truth images
        crop_border: Number of pixels to crop from border (default: 0)
        
    Returns:
        Dictionary with metrics for each image and average metrics
    """
    # Expand paths
    output_dir = os.path.expanduser(output_dir)
    gt_img_dir = os.path.expanduser(gt_img_dir)
    
    output_path = Path(output_dir)
    gt_path = Path(gt_img_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return None
        
    if not gt_path.exists():
        print(f"❌ GT directory does not exist: {gt_img_dir}")
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
    
    if len(generated_images) == 0:
        print(f"⚠ No PNG images found in: {output_dir}")
        return None
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    print(f"Processing {len(generated_images)} images...")
    
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
                print(f"⚠ GT image not found for {img_base_name}")
                continue
        
        # Read images
        gen_img = cv2.imread(str(gen_img_path))
        gt_img = cv2.imread(str(gt_img_path))
        
        if gen_img is None or gt_img is None:
            print(f"⚠ Could not read images for {img_base_name}")
            continue
        
        # Ensure same size
        if gen_img.shape != gt_img.shape:
            print(f"⚠ Size mismatch for {img_base_name}: {gen_img.shape} vs {gt_img.shape}")
            # Resize to match
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))
        
        # Calculate PSNR using basicsr
        # Images are already in [0, 255] range from cv2.imread
        img_psnr = calculate_psnr(gen_img, gt_img, crop_border=crop_border, 
                                   input_order='HWC', test_y_channel=False)
        
        # Calculate SSIM using basicsr
        img_ssim = calculate_ssim(gen_img, gt_img, crop_border=crop_border, 
                                   input_order='HWC', test_y_channel=False)
        
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
                print(f"⚠ LPIPS calculation failed for {img_base_name}: {e}")
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
        print(f"✓ Metrics saved to: {json_path}")
        
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
                writer.writerow(['Average', f"{metrics['average_psnr']:.4f}", 
                               f"{metrics['average_ssim']:.4f}", avg_lpips])
            else:
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM'])
                for img_metrics in metrics['images']:
                    writer.writerow([
                        img_metrics['image_name'],
                        f"{img_metrics['psnr']:.4f}",
                        f"{img_metrics['ssim']:.4f}"
                    ])
                writer.writerow(['Average', f"{metrics['average_psnr']:.4f}", 
                               f"{metrics['average_ssim']:.4f}"])
        print(f"✓ Metrics saved to: {csv_path}")
        
        return metrics
    else:
        print("❌ No valid image pairs found for evaluation")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Calculate PSNR, SSIM, and LPIPS metrics for super-resolution results'
    )
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing generated SR images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth HR images')
    parser.add_argument('--crop_border', type=int, default=0,
                       help='Number of pixels to crop from border (default: 0)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Standalone Metrics Calculation")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Crop border: {args.crop_border}")
    if lpips_available:
        print(f"LPIPS: Available (using taming LPIPS on {device})")
    else:
        print(f"LPIPS: Not available")
    print("="*60)
    print()
    
    # Calculate metrics
    metrics = calculate_metrics(args.output_dir, args.gt_dir, args.crop_border)
    
    if metrics is None:
        print("\n❌ Metrics calculation failed")
        sys.exit(1)
    
    print("\n✓ Metrics calculation completed successfully")
    sys.exit(0)


if __name__ == '__main__':
    main()

