#!/usr/bin/env python3
"""
Standalone script to calculate PSNR and SSIM metrics for existing results
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
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import basicsr metrics
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# Suppress warnings for LPIPS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips.lpips")

# Load LPIPS model
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None
try:
    import lpips
    lpips_model = lpips.LPIPS(net='alex')
    lpips_available = True
except ImportError:
    lpips_available = False
    lpips_model = None
builtins.print = _original_print

# LPIPS transform
lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def calculate_metrics(output_dir, gt_img_dir, crop_border=0, test_y_channel=False):
    """
    Calculate PSNR and SSIM between generated images and ground truth
    
    Args:
        output_dir: Directory containing generated images
        gt_img_dir: Directory containing ground truth images
        crop_border: Cropped pixels in each edge (default: 0)
        test_y_channel: Test on Y channel of YCbCr (default: False)
        
    Returns:
        Dictionary with metrics for each image and average metrics
    """
    output_path = Path(output_dir)
    gt_path = Path(gt_img_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return None
        
    if not gt_path.exists():
        print(f"Error: GT directory does not exist: {gt_img_dir}")
        return None
    
    metrics = {
        'images': [],
        'average_psnr': 0.0,
        'average_ssim': 0.0,
        'average_lpips': 0.0,
        'total_images': 0
    }
    
    # Find all generated images
    generated_images = sorted(output_path.glob('*.png'))
    
    if not generated_images:
        print(f"Warning: No PNG images found in {output_dir}")
        return None
    
    print(f"Found {len(generated_images)} images in output directory")
    if lpips_available:
        print("LPIPS calculation enabled")
    else:
        print("LPIPS not available (install with: pip install lpips)")
    
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
            # Try jpg extension
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
            print(f"Info: Resizing GT image for {img_base_name}: {gt_img.shape} -> {gen_img.shape}")
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))
        
        # Calculate PSNR using basicsr
        # Images are already in [0, 255] range from cv2.imread
        img_psnr = calculate_psnr(gen_img, gt_img, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        
        # Calculate SSIM using basicsr
        img_ssim = calculate_ssim(gen_img, gt_img, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        
        # Calculate LPIPS if available
        img_lpips = 0.0
        if lpips_available and lpips_model is not None:
            try:
                gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
                gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                
                gen_pil = Image.fromarray(gen_rgb)
                gt_pil = Image.fromarray(gt_rgb)
                
                gen_tensor = lpips_transform(gen_pil).unsqueeze(0)
                gt_tensor = lpips_transform(gt_pil).unsqueeze(0)
                
                with torch.no_grad():
                    img_lpips = lpips_model(gen_tensor, gt_tensor).item()
            except Exception as e:
                print(f"Warning: LPIPS calculation failed for {img_base_name}: {e}")
                img_lpips = -1.0
        
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
        
        if lpips_available:
            print(f"  {img_name}: PSNR={img_psnr:.4f} dB, SSIM={img_ssim:.4f}, LPIPS={img_lpips:.4f}")
        else:
            print(f"  {img_name}: PSNR={img_psnr:.4f} dB, SSIM={img_ssim:.4f}")
    
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
        print(f"✓ Metrics saved to: {csv_path}")
        
        return metrics
    else:
        print("Error: No valid image pairs found for evaluation")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR and SSIM metrics for super-resolution results"
    )
    
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--gt_dir", type=str,
                       default=os.path.expanduser("~/nas/test_dataset/512x512_White_GT"),
                       help="Directory containing ground truth images")
    parser.add_argument("--recursive", action="store_true",
                       help="Process all subdirectories recursively")
    parser.add_argument("--crop_border", type=int, default=0,
                       help="Cropped pixels in each edge (default: 0, common: 4)")
    parser.add_argument("--test_y_channel", action="store_true",
                       help="Test on Y channel of YCbCr (default: False)")
    
    args = parser.parse_args()
    
    # Expand paths
    output_dir = os.path.expanduser(args.output_dir)
    gt_dir = os.path.expanduser(args.gt_dir)
    
    if args.recursive:
        # Process all subdirectories
        output_path = Path(output_dir)
        subdirs = [d for d in output_path.rglob('*') if d.is_dir() and any(d.glob('*.png'))]
        
        if not subdirs:
            print(f"No subdirectories with images found in {output_dir}")
            return
        
        print(f"Found {len(subdirs)} directories to process\n")
        
        for subdir in subdirs:
            print(f"\n{'='*80}")
            print(f"Processing: {subdir.relative_to(output_path)}")
            print(f"{'='*80}")
            calculate_metrics(str(subdir), gt_dir, args.crop_border, args.test_y_channel)
    else:
        # Process single directory
        print(f"Processing: {output_dir}")
        print(f"GT directory: {gt_dir}")
        print(f"Crop border: {args.crop_border}")
        print(f"Test Y channel: {args.test_y_channel}\n")
        calculate_metrics(output_dir, gt_dir, args.crop_border, args.test_y_channel)


if __name__ == "__main__":
    main()

