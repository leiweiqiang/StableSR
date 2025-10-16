#!/usr/bin/env python3
"""
Create comparison grid image combining:
1. GT (Ground Truth)
2. StableSR inference result
3. Edge inference result
4. No Edge inference result
5. Dummy Edge inference result
6. Edge map from Edge inference

Each image is labeled with its name and quality metrics.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def find_corresponding_images(base_path, epoch_num, image_name):
    """
    Find all corresponding images for a given image name across different modes.
    
    Returns dict with keys: gt, stablesr, edge, no_edge, dummy_edge, edge_map
    """
    results = {}
    
    # GT image path (from GT directory)
    # This should be passed as a parameter
    
    # StableSR path (try both possible locations)
    stablesr_path = os.path.join(base_path, "stablesr", "epochs_baseline")
    if not os.path.exists(stablesr_path):
        stablesr_path = os.path.join(base_path, "stablesr", "baseline")
    
    # Edge mode path
    edge_path = os.path.join(base_path, "edge", f"epochs_{epoch_num}")
    
    # No edge mode path
    no_edge_path = os.path.join(base_path, "no_edge", f"epochs_{epoch_num}")
    
    # Dummy edge mode path
    dummy_edge_path = os.path.join(base_path, "dummy_edge", f"epochs_{epoch_num}")
    
    # Extract base name without _edge suffix for matching
    # e.g., "0801_edge.png" -> "0801"
    if image_name.endswith('_edge.png'):
        base_name = image_name.replace('_edge.png', '')
    else:
        base_name = image_name.replace('.png', '')
    
    # Find StableSR result (no _edge suffix)
    stablesr_file = os.path.join(stablesr_path, f"{base_name}.png")
    if os.path.exists(stablesr_file):
        results['stablesr'] = stablesr_file
    
    # Find Edge result
    edge_file = os.path.join(edge_path, image_name)
    if os.path.exists(edge_file):
        results['edge'] = edge_file
    
    # Find Edge map (in edge_map subdirectory)
    edge_map_file = os.path.join(edge_path, "edge_map", image_name)
    if os.path.exists(edge_map_file):
        results['edge_map'] = edge_map_file
    
    # Find No Edge result (with _edge suffix)
    no_edge_file = os.path.join(no_edge_path, image_name)
    if os.path.exists(no_edge_file):
        results['no_edge'] = no_edge_file
    
    # Find Dummy Edge result (with _edge suffix)
    dummy_edge_file = os.path.join(dummy_edge_path, image_name)
    if os.path.exists(dummy_edge_file):
        results['dummy_edge'] = dummy_edge_file
    
    return results


def load_metrics_for_image(metrics_json_path, image_name):
    """
    Load metrics for a specific image from metrics.json.
    
    Returns dict with metrics or None if not found.
    """
    if not os.path.exists(metrics_json_path):
        return None
    
    try:
        with open(metrics_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract base name without _edge suffix
        if image_name.endswith('_edge.png'):
            base_name = image_name.replace('_edge.png', '.png')
        else:
            base_name = image_name
        
        # Try to find in per_image_metrics (old format)
        if 'per_image_metrics' in data and base_name in data['per_image_metrics']:
            return data['per_image_metrics'][base_name]
        
        # Try to find in images array (new format)
        if 'images' in data:
            for img_data in data['images']:
                if img_data.get('image_name') == base_name:
                    return img_data
        
        return None
    except Exception as e:
        print(f"  ⚠ Error reading metrics from {metrics_json_path}: {e}")
        return None


def format_metrics_text(metrics):
    """
    Format metrics dict into display text.
    """
    if not metrics:
        return ""
    
    lines = []
    
    # Main quality metrics
    if 'psnr' in metrics:
        lines.append(f"PSNR: {metrics['psnr']:.2f}")
    if 'ssim' in metrics:
        lines.append(f"SSIM: {metrics['ssim']:.4f}")
    if 'lpips' in metrics:
        lines.append(f"LPIPS: {metrics['lpips']:.4f}")
    
    # Edge metrics
    if 'edge_psnr' in metrics:
        lines.append(f"Edge PSNR: {metrics['edge_psnr']:.2f}")
    if 'edge_overlap' in metrics:
        lines.append(f"Edge Overlap: {metrics['edge_overlap']:.4f}")
    
    return " | ".join(lines)


def add_label_to_image(img, title, metrics_text="", title_font_size=22, metrics_font_size=13, fixed_label_height=80):
    """
    Add a fixed-height label area above the image with title and metrics.
    Image size remains unchanged (512x512).
    Label area has fixed height for all images.
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_font_size)
        metrics_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", metrics_font_size)
    except:
        title_font = ImageFont.load_default()
        metrics_font = ImageFont.load_default()
    
    # Create temporary draw to measure text
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    
    # Measure title
    title_bbox = temp_draw.textbbox((0, 0), title, font=title_font)
    title_height = title_bbox[3] - title_bbox[1]
    
    # Fixed label area height
    label_height = fixed_label_height
    
    # Split metrics into multiple lines if needed
    if metrics_text:
        max_width = img.width - 20
        metrics_lines = []
        current_line = ""
        
        for part in metrics_text.split(" | "):
            test_line = current_line + (" | " if current_line else "") + part
            test_bbox = temp_draw.textbbox((0, 0), test_line, font=metrics_font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width > max_width and current_line:
                metrics_lines.append(current_line)
                current_line = part
            else:
                current_line = test_line
        
        if current_line:
            metrics_lines.append(current_line)
    else:
        metrics_lines = []
    
    # Create new image: fixed label area above + original image below
    new_height = label_height + img.height
    new_img = Image.new('RGB', (img.width, new_height), color=(250, 250, 250))
    
    # Paste original image at the bottom (unchanged)
    new_img.paste(img, (0, label_height))
    
    # Draw label area
    draw = ImageDraw.Draw(new_img)
    
    # Calculate vertical centering for content within fixed label area
    padding_top = 12
    title_area_height = title_height + 8
    
    # Draw title (centered horizontally, positioned at padding_top)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (img.width - title_width) // 2
    title_y = padding_top
    draw.text((title_x, title_y), title, fill=(0, 0, 0), font=title_font)
    
    # Draw metrics (centered, smaller font)
    if metrics_lines:
        line_spacing = 6
        metrics_y = title_y + title_area_height
        for line in metrics_lines:
            # Center each metrics line
            line_bbox = temp_draw.textbbox((0, 0), line, font=metrics_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = (img.width - line_width) // 2
            draw.text((line_x, metrics_y), line, fill=(70, 70, 70), font=metrics_font)
            metrics_y += metrics_font_size + line_spacing
    
    # Draw a border line between label and image
    draw.line([(0, label_height-1), (img.width, label_height-1)], fill=(200, 200, 200), width=2)
    
    return new_img


def create_comparison_grid(gt_img_path, image_paths, output_path, image_name, metrics_paths, 
                          exp_name="", epoch_num="", base_path=""):
    """
    Create a 2x3 grid with labeled images and metrics.
    Includes a header with experiment name, checkpoint info, and generation time.
    
    Layout:
    [Header: Experiment Name | Checkpoint | Generation Time]
    [GT]              [StableSR]        [Edge]
    [No Edge]         [Dummy Edge]      [Edge Map]
    """
    # Load GT image
    if not os.path.exists(gt_img_path):
        print(f"  ⚠ GT image not found: {gt_img_path}")
        return False
    
    gt_img = Image.open(gt_img_path)
    
    # Check if all required images exist
    required_keys = ['stablesr', 'edge', 'no_edge', 'dummy_edge', 'edge_map']
    missing = [k for k in required_keys if k not in image_paths]
    
    if missing:
        print(f"  ⚠ Missing images for {image_name}: {', '.join(missing)}")
        return False
    
    # Load all images
    images = {
        'gt': gt_img,
        'stablesr': Image.open(image_paths['stablesr']),
        'edge': Image.open(image_paths['edge']),
        'no_edge': Image.open(image_paths['no_edge']),
        'dummy_edge': Image.open(image_paths['dummy_edge']),
        'edge_map': Image.open(image_paths['edge_map'])
    }
    
    # Load metrics for each mode
    metrics = {}
    for mode in ['stablesr', 'edge', 'no_edge', 'dummy_edge']:
        if mode in metrics_paths:
            mode_metrics = load_metrics_for_image(metrics_paths[mode], image_name)
            metrics[mode] = format_metrics_text(mode_metrics) if mode_metrics else ""
        else:
            metrics[mode] = ""
    
    # Add labels with metrics
    labeled_images = {
        'gt': add_label_to_image(images['gt'], 'GT (Ground Truth)', ""),
        'stablesr': add_label_to_image(images['stablesr'], 'StableSR', metrics.get('stablesr', '')),
        'edge': add_label_to_image(images['edge'], 'Edge', metrics.get('edge', '')),
        'no_edge': add_label_to_image(images['no_edge'], 'No Edge', metrics.get('no_edge', '')),
        'dummy_edge': add_label_to_image(images['dummy_edge'], 'Dummy Edge', metrics.get('dummy_edge', '')),
        'edge_map': add_label_to_image(images['edge_map'], 'Edge Map', "")
    }
    
    # Get dimensions (all should be the same after adding labels)
    img_width = labeled_images['gt'].width
    img_height = labeled_images['gt'].height
    
    # Define spacing between images
    h_spacing = 10  # horizontal spacing
    v_spacing = 10  # vertical spacing
    
    # Calculate grid dimensions (without header)
    grid_content_width = img_width * 3 + h_spacing * 2
    grid_content_height = img_height * 2 + v_spacing * 1
    
    # Create header with experiment info
    header_height = 60
    try:
        header_title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        header_info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        header_title_font = ImageFont.load_default()
        header_info_font = ImageFont.load_default()
    
    # Generate header text
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract experiment name from base_path if not provided
    if not exp_name and base_path:
        exp_name = os.path.basename(base_path)
    
    # Clean up image name for display (remove _edge suffix if present)
    display_image_name = image_name.replace('_edge.png', '.png') if image_name.endswith('_edge.png') else image_name
    
    header_line1 = f"Experiment: {exp_name}" if exp_name else "Comparison Grid"
    header_line2 = f"Epoch: {epoch_num}  |  Image: {display_image_name}  |  Generated: {current_time}"
    
    # Create full grid with header
    grid_width = grid_content_width
    grid_height = header_height + grid_content_height
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(240, 240, 240))
    
    # Draw header
    draw = ImageDraw.Draw(grid_img)
    
    # Header background
    draw.rectangle([(0, 0), (grid_width, header_height)], fill=(255, 255, 255))
    draw.line([(0, header_height-1), (grid_width, header_height-1)], fill=(180, 180, 180), width=2)
    
    # Draw header text
    # Line 1: Experiment name (centered, bold)
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    line1_bbox = temp_draw.textbbox((0, 0), header_line1, font=header_title_font)
    line1_width = line1_bbox[2] - line1_bbox[0]
    line1_x = (grid_width - line1_width) // 2
    draw.text((line1_x, 10), header_line1, fill=(0, 0, 0), font=header_title_font)
    
    # Line 2: Details (centered, normal)
    line2_bbox = temp_draw.textbbox((0, 0), header_line2, font=header_info_font)
    line2_width = line2_bbox[2] - line2_bbox[0]
    line2_x = (grid_width - line2_width) // 2
    draw.text((line2_x, 35), header_line2, fill=(60, 60, 60), font=header_info_font)
    
    # Paste images in grid with spacing (offset by header height)
    y_offset = header_height
    
    # Row 1: GT, StableSR, Edge
    grid_img.paste(labeled_images['gt'], (0, y_offset))
    grid_img.paste(labeled_images['stablesr'], (img_width + h_spacing, y_offset))
    grid_img.paste(labeled_images['edge'], (img_width * 2 + h_spacing * 2, y_offset))
    
    # Row 2: No Edge, Dummy Edge, Edge Map
    grid_img.paste(labeled_images['no_edge'], (0, y_offset + img_height + v_spacing))
    grid_img.paste(labeled_images['dummy_edge'], (img_width + h_spacing, y_offset + img_height + v_spacing))
    grid_img.paste(labeled_images['edge_map'], (img_width * 2 + h_spacing * 2, y_offset + img_height + v_spacing))
    
    # Save grid
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path, quality=95)
    
    return True


def process_checkpoint_comparisons(base_path, gt_dir, epoch_num):
    """
    Process all images for a given checkpoint and create comparison grids.
    """
    # Get edge results directory
    edge_dir = os.path.join(base_path, "edge", f"epochs_{epoch_num}")
    
    if not os.path.exists(edge_dir):
        print(f"❌ Edge directory not found: {edge_dir}")
        return 0
    
    # Get list of result images (only PNG files in the main directory, not subdirectories)
    all_files = []
    try:
        all_files = [f for f in os.listdir(edge_dir) if f.endswith('.png')]
    except Exception as e:
        print(f"❌ Error reading directory {edge_dir}: {e}")
        return 0
    
    # Filter out non-image files (like config files)
    image_files = [f for f in all_files if f[0].isdigit()]  # Images typically start with digits
    
    if not image_files:
        print(f"  ℹ No images found in {edge_dir}")
        return 0
    
    # Build metrics paths dict
    metrics_paths = {}
    
    # StableSR metrics (try both possible locations)
    stablesr_metrics = os.path.join(base_path, "stablesr", "epochs_baseline", "metrics.json")
    if not os.path.exists(stablesr_metrics):
        stablesr_metrics = os.path.join(base_path, "stablesr", "baseline", "metrics.json")
    if os.path.exists(stablesr_metrics):
        metrics_paths['stablesr'] = stablesr_metrics
    
    # Edge metrics
    edge_metrics = os.path.join(base_path, "edge", f"epochs_{epoch_num}", "metrics.json")
    if os.path.exists(edge_metrics):
        metrics_paths['edge'] = edge_metrics
    
    # No edge metrics
    no_edge_metrics = os.path.join(base_path, "no_edge", f"epochs_{epoch_num}", "metrics.json")
    if os.path.exists(no_edge_metrics):
        metrics_paths['no_edge'] = no_edge_metrics
    
    # Dummy edge metrics
    dummy_edge_metrics = os.path.join(base_path, "dummy_edge", f"epochs_{epoch_num}", "metrics.json")
    if os.path.exists(dummy_edge_metrics):
        metrics_paths['dummy_edge'] = dummy_edge_metrics
    
    # Create output directory for comparison grids
    comparison_dir = os.path.join(base_path, "comparisons", f"epochs_{epoch_num}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"  📊 Creating comparison grids for epoch {epoch_num}...")
    print(f"     Found {len(image_files)} images to process")
    print(f"     Loaded metrics from {len(metrics_paths)} modes")
    
    success_count = 0
    
    for image_name in sorted(image_files):
        # Find corresponding images
        image_paths = find_corresponding_images(base_path, epoch_num, image_name)
        
        # GT image path (remove _edge suffix if present)
        if image_name.endswith('_edge.png'):
            gt_filename = image_name.replace('_edge.png', '.png')
        else:
            gt_filename = image_name
        gt_img_path = os.path.join(gt_dir, gt_filename)
        
        # Output path (base name without _edge suffix)
        if image_name.endswith('_edge.png'):
            base_output_name = image_name.replace('_edge.png', '_comparison.png')
        else:
            base_output_name = image_name.replace('.png', '_comparison.png')
        output_path = os.path.join(comparison_dir, base_output_name)
        
        # Skip if already exists
        if os.path.exists(output_path):
            success_count += 1
            continue
        
        # Create comparison grid (with experiment info)
        exp_name = os.path.basename(base_path) if base_path else ""
        if create_comparison_grid(gt_img_path, image_paths, output_path, image_name, metrics_paths,
                                 exp_name=exp_name, epoch_num=epoch_num, base_path=base_path):
            success_count += 1
    
    print(f"  ✓ Created {success_count}/{len(image_files)} comparison grids")
    print(f"     Output: {comparison_dir}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Create comparison grids for inference results')
    parser.add_argument('base_path', type=str, help='Base path containing inference results')
    parser.add_argument('gt_dir', type=str, help='Directory containing ground truth images')
    parser.add_argument('epoch_num', type=str, help='Epoch number to process')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.base_path):
        print(f"❌ Base path not found: {args.base_path}")
        return 1
    
    if not os.path.exists(args.gt_dir):
        print(f"❌ GT directory not found: {args.gt_dir}")
        return 1
    
    # Process checkpoint
    success_count = process_checkpoint_comparisons(args.base_path, args.gt_dir, args.epoch_num)
    
    if success_count > 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

