"""
Extract Canny Edge Maps from Images

This utility script extracts Canny edges from images for testing edge-to-image generation.
Useful for creating test datasets or preprocessing images.

Usage:
    python scripts/extract_canny_edges.py \
        --input path/to/images/ \
        --output path/to/edges/ \
        --low_threshold 50 \
        --high_threshold 150
"""

import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_canny_edges(image_path, low_threshold=50, high_threshold=150, 
                       blur_sigma=1.4, morph_close=True):
    """
    Extract Canny edges from an image
    
    Args:
        image_path: Path to input image
        low_threshold: Lower threshold for Canny (default 50)
        high_threshold: Upper threshold for Canny (default 150)
        blur_sigma: Gaussian blur sigma before Canny (default 1.4)
        morph_close: Apply morphological closing to connect edges (default True)
        
    Returns:
        edge_map: Binary edge map as numpy array (0-255)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    if blur_sigma > 0:
        blur_size = int(2 * np.ceil(3 * blur_sigma) + 1)  # Odd kernel size
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_sigma)
    else:
        blurred = gray
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Apply morphological closing to connect nearby edges
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges


def extract_canny_adaptive(image_path, blur_sigma=1.4, morph_close=True, 
                          threshold_ratio=(0.7, 1.3)):
    """
    Extract Canny edges with adaptive thresholds based on image statistics
    
    This matches the training-time edge generation more closely.
    
    Args:
        image_path: Path to input image
        blur_sigma: Gaussian blur sigma (default 1.4, matches training)
        morph_close: Apply morphological closing (default True)
        threshold_ratio: (lower_ratio, upper_ratio) for adaptive thresholds
        
    Returns:
        edge_map: Binary edge map as numpy array (0-255)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur_size = int(2 * np.ceil(3 * blur_sigma) + 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_sigma)
    
    # Calculate adaptive thresholds based on median
    median_val = np.median(blurred)
    lower_threshold = max(0, int(threshold_ratio[0] * median_val))
    upper_threshold = min(255, int(threshold_ratio[1] * median_val))
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    # Apply morphological closing
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges


def main():
    parser = argparse.ArgumentParser(description="Extract Canny edges from images")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory or image file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for edge maps"
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=50,
        help="Lower threshold for Canny (ignored if --adaptive is used)"
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=150,
        help="Upper threshold for Canny (ignored if --adaptive is used)"
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.4,
        help="Gaussian blur sigma before edge detection"
    )
    parser.add_argument(
        "--no_morph_close",
        action="store_true",
        help="Disable morphological closing"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive thresholds based on image median (matches training)"
    )
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="Save as 3-channel RGB instead of grayscale"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert edge map (white background, black edges)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find input images
    if os.path.isfile(args.input):
        image_files = [args.input]
    else:
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(args.input, ext)))
        image_files.sort()
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.input}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {args.output}")
    print(f"Method: {'Adaptive' if args.adaptive else 'Fixed'} thresholds")
    if not args.adaptive:
        print(f"Thresholds: [{args.low_threshold}, {args.high_threshold}]")
    print(f"Blur sigma: {args.blur_sigma}")
    print(f"Morphological closing: {not args.no_morph_close}")
    print(f"Output format: {'RGB' if args.save_rgb else 'Grayscale'}")
    print("")
    
    # Process images
    success_count = 0
    for img_path in tqdm(image_files, desc="Extracting edges"):
        try:
            # Extract edges
            if args.adaptive:
                edges = extract_canny_adaptive(
                    img_path,
                    blur_sigma=args.blur_sigma,
                    morph_close=not args.no_morph_close
                )
            else:
                edges = extract_canny_edges(
                    img_path,
                    low_threshold=args.low_threshold,
                    high_threshold=args.high_threshold,
                    blur_sigma=args.blur_sigma,
                    morph_close=not args.no_morph_close
                )
            
            # Invert if requested
            if args.invert:
                edges = 255 - edges
            
            # Convert to RGB if requested
            if args.save_rgb:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Save edge map
            basename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(args.output, f"{basename}_edge.png")
            cv2.imwrite(output_path, edges)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print(f"\nSuccessfully processed {success_count}/{len(image_files)} images")
    print(f"Edge maps saved to: {args.output}")


if __name__ == "__main__":
    main()

