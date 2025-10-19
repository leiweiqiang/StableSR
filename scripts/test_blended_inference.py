#!/usr/bin/env python3
"""
Quick test script for blended input inference
Creates dummy LR and edge inputs for testing
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Add project root to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def create_dummy_lr_image(size=32, save_path=None):
    """Create a dummy LR image for testing"""
    # Create a simple gradient pattern
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # RGB gradient
    r = xx
    g = yy
    b = 1.0 - (xx + yy) / 2.0
    
    image = np.stack([r, g, b], axis=2)
    image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    if save_path:
        pil_image.save(save_path)
        print(f"Created dummy LR image: {save_path}")
    
    return pil_image


def create_dummy_edge_map(size=512, save_path=None):
    """Create a dummy edge map for testing"""
    # Create a simple edge pattern (grid)
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Draw grid lines as edges
    spacing = size // 8
    for i in range(0, size, spacing):
        image[i, :, :] = 0  # Horizontal lines
        image[:, i, :] = 0  # Vertical lines
    
    # Draw diagonal
    for i in range(size):
        if i < size:
            image[i, i, :] = 0
    
    pil_image = Image.fromarray(image)
    
    if save_path:
        pil_image.save(save_path)
        print(f"Created dummy edge map: {save_path}")
    
    return pil_image


def test_blending():
    """Test the blending logic with dummy data"""
    print("\n" + "="*60)
    print("Testing Blended Input Creation")
    print("="*60 + "\n")
    
    # Create dummy inputs
    lr_size = 32
    output_size = 512
    blend_alpha = 0.5
    
    # LR image as tensor
    lr_np = np.random.rand(1, 3, lr_size, lr_size).astype(np.float32)
    lr_tensor = torch.from_numpy(lr_np) * 2.0 - 1.0  # [-1, 1]
    
    # Edge map as tensor
    edge_np = np.random.rand(1, 3, output_size, output_size).astype(np.float32)
    edge_tensor = torch.from_numpy(edge_np) * 2.0 - 1.0  # [-1, 1]
    
    print(f"LR tensor shape: {lr_tensor.shape}, range: [{lr_tensor.min():.3f}, {lr_tensor.max():.3f}]")
    print(f"Edge tensor shape: {edge_tensor.shape}, range: [{edge_tensor.min():.3f}, {edge_tensor.max():.3f}]")
    
    # Test upscaling
    print("\nUpscaling LR to 512x512...")
    lr_upscaled = F.interpolate(lr_tensor, size=(output_size, output_size), 
                                mode='bicubic', align_corners=False)
    print(f"  Upscaled shape: {lr_upscaled.shape}")
    print(f"  Upscaled range: [{lr_upscaled.min():.3f}, {lr_upscaled.max():.3f}]")
    
    # Test blending
    print(f"\nBlending with alpha={blend_alpha}...")
    lr_01 = (lr_upscaled + 1.0) / 2.0
    edge_01 = (edge_tensor + 1.0) / 2.0
    blended = blend_alpha * lr_01 + (1.0 - blend_alpha) * edge_01
    blended = 2.0 * blended - 1.0
    blended = torch.clamp(blended, -1.0, 1.0)
    
    print(f"  Blended shape: {blended.shape}")
    print(f"  Blended range: [{blended.min():.3f}, {blended.max():.3f}]")
    
    # Verify properties
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    
    checks_passed = 0
    total_checks = 3
    
    if blended.shape == (1, 3, output_size, output_size):
        print("✓ Shape is correct: (1, 3, 512, 512)")
        checks_passed += 1
    else:
        print(f"✗ Shape is incorrect: {blended.shape}")
    
    if -1.0 <= blended.min() and blended.max() <= 1.0:
        print(f"✓ Range is correct: [{blended.min():.3f}, {blended.max():.3f}]")
        checks_passed += 1
    else:
        print(f"✗ Range is incorrect: [{blended.min():.3f}, {blended.max():.3f}]")
    
    if not torch.allclose(blended, lr_upscaled) and not torch.allclose(blended, edge_tensor):
        print("✓ Blending is working (output differs from both inputs)")
        checks_passed += 1
    else:
        print("✗ Blending may not be working correctly")
    
    print(f"\n{checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n✓ All tests passed! Blending logic is working correctly.\n")
        return True
    else:
        print("\n✗ Some tests failed. Please check the implementation.\n")
        return False


def create_test_inputs(output_dir="test_inputs"):
    """Create test LR and edge images"""
    print("\n" + "="*60)
    print("Creating Test Inputs")
    print("="*60 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    lr_path = os.path.join(output_dir, "test_lr_32x32.png")
    edge_path = os.path.join(output_dir, "test_edge_512x512.png")
    
    create_dummy_lr_image(size=32, save_path=lr_path)
    create_dummy_edge_map(size=512, save_path=edge_path)
    
    print(f"\n✓ Test inputs created in: {output_dir}/")
    print(f"  - LR image: {lr_path}")
    print(f"  - Edge map: {edge_path}")
    print("\nYou can now run inference with:")
    print(f"  python scripts/inference_blended_input.py \\")
    print(f"    --lr-img {lr_path} \\")
    print(f"    --edge-img {edge_path} \\")
    print(f"    --outdir outputs/test/ \\")
    print(f"    --ckpt path/to/checkpoint.ckpt\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-test-inputs", action="store_true",
                       help="Create dummy test inputs")
    parser.add_argument("--test-blending", action="store_true",
                       help="Test blending logic")
    
    args = parser.parse_args()
    
    if args.create_test_inputs:
        create_test_inputs()
    elif args.test_blending:
        test_blending()
    else:
        print("Please specify --create-test-inputs or --test-blending")
        print("\nExamples:")
        print("  python scripts/test_blended_inference.py --test-blending")
        print("  python scripts/test_blended_inference.py --create-test-inputs")

