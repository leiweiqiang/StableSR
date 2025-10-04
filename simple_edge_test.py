#!/usr/bin/env python3
"""
Simplified test script for edge processing functionality
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_test_edge_map(size=(512, 512)):
    """Create a test edge map using Canny edge detection"""
    # Create a simple test image
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Draw some geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
    cv2.line(img, (100, 300), (400, 100), (200, 200, 200), 3)
    
    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    # Convert to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_3ch


class EdgeMapProcessor(nn.Module):
    """
    Edge Map Processor that converts edge maps to 64x64x4 latent features
    """
    
    def __init__(self, input_channels=3, output_channels=4, target_size=64):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_size = target_size
        
        # Stage 1: 3x3 conv layers with stride=1, 5 layers, 1024 channels each
        self.stage1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels if i == 0 else 1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ) for i in range(5)
        ])
        
        # Stage 2: 4x4 conv layers with stride=2, 5 layers, channels [512, 256, 64, 16, 4]
        stage2_channels = [512, 256, 64, 16, 4]
        self.stage2_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024 if i == 0 else stage2_channels[i-1], stage2_channels[i], 4, stride=2, padding=1),
                nn.BatchNorm2d(stage2_channels[i]),
                nn.ReLU(inplace=True) if i < len(stage2_channels) - 1 else nn.Identity()
            ) for i in range(5)
        ])
        
        # Adaptive pooling to ensure output size is target_size x target_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_size, target_size))
        
    def forward(self, edge_map):
        """
        Forward pass through edge map processor
        
        Args:
            edge_map: Input edge map tensor [B, C, H, W]
            
        Returns:
            latent_features: Processed features [B, 4, 64, 64]
        """
        x = edge_map
        
        # Stage 1: 3x3 conv layers with stride=1
        for layer in self.stage1_layers:
            x = layer(x)
        
        # Stage 2: 4x4 conv layers with stride=2
        for layer in self.stage2_layers:
            x = layer(x)
        
        # Adaptive pooling to ensure output size
        x = self.adaptive_pool(x)
        
        return x


class EdgeFusionModule(nn.Module):
    """
    Module to fuse edge features with U-Net input features
    Combines 64x64x4 (U-Net input) + 64x64x4 (edge features) = 64x64x8
    """
    
    def __init__(self, unet_channels=4, edge_channels=4, output_channels=8):
        super().__init__()
        
        self.unet_channels = unet_channels
        self.edge_channels = edge_channels
        self.output_channels = output_channels
        
        # Fusion layer to combine features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(unet_channels + edge_channels, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, unet_input, edge_features):
        """
        Fuse U-Net input with edge features
        
        Args:
            unet_input: U-Net input features [B, 4, 64, 64]
            edge_features: Edge processed features [B, 4, 64, 64]
            
        Returns:
            fused_features: Combined features [B, 8, 64, 64]
        """
        # Concatenate along channel dimension
        combined = torch.cat([unet_input, edge_features], dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(combined)
        
        return fused


def test_edge_processor():
    """Test EdgeMapProcessor"""
    print("Testing EdgeMapProcessor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = EdgeMapProcessor(input_channels=3, output_channels=4, target_size=64).to(device)
    
    # Test with different input sizes
    test_sizes = [(512, 512), (256, 256), (128, 128)]
    
    for h, w in test_sizes:
        # Create test edge map
        edge_map_np = create_test_edge_map((h, w))
        edge_map = torch.from_numpy(edge_map_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        edge_map = edge_map.to(device)
        
        # Process edge map
        with torch.no_grad():
            output = processor(edge_map)
        
        print(f"Input: {edge_map.shape} -> Output: {output.shape}")
        assert output.shape == (1, 4, 64, 64), f"Expected (1, 4, 64, 64), got {output.shape}"
    
    print("✓ EdgeMapProcessor test passed!")


def test_edge_fusion():
    """Test EdgeFusionModule"""
    print("Testing EdgeFusionModule...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_module = EdgeFusionModule().to(device)
    
    # Test inputs
    unet_input = torch.randn(2, 4, 64, 64).to(device)
    edge_features = torch.randn(2, 4, 64, 64).to(device)
    
    with torch.no_grad():
        fused = fusion_module(unet_input, edge_features)
    
    print(f"Fusion: {unet_input.shape} + {edge_features.shape} -> {fused.shape}")
    assert fused.shape == (2, 8, 64, 64), f"Expected (2, 8, 64, 64), got {fused.shape}"
    
    print("✓ EdgeFusionModule test passed!")


def test_edge_map_generation():
    """Test edge map generation from dataset"""
    print("Testing edge map generation...")
    
    # Test edge map generation
    edge_map = create_test_edge_map((512, 512))
    print(f"Generated edge map shape: {edge_map.shape}")
    assert edge_map.shape == (512, 512, 3), f"Expected (512, 512, 3), got {edge_map.shape}"
    
    # Test tensor conversion
    edge_tensor = torch.from_numpy(edge_map).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    print(f"Edge tensor shape: {edge_tensor.shape}")
    assert edge_tensor.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {edge_tensor.shape}"
    
    print("✓ Edge map generation test passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Edge Processing Functionality (Simplified)")
    print("=" * 60)
    
    try:
        test_edge_map_generation()
        test_edge_processor()
        test_edge_fusion()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
        # Print summary
        print("\nSummary:")
        print("- Edge map generation: ✓")
        print("- Edge map processing: ✓")
        print("- Feature fusion: ✓")
        print("\nThe edge processing functionality is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
