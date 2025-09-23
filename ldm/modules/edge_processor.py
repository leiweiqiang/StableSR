"""
Edge Image Processor for StableSR
Processes 2Kx2K edge images to generate 64x64x4 features for fusion with SD U-Net input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EdgeProcessor(nn.Module):
    """
    Edge image processor that converts 512x512 edge images to 64x64x4 features
    
    Architecture:
    1. First stage: 3 layers of 3x3 conv with stride=1, channels: 1->128->256->256
    2. Second stage: 3 layers of 4x4 conv with stride=2, channels: 256->128->64->4
    3. Output: 64x64x4 features ready for fusion with SD U-Net input
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 4):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # First stage: 3x3 conv layers with stride=1, 256 channels each (adapted for 512x512 input)
        self.first_stage = nn.Sequential(
            # Layer 1: 512x512 -> 512x512
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 2: 512x512 -> 512x512
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 3: 512x512 -> 512x512
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Second stage: 4x4 conv layers with stride=2, decreasing channels (adapted for 512x512 input)
        self.second_stage = nn.Sequential(
            # Layer 1: 512x512 -> 256x256, 256->128 channels
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 2: 256x256 -> 128x128, 128->64 channels
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: 128x128 -> 64x64, 64->output_channels
            nn.Conv2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the edge processor
        
        Args:
            x: Input edge image tensor of shape (B, C, H, W) where H=W=2048
            
        Returns:
            Edge features tensor of shape (B, 4, 64, 64)
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        B, C, H, W = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {C}")
        
        # First stage: 3x3 convolutions
        x = self.first_stage(x)
        
        # Second stage: 4x4 convolutions with stride=2
        x = self.second_stage(x)
        
        # Final output should be (B, 4, 64, 64)
        expected_shape = (B, self.output_channels, 64, 64)
        if x.shape != expected_shape:
            raise RuntimeError(f"Expected output shape {expected_shape}, got {x.shape}")
        
        return x


class EdgeFeatureFusion(nn.Module):
    """
    Fusion module that combines edge features with SD U-Net input
    """
    
    def __init__(self, sd_input_channels: int = 4, edge_channels: int = 4):
        super().__init__()
        
        self.sd_input_channels = sd_input_channels
        self.edge_channels = edge_channels
        self.fused_channels = sd_input_channels + edge_channels
        
        # Optional: Add a fusion layer to better combine the features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.fused_channels, self.fused_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.fused_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, sd_input: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse SD U-Net input with edge features
        
        Args:
            sd_input: SD U-Net input tensor of shape (B, 4, 64, 64)
            edge_features: Edge features tensor of shape (B, 4, 64, 64)
            
        Returns:
            Fused features tensor of shape (B, 8, 64, 64)
        """
        # Validate input shapes
        if sd_input.shape != edge_features.shape:
            raise ValueError(f"SD input shape {sd_input.shape} != edge features shape {edge_features.shape}")
        
        if sd_input.shape[1] != self.sd_input_channels:
            raise ValueError(f"Expected SD input to have {self.sd_input_channels} channels, got {sd_input.shape[1]}")
        
        if edge_features.shape[1] != self.edge_channels:
            raise ValueError(f"Expected edge features to have {self.edge_channels} channels, got {edge_features.shape[1]}")
        
        # Concatenate along channel dimension
        fused = torch.cat([sd_input, edge_features], dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(fused)
        
        return fused


def test_edge_processor():
    """Test function for the edge processor"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create edge processor
    edge_processor = EdgeProcessor(input_channels=1, output_channels=4).to(device)
    
    # Create test input: batch of 2Kx2K edge images
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 2048, 2048).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        edge_features = edge_processor(test_input)
    
    print(f"Edge features shape: {edge_features.shape}")
    
    # Test fusion
    fusion_module = EdgeFeatureFusion().to(device)
    sd_input = torch.randn(batch_size, 4, 64, 64).to(device)
    
    with torch.no_grad():
        fused_features = fusion_module(sd_input, edge_features)
    
    print(f"Fused features shape: {fused_features.shape}")
    
    print("✓ Edge processor test passed!")


if __name__ == "__main__":
    test_edge_processor()
