"""
Edge Map Processor for StableSR
Processes edge maps to generate 64x64x4 latent features for fusion with U-Net input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EdgeMapProcessor(nn.Module):
    """
    Memory-optimized Edge Map Processor that converts edge maps to 64x64x4 latent features
    
    Architecture:
    1. Initial downsampling to reduce memory usage
    2. 3x3 conv layers with reduced channels: 3 layers with 256 channels each
    3. 4x4 conv layers (stride=2): 4 layers with channels [128, 64, 16, 4]
    """
    
    def __init__(self, input_channels=3, output_channels=4, target_size=64, use_checkpoint=False):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_size = target_size
        self.use_checkpoint = use_checkpoint
        
        # Initial downsampling to reduce memory usage (stride=2)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 3x3 conv layers with reduced channels for memory efficiency
        self.stage1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if i == 0 else 256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for i in range(3)  # Reduced from 5 to 3 layers
        ])
        
        # Stage 2: 4x4 conv layers with stride=2, reduced channels
        stage2_channels = [128, 64, 16, 4]  # Reduced from [512, 256, 64, 16, 4]
        self.stage2_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 if i == 0 else stage2_channels[i-1], stage2_channels[i], 4, stride=2, padding=1),
                nn.BatchNorm2d(stage2_channels[i]),
                nn.ReLU(inplace=True) if i < len(stage2_channels) - 1 else nn.Identity()
            ) for i in range(4)  # Reduced from 5 to 4 layers
        ])
        
        # Adaptive pooling to ensure output size is target_size x target_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_size, target_size))
        
    def forward(self, edge_map):
        """
        Forward pass through edge map processor with gradient checkpointing support
        
        Args:
            edge_map: Input edge map tensor [B, C, H, W]
            
        Returns:
            latent_features: Processed features [B, 4, 64, 64]
        """
        x = edge_map
        
        # Initial downsampling
        x = self.initial_conv(x)
        
        # Stage 1: 3x3 conv layers with gradient checkpointing for memory efficiency
        for i, layer in enumerate(self.stage1_layers):
            # Temporarily disable checkpointing to avoid gradient issues
            # if self.use_checkpoint and self.training:
            #     x = checkpoint(layer, x)
            # else:
            x = layer(x)
        
        # Stage 2: 4x4 conv layers with gradient checkpointing
        for i, layer in enumerate(self.stage2_layers):
            # Temporarily disable checkpointing to avoid gradient issues
            # if self.use_checkpoint and self.training:
            #     x = checkpoint(layer, x)
            # else:
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
    """Test function for EdgeMapProcessor with memory optimization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create processor with gradient checkpointing enabled
    processor = EdgeMapProcessor(
        input_channels=3, 
        output_channels=4, 
        target_size=64,
        use_checkpoint=True
    ).to(device)
    
    print(f"Testing EdgeMapProcessor on device: {device}")
    
    # Test with smaller batch size to avoid memory issues
    test_sizes = [(512, 512), (256, 256), (128, 128)]
    
    for h, w in test_sizes:
        # Create dummy edge map with smaller batch size
        edge_map = torch.randn(1, 3, h, w).to(device)
        
        # Process edge map
        with torch.no_grad():
            output = processor(edge_map)
        
        print(f"Input: {edge_map.shape} -> Output: {output.shape}")
        assert output.shape == (1, 4, 64, 64), f"Expected (1, 4, 64, 64), got {output.shape}"
        
        # Clear memory after each test
        del edge_map, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("EdgeMapProcessor test passed!")
    
    # Test fusion module
    fusion_module = EdgeFusionModule().to(device)
    unet_input = torch.randn(1, 4, 64, 64).to(device)
    edge_features = torch.randn(1, 4, 64, 64).to(device)
    
    with torch.no_grad():
        fused = fusion_module(unet_input, edge_features)
    
    print(f"Fusion: {unet_input.shape} + {edge_features.shape} -> {fused.shape}")
    assert fused.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {fused.shape}"
    
    print("EdgeFusionModule test passed!")


if __name__ == "__main__":
    test_edge_processor()
