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
    Encode a 3-channel image of any HxW into a 4x64x64 feature map.
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 下采样 /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 下采样 /2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 -> 128 (增强表达)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.to_four = nn.Conv2d(128, 4, kernel_size=1, bias=True)

        self.pool = nn.AdaptiveAvgPool2d((64, 64))

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, 4, 64, 64)
        """
        feat = self.backbone(x)          # (B, 128, H', W')
        feat4 = self.to_four(feat)       # (B, 4,   H', W')
        out = self.pool(feat4)           # (B, 4,   64, 64)
        return out


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
