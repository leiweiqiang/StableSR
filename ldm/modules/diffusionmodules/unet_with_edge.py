"""
UNetModelDualcondV2 with Edge Processing Support
Extends the original UNetModelDualcondV2 to support edge map processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .openaimodel import UNetModelDualcondV2
from .edge_processor import EdgeMapProcessor, EdgeFusionModule
from .openaimodel import conv_nd, TimestepEmbedSequential


class UNetModelDualcondV2WithEdge(UNetModelDualcondV2):
    """
    Extended UNetModelDualcondV2 with edge map processing support
    
    Key features:
    - Processes edge maps to generate 64x64x4 latent features
    - Fuses edge features with U-Net input (64x64x4 + 64x64x4 = 64x64x8)
    - Maintains compatibility with original UNetModelDualcondV2
    """
    
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        semb_channels=None,
        use_edge_processing=False,
        edge_input_channels=3,
    ):
        # Store original in_channels for edge fusion
        self.original_in_channels = in_channels
        
        # Initialize parent class with original in_channels
        # We'll handle edge fusion in the forward pass instead of changing the architecture
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
            disable_self_attentions=disable_self_attentions,
            num_attention_blocks=num_attention_blocks,
            disable_middle_self_attn=disable_middle_self_attn,
            use_linear_in_transformer=use_linear_in_transformer,
            semb_channels=semb_channels,
        )
        
        # Initialize edge processing modules if enabled (after super().__init__)
        if use_edge_processing:
            self.edge_processor = EdgeMapProcessor(
                input_channels=edge_input_channels,
                output_channels=4,  # Output 4 channels to match U-Net input
                target_size=64,
                use_checkpoint=False  # Disable gradient checkpointing for edge processor to avoid gradient issues
            )
            self.edge_fusion = EdgeFusionModule(
                unet_channels=in_channels,
                edge_channels=4,
                output_channels=in_channels + 4  # Fuse to 8 channels total
            )
            
            # Replace the first input block to handle fused input
            # The original first block expects in_channels, but we need it to handle in_channels + 4
            fused_channels = in_channels + 4
            self.input_blocks[0] = TimestepEmbedSequential(
                conv_nd(dims, fused_channels, model_channels, 3, padding=1)
            )
            
            # Also need to replace the first ResBlockDual to handle the correct input channels
            # The first ResBlockDual should expect model_channels input (from the conv layer above)
            # but we need to make sure its skip connection is set up correctly
            # Actually, the first ResBlockDual should be fine since it expects model_channels input
            # The issue might be elsewhere - let's check the channel flow
        else:
            self.edge_processor = None
            self.edge_fusion = None
        
        # Store edge processing configuration
        self.use_edge_processing = use_edge_processing
        self.edge_input_channels = edge_input_channels
        
        # Ensure edge processing modules require gradients
        if use_edge_processing:
            self._ensure_edge_modules_require_grad()
    
    def forward(self, x, timesteps=None, context=None, struct_cond=None, edge_map=None, y=None, **kwargs):
        """
        Forward pass with edge map processing support
        
        Args:
            x: Input tensor [B, C, H, W] - U-Net input (usually 64x64x4)
            timesteps: Diffusion timesteps
            context: Text conditioning
            struct_cond: Structural conditioning
            edge_map: Edge map tensor [B, 3, H, W] - Optional edge map input
            y: Class conditioning
            **kwargs: Additional arguments
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Process edge map if provided and edge processing is enabled
        if self.use_edge_processing and edge_map is not None:
            # Process edge map to get 64x64x4 features
            edge_features = self.edge_processor(edge_map)
            
            # Fuse U-Net input with edge features
            x = self.edge_fusion(x, edge_features)
        
        # Call parent forward method
        return super().forward(
            x=x,
            timesteps=timesteps,
            context=context,
            struct_cond=struct_cond,
            y=y,
            **kwargs
        )
    
    def get_edge_features(self, edge_map):
        """
        Get processed edge features without fusion
        
        Args:
            edge_map: Edge map tensor [B, 3, H, W]
            
        Returns:
            edge_features: Processed edge features [B, 4, 64, 64]
        """
        if not self.use_edge_processing:
            raise ValueError("Edge processing is not enabled")
        
        return self.edge_processor(edge_map)
    
    def fuse_with_edge(self, unet_input, edge_map):
        """
        Fuse U-Net input with edge map
        
        Args:
            unet_input: U-Net input tensor [B, 4, 64, 64]
            edge_map: Edge map tensor [B, 3, H, W]
            
        Returns:
            fused_input: Fused input tensor [B, 8, 64, 64]
        """
        if not self.use_edge_processing:
            raise ValueError("Edge processing is not enabled")
        
        edge_features = self.edge_processor(edge_map)
        return self.edge_fusion(unet_input, edge_features)
    
    def _ensure_edge_modules_require_grad(self):
        """
        Ensure all edge processing modules require gradients for training
        """
        if self.edge_processor is not None:
            for param in self.edge_processor.parameters():
                param.requires_grad = True
        
        if self.edge_fusion is not None:
            for param in self.edge_fusion.parameters():
                param.requires_grad = True
    


def test_unet_with_edge():
    """Test function for UNetModelDualcondV2WithEdge"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with edge processing
    model = UNetModelDualcondV2WithEdge(
        image_size=32,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4, 4],
        num_head_channels=64,
        use_spatial_transformer=True,
        context_dim=1024,
        semb_channels=256,
        use_edge_processing=True,
        edge_input_channels=3,
    ).to(device)
    
    # Test inputs
    batch_size = 2
    unet_input = torch.randn(batch_size, 4, 64, 64).to(device)
    edge_map = torch.randn(batch_size, 3, 512, 512).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 77, 1024).to(device)
    struct_cond = torch.randn(batch_size, 256, 96, 96).to(device)
    
    # Test forward pass with edge map
    with torch.no_grad():
        output = model(
            x=unet_input,
            timesteps=timesteps,
            context=context,
            struct_cond=struct_cond,
            edge_map=edge_map
        )
    
    print(f"Input: {unet_input.shape}")
    print(f"Edge map: {edge_map.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == unet_input.shape, f"Expected {unet_input.shape}, got {output.shape}"
    
    # Test edge features extraction
    edge_features = model.get_edge_features(edge_map)
    print(f"Edge features: {edge_features.shape}")
    assert edge_features.shape == (batch_size, 4, 64, 64), f"Expected {(batch_size, 4, 64, 64)}, got {edge_features.shape}"
    
    # Test fusion
    fused = model.fuse_with_edge(unet_input, edge_map)
    print(f"Fused input: {fused.shape}")
    assert fused.shape == (batch_size, 8, 64, 64), f"Expected {(batch_size, 8, 64, 64)}, got {fused.shape}"
    
    print("UNetModelDualcondV2WithEdge test passed!")


if __name__ == "__main__":
    test_unet_with_edge()
