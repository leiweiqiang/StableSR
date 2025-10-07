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
    - Concatenates edge features with U-Net input (64x64x4 + 64x64x4 = 64x64x8)
    - Uses 8-channel input for the U-Net to accommodate edge concatenation
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
        edge_weight: float = 0.1,
    ):
        # Store original in_channels for edge fusion
        self.original_in_channels = in_channels
        
        # Initialize parent class with the provided in_channels (8 for concatenation)
        # This creates the U-Net with 8 input channels for edge concatenation
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,  # Use provided in_channels (8)
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
            
            # Use additive fusion instead of concatenation to avoid channel mismatch
            # This allows us to keep the original input channels and add edge features
            self.edge_weight = float(edge_weight) if edge_weight is not None else 0.1
            
            # Ensure edge processing modules require gradients
            self.edge_processor.train()
        else:
            self.edge_processor = None
        
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
            x: Input tensor [B, 4, H, W] - U-Net input (64x64x4)
            timesteps: Diffusion timesteps
            context: Text conditioning
            struct_cond: Structural conditioning
            edge_map: Edge map tensor [B, 3, H, W] - Optional edge map input
            y: Class conditioning
            **kwargs: Additional arguments
            
        Returns:
            Output tensor [B, 4, H, W]
            
        Note:
            When edge_map is provided, x and edge_features are concatenated
            to create 8-channel input [B, 8, H, W] for the U-Net
        """
        # Process edge map if provided and edge processing is enabled
        if self.use_edge_processing and edge_map is not None:
            # Process edge map to get 64x64x4 features
            edge_features = self.edge_processor(edge_map)
            
            # Concatenate edge features with U-Net input (4 + 4 = 8 channels)
            # This creates 8-channel input for the U-Net
            x = torch.cat([x, edge_features], dim=1)
        
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
    
    def _ensure_edge_modules_require_grad(self):
        """
        Ensure all edge processing modules require gradients for training
        """
        if self.edge_processor is not None:
            for param in self.edge_processor.parameters():
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
    
    print("UNetModelDualcondV2WithEdge test passed!")


if __name__ == "__main__":
    test_unet_with_edge()
