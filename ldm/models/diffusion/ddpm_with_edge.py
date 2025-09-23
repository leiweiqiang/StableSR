"""
StableSR with Edge Processing Integration
Extends the original StableSR model to include edge image processing capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT
from ldm.modules.edge_processor import EdgeProcessor, EdgeFeatureFusion
from ldm.util import instantiate_from_config


class LatentDiffusionSRTextWTWithEdge(LatentDiffusionSRTextWT):
    """
    StableSR model with edge processing capabilities
    
    This model extends the original StableSR to:
    1. Accept 2Kx2K edge images as additional input
    2. Process edge images through the EdgeProcessor to generate 64x64x4 features
    3. Fuse edge features with SD U-Net input to create 64x64x8 features
    4. Use the fused features for super-resolution
    """
    
    def __init__(self, 
                 edge_processor_config=None,
                 use_edge_fusion=True,
                 *args, **kwargs):
        """
        Initialize StableSR with edge processing
        
        Args:
            edge_processor_config: Configuration for edge processor
            use_edge_fusion: Whether to use edge feature fusion
            *args, **kwargs: Arguments for parent class
        """
        # Temporarily set use_edge_fusion attribute so init_from_ckpt can detect it
        self.use_edge_fusion_temp = use_edge_fusion
        print(f"DEBUG: __init__ - use_edge_fusion: {use_edge_fusion}")
        print(f"DEBUG: __init__ - use_edge_fusion_temp: {self.use_edge_fusion_temp}")
        
        # Call parent constructor first
        super().__init__(*args, **kwargs)
        
        # Now we can safely assign module attributes
        self.use_edge_fusion = use_edge_fusion
        
        # Initialize edge processor
        if edge_processor_config is not None:
            self.edge_processor = instantiate_from_config(edge_processor_config)
        else:
            # Default edge processor configuration
            self.edge_processor = EdgeProcessor(
                input_channels=1,  # Assuming grayscale edge images
                output_channels=4
            )
        
        # Initialize edge feature fusion module
        if self.use_edge_fusion:
            self.edge_fusion = EdgeFeatureFusion(
                sd_input_channels=4,
                edge_channels=4
            )
            print(f"DEBUG: __init__ - Created edge_fusion: {type(self.edge_fusion)}")
            print(f"DEBUG: __init__ - edge_processor created: {hasattr(self, 'edge_processor')}")
            print(f"DEBUG: __init__ - edge_fusion created: {hasattr(self, 'edge_fusion')}")
            # Edge fusion components will be moved to device later
        else:
            print("DEBUG: __init__ - use_edge_fusion is False, not creating edge_fusion")
            
        # Final verification
        print(f"DEBUG: __init__ - Final check - use_edge_fusion: {self.use_edge_fusion}")
        print(f"DEBUG: __init__ - Final check - has edge_processor: {hasattr(self, 'edge_processor')}")
        print(f"DEBUG: __init__ - Final check - has edge_fusion: {hasattr(self, 'edge_fusion')}")
        if hasattr(self, 'edge_processor'):
            print(f"DEBUG: __init__ - edge_processor type: {type(self.edge_processor)}")
        if hasattr(self, 'edge_fusion'):
            print(f"DEBUG: __init__ - edge_fusion type: {type(self.edge_fusion)}")
        
        # Store original model for comparison
        self.original_model = self.model
        
        # Note: We don't need to create an extended model anymore
        # The checkpoint loading already handled the channel extension
        # Just ensure the model is properly configured for edge fusion
        if self.use_edge_fusion:
            print("✓ Edge fusion enabled - model ready for 8-channel input")
    
    def _create_extended_model(self):
        """
        Create an extended U-Net model that can handle 8-channel input instead of 4
        """
        # Get the original model configuration from the parent class
        # The model config is stored in the parent class, not in the DiffusionWrapper
        if hasattr(self, 'unet_config'):
            original_config = self.unet_config.copy()
        else:
            # Fallback: create a basic config based on the current model structure
            original_config = {
                'target': 'ldm.modules.diffusionmodules.openaimodel.UNetModelDualcondV2',
                'params': {
                    'image_size': 32,
                    'in_channels': 4,  # Will be updated to 8
                    'out_channels': 4,
                    'model_channels': 320,
                    'attention_resolutions': [4, 2, 1],
                    'num_res_blocks': 2,
                    'channel_mult': [1, 2, 4, 4],
                    'num_head_channels': 64,
                    'use_spatial_transformer': True,
                    'use_linear_in_transformer': True,
                    'transformer_depth': 1,
                    'context_dim': 1024,
                    'use_checkpoint': False,
                    'legacy': False,
                    'semb_channels': 256,
                }
            }
        
        # Create a new model with extended input channels
        extended_config = original_config.copy()
        extended_config['params']['in_channels'] = 8  # 4 (original) + 4 (edge features)
        
        # Instantiate the extended model
        self.model = instantiate_from_config(extended_config)
        
        # Copy weights from original model for the first 4 channels
        self._transfer_weights()
    
    def _transfer_weights(self):
        """
        Transfer weights from original 4-channel model to extended 8-channel model
        """
        with torch.no_grad():
            # Copy weights for the first 4 channels
            try:
                # Handle nested model structure
                original_conv = self.original_model.model.diffusion_model.input_blocks[0][0]
                extended_conv = self.model.model.diffusion_model.input_blocks[0][0]
                
                # Copy weights for first 4 channels
                extended_conv.weight.data[:, :4, :, :] = original_conv.weight.data
                
                # Initialize remaining 4 channels with small random values
                # Use Xavier initialization for better training stability
                torch.nn.init.xavier_uniform_(extended_conv.weight.data[:, 4:, :, :], gain=0.01)
                
                print(f"✓ Weight transfer completed:")
                print(f"  - Original channels (0-3): Copied from pretrained model")
                print(f"  - New channels (4-7): Initialized with Xavier uniform (gain=0.01)")
                print(f"  - Weight shape: {extended_conv.weight.shape}")
            except Exception as e:
                print(f"⚠️ Warning: Could not transfer weights: {e}")
                print("  - Original model structure:", hasattr(self, 'original_model'))
                print("  - Extended model structure:", hasattr(self.model, 'model'))
                print("  - This is expected if weights were already transferred during checkpoint loading")
    
    @torch.no_grad()
    def get_input(self, batch, k=None, return_first_stage_outputs=False, 
                  force_c_encode=False, cond_key=None, return_original_cond=False, 
                  bs=None, val=False, text_cond=[''], return_gt=False, resize_lq=True):
        """
        Enhanced get_input method that handles edge images
        
        Args:
            batch: Input batch containing 'lq', 'gt', and optionally 'edge_map'
            k: Key for input data
            return_first_stage_outputs: Whether to return first stage outputs
            force_c_encode: Force encoding
            cond_key: Conditioning key
            return_original_cond: Return original conditioning
            bs: Batch size
            val: Validation mode
            text_cond: Text conditioning
            return_gt: Return ground truth
            resize_lq: Resize low quality image
            
        Returns:
            Processed inputs with optional edge features
        """
        # Get the original input processing
        result = super().get_input(
            batch, k, return_first_stage_outputs, force_c_encode,
            cond_key, return_original_cond, bs, val, text_cond, return_gt, resize_lq
        )
        
        # Process edge images if available
        # Ensure edge fusion components are on the correct device
        if hasattr(self, 'edge_processor') and hasattr(self, 'device'):
            if next(self.edge_processor.parameters()).device != self.device:
                self.edge_processor = self.edge_processor.to(self.device)
        if hasattr(self, 'edge_fusion') and hasattr(self, 'device'):
            if next(self.edge_fusion.parameters()).device != self.device:
                self.edge_fusion = self.edge_fusion.to(self.device)
        
        if self.use_edge_fusion and len(result) >= 1:
            latent = result[0]  # First element is usually the latent
            
            if 'edge_map' in batch:
                edge_images = batch['edge_map'].cuda()
                edge_images = edge_images.to(memory_format=torch.contiguous_format).float()
                
                # Ensure edge images are in the correct range [-1, 1]
                if edge_images.max() <= 1.0 and edge_images.min() >= 0.0:
                    edge_images = edge_images * 2.0 - 1.0
                
                # Process edge images through edge processor
                edge_features = self.edge_processor(edge_images)
            else:
                # Create dummy edge features with the same spatial dimensions as latent
                batch_size, _, height, width = latent.shape
                edge_features = torch.zeros(batch_size, 4, height, width, device=latent.device, dtype=latent.dtype)
            
            # Apply edge fusion
            fused_latent = self.edge_fusion(latent, edge_features)
            # Replace first element in the list
            result[0] = fused_latent
        
        return result
    
    def forward(self, x, c, gt, *args, **kwargs):
        """
        Enhanced forward pass with edge processing
        """
        # The model is already configured for 8-channel input if edge fusion is enabled
        # Just call the parent forward method
        return super().forward(x, c, gt, *args, **kwargs)
    
    def apply_model(self, x_noisy, t, cond, struct_cond, return_ids=False):
        """
        Enhanced apply_model with edge processing support
        """
        # The model is already configured for 8-channel input if edge fusion is enabled
        # Just call the parent apply_model method
        return super().apply_model(x_noisy, t, cond, struct_cond, return_ids)
    
    def sample(self, cond, struct_cond, batch_size=1, timesteps=None, 
               time_replace=None, x_T=None, return_intermediates=False, 
               edge_features=None, **kwargs):
        """
        Enhanced sampling method with edge processing
        
        Args:
            cond: Conditioning
            struct_cond: Structural conditioning
            batch_size: Batch size
            timesteps: Number of timesteps
            time_replace: Time replacement
            x_T: Initial noise
            return_intermediates: Return intermediate results
            edge_features: Edge features tensor
            **kwargs: Additional arguments
        """
        # Store edge features if provided
        if edge_features is not None:
            self.edge_features = edge_features
        
        # Call parent sampling method
        return super().sample(
            cond, struct_cond, batch_size, timesteps, time_replace,
            x_T, return_intermediates, **kwargs
        )
    
    def sample_canvas(self, cond, struct_cond, batch_size=1, timesteps=None,
                      time_replace=None, x_T=None, return_intermediates=False,
                      tile_size=64, tile_overlap=32, batch_size_sample=1,
                      edge_features=None, **kwargs):
        """
        Enhanced canvas sampling with edge processing
        
        Args:
            cond: Conditioning
            struct_cond: Structural conditioning
            batch_size: Batch size
            timesteps: Number of timesteps
            time_replace: Time replacement
            x_T: Initial noise
            return_intermediates: Return intermediate results
            tile_size: Tile size for processing
            tile_overlap: Tile overlap
            batch_size_sample: Batch size for sampling
            edge_features: Edge features tensor
            **kwargs: Additional arguments
        """
        # Store edge features if provided
        if edge_features is not None:
            self.edge_features = edge_features
        
        # Call parent canvas sampling method
        return super().sample_canvas(
            cond, struct_cond, batch_size, timesteps, time_replace,
            x_T, return_intermediates, tile_size, tile_overlap,
            batch_size_sample, **kwargs
        )


def create_stablesr_with_edge_config():
    """
    Create a configuration for StableSR with edge processing
    """
    config = {
        "target": "ldm.models.diffusion.ddpm_with_edge.LatentDiffusionSRTextWTWithEdge",
        "params": {
            "first_stage_config": {
                "target": "ldm.models.autoencoder.AutoencoderKLResi",
                "params": {
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    },
                    "embed_dim": 4,
                    "ckpt_path": "vqgan_cfw_00011.ckpt"
                }
            },
            "cond_stage_config": {
                "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder"
            },
            "structcond_stage_config": {
                "target": "ldm.modules.encoders.modules.ConcatTimestepEmbedder",
                "params": {
                    "outdim": 256
                }
            },
            "edge_processor_config": {
                "target": "ldm.modules.edge_processor.EdgeProcessor",
                "params": {
                    "input_channels": 1,
                    "output_channels": 4
                }
            },
            "use_edge_fusion": True,
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "image_size": 32,
                    "in_channels": 8,  # Extended to handle edge features
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [4, 2, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 4, 4],
                    "num_heads": 8,
                    "use_spatial_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 768,
                    "use_checkpoint": True,
                    "legacy": False
                }
            },
            "timesteps": 1000,
            "beta_schedule": "linear",
            "linear_start": 0.00085,
            "linear_end": 0.0120,
            "cosine_s": 8e-3,
            "given_betas": None,
            "original_elbo_weight": 0.0,
            "v_posterior": 0.0,
            "l_simple_weight": 1.0,
            "parameterization": "eps",
            "learn_logvar": False,
            "logvar_init": 0.0,
            "use_ema": False,
            "lr": 1e-4,
            "ckpt_path": "stablesr_000117.ckpt"
        }
    }
    return config
