"""
LatentDiffusionSRTextWT with Edge Processing Support
Extends the original LatentDiffusionSRTextWT to support edge map processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .ddpm import LatentDiffusionSRTextWT
from ldm.modules.diffusionmodules.unet_with_edge import UNetModelDualcondV2WithEdge
from ldm.util import default
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import noise_like
import pytorch_lightning as pl


class LatentDiffusionSRTextWTWithEdge(LatentDiffusionSRTextWT):
    """
    Extended LatentDiffusionSRTextWT with edge map processing support
    
    Key features:
    - Processes edge maps from dataset
    - Fuses edge features with U-Net input
    - Maintains compatibility with original training pipeline
    """
    
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        structcond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        unfrozen_diff=False,
        random_size=False,
        test_gt=False,
        p2_gamma=None,
        p2_k=None,
        time_replace=None,
        use_usm=False,
        mix_ratio=0.0,
        use_edge_processing=False,
        edge_input_channels=3,
        *args, **kwargs
    ):
        # Store edge processing configuration
        self.use_edge_processing = use_edge_processing
        self.edge_input_channels = edge_input_channels
        # Consume optional edge_weight from config to avoid passing to base class
        edge_weight = kwargs.pop('edge_weight', None)
        
        # No need to ignore keys since we're using additive fusion instead of concatenation
        # This ensures compatibility with pre-trained weights
        
        # Modify unet_config to use edge-enabled UNet if it exists in kwargs
        if use_edge_processing and 'unet_config' in kwargs:
            unet_config = kwargs['unet_config']
            if hasattr(unet_config, 'target'):
                unet_config.target = 'ldm.modules.diffusionmodules.unet_with_edge.UNetModelDualcondV2WithEdge'
            if hasattr(unet_config, 'params'):
                unet_config.params.use_edge_processing = True
                unet_config.params.edge_input_channels = edge_input_channels
                # Forward edge_weight to UNet if provided
                if edge_weight is not None:
                    unet_config.params.edge_weight = edge_weight
        
        # Initialize parent class
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            structcond_stage_config=structcond_stage_config,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            unfrozen_diff=unfrozen_diff,
            random_size=random_size,
            test_gt=test_gt,
            p2_gamma=p2_gamma,
            p2_k=p2_k,
            time_replace=time_replace,
            use_usm=use_usm,
            mix_ratio=mix_ratio,
            *args, **kwargs
        )
        
        # No need to initialize edge input weights since we're using additive fusion
    
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """
        Override init_from_ckpt to handle edge processing weight loading
        """
        # Add keys to ignore when loading checkpoint for edge processing models
        # This handles the case where the checkpoint was saved with concatenation approach (8 channels)
        # but we're now using additive fusion approach (4 channels)
        if self.use_edge_processing:
            edge_ignore_keys = [
                'model.diffusion_model.input_blocks.0.0.weight',
                'model.diffusion_model.input_blocks.0.0.bias',
            ]
            ignore_keys = ignore_keys + edge_ignore_keys
        
        # Call parent method with updated ignore_keys
        super().init_from_ckpt(path, ignore_keys, only_model)
    
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, val=False, text_cond=[''], return_gt=False, resize_lq=True):
        """
        Override get_input to handle edge map processing
        """
        # Call parent get_input method
        result = super().get_input(
            batch=batch,
            k=k,
            return_first_stage_outputs=return_first_stage_outputs,
            force_c_encode=force_c_encode,
            cond_key=cond_key,
            return_original_cond=return_original_cond,
            bs=bs,
            val=val,
            text_cond=text_cond,
            return_gt=return_gt,
            resize_lq=resize_lq
        )
        
        # Add edge map to result if available and edge processing is enabled
        if self.use_edge_processing and 'img_edge' in batch:
            edge_map = batch['img_edge'].cuda()
            edge_map = edge_map.to(memory_format=torch.contiguous_format).float()
            
            # Edge map should already be in [0, 1] range from dataset
            # Only normalize to [-1, 1] if it's not already in that range
            if edge_map.max() <= 1.0 and edge_map.min() >= 0.0:
                edge_map = edge_map * 2.0 - 1.0
            
            # Do NOT make edge_map require gradients - it should be treated as input data
            # edge_map = edge_map.requires_grad_(True)  # Remove this line
            
            if bs is not None:
                edge_map = edge_map[:bs]
            
            # Add edge map to result
            if isinstance(result, list):
                result.append(edge_map)
            else:
                result = [result, edge_map]
        
        return result
    
    def apply_model(self, x_noisy, t, cond, struct_cond, edge_map=None, **kwargs):
        """
        Override apply_model to pass edge_map to the UNet
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # Add edge_map to cond dict if provided
        if self.use_edge_processing and edge_map is not None:
            cond['edge_map'] = edge_map
        
        # Call the model with the same pattern as the original
        return self.model(x_noisy, t, **cond, struct_cond=struct_cond)
    
    def p_mean_variance(self, x, c, struct_cond, t, clip_denoised=True, return_codebook_ids=False, quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None, edge_map=None):
        """
        Override p_mean_variance to handle edge_map parameter
        """
        t_in = t
        model_out = self.apply_model(x, t_in, c, struct_cond, return_codebook_ids=return_codebook_ids, edge_map=edge_map)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, *_ = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, model_out
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, x, c, struct_cond, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, t_replace=None,
                 unconditional_conditioning=None, unconditional_guidance_scale=None,
                 reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000], edge_map=None):
        """
        Override p_sample to handle edge_map parameter
        """
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, struct_cond=struct_cond, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       edge_map=edge_map)
        if return_codebook_ids:
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x_recon = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise, logits
        elif return_x0:
            return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise, x_recon
        else:
            return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def p_losses(self, x_start, cond, struct_cond, t, t_ori, z_gt, noise=None, edge_map=None):
        """
        Override p_losses to handle edge map in loss computation
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.mix_ratio > 0:
            if random.random() < self.mix_ratio:
                noise_new = default(noise, lambda: torch.randn_like(x_start))
                noise = noise_new * 0.5 + noise * 0.5
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Apply model with edge map if available
        if self.use_edge_processing and edge_map is not None:
            model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond, edge_map=edge_map)
        else:
            model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        model_output_ = model_output

        loss_simple = self.get_loss(model_output_, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # P2 weighting
        if self.snr is not None:
            self.snr = self.snr.to(loss_simple.device)
            weight = self.extract_into_tensor(1 / (self.p2_k + self.snr)**self.p2_gamma, t, target.shape)
            loss_simple = weight * loss_simple

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output_, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        loss_dict.update({f'{prefix}/loss_simple_ema': loss_simple.mean().detach()})

        return loss, loss_dict
    
    def shared_step(self, batch, **kwargs):
        """
        Override shared_step to handle edge map in validation
        """
        # Get input with edge map
        if self.use_edge_processing and 'img_edge' in batch:
            x, c, gt, edge_map = self.get_input(batch, self.first_stage_key)
            loss = self(x, c, gt, edge_map=edge_map)
        else:
            x, c, gt = self.get_input(batch, self.first_stage_key)
            loss = self(x, c, gt)
        return loss
    
    def forward(self, x, c, gt, edge_map=None, *args, **kwargs):
        """
        Override forward to handle edge map parameter
        """
        index = np.random.randint(0, self.num_timesteps, size=x.size(0))
        t = torch.from_numpy(index)
        t = t.to(self.device).long()

        t_ori = torch.tensor([self.ori_timesteps[index_i] for index_i in index])
        t_ori = t_ori.long().to(x.device)

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            else:
                c = self.cond_stage_model(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                print(s)
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        # Create structural conditioning from latent representation
        # Check if inputs are already in latent space (4 channels) or RGB space (3 channels)
        if self.test_gt:
            # Check if gt is already in latent space (4 channels) or RGB space (3 channels)
            if gt.shape[1] == 4:
                # Already in latent space
                struc_c = self.structcond_stage_model(gt, t_ori)
            else:
                # RGB space, need to encode to latent space
                encoder_posterior_gt = self.encode_first_stage(gt)
                z_gt = self.get_first_stage_encoding(encoder_posterior_gt).detach()
                struc_c = self.structcond_stage_model(z_gt, t_ori)
        else:
            # Check if x is already in latent space (4 channels) or RGB space (3 channels)
            if x.shape[1] == 4:
                # Already in latent space
                struc_c = self.structcond_stage_model(x, t_ori)
            else:
                # RGB space, need to encode to latent space
                encoder_posterior_x = self.encode_first_stage(x)
                z_x = self.get_first_stage_encoding(encoder_posterior_x).detach()
                struc_c = self.structcond_stage_model(z_x, t_ori)
        
        # Call p_losses with edge_map if available
        # Note: p_losses expects (x_start, cond, struct_cond, t, t_ori, z_gt, noise=None)
        # where x_start should be the latent representation
        if x.shape[1] == 4:
            # x is already in latent space, use it directly
            x_start = x
            z_gt = gt  # gt is also in latent space
        else:
            # x is RGB, need to encode to latent space
            encoder_posterior_x = self.encode_first_stage(x)
            x_start = self.get_first_stage_encoding(encoder_posterior_x).detach()
            encoder_posterior_gt = self.encode_first_stage(gt)
            z_gt = self.get_first_stage_encoding(encoder_posterior_gt).detach()
        
        if edge_map is not None:
            return self.p_losses(x_start, c, struc_c, t, t_ori, z_gt, edge_map=edge_map)
        else:
            return self.p_losses(x_start, c, struc_c, t, t_ori, z_gt)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Override training_step to handle edge map in training
        """
        if optimizer_idx == 0:
            # Get input with edge map
            if self.use_edge_processing and 'img_edge' in batch:
                result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
                z, c, z_gt, x, gt, xrec, edge_map = result
            else:
                result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
                z, c, z_gt, x, gt, xrec = result
                edge_map = None
            
            # Get time steps
            t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
            t_ori = t
            
            # Create structural conditioning from latent representation
            # Since z and z_gt are detached from get_input, we need to re-encode with gradients
            if self.test_gt:
                # Re-encode ground truth with gradients for training
                encoder_posterior_gt = self.encode_first_stage(gt)
                z_gt_with_grad = self.get_first_stage_encoding(encoder_posterior_gt)
                struct_cond = self.structcond_stage_model(z_gt_with_grad, t_ori)
            else:
                # Re-encode low-quality input with gradients for training
                encoder_posterior_x = self.encode_first_stage(x)
                z_with_grad = self.get_first_stage_encoding(encoder_posterior_x)
                struct_cond = self.structcond_stage_model(z_with_grad, t_ori)
            
            # Process text conditioning to convert from strings to tensors
            if isinstance(c, list) and len(c) > 0 and isinstance(c[0], str):
                # c is a list of strings, need to process through cond_stage_model
                c = self.cond_stage_model(c)
            
            # Compute loss with edge map
            if self.use_edge_processing and edge_map is not None:
                loss, loss_dict = self.p_losses(z, c, struct_cond, t, t_ori, z_gt, edge_map=edge_map)
            else:
                loss, loss_dict = self.p_losses(z, c, struct_cond, t, t_ori, z_gt)
            
            return loss
        else:
            # Discriminator training (if applicable)
            return super().training_step(batch, batch_idx)


def test_latent_diffusion_with_edge():
    """Test function for LatentDiffusionSRTextWTWithEdge"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple config for testing
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'unet_config': {
            'target': 'ldm.modules.diffusionmodules.unet_with_edge.UNetModelDualcondV2WithEdge',
            'params': {
                'image_size': 32,
                'in_channels': 4,
                'model_channels': 320,
                'out_channels': 4,
                'num_res_blocks': 2,
                'attention_resolutions': [4, 2, 1],
                'channel_mult': [1, 2, 4, 4],
                'num_head_channels': 64,
                'use_spatial_transformer': True,
                'context_dim': 1024,
                'semb_channels': 256,
                'use_edge_processing': True,
                'edge_input_channels': 3,
            }
        },
        'first_stage_config': {
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 4,
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 4,
                    'resolution': 512,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 4, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0,
                },
                'lossconfig': {
                    'target': 'torch.nn.Identity'
                }
            }
        },
        'cond_stage_config': {
            'target': 'ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder',
            'params': {
                'freeze': True,
                'layer': 'penultimate'
            }
        },
        'structcond_stage_config': {
            'target': 'ldm.modules.diffusionmodules.openaimodel.EncoderUNetModelWT',
            'params': {
                'image_size': 96,
                'in_channels': 4,
                'model_channels': 256,
                'out_channels': 256,
                'num_res_blocks': 2,
                'attention_resolutions': [4, 2, 1],
                'dropout': 0,
                'channel_mult': [1, 1, 2, 2],
                'conv_resample': True,
                'dims': 2,
                'use_checkpoint': False,
                'use_fp16': False,
                'num_heads': 4,
                'num_head_channels': -1,
                'num_heads_upsample': -1,
                'use_scale_shift_norm': False,
                'resblock_updown': False,
                'use_new_attention_order': False,
            }
        }
    })
    
    # Create model
    model = LatentDiffusionSRTextWTWithEdge(
        first_stage_config=config.first_stage_config,
        cond_stage_config=config.cond_stage_config,
        structcond_stage_config=config.structcond_stage_config,
        use_edge_processing=True,
        edge_input_channels=3,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        first_stage_key="image",
        cond_stage_key="caption",
        image_size=512,
        channels=4,
        cond_stage_trainable=False,
        conditioning_key="crossattn",
        scale_factor=0.18215,
        use_ema=False,
        unfrozen_diff=False,
        random_size=False,
        time_replace=1000,
        use_usm=True,
    ).to(device)
    
    # Test with dummy batch
    batch_size = 2
    batch = {
        'gt': torch.randn(batch_size, 3, 512, 512).to(device),
        'img_edge': torch.randn(batch_size, 3, 512, 512).to(device),
        'kernel1': torch.randn(batch_size, 1, 21, 21).to(device),
        'kernel2': torch.randn(batch_size, 1, 11, 11).to(device),
        'sinc_kernel': torch.randn(batch_size, 1, 21, 21).to(device),
    }
    
    # Test get_input
    with torch.no_grad():
        result = model.get_input(batch, return_first_stage_outputs=True)
        print(f"get_input result length: {len(result)}")
        print(f"z shape: {result[0].shape}")
        print(f"edge_map shape: {result[-1].shape}")
    
    print("LatentDiffusionSRTextWTWithEdge test passed!")


if __name__ == "__main__":
    test_latent_diffusion_with_edge()
