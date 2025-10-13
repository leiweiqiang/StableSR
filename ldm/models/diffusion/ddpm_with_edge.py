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
        edge_loss_weight=0,
        *args, **kwargs
    ):
        # Store edge processing configuration
        self.use_edge_processing = use_edge_processing
        self.edge_input_channels = edge_input_channels
        self.edge_loss_weight = edge_loss_weight
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
        
        # 🔥 CRITICAL FIX: Ensure edge_processor parameters are trainable
        # Parent class freezes all non-spade parameters, but we need edge_processor to train
        if self.use_edge_processing:
            self._ensure_edge_processor_trainable()
    
    def _ensure_edge_processor_trainable(self):
        """
        Ensure edge_processor parameters are trainable
        
        This is called after parent class __init__ which may freeze these parameters
        """
        if hasattr(self.model, 'diffusion_model') and hasattr(self.model.diffusion_model, 'edge_processor'):
            edge_processor = self.model.diffusion_model.edge_processor
            if edge_processor is not None:
                edge_processor.train()  # Set to train mode
                for name, param in edge_processor.named_parameters():
                    param.requires_grad = True
                    
                print("🔥 Edge Processor - Trainable Parameters:")
                for name, param in edge_processor.named_parameters():
                    print(f"  ✅ {name}: requires_grad={param.requires_grad}")
    
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """
        Override init_from_ckpt to handle edge processing weight loading
        Properly extends 4-channel input weights to 8-channel for edge processing
        """
        import torch
        
        print(f"Loading model from {path}")
        pl_sd = torch.load(path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        
        # Handle 4->8 channel conversion for edge processing
        if self.use_edge_processing:
            # Check if the first conv layer needs channel expansion
            first_conv_key = 'model.diffusion_model.input_blocks.0.0.weight'
            if first_conv_key in sd:
                pretrained_weight = sd[first_conv_key]  # [out_ch, 4, k, k]
                
                # Check if our model expects 8 channels but checkpoint has 4
                current_weight = self.model.diffusion_model.input_blocks[0][0].weight
                if current_weight.shape[1] == 8 and pretrained_weight.shape[1] == 4:
                    print(f"Expanding input conv from 4 to 8 channels for edge processing")
                    # Create new 8-channel weight
                    # First 4 channels: use pretrained weights
                    # Last 4 channels: initialize with zeros (edge features start with no influence)
                    new_weight = torch.zeros(
                        pretrained_weight.shape[0],  # out_channels
                        8,  # in_channels (4 latent + 4 edge)
                        pretrained_weight.shape[2],  # kernel_h
                        pretrained_weight.shape[3],  # kernel_w
                        dtype=pretrained_weight.dtype
                    )
                    # Copy pretrained weights to first 4 channels
                    new_weight[:, :4, :, :] = pretrained_weight
                    # Initialize edge channels with larger random values (改进: 0.01 → 0.1)
                    # 这样edge特征在训练初期就能有更大的影响力
                    new_weight[:, 4:, :, :] = torch.randn_like(new_weight[:, 4:, :, :]) * 0.1
                    
                    sd[first_conv_key] = new_weight
                    print(f"✓ Extended {first_conv_key} from {pretrained_weight.shape} to {new_weight.shape}")
        
        # Load the state dict
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        m, u = self.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
        
        print(f"✓ Model loaded successfully from {path}")
    
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
    
    @torch.no_grad()
    def p_sample_loop(self, cond, struct_cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, time_replace=None, adain_fea=None, interfea_path=None,
                      unconditional_conditioning=None,
                      unconditional_guidance_scale=None,
                      reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
                      edge_map=None):
        """
        Override p_sample_loop to handle edge_map parameter
        """
        from tqdm import tqdm
        from einops import repeat
        
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        batch_list = []
        for i in iterator:
            # Handle time_replace logic like the original implementation
            if time_replace is None or time_replace == 1000:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace = None
            else:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace = repeat(torch.tensor([self.ori_timesteps[i]]), '1 -> b', b=img.size(0))
                t_replace = t_replace.long().to(device)
            
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            # Generate structural conditioning based on t_replace
            if t_replace is not None:
                if start_T is not None:
                    if self.ori_timesteps[i] > start_T:
                        continue
                struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
            else:
                if start_T is not None:
                    if i > start_T:
                        continue
                struct_cond_input = self.structcond_stage_model(struct_cond, ts)

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            img = self.p_sample(img, cond, struct_cond_input, ts, clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised, t_replace=t_replace,
                                unconditional_conditioning=unconditional_conditioning,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                reference_sr=reference_sr,
                                reference_lr=reference_lr,
                                reference_step=reference_step,
                                reference_range=reference_range,
                                edge_map=edge_map)

            if adain_fea is not None:
                if i < 1:
                    from scripts.wavelet_color_fix import adaptive_instance_normalization
                    img = adaptive_instance_normalization(img, adain_fea)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, struct_cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, time_replace=None, adain_fea=None, interfea_path=None, start_T=None,
               unconditional_conditioning=None,
               unconditional_guidance_scale=None,
               reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
               edge_map=None,
               **kwargs):
        """
        Override sample to handle edge_map parameter
        """
        if shape is None:
            shape = (batch_size, self.channels, self.image_size//8, self.image_size//8)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  struct_cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, time_replace=time_replace, adain_fea=adain_fea, interfea_path=interfea_path, start_T=start_T,
                                  unconditional_conditioning=unconditional_conditioning,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range,
                                  edge_map=edge_map)
    
    @torch.no_grad()
    def sample_canvas(self, cond, struct_cond, batch_size=16, return_intermediates=False, x_T=None,
                      verbose=True, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, shape=None, time_replace=None, adain_fea=None, interfea_path=None, 
                      tile_size=64, tile_overlap=32, batch_size_sample=4, log_every_t=None,
                      unconditional_conditioning=None, unconditional_guidance_scale=None, 
                      edge_map=None, **kwargs):
        """
        Override sample_canvas to handle edge_map parameter
        
        🔥 CRITICAL FIX: Parent class sample_canvas doesn't pass edge_map to p_sample_loop_canvas
        This override ensures edge_map is properly propagated during tile-based inference
        """
        if shape is None:
            shape = (batch_size, self.channels, self.image_size//8, self.image_size//8)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key] if not isinstance(cond[key], list) else
                list(map(lambda x: x, cond[key])) for key in cond}
            else:
                cond = [c for c in cond] if isinstance(cond, list) else cond
        return self.p_sample_loop_canvas(cond,
                                  struct_cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, time_replace=time_replace, adain_fea=adain_fea, interfea_path=interfea_path, 
                                  tile_size=tile_size, tile_overlap=tile_overlap,
                                  unconditional_conditioning=unconditional_conditioning, 
                                  unconditional_guidance_scale=unconditional_guidance_scale, 
                                  batch_size=batch_size_sample, log_every_t=log_every_t,
                                  edge_map=edge_map)  # 🔥 Pass edge_map!
    
    @torch.no_grad()
    def p_sample_loop_canvas(self, cond, struct_cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, time_replace=None, adain_fea=None, interfea_path=None, 
                      tile_size=64, tile_overlap=32, batch_size=4,
                      unconditional_conditioning=None, unconditional_guidance_scale=None,
                      reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
                      edge_map=None):
        """
        Override p_sample_loop_canvas to handle edge_map parameter
        
        🔥 CRITICAL FIX: Parent class doesn't support edge_map in canvas sampling
        """
        from tqdm import tqdm
        from einops import repeat
        
        assert tile_size is not None

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = batch_size
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        for i in iterator:
            if time_replace is None or time_replace == 1000:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace=None
            else:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace = repeat(torch.tensor([self.ori_timesteps[i]]), '1 -> b', b=batch_size)
                t_replace = t_replace.long().to(device)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            # Generate structural conditioning based on t_replace
            if t_replace is not None:
                struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
            else:
                struct_cond_input = self.structcond_stage_model(struct_cond, ts)

            # Call p_sample_canvas with edge_map
            img = self.p_sample_canvas(img, cond, struct_cond_input, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised, t_replace=t_replace,
                                tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size, tile_weights=tile_weights,
                                unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale,
                                reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range,
                                edge_map=edge_map)  # 🔥 Pass edge_map!

            if adain_fea is not None:
                if i < 1:
                    from scripts.wavelet_color_fix import adaptive_instance_normalization
                    img = adaptive_instance_normalization(img, adain_fea)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img
    
    @torch.no_grad()
    def p_sample_canvas(self, x, c, struct_cond, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, t_replace=None,
                 tile_size=64, tile_overlap=32, batch_size=4, tile_weights=None, 
                 unconditional_conditioning=None, unconditional_guidance_scale=None,
                 reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
                 edge_map=None):
        """
        Override p_sample_canvas to handle edge_map parameter
        
        🔥 CRITICAL FIX: Propagate edge_map to p_mean_variance_canvas
        """
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance_canvas(x=x, c=c, struct_cond=struct_cond, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, t_replace=t_replace,
                                       tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size, tile_weights=tile_weights,
                                       unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale,
                                       reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range,
                                       edge_map=edge_map)  # 🔥 Pass edge_map!
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_mean_variance_canvas(self, x, c, struct_cond, t, clip_denoised: bool, return_codebook_ids=False, 
                               quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None, 
                               t_replace=None, tile_size=64, tile_overlap=32, batch_size=4, tile_weights=None,
                               unconditional_conditioning=None, unconditional_guidance_scale=None,
                               reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
                               edge_map=None):
        """
        Override p_mean_variance_canvas to handle edge_map parameter
        
        🔥 CRITICAL FIX: This is where edge_map is finally passed to apply_model
        """
        # Call parent implementation but with edge_map in the apply_model call
        # We need to implement tile-based processing with edge support
        from einops import rearrange
        
        b, c_dim, h, w = x.shape
        
        # Calculate tile grid
        num_h = (h - tile_overlap) // (tile_size - tile_overlap)
        num_w = (w - tile_overlap) // (tile_size - tile_overlap)
        
        # Initialize output tensor
        model_output = torch.zeros_like(x)
        weight_sum = torch.zeros_like(x)
        
        # Process tiles
        for h_idx in range(num_h):
            for w_idx in range(num_w):
                h_start = h_idx * (tile_size - tile_overlap)
                h_end = min(h_start + tile_size, h)
                w_start = w_idx * (tile_size - tile_overlap)
                w_end = min(w_start + tile_size, w)
                
                # Extract tile
                x_tile = x[:, :, h_start:h_end, w_start:w_end]
                struct_tile = struct_cond[:, :, h_start:h_end, w_start:w_end] if struct_cond.shape[-2:] == x.shape[-2:] else struct_cond
                
                # Extract edge tile if edge_map is provided
                edge_tile = None
                if self.use_edge_processing and edge_map is not None:
                    # Scale edge_map coordinates to match input size
                    edge_h_start = int(h_start * edge_map.shape[-2] / h)
                    edge_h_end = int(h_end * edge_map.shape[-2] / h)
                    edge_w_start = int(w_start * edge_map.shape[-1] / w)
                    edge_w_end = int(w_end * edge_map.shape[-1] / w)
                    edge_tile = edge_map[:, :, edge_h_start:edge_h_end, edge_w_start:edge_w_end]
                
                # Apply model to tile with edge_map
                t_tile = t if t.dim() == 1 else t
                model_out_tile = self.apply_model(x_tile, t_tile, c, struct_tile, 
                                                  return_codebook_ids=return_codebook_ids,
                                                  edge_map=edge_tile)  # 🔥 Pass edge_map!
                
                # Apply tile weights
                tile_h, tile_w = h_end - h_start, w_end - w_start
                if tile_weights is not None and tile_h == tile_size and tile_w == tile_size:
                    weights = tile_weights[:tile_h, :tile_w].unsqueeze(0).unsqueeze(0).to(x.device)
                else:
                    weights = torch.ones(1, 1, tile_h, tile_w, device=x.device)
                
                # Accumulate weighted output
                model_output[:, :, h_start:h_end, w_start:w_end] += model_out_tile * weights
                weight_sum[:, :, h_start:h_end, w_start:w_end] += weights
        
        # Normalize by weight sum
        model_output = model_output / (weight_sum + 1e-8)
        
        # Post-process like parent method
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.parameterization == "x0":
            x_recon = model_output
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, *_ = self.first_stage_model.quantize(x_recon)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, None
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    
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

        # Add edge loss if edge_map is provided
        if self.use_edge_processing and edge_map is not None:
            # Encode edge_map to latent space for comparison with model_output_
            with torch.no_grad():
                edge_latent = self.encode_first_stage(edge_map)
                edge_latent = self.get_first_stage_encoding(edge_latent).detach()
            
            # Calculate MSE (L2) loss between model output and edge latent
            edge_loss = F.mse_loss(model_output_, edge_latent, reduction='mean')
            
            # Add weighted edge loss to total loss
            loss = loss + self.edge_loss_weight * edge_loss
            
            # Log edge loss
            loss_dict.update({f'{prefix}/edge_loss': edge_loss.detach()})
            loss_dict.update({f'{prefix}/edge_loss_weighted': (self.edge_loss_weight * edge_loss).detach()})

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
