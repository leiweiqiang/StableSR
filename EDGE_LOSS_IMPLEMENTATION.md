# Edge Loss Implementation Summary

## Overview
This document describes the implementation of edge loss in the StableSR training pipeline, as requested in the v2 loss requirements.

## Changes Made

### 1. Modified `ddpm_with_edge.py`

#### Added `edge_loss_weight` parameter to `__init__`:
- **Location**: Line 52
- **Default value**: 0.1
- **Purpose**: Controls the weight of edge loss in the total training loss
- **Configuration**: Can be adjusted via config file parameter `edge_loss_weight`

#### Implemented edge loss calculation in `p_losses()`:
- **Location**: Lines 473-488
- **Implementation details**:
  1. Encodes `edge_map` from image space to latent space using the VAE encoder
  2. Calculates MSE (L2) loss between `model_output_` and encoded `edge_latent`
  3. Adds weighted edge loss to total loss: `loss = loss + edge_loss_weight * edge_loss`
  4. Logs both raw edge loss and weighted edge loss for monitoring

### 2. Updated Configuration Files

Updated the following config files to include the `edge_loss_weight` parameter:

1. **`v2-finetune_text_T_512_edge_loss.yaml`** (already had it)
   - `edge_loss_weight: 0.1` at line 34

2. **`v2-finetune_text_T_512_edge.yaml`** (added)
   - `edge_loss_weight: 0.1` at line 30

3. **`v2-finetune_text_T_512_edge_fixed.yaml`** (added)
   - `edge_loss_weight: 0.1` at line 31

## How It Works

### Training Flow:
1. During training, `edge_map` is passed as input argument to `p_losses()`
2. The model generates `model_output_` (denoised prediction in latent space)
3. `edge_map` (in image space) is encoded to latent space using VAE encoder
4. MSE loss is calculated between `model_output_` and `edge_latent`
5. Weighted edge loss is added to the total loss: `total_loss = diffusion_loss + edge_loss_weight * edge_loss`

### Loss Components:
- **Primary loss**: Standard diffusion loss (noise prediction or x0 prediction)
- **Edge loss**: MSE between model output and edge map (both in latent space)
- **Total loss**: `loss = diffusion_loss + edge_loss_weight * edge_loss`

### Monitoring:
The following metrics are logged during training:
- `train/edge_loss`: Raw MSE loss between model output and edge latent
- `train/edge_loss_weighted`: Weighted edge loss that contributes to total loss

## Configuration and Tuning

### Adjusting Edge Loss Weight:
You can modify the `edge_loss_weight` parameter in your config file to experiment with different weights:

```yaml
model:
  params:
    edge_loss_weight: 0.1  # Start with 0.1, adjust based on training results
```

### Recommended Starting Values:
- **Initial**: 0.1 (as specified in requirements)
- **If edge details are weak**: Increase to 0.2 or 0.5
- **If edge details are too sharp/artifacts**: Decrease to 0.05 or 0.01

## Technical Notes

### Why encode edge_map to latent space?
- `model_output_` is in latent space (4 channels, 64x64)
- `edge_map` is in image space (3 channels, 512x512)
- To calculate loss, both tensors must be in the same space
- We encode `edge_map` to latent space using `encode_first_stage()` and `get_first_stage_encoding()`

### Gradient Flow:
- Edge encoding is done with `torch.no_grad()` to prevent gradients from flowing through VAE
- Only the diffusion model (UNet) receives gradients from edge loss
- This ensures edge loss guides the denoising process without affecting VAE

## Testing and Validation

To verify the implementation is working:
1. Check training logs for `train/edge_loss` and `train/edge_loss_weighted` metrics
2. These values should be logged at each training step
3. Monitor how edge loss changes during training
4. Compare generated images with and without edge loss to evaluate effectiveness

## Next Steps

1. Start training with `edge_loss_weight: 0.1`
2. Monitor training metrics and generated samples
3. Adjust `edge_loss_weight` based on results:
   - If edges are not sharp enough, increase weight
   - If there are edge artifacts, decrease weight
4. Experiment with different values to find optimal balance

