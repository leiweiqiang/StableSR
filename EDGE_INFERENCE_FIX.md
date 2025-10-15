# Edge Inference Fix Summary

## Issues Found and Fixed

### 1. **Module Import Issue**
**Problem:** The script was importing modules from `StableSR_Edge_v2` instead of `StableSR_Edge_v3`, causing incompatibility with edge support.

**Root Cause:**
- Missing `__init__.py` files in the `ldm/` and `ldm/modules/` directories
- Python was falling back to importing from v2 which was in the path

**Fixes Applied:**
- Created `/root/dp/StableSR_Edge_v3/ldm/__init__.py`
- Created `/root/dp/StableSR_Edge_v3/ldm/modules/__init__.py`
- Added `sys.path` manipulation at the start of the script to prioritize v3 imports
- Added debug prints to verify which modules are being loaded

### 2. **Model Architecture Mismatch**
**Problem:** The `structcond_stage_model` expects 8 input channels (4 from latent + 4 from edge features) but was receiving only 4 channels.

**Root Cause:**
- The v3 model has an `EdgeMapProcessor` that converts 3-channel edge maps to 4-channel edge features
- The v2 model doesn't have this processor
- When importing from v2, the edge processing wasn't happening

**Current State:**
- v3 model properly has `EdgeMapProcessor` that converts edge_map [B, 3, H, W] → edge_feat [B, 4, 64, 64]
- Edge features are concatenated with struct_cond [B, 4, H, W] to create 8-channel input
- The forward method signature is: `forward(self, x, edge_map, timesteps)`

### 3. **Config Issue for Inference**
**Problem:** The training config has `ignore_keys: ['structcond_stage_model']` which prevents loading trained weights.

**Fix:**
- Created a new inference-specific config: `configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml`
- This config has `ignore_keys: []` to load all available weights

## How to Use

### Option 1: Use the inference config (Recommended)
```bash
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/your/edge_checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img /path/to/input/images \
    --outdir /path/to/output \
    --ddpm_steps 200
```

### Option 2: Use the training config (if structcond weights are not trained)
```bash
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge_800_32x32.yaml \
    --ckpt /path/to/your/checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img /path/to/input/images \
    --outdir /path/to/output \
    --ddpm_steps 200
```

## Expected Output on Run

When you run the script, you should see debug output like:
```
[INFO] Project root added to sys.path: /root/dp/StableSR_Edge_v3
[INFO] sys.path[0]: /root/dp/StableSR_Edge_v3
[INFO] ldm.models.diffusion.ddpm loaded from: /root/dp/StableSR_Edge_v3/ldm/models/diffusion/ddpm.py
[INFO] ldm.modules.diffusionmodules.openaimodel loaded from: /root/dp/StableSR_Edge_v3/ldm/modules/diffusionmodules/openaimodel.py
```

This confirms that the v3 modules are being used.

## Checkpoint Considerations

### If your checkpoint was trained WITH edge support:
- Use the inference config (Option 1)
- The checkpoint should have `structcond_stage_model` weights with `in_channels=8`
- If `edge_processor` weights are missing, they will be randomly initialized (may need fine-tuning)

### If your checkpoint was trained WITHOUT edge support:
- You'll need to either:
  1. Train/fine-tune the model with edge support first
  2. Use the model without `--use_edge` flag
  3. Fine-tune only the `edge_processor` and `structcond_stage_model` components

## Troubleshooting

### If you still see errors from v2 paths:
```bash
# Clear Python cache and retry
find /root/dp/StableSR_Edge_v3 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py --use_edge [other args]
```

### If you get "missing keys" warnings:
This is normal if the checkpoint doesn't have `edge_processor` weights. The processor will be randomly initialized. For best results, you should fine-tune the model or use a checkpoint that was trained with edge support.

### If you get "unexpected keys" warnings:
This means the checkpoint has extra keys that the current model doesn't expect. Usually safe to ignore.

## Next Steps

1. **Test the fix:**
   ```bash
   cd /root/dp/StableSR_Edge_v3
   python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
       --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
       --ckpt [YOUR_CHECKPOINT_PATH] \
       --use_edge \
       --save_edge \
       --init-img inputs/user_upload \
       --outdir outputs/user_upload \
       --ddpm_steps 200
   ```

2. **Check the debug output** to confirm v3 modules are loaded

3. **Check the edge maps** in the output directory to verify edge detection is working

4. **If results are poor**, you may need to fine-tune the edge_processor component

## Files Modified

1. `/root/dp/StableSR_Edge_v3/scripts/sr_val_ddpm_text_T_vqganfin_edge.py`
   - Added sys.path manipulation
   - Added debug logging

2. `/root/dp/StableSR_Edge_v3/ldm/__init__.py` (created)

3. `/root/dp/StableSR_Edge_v3/ldm/modules/__init__.py` (created)

4. `/root/dp/StableSR_Edge_v3/configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml` (created)




