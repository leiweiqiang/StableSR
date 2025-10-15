# StableSR Edge v3 - Fixes Applied ✓

## Summary

Your StableSR Edge v3 model had a channel mismatch error because it was importing modules from the v2 directory instead of v3. This has been **FIXED** and **VERIFIED**.

## Test Results ✓

```
✓ SUCCESS: Using v3 modules
✓ EdgeMapProcessor class found and working
✓ EncoderUNetModelWT forward pass works with edge_map
✓ All architecture components verified
✓ Config loading successful
```

## What Was Fixed

### 1. **Python Module Import Path** ✓
- **Issue**: Script was importing from `StableSR_Edge_v2` instead of `v3`
- **Fix**: 
  - Created `/root/dp/StableSR_Edge_v3/ldm/__init__.py`
  - Created `/root/dp/StableSR_Edge_v3/ldm/modules/__init__.py`
  - Added `sys.path.insert(0, project_root)` to the script
  - Added debug logging to verify correct imports

### 2. **Edge Model Architecture** ✓
The v3 model now properly includes:
- `EdgeMapProcessor`: Converts 3-channel edge maps → 4-channel features (512×512 → 64×64)
- `EncoderUNetModelWT`: Accepts 8 channels (4 latent + 4 edge features)
- Proper forward signature: `forward(self, x, edge_map, timesteps)`

### 3. **Inference Configuration** ✓
- Created `configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml`
- Removed `ignore_keys` to allow loading trained weights
- Optimized for inference (no training-specific parameters)

## How to Run Inference

### Option 1: With Edge Support (Recommended)

```bash
cd /root/dp/StableSR_Edge_v3

# Activate environment
source /root/miniconda/etc/profile.d/conda.sh
conda activate sr_edge

# Run inference
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/your/checkpoint.ckpt \
    --vqgan_ckpt /path/to/your/vqgan_checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img inputs/user_upload \
    --outdir outputs/edge_test \
    --ddpm_steps 200 \
    --n_samples 1 \
    --input_size 512
```

### Option 2: Without Edge Support (Fallback)

If you want to test without edge features:

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/your/checkpoint.ckpt \
    --vqgan_ckpt /path/to/your/vqgan_checkpoint.ckpt \
    --init-img inputs/user_upload \
    --outdir outputs/no_edge_test \
    --ddpm_steps 200 \
    --n_samples 1
```
(Note: Without `--use_edge`, the model will use zero edge features)

## Debug Output

When you run the script, you should see:

```
[INFO] Project root added to sys.path: /root/dp/StableSR_Edge_v3
[INFO] sys.path[0]: /root/dp/StableSR_Edge_v3
[INFO] ldm.models.diffusion.ddpm loaded from: /root/dp/StableSR_Edge_v3/ldm/models/diffusion/ddpm.py
[INFO] ldm.modules.diffusionmodules.openaimodel loaded from: /root/dp/StableSR_Edge_v3/ldm/modules/diffusionmodules/openaimodel.py
```

This confirms v3 modules are being used.

## Checkpoint Considerations

### Scenario 1: Checkpoint trained WITH EdgeMapProcessor
- Use inference config (as shown above)
- All components will load correctly
- Edge features will enhance the output

### Scenario 2: Checkpoint trained WITHOUT EdgeMapProcessor
- The checkpoint may be missing `structcond_stage_model.edge_processor.*` weights
- You'll see warnings like: `missing keys: ['structcond_stage_model.edge_processor.backbone.0.weight', ...]`
- The EdgeMapProcessor will use randomly initialized weights
- **Results may be suboptimal** - consider fine-tuning or using a checkpoint with edge support

### Scenario 3: Checkpoint has no structcond_stage_model at all
- Use the training config instead: `configs/stableSRNew/v2-finetune_text_T_512_edge_800_32x32.yaml`
- All structcond components will be randomly initialized
- **Not recommended for inference** - you'll need to train first

## Verification

To verify everything is working before running full inference:

```bash
cd /root/dp/StableSR_Edge_v3
source /root/miniconda/etc/profile.d/conda.sh
conda activate sr_edge
python test_edge_model_loading.py
```

Expected output: All checks should pass with ✓

## Files Created/Modified

### Created:
1. `/root/dp/StableSR_Edge_v3/ldm/__init__.py`
2. `/root/dp/StableSR_Edge_v3/ldm/modules/__init__.py`
3. `/root/dp/StableSR_Edge_v3/configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml`
4. `/root/dp/StableSR_Edge_v3/test_edge_model_loading.py`
5. `/root/dp/StableSR_Edge_v3/EDGE_INFERENCE_FIX.md`
6. `/root/dp/StableSR_Edge_v3/FIXES_APPLIED.md` (this file)

### Modified:
1. `/root/dp/StableSR_Edge_v3/scripts/sr_val_ddpm_text_T_vqganfin_edge.py`
   - Added sys.path manipulation (lines 5-10)
   - Added module verification logging (lines 30-33)

## Troubleshooting

### Issue: Still seeing v2 paths in error messages
**Solution**: Clear Python cache
```bash
cd /root/dp/StableSR_Edge_v3
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

### Issue: "missing keys" warnings for edge_processor
**Expected**: If checkpoint doesn't have edge_processor weights
**Impact**: Edge features will use random initialization - may need fine-tuning
**Solution**: Use a checkpoint trained with edge support, or fine-tune the model

### Issue: RuntimeError about channel mismatch
**Cause**: Likely still using v2 modules
**Solution**: Run `python test_edge_model_loading.py` to verify module paths

## Next Steps

1. ✅ **Verification Complete** - Test script passed
2. ⏭️ **Run Inference** - Use the command above with your checkpoint
3. 📊 **Check Results** - Verify edge maps are saved and output quality
4. 🔧 **Fine-tune** (if needed) - If using checkpoint without edge_processor weights

## Architecture Details

For reference, here's how edge processing works:

```
Input Image [B, 3, H, W] 
    ↓
Edge Extraction (Canny) [B, 3, H, W]
    ↓
EdgeMapProcessor [B, 3, H, W] → [B, 4, 64, 64]
    ↓
Concatenate with struct_cond [B, 4, 64, 64]
    ↓
Combined Input [B, 8, 64, 64]
    ↓
EncoderUNetModelWT → Features
    ↓
UNet Denoising
    ↓
Output [B, 4, 64, 64]
```

The fix ensures this pipeline works correctly by loading v3 modules.

---

**Status**: ✅ FIXED AND VERIFIED

You can now run inference with edge support!




