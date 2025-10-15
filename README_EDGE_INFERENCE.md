# StableSR Edge v3 - Quick Start Guide

## 🎉 Your Issue Has Been Fixed!

The channel mismatch error you encountered has been **resolved**. The problem was that Python was importing modules from `StableSR_Edge_v2` instead of `v3`. All necessary fixes have been applied and verified.

---

## 📋 Quick Start (3 Steps)

### Step 1: Verify Setup
```bash
cd /root/dp/StableSR_Edge_v3
source /root/miniconda/etc/profile.d/conda.sh
conda activate sr_edge
python test_edge_model_loading.py
```

**Expected**: All checks pass with ✓

### Step 2: Inspect Your Checkpoint (Optional but Recommended)
```bash
python inspect_checkpoint.py /path/to/your/checkpoint.ckpt
```

This will tell you if your checkpoint supports edge features.

### Step 3: Run Inference
```bash
python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/your/checkpoint.ckpt \
    --vqgan_ckpt /path/to/your/vqgan_checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img /path/to/input/images \
    --outdir /path/to/output \
    --ddpm_steps 200
```

---

## 📚 Detailed Documentation

- **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - Complete list of fixes and how to use them
- **[EDGE_INFERENCE_FIX.md](EDGE_INFERENCE_FIX.md)** - Technical details about the fix
- **[test_edge_model_loading.py](test_edge_model_loading.py)** - Verification script
- **[inspect_checkpoint.py](inspect_checkpoint.py)** - Checkpoint inspection tool

---

## 🔧 What Was Fixed

1. **Module Import Path** ✓
   - Created missing `__init__.py` files
   - Added `sys.path` manipulation to prioritize v3 imports
   - Added verification logging

2. **Edge Architecture** ✓
   - Verified `EdgeMapProcessor` works correctly
   - Verified `EncoderUNetModelWT` accepts edge_map parameter
   - Confirmed 8-channel input (4 latent + 4 edge)

3. **Configuration** ✓
   - Created inference-optimized config
   - Removed `ignore_keys` to load trained weights

---

## 🎯 Expected Behavior

When you run the inference script, you should see:

```
[INFO] Project root added to sys.path: /root/dp/StableSR_Edge_v3
[INFO] sys.path[0]: /root/dp/StableSR_Edge_v3
[INFO] ldm.models.diffusion.ddpm loaded from: /root/dp/StableSR_Edge_v3/...
[INFO] ldm.modules.diffusionmodules.openaimodel loaded from: /root/dp/StableSR_Edge_v3/...
```

This confirms v3 modules are being used.

---

## 🚨 Common Issues & Solutions

### Issue: "missing keys" warnings for edge_processor

**Cause**: Your checkpoint doesn't have `edge_processor` weights

**Impact**: Edge features will use random initialization

**Solutions**:
1. Use a checkpoint that was trained with edge support
2. Fine-tune the model to train the edge_processor
3. Accept potentially suboptimal results

### Issue: Still seeing v2 paths in errors

**Solution**: Clear Python cache
```bash
find /root/dp/StableSR_Edge_v3 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Input Image [B, 3, 512, 512]                           │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐         ┌────────────────┐
│ VAE Encoder  │         │ Canny Edge     │
│              │         │ Detection      │
└──────┬───────┘         └────────┬───────┘
       │                          │
       ▼                          ▼
┌──────────────┐         ┌────────────────┐
│ Latent       │         │ Edge Map       │
│ [B,4,64,64]  │         │ [B,3,512,512]  │
└──────┬───────┘         └────────┬───────┘
       │                          │
       │                          ▼
       │                  ┌────────────────┐
       │                  │ EdgeMap-       │
       │                  │ Processor      │
       │                  └────────┬───────┘
       │                           │
       │                           ▼
       │                  ┌────────────────┐
       │                  │ Edge Features  │
       │                  │ [B,4,64,64]    │
       │                  └────────┬───────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Concatenate    │
          │ [B,8,64,64]    │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ EncoderUNet-   │
          │ ModelWT        │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ Structural     │
          │ Features       │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ Main UNet      │
          │ Denoising      │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ VAE Decoder    │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ Output Image   │
          │ [B,3,512,512]  │
          └────────────────┘
```

---

## 🧪 Testing Your Setup

### Test 1: Module Loading
```bash
python test_edge_model_loading.py
```
All checks should pass.

### Test 2: Checkpoint Inspection
```bash
python inspect_checkpoint.py /path/to/checkpoint.ckpt
```
This tells you what your checkpoint contains.

### Test 3: Small Inference Run
```bash
# Create a test directory with one image
mkdir -p test_input test_output
# Copy one test image to test_input/

python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/checkpoint.ckpt \
    --vqgan_ckpt /path/to/vqgan_checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img test_input \
    --outdir test_output \
    --ddpm_steps 50 \
    --n_samples 1
```

---

## 📝 Command Reference

### Full Inference Command
```bash
cd /root/dp/StableSR_Edge_v3
source /root/miniconda/etc/profile.d/conda.sh
conda activate sr_edge

python scripts/sr_val_ddpm_text_T_vqganfin_edge.py \
    --config configs/stableSRNew/v2-inference_text_T_512_edge_32x32.yaml \
    --ckpt /path/to/checkpoint.ckpt \
    --vqgan_ckpt /path/to/vqgan_checkpoint.ckpt \
    --use_edge \
    --save_edge \
    --init-img /path/to/input \
    --outdir /path/to/output \
    --ddpm_steps 200 \
    --n_samples 1 \
    --input_size 512 \
    --dec_w 0.5 \
    --colorfix_type adain
```

### Parameters Explained
- `--config`: Model configuration file
- `--ckpt`: Main model checkpoint
- `--vqgan_ckpt`: VAE checkpoint
- `--use_edge`: Enable edge-guided super-resolution
- `--save_edge`: Save extracted edge maps
- `--init-img`: Input image directory
- `--outdir`: Output directory
- `--ddpm_steps`: Number of denoising steps (more = slower but potentially better)
- `--n_samples`: Batch size
- `--input_size`: Input image size (will be resized)
- `--dec_w`: Weight for VQGAN decoder fusion (0.0-1.0)
- `--colorfix_type`: Color correction method (adain, wavelet, or nofix)

---

## ✅ Success Indicators

Your inference is working correctly if you see:

1. ✓ Debug output showing v3 module paths
2. ✓ Edge maps saved in output directory (if --save_edge)
3. ✓ Super-resolved images in output directory
4. ✓ No channel mismatch errors

---

## 🆘 Need Help?

1. Run `python test_edge_model_loading.py` to verify setup
2. Run `python inspect_checkpoint.py your_checkpoint.ckpt` to check checkpoint
3. Check `FIXES_APPLIED.md` for detailed troubleshooting
4. Review error messages for clues

---

## 📈 Next Steps After Success

1. **Experiment with parameters**:
   - Try different `--ddpm_steps` (50, 100, 200)
   - Try different `--dec_w` values (0.0, 0.5, 1.0)
   - Try different color fix methods

2. **Evaluate results**:
   - Compare outputs with and without `--use_edge`
   - Check edge maps to ensure they're meaningful
   - Assess visual quality

3. **Optimize**:
   - If results are poor, consider fine-tuning
   - If speed is an issue, reduce `--ddpm_steps`

---

**Status**: ✅ **READY TO USE**

All fixes have been applied and verified. You can now run edge-guided super-resolution inference!




