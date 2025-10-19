# Training Readiness Checklist for Blended Input

## Executive Summary

**Current Status**: ⚠️ **Additional changes required before training**

The core blended input logic is implemented, but there are critical issues in the `log_images()` method that will cause training to crash during logging/validation.

---

## ✅ What's Already Working

1. ✅ **Core Training Loop**
   - `get_input()` creates blended images correctly
   - `forward()` passes `z_blend` to time-aware encoder
   - `p_losses()` computes loss correctly
   - Optimizer configured properly

2. ✅ **Blended Input Creation**
   - LR downsampling (512→32)
   - Upscaling (32→512)
   - Alpha blending with edge map
   - VAE encoding to `z_blend`

3. ✅ **Model Architecture**
   - Time-aware encoder receives `z_blend`
   - UNET receives `struct_cond` from time-aware encoder
   - All signatures updated

---

## ⚠️ Critical Issues Found

### **Issue 1: log_images() Variable Unpacking** 🔴 **WILL CRASH**

**Location**: Lines 3290, 6765 (both LatentDiffusionSRTextWT classes)

**Problem:**
```python
# Current (WRONG):
z, c_lq, z_gt, edge, x, gt, yrec, xc = self.get_input(...)

# What get_input actually returns:
# [z, text_cond, z_gt, z_blend] + [x, self.gt, xrec] + [xc]
#  └─────┬──────┘  └────┬─────┘     └───────┬────────┘   └┬┘
#     base returns    blend latent    first_stage_outputs  original_cond

# Should be:
z, c_lq, z_gt, z_blend, x, gt, yrec, xc = self.get_input(...)
```

**Impact**: Training will crash when logging images (typically every N steps)

**Fix Required:**
```python
z, c_lq, z_gt, z_blend, x, gt, yrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, val=True)
```

---

### **Issue 2: struct_cond Computation in log_images()** 🔴 **INCORRECT BEHAVIOR**

**Location**: Lines 3304-3308, 6779-6783

**Problem:**
```python
# Current (WRONG):
c = self.cond_stage_model(c_lq)
if self.test_gt:
    struct_cond = z_gt  # ❌ This is just the GT latent!
else:
    struct_cond = z  # ❌ This is just the LR latent!
```

**Why it's wrong**: 
- In training, `struct_cond` is computed by the time-aware encoder from `z_blend`
- In log_images, it's incorrectly using raw latents instead
- This means validation/logging uses different conditioning than training!

**Fix Required:**
```python
c = self.cond_stage_model(c_lq)
# Compute struct_cond using time-aware encoder (matching training)
t_ori = torch.tensor([self.ori_timesteps[0]] * N).long().to(self.device)
if self.test_gt:
    struct_cond = self.structcond_stage_model(z_gt, z_blend, t_ori)
else:
    struct_cond = self.structcond_stage_model(z, z_blend, t_ori)
```

---

### **Issue 3: edge_map References in Sampling Calls** 🟡 **LEGACY PARAMETER**

**Location**: Multiple lines in both classes

**Examples:**
```python
# Line 3338, 6813:
samples, z_denoise_row = self.sample(..., edge_map=edge)

# Line 3350, 6825:
samples, z_denoise_row = self.sample_log(..., edge_map=edge)

# Lines 3365, 3373, 6840, 6848:
samples, _ = self.sample_log(..., edge_map=edge)
```

**Problem**: 
- Variable `edge` doesn't exist after we rename it to `z_blend`
- `edge_map` parameter is legacy and not actually used in current sampling logic

**Fix Options:**
- **Option A**: Remove `edge_map=edge` from all calls (recommended)
- **Option B**: Pass `edge_map=z_blend` if keeping parameter for compatibility

---

## 📋 Required Changes Summary

| # | Change | Location | Priority | Blocks Training? |
|---|--------|----------|----------|-----------------|
| 1 | Fix log_images() unpacking: `edge` → `z_blend` | Lines 3290, 6765 | 🔴 Critical | **YES** |
| 2 | Fix struct_cond computation in log_images() | Lines 3304-3308, 6779-6783 | 🔴 Critical | Incorrect validation |
| 3 | Add blended_input to log dict | Lines 3302, 6777 | 🟡 Nice to have | No |
| 4 | Update edge_map→z_blend in sample() calls | Lines 3338, 6813 | 🔴 Critical | **YES** |
| 5 | Update edge_map in sample_log() calls | Lines 3350, 6825, 3365, 3373, 6840, 6848 | 🔴 Critical | **YES** |

---

## Detailed Fix Plan

### **Fix 1: First LatentDiffusionSRTextWT.log_images() (Line ~3290)**

```python
# Change line 3290:
z, c_lq, z_gt, z_blend, x, gt, yrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, val=True)

# Add after line 3302:
log["blended_input"] = yrec  # Show the blended image

# Replace lines 3304-3308:
c = self.cond_stage_model(c_lq)
t_ori = torch.tensor([self.ori_timesteps[0]] * N).long().to(self.device)
if self.test_gt:
    struct_cond = self.structcond_stage_model(z_gt, z_blend, t_ori)
else:
    struct_cond = self.structcond_stage_model(z, z_blend, t_ori)

# Update line 3338 (remove edge_map):
samples, z_denoise_row = self.sample(cond=c, struct_cond=struct_cond, batch_size=N, 
                                     timesteps=cur_time_step, return_intermediates=True, 
                                     time_replace=self.time_replace)

# Update line 3350 (remove edge_map):
samples, z_denoise_row = self.sample_log(cond=c,struct_cond=struct_cond,batch_size=N,ddim=use_ddim,
                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                         quantize_denoised=True, x_T=x_T)

# Update lines 3365, 3373 (remove edge_map):
samples, _ = self.sample_log(cond=c, struct_cond=struct_cond, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
```

### **Fix 2: Second LatentDiffusionSRTextWT.log_images() (Line ~6765)**

**Apply identical changes to:**
- Line 6765: unpacking
- Lines 6779-6783: struct_cond computation
- Lines 6813, 6825, 6840, 6848: remove edge_map parameter

---

## Implementation Strategy

### **Recommended Approach:**

Since the changes are in 2 classes (first and second LatentDiffusionSRTextWT), and each has identical patterns:

1. **Manually edit using line numbers** for precision
2. **Test after each class** to ensure no syntax errors
3. **Remove edge_map parameters systematically**

### **Alternative Approach:**

Use `sed` or direct line-based editing to make precise changes at specific line numbers.

---

## After These Changes

### **Training will:**
✅ Start successfully  
✅ Compute loss correctly  
✅ Log images without crashing  
✅ Use correct struct_cond during validation  
✅ Save checkpoints properly

### **You'll be able to:**
✅ Monitor training progress  
✅ View blended inputs in logs  
✅ Compare inputs vs outputs  
✅ Validate the model properly

---

## Testing After Changes

```bash
# Quick syntax check
python -c "from ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT; print('✓ Import successful')"

# Start training (will test logging at first checkpoint)
python main.py --base configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml -t --gpus 0,
```

---

## Would You Like Me To:

1. ✅ **Implement all critical fixes now** (recommended)
2. ⏸️ Wait for your review before proceeding
3. 📝 Create a detailed patch file for manual application

Please let me know how you'd like to proceed!

