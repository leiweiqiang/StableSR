# Complete Device Error Resolution Summary

## Issue Overview

Multiple device mismatch errors occurred because tensors and model modules were being created/moved to devices incorrectly.

---

## All Fixes Applied

### **Fix 1: Inference Script - Timestep Tensor Creation**
**File**: `scripts/inference_blended_input.py`
**Line**: 353-354

```python
# BEFORE (caused error):
t_start = repeat(torch.tensor([t_start_idx]), '1 -> b', b=1).to(device).long()

# AFTER (fixed):
t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
t_start = repeat(t_start, '1 -> b', b=1)
```

---

### **Fix 2: Inference Script - Schedule Buffers**
**File**: `scripts/inference_blended_input.py`
**Lines**: 188-202

```python
# After register_schedule(), ensure buffers are on device
device = next(model.parameters()).device

# Move schedule buffers to device
if hasattr(model, 'sqrt_alphas_cumprod'):
    model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
if hasattr(model, 'sqrt_one_minus_alphas_cumprod'):
    model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
if hasattr(model, 'alphas_cumprod'):
    model.alphas_cumprod = model.alphas_cumprod.to(device)

# Ensure entire model (including sub-modules) is on device
# Critical for structcond_stage_model's time_embed module
model = model.to(device)
if hasattr(model, 'structcond_stage_model'):
    model.structcond_stage_model = model.structcond_stage_model.to(device)
```

**Why this is needed**: 
- `register_schedule()` creates new buffer tensors on CPU
- Sub-modules like `time_embed` need to be explicitly moved to device
- `model.to(device)` ensures all sub-modules are on CUDA

---

### **Fix 3: Core Model - t_replace in Sampling Loops**  
**File**: `ldm/models/diffusion/ddpm.py`
**Lines**: 3029, 3173, 6515, 6648 (4 locations)

```python
# BEFORE (caused error):
t_replace = repeat(torch.tensor([self.ori_timesteps[i]]), '1 -> b', b=img.size(0))
t_replace = t_replace.long().to(device)

# AFTER (fixed):
t_replace = torch.tensor([self.ori_timesteps[i]], device=device, dtype=torch.long)
t_replace = repeat(t_replace, '1 -> b', b=img.size(0))
```

**Locations fixed**:
1. First LatentDiffusionSRTextWT - p_sample_loop (line 3029)
2. First LatentDiffusionSRTextWT - progressive_denoising (line 3173)
3. Second LatentDiffusionSRTextWT - p_sample_loop (line 6515)
4. Second LatentDiffusionSRTextWT - progressive_denoising (line 6648)

---

## Root Causes Explained

### **Problem 1: Tensor Creation Pattern**

**Bad pattern** (creates on CPU first):
```python
t = torch.tensor([value])      # Created on CPU by default
t = t.long()                   # Still on CPU
t = t.to(device)               # Move to device (can be delayed)
```

**Good pattern** (creates directly on target device):
```python
t = torch.tensor([value], device=device, dtype=torch.long)
```

---

### **Problem 2: register_schedule() Side Effects**

When `model.register_schedule()` is called:
1. It creates new tensors from numpy arrays
2. These tensors default to **CPU**
3. Even though the model is on CUDA, the new buffers are on CPU

**Solution**: Explicitly move buffers after registration:
```python
model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
```

---

### **Problem 3: Sub-Module Placement**

After operations that modify the model:
- Main model might be on CUDA
- But sub-modules (like `time_embed`, `structcond_stage_model`) might drift to CPU

**Solution**: Re-apply `.to(device)` to entire model and sub-modules:
```python
model = model.to(device)
model.structcond_stage_model = model.structcond_stage_model.to(device)
```

---

## Error Progression & Fixes

### **Error 1: Index Out of Bounds**
```
Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed
```
- **Cause**: Used timestep index 999 after shortening schedule to 200 steps
- **Fix**: Use `len(model.ori_timesteps) - 1` instead of hardcoded 999

---

### **Error 2: Device Mismatch in q_sample**
```
Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```
- **Cause**: `t_start` tensor created on CPU
- **Fix**: Create tensor with `device=device` parameter

---

### **Error 3: Device Mismatch in structcond_stage_model**
```
Expected all tensors to be on the same device, but found cuda:0 and cpu!
```
- **Cause**: `structcond_stage_model.time_embed` module on CPU
- **Fix**: Explicitly move model and sub-modules to device after setup

---

## Complete Fix Checklist

- [x] **Fix 1**: Timestep tensor in inference script (line 353)
- [x] **Fix 2**: Schedule buffers in setup_timesteps (lines 191-196)
- [x] **Fix 3**: Model sub-modules to device (lines 200-202)
- [x] **Fix 4**: t_replace in ddpm.py p_sample_loop (line 3029)
- [x] **Fix 5**: t_replace in ddpm.py p_sample_loop (line 6515)
- [x] **Fix 6**: t_replace in ddpm.py progressive_denoising (line 3173)
- [x] **Fix 7**: t_replace in ddpm.py progressive_denoising (line 6648)

**Total**: 7 fixes across 2 files

---

## Testing

After all fixes, run:

```bash
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/edges_512/ \
    --outdir outputs/test/ \
    --ckpt checkpoints/model.ckpt \
    --blend-alpha 0.5 \
    --ddpm-steps 50  # Start with fewer steps for faster testing
```

**Expected**: Should complete without device errors

---

## Best Practices Learned

### **1. Always specify device when creating tensors**
```python
# ✅ GOOD:
t = torch.tensor([1, 2, 3], device='cuda', dtype=torch.long)
t = torch.zeros((10, 10), device=device)

# ❌ BAD:
t = torch.tensor([1, 2, 3]).to(device)
```

### **2. Move model to device after any structural changes**
```python
# After register_schedule, instantiation, or other modifications:
model = model.to(device)
```

### **3. Explicitly move sub-modules if needed**
```python
model.sub_module = model.sub_module.to(device)
```

### **4. Check device placement in debugging**
```python
print(f"Model device: {next(model.parameters()).device}")
print(f"Tensor device: {my_tensor.device}")
print(f"Sub-module device: {next(model.sub_module.parameters()).device}")
```

---

## Files Modified

| File | Changes | Purpose |
|------|---------|---------|
| `scripts/inference_blended_input.py` | Lines 353, 188-202 | Fix inference script device issues |
| `ldm/models/diffusion/ddpm.py` | Lines 3029, 3173, 6515, 6648 | Fix core sampling device issues |

---

## Summary

**All device mismatch errors should now be resolved!** ✅

The fixes ensure:
- ✅ All tensors created on correct device
- ✅ All schedule buffers on CUDA
- ✅ All model sub-modules on CUDA
- ✅ Works with any ddpm_steps value

**The inference should now run smoothly from start to finish!** 🚀

