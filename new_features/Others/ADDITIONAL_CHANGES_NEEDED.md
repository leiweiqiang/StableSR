# Additional Changes Needed for Training

## Analysis Results

After reviewing the codebase, here are the additional modifications needed to fully support the blended input approach:

---

## ✅ Already Working (No Changes Needed)

### **1. Training Loop**
- ✅ `shared_step()` - Updated to use `z_blend`
- ✅ `forward()` - Updated to use `z_blend`
- ✅ `p_losses()` - Updated signature
- ✅ `get_input()` - Creates blended input
- ✅ `structcond_stage_model` call - Receives `z_blend`

### **2. Optimizer Configuration**
- ✅ `configure_optimizers()` - No changes needed
- ✅ Already optimizes: UNET + text encoder + struct encoder

---

## ⚠️ Changes Required

### **1. log_images() Method** ⚠️ **CRITICAL**

**Location**: Lines 3290, 6765 (both LatentDiffusionSRTextWT classes)

**Current Code:**
```python
z, c_lq, z_gt, edge, x, gt, yrec, xc = self.get_input(batch, ...)
```

**Issue**: Variable named `edge` but it's actually `z_blend` now!

**Required Change:**
```python
z, c_lq, z_gt, z_blend, x, gt, yrec, xc = self.get_input(batch, ...)
```

**Additional Issue in struct_cond assignment:**
```python
# Current (WRONG for our new setup):
if self.test_gt:
    struct_cond = z_gt
else:
    struct_cond = z

# Should be (to match training):
t_ori = torch.tensor([self.ori_timesteps[0]] * N).long().to(self.device)
if self.test_gt:
    struct_cond = self.structcond_stage_model(z_gt, z_blend, t_ori)
else:
    struct_cond = self.structcond_stage_model(z, z_blend, t_ori)
```

**Why**: In training, struct_cond is computed by time-aware encoder. In log_images, it should be too!

---

### **2. Sampling Methods with edge_map Parameter** ⚠️

**Files to check**: All sampling methods that pass `edge_map`

**Found references in:**
- `sample()` method - line 3338, 6813
- `sample_log()` method - line 3350, 6825
- Inpainting paths - lines 3365, 3373, 6840, 6848

**Current Pattern:**
```python
samples = self.sample(..., edge_map=edge)
```

**Issue**: These methods have `edge_map` parameter but:
1. It's not currently used in the sampling logic (only passed around)
2. With our changes, we don't need to pass it separately anymore
3. The struct_cond is already computed from z_blend

**Options:**
- **Option A** (Recommended): Remove `edge_map` parameter from all sampling methods since struct_cond already contains the information
- **Option B**: Keep for backward compatibility but ignore it

---

### **3. Canvas/Tiled Inference Methods**

**Location**: `p_mean_variance_canvas()` and related methods

**Found**: Lines with edge_map parameter handling

**Current Code** (example line 2679, 6178):
```python
def p_mean_variance_canvas(..., edge_map=None):
```

**Issue**: These methods check for `edge_map` and process it for tiling

**Required**: 
- These don't need changes for basic training
- Only needed if you use tiled/canvas inference
- Can be addressed later when doing inference

---

## 📋 Detailed Change Plan

### **Priority 1: Critical for Training** 🔴

#### **Change 1.1: Fix log_images() unpacking**
**Both classes at lines ~3290 and ~6765**

```python
# OLD:
z, c_lq, z_gt, edge, x, gt, yrec, xc = self.get_input(...)

# NEW:
z, c_lq, z_gt, z_blend, x, gt, yrec, xc = self.get_input(...)
```

#### **Change 1.2: Fix struct_cond computation in log_images()**
**Both classes at lines ~3304-3308 and ~6779-6783**

```python
# OLD:
c = self.cond_stage_model(c_lq)
if self.test_gt:
    struct_cond = z_gt
else:
    struct_cond = z

# NEW:
c = self.cond_stage_model(c_lq)
# Compute struct_cond using time-aware encoder (same as training)
t_ori = torch.tensor([self.ori_timesteps[0]] * N).long().to(self.device)
if self.test_gt:
    struct_cond = self.structcond_stage_model(z_gt, z_blend, t_ori)
else:
    struct_cond = self.structcond_stage_model(z, z_blend, t_ori)
```

#### **Change 1.3: Update edge variable references in log_images()**
**Lines ~3338, 3350, 3365, 3373 (first class)**
**Lines ~6813, 6825, 6840, 6848 (second class)**

```python
# OLD:
..., edge_map=edge)

# NEW: Remove edge_map parameter from calls
# The struct_cond already contains the blended information
...) # Remove edge_map parameter
```

---

### **Priority 2: Nice to Have** 🟡

#### **Change 2.1: Log blended image in visualization**

Add to `log_images()`:
```python
# Decode z_blend to show the blended input
log["blended_input"] = self.decode_first_stage(z_blend)
```

#### **Change 2.2: Clean up edge_map parameters from method signatures**

Once confirmed working, remove unused `edge_map` parameters from:
- `sample()` method
- `sample_log()` method  
- `p_sample()` method
- `p_sample_canvas()` method
- `p_mean_variance_canvas()` method

---

### **Priority 3: For Inference (Not Training)** 🔵

#### **Change 3.1: Update inference scripts**

Check these files:
- `scripts/sr_val_ddpm_text_T_vqganfin_old.py`
- `scripts/sr_val_edge_inference.py`
- `scripts/inference_edge_to_image.py`
- `app.py` (if it exists)

These might call the model differently and need z_blend handling.

---

## Summary Table

| Change | Priority | Location | Required for Training? |
|--------|----------|----------|----------------------|
| Fix log_images() unpacking | 🔴 Critical | Lines 3290, 6765 | **YES** |
| Fix struct_cond in log_images() | 🔴 Critical | Lines 3304-3308, 6779-6783 | **YES** |
| Update edge_map→z_blend in calls | 🔴 Critical | Multiple sample calls | **YES** |
| Log blended image | 🟡 Nice to have | log_images() | No |
| Clean up edge_map params | 🟡 Nice to have | Multiple methods | No |
| Update inference scripts | 🔵 For inference | scripts/ folder | No (only for inference) |

---

## Quick Diagnosis

**Can you train without these changes?**
- ❌ **NO** - The log_images() method will crash due to incorrect unpacking
- Training will fail when trying to log samples

**What must be fixed first:**
1. ✅ log_images() unpacking (both classes)
2. ✅ struct_cond computation in log_images() (both classes)
3. ✅ Remove edge_map from sampling calls in log_images()

---

## Next Steps

I'll implement the **Priority 1 (Critical)** changes now so training can work properly.

