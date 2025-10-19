# struct_cond Fix: Pass Latent, Not Features

## The Error

```
AttributeError: 'dict' object has no attribute 'type'
File "ldm/models/diffusion/ddpm.py", line 6532, in p_sample_loop
    struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
File "ldm/modules/diffusionmodules/openaimodel.py", line 1602, in forward
    h = h_input.type(self.dtype)
```

---

## Root Cause

### **The Confusion:**

In the inference script, we were **pre-computing** the structural features:

```python
# WRONG approach:
# Step 1: Compute features beforehand
struct_cond = model.structcond_stage_model(z_blend, t_ori)  # Returns dict

# Step 2: Pass dict to sample()
model.sample(cond=semantic_c, struct_cond=struct_cond, ...)
                              ^^^^^^^^^^^
                              This is a dict!

# Step 3: In sampling loop, tries to call structcond_stage_model again
struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
                                                 ^^^^^^^^^^^
                                                 But struct_cond is a dict, not a tensor!
```

---

## The Correct Approach

### **What sample() Expects:**

The `sample()` method expects `struct_cond` to be the **input latent** (z_blend), not pre-computed features.

**Why?** The time-aware encoder needs to be called **at each timestep** during sampling with the current timestep value.

```python
# CORRECT approach:
# Pass z_blend as struct_cond
model.sample(
    cond=semantic_c,
    struct_cond=z_blend,  # ← Pass the latent, not features!
    ...
)

# Inside sampling loop (p_sample_loop):
for i in range(timesteps):
    t_replace = torch.tensor([ori_timesteps[i]], device=device)
    
    # Call time-aware encoder at THIS timestep
    struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
    #                                                ^^^^^^^^^^   ^^^^^^^^^
    #                                                z_blend      current timestep
    
    # Use features for this sampling step
    ...
```

---

## Understanding the Sampling Loop

### **Why Time-Aware Encoder is Called in Loop:**

```
Sampling Loop (200 steps):
├─ Step 0 (t=999):
│  ├─ Call structcond_stage_model(z_blend, t=999)
│  ├─ Get features conditioned on t=999
│  └─ Denoise with these features
│
├─ Step 1 (t=995):
│  ├─ Call structcond_stage_model(z_blend, t=995)  ← Different timestep!
│  ├─ Get features conditioned on t=995
│  └─ Denoise with these features
│
├─ Step 2 (t=990):
│  ├─ Call structcond_stage_model(z_blend, t=990)
│  └─ ...
│
└─ ... continues for all timesteps
```

**Key insight**: The time-aware encoder produces **different features at different timesteps**, so it must be called at each step!

---

## The Fix

### **Before (WRONG):**

```python
# Pre-compute features
struct_cond = model.structcond_stage_model(z_blend, t_ori)  # Dict

# Pass dict to sample
model.sample(cond=semantic_c, struct_cond=struct_cond, ...)
```

### **After (CORRECT):**

```python
# Don't pre-compute - just prepare the latent
# No need to call structcond_stage_model here!

# Pass latent to sample - it will call structcond_stage_model internally
model.sample(cond=semantic_c, struct_cond=z_blend, ...)
              ^^^^^^^^^^^
              Pass z_blend directly!
```

---

## Updated Inference Flow

```
1. Load LR (32×32) and Edge (512×512)
2. Create blended image (512×512)
3. Encode to latent → z_blend (4ch, 64×64)
4. Get text conditioning
5. Setup timesteps
6. Create initial noise
7. Sample:
   ├─ Pass z_blend as struct_cond
   ├─ Sampling loop calls structcond_stage_model(z_blend, t) at each step
   └─ Returns denoised samples
8. Decode to image
9. Save output
```

---

## Code Changes

### **In inference_single_image() function:**

**Removed:**
```python
# OLD (removed this):
t_ori = torch.tensor([model.ori_timesteps[0]]).long().to(device)
with torch.no_grad():
    struct_cond = model.structcond_stage_model(z_blend, t_ori)
print(f"  Struct_cond resolutions: {list(struct_cond.keys())}")
```

**Changed:**
```python
# NEW (simplified):
# Just note that z_blend will be used
print("Preparing structural conditioning input...")
print("Using blended latent (z_blend) as structural conditioning input")
print("The time-aware encoder will be called during sampling loop")

# Later, pass z_blend to sample():
model.sample(cond=semantic_c, struct_cond=z_blend, ...)
```

---

## Why This is Correct

### **Training vs Inference:**

**Training (forward pass):**
```python
def forward(self, x, c, z_blend, gt):
    # Compute struct_cond once for this batch
    struc_c = self.structcond_stage_model(x, z_blend, t_ori)
    return self.p_losses(gt, c, struc_c, ...)
```
→ Computes once per forward pass

**Inference (sampling):**
```python
def p_sample_loop(..., struct_cond):  # struct_cond = z_blend
    for i in timesteps:
        t_replace = current_timestep[i]
        # Compute features at THIS timestep
        struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
        # Denoise one step
        img = self.p_sample(img, cond, struct_cond_input, ...)
```
→ Computes at each timestep with time-varying features

---

## Summary

**The fix:** Pass `z_blend` as `struct_cond` to `model.sample()`, not pre-computed features.

**Why:** The sampling loop needs to call the time-aware encoder at each timestep with the appropriate timestep value.

**Result:** The time-aware encoder produces time-varying structural features throughout the diffusion process. ✅

---

This should resolve the AttributeError and allow inference to complete successfully!

