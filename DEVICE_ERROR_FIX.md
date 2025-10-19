# Device Mismatch Error - Complete Fix

## The Error

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

**Location**: When calling `model.q_sample()` during initial noise creation

---

## Root Cause

The `model.register_schedule()` function creates new tensors from numpy arrays, and **these tensors default to CPU**:

```python
# In setup_timesteps():
new_betas = [beta.data.cpu().numpy() for beta in new_betas]  # Convert to numpy
model.register_schedule(given_betas=np.array(new_betas), ...)  # Creates new tensors
# ↑ New tensors created on CPU by default!

# Later in q_sample():
model.sqrt_alphas_cumprod  # ← On CPU! ❌
z_blend  # ← On CUDA ✓
t_start  # ← On CUDA ✓

extract_into_tensor(self.sqrt_alphas_cumprod, t, ...)
# ↑ CPU tensor                         ↑ CUDA tensor
# → Device mismatch error!
```

---

## Complete Fix Applied

### **Fix 1: Create timestep tensor on device directly**

```python
# Line 353:
t_start = torch.tensor([t_start_idx], device=device, dtype=torch.long)
#                                     ^^^^^^^^^^^^^^ Create on CUDA!
t_start = repeat(t_start, '1 -> b', b=1)
```

### **Fix 2: Move schedule buffers to device after registration**

```python
# Lines 188-196 in setup_timesteps():
# Ensure schedule buffers are on correct device after registration
device = next(model.parameters()).device
if hasattr(model, 'sqrt_alphas_cumprod'):
    model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
if hasattr(model, 'sqrt_one_minus_alphas_cumprod'):
    model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
if hasattr(model, 'alphas_cumprod'):
    model.alphas_cumprod = model.alphas_cumprod.to(device)
```

---

## Why Both Fixes Are Needed

### **Fix 1 alone** (timestep on device):
- ✓ `t_start` is on CUDA
- ✓ `z_blend` is on CUDA
- ❌ `model.sqrt_alphas_cumprod` still on CPU
- **Still fails!**

### **Fix 2 alone** (schedule on device):
- ✓ `model.sqrt_alphas_cumprod` is on CUDA
- ✓ `z_blend` is on CUDA
- ❌ `t_start` might be on CPU (depending on how it's created)
- **Still fails!**

### **Both fixes together**:
- ✓ `t_start` on CUDA
- ✓ `z_blend` on CUDA
- ✓ `model.sqrt_alphas_cumprod` on CUDA
- ✅ **Works!**

---

## Verification

After applying both fixes, all tensors are on the same device:

```python
print(f"z_blend device: {z_blend.device}")  # cuda:0
print(f"t_start device: {t_start.device}")  # cuda:0
print(f"sqrt_alphas_cumprod device: {model.sqrt_alphas_cumprod.device}")  # cuda:0
# ✅ All on CUDA!
```

---

## Testing

The script should now work:

```bash
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/edges_512/ \
    --outdir outputs/results/ \
    --ckpt checkpoints/your_model.ckpt \
    --blend-alpha 0.5
```

**Expected output:**
```
📁 Detected directory inputs - automatically enabling batch mode
Found 10 LR images and 10 edge maps

Processing 1/10: 0810.png
==========================================
Step 1/8: Loading LR image and edge map...
Step 2/8: Creating blended image (alpha=0.5)...
Step 3/8: Encoding blended image to latent space...
Step 4/8: Computing structural conditioning...
Step 5/8: Preparing text conditioning...
Step 6/8: Setting up timestep schedule (200 steps)...
Step 7/8: Preparing initial state...  ← Should work now!
Step 8/8: Running diffusion sampling...
Decoding to image space...

✓ Saved output: outputs/results/0810.png
✓ Saved blended: outputs/results/0810_blended.png
```

---

## Common Device Issues in PyTorch

### **Issue 1: Tensor created on wrong device**
```python
# WRONG:
t = torch.tensor([1, 2, 3]).to(device)  # Created on CPU, then moved

# CORRECT:
t = torch.tensor([1, 2, 3], device=device)  # Created directly on target device
```

### **Issue 2: Model buffers on CPU after modifications**
```python
# After operations that create new tensors:
model.some_buffer = torch.tensor([...])  # ← Defaults to CPU!

# Fix:
model.some_buffer = torch.tensor([...], device=device)
# Or:
model.some_buffer = model.some_buffer.to(device)
```

### **Issue 3: Numpy → Torch conversion**
```python
# WRONG:
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)  # ← On CPU!

# CORRECT:
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr).to(device)  # Move to device
```

---

## Applied Fixes Summary

| Location | Fix | Why |
|----------|-----|-----|
| Line 353 | Create `t_start` with `device=device` | Ensure timestep tensor on CUDA |
| Lines 188-196 | Move schedule buffers to device | Ensure model buffers on CUDA after registration |

---

**The script is now fully fixed and should work without device errors!** ✅

