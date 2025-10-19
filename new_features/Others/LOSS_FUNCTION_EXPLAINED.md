# Loss Function Explanation

## Where is the Loss Function?

The loss function is in the **`p_losses()` method** of the `LatentDiffusionSRTextWT` class.

**Location**: `ldm/models/diffusion/ddpm.py`
- First LatentDiffusionSRTextWT class: **Line 2519**
- Second LatentDiffusionSRTextWT class: **Line 6019**

---

## Method Signature

```python
def p_losses(self, x_start, cond, struct_cond, t, t_ori, z_gt, z_blend=None, noise=None):
```

**Parameters:**
- `x_start`: Ground truth latent (z_gt)
- `cond`: Text conditioning embeddings
- `struct_cond`: Structural conditioning from time-aware encoder
- `t`: Timestep for diffusion
- `t_ori`: Original timestep for time-aware encoder
- `z_gt`: Ground truth latent (same as x_start)
- `z_blend`: Blended input latent (currently not used in loss, but available)
- `noise`: Optional pre-generated noise

---

## Loss Function Breakdown

### **Step 1: Create Noisy Input**
```python
noise = torch.randn_like(x_start)  # Random Gaussian noise
x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
```
- Forward diffusion: Add noise to ground truth
- Formula: `x_noisy = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise`

### **Step 2: UNET Prediction**
```python
model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)
```
- Pass noisy latent through UNET
- UNET tries to predict either noise, clean image, or velocity

### **Step 3: Determine Target**
```python
if self.parameterization == "x0":
    target = x_start  # Predict clean image directly
elif self.parameterization == "eps":
    target = noise  # Predict the noise (most common)
elif self.parameterization == "v":
    target = self.get_v(x_start, noise, t)  # Predict velocity
```

**Default**: Usually "eps" (noise prediction)

### **Step 4: Compute Main Loss**
```python
loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
```

**`get_loss()` function** (Line 419):
```python
def get_loss(self, pred, target, mean=True):
    if self.loss_type == 'l1':
        loss = (target - pred).abs()  # L1 / MAE
    elif self.loss_type == 'l2':
        loss = F.mse_loss(target, pred)  # L2 / MSE
```

**Common**: L2 loss (MSE) is typically used

### **Step 5: Apply Weighting**
```python
logvar_t = self.logvar[t].to(self.device)
loss = loss_simple / torch.exp(logvar_t) + logvar_t
loss = self.l_simple_weight * loss.mean()
```
- Adaptive loss weighting based on timestep
- `l_simple_weight`: Scaling factor (usually 1.0)

### **Step 6: Add VLB Loss**
```python
loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
loss += (self.original_elbo_weight * loss_vlb)
```
- Variational Lower Bound loss
- Usually weighted very low (original_elbo_weight ≈ 0)

### **Step 7: Optional Edge Loss** (if enabled)
```python
if self.use_edge_loss and self.training:
    # Decode predicted and GT to image space
    pred_img = self.differentiable_decode_first_stage(pred_x0)
    gt_img = self.differentiable_decode_first_stage(x_start)
    
    # Extract edge maps
    pred_edge = self.edge_generator.generate_from_tensor(pred_img.detach())
    gt_edge = self.edge_generator.generate_from_tensor(gt_img.detach())
    
    # Compute edge loss
    edge_loss = F.mse_loss(pred_edge_norm, gt_edge_norm)
    
    # Add to total loss
    loss = loss + self.edge_loss_weight * edge_loss
```
- **Optional**: Controlled by `edge_loss_weight` parameter
- Ensures predicted edges match ground truth edges

---

## Complete Loss Formula

```
Total Loss = l_simple_weight * (weighted_diffusion_loss) 
           + original_elbo_weight * loss_vlb
           + edge_loss_weight * edge_loss (if enabled)

Where:
  weighted_diffusion_loss = (loss_simple / exp(logvar_t) + logvar_t)
  loss_simple = MSE(model_output, target) or L1(model_output, target)
  target = noise (if parameterization=="eps")
         = x_start (if parameterization=="x0")
         = v (if parameterization=="v")
```

---

## Loss Function Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    p_losses()                            │
│                                                          │
│  1. Create Noisy Input                                  │
│     x_start (GT) + noise → x_noisy                      │
│                                                          │
│  2. UNET Forward                                        │
│     UNET(x_noisy, t, text_cond, struct_cond)           │
│         ↓                                                │
│     model_output (predicted noise/x0/v)                 │
│                                                          │
│  3. Compute Target                                      │
│     target = noise (or x_start or v)                    │
│                                                          │
│  4. Main Diffusion Loss                                 │
│     loss_simple = MSE(model_output, target)             │
│                                                          │
│  5. Apply Weighting                                     │
│     loss = loss_simple / exp(logvar_t) + logvar_t       │
│                                                          │
│  6. Add VLB Loss                                        │
│     loss += original_elbo_weight * loss_vlb             │
│                                                          │
│  7. Optional Edge Loss                                  │
│     if use_edge_loss:                                   │
│         loss += edge_loss_weight * edge_loss            │
│                                                          │
│  8. Return                                              │
│     return loss, loss_dict                              │
└─────────────────────────────────────────────────────────┘
```

---

## Loss Components Explained

### **1. Diffusion Loss (Main Loss)**
- **Purpose**: Train UNET to denoise/predict
- **Type**: MSE (L2) or MAE (L1)
- **Target**: Depends on parameterization
  - "eps": Predict noise → MSE(predicted_noise, actual_noise)
  - "x0": Predict clean image → MSE(predicted_x0, actual_x0)
- **Weight**: Typically 1.0

### **2. VLB Loss** 
- **Purpose**: Variational lower bound from diffusion theory
- **Type**: Same as main loss
- **Weight**: Very small (often ~0.001)
- **Impact**: Minimal in practice

### **3. Edge Loss** (Optional)
- **Purpose**: Encourage edge preservation
- **Type**: MSE between edge maps
- **Enabled**: If `edge_loss_weight > 0` in config
- **Weight**: Configurable (e.g., 0.1)

---

## How Loss is Used

### **Training Flow:**
```python
# In training_step():
loss, loss_dict = self.shared_step(batch)  # Calls p_losses internally
# PyTorch Lightning automatically:
# 1. Calls loss.backward()
# 2. Updates optimizer
# 3. Logs loss_dict values
```

### **Call Stack:**
```
training_step()
    ↓
shared_step(batch)
    ↓
get_input(batch) → [x, c, gt, z_blend]
    ↓
forward(x, c, z_blend, gt)
    ↓
p_losses(gt, c, struct_cond, t, t_ori, x, z_blend)
    ↓
loss, loss_dict
```

---

## Logged Loss Values

```python
loss_dict = {
    'train/loss_simple': simple_loss.mean(),      # Main MSE/L1 loss
    'train/loss_vlb': vlb_loss,                   # VLB term
    'train/loss': total_loss,                     # Total combined loss
    'train/loss_edge': edge_loss (if enabled),    # Optional edge loss
    'logvar': self.logvar.data.mean() (if learning)
}
```

---

## Configuration

**In config file:**
```yaml
model:
  params:
    # Loss type (typically not specified, defaults to 'l2')
    # Parameterization (usually 'eps' for noise prediction)
    
    # Optional edge loss
    edge_loss_weight: 0.0  # Set > 0 to enable edge loss
```

**Default behavior**: MSE loss on noise prediction (eps parameterization)

---

## Summary

**The loss function is located in `p_losses()` method at line 2519.**

**It computes:**
```
Total Loss = MSE(predicted_noise, actual_noise)  ← Main loss
           + small_weight * VLB_loss             ← Theoretical term
           + edge_weight * edge_loss (optional)  ← Edge preservation
```

**The loss trains the UNET to predict noise, which allows it to denoise images and perform super-resolution guided by the blended input (LR upscaled + edge) through struct_cond.**


