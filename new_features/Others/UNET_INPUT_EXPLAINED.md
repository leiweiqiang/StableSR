# What is the Input to Stable Diffusion UNET?

## Direct Answer

The **Stable Diffusion UNET** receives the following inputs:

### **Primary Input (x)**
```python
x_noisy  # Shape: [N, 4, 64, 64]
```
- **What it is**: Noised ground truth latent
- **How it's created**: `x_noisy = q_sample(x_start=z_gt, t=t, noise=noise)`
- **Purpose**: The corrupted latent that UNET needs to denoise
- **Channels**: 4 (latent space channels from VAE)
- **Resolution**: 64×64 (for 512×512 output images)

---

## Complete UNET Input Specification

### **UNetModelDualcondV2.forward() Signature**
```python
def forward(self, x, timesteps=None, context=None, struct_cond=None, y=None, **kwargs):
    """
    :param x: [N x 4 x 64 x 64] - Noisy latent tensor (PRIMARY INPUT)
    :param timesteps: [N] - Diffusion timesteps (which noise level)
    :param context: [N x 77 x 1024] - Text embeddings (for cross-attention)
    :param struct_cond: dict of multi-scale features (structural conditioning)
    :param y: [N] - Class labels (if class-conditional, not used here)
    """
```

---

## Detailed Breakdown

### **1. Primary Input: x (Noisy Latent)**

**What is x?**
```
Ground Truth Image (512×512, 3ch)
        ↓
    VAE Encoder
        ↓
z_gt (64×64, 4ch)  ← Ground truth in latent space
        ↓
Add Noise (forward diffusion)
        ↓
x_noisy (64×64, 4ch) ← THIS IS THE PRIMARY INPUT TO UNET
```

**Code:**
```python
# In p_losses() method
x_start = z_gt  # GT latent (4ch, 64×64)
noise = torch.randn_like(x_start)  # Random noise
x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # Add noise

# Then pass to UNET
model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)
```

**Properties:**
- **Shape**: `[Batch, 4, 64, 64]`
- **Type**: Noised latent representation
- **Value Range**: Approximately [-4, 4] (latent space, not normalized)
- **Physical Meaning**: GT image with added Gaussian noise at timestep t

---

### **2. Timesteps: t**

**What is it?**
- Indicates the noise level in the diffusion process
- **Range**: 0 to 999 (for 1000-step diffusion)
- **t=0**: Almost no noise (clean image)
- **t=999**: Maximum noise (pure Gaussian noise)

**Usage in UNET:**
```python
t_emb = timestep_embedding(timesteps, model_channels)  # Sinusoidal embedding
emb = time_embed(t_emb)  # MLP projection
# Then injected into every ResBlock
```

---

### **3. Context: Text Embeddings (Cross-Attention)**

**What is it?**
- Text prompt embeddings from CLIP or similar encoder
- **Shape**: `[Batch, 77, 1024]`
  - 77 = max token length
  - 1024 = CLIP embedding dimension
- **Purpose**: Semantic guidance ("high quality", "detailed", etc.)

**Usage in UNET:**
- Passed to **SpatialTransformer** blocks
- Used in **cross-attention** layers
- Allows text to guide the generation/denoising

---

### **4. Struct_Cond: Structural Conditioning** ⭐

**What is it?**
- Multi-scale features from the time-aware encoder
- Contains structural information from blended input (LR upscaled + edge)

**Structure:**
```python
struct_cond = {
    '64': features_64,  # [N, 256, 64, 64]
    '32': features_32,  # [N, 256, 32, 32]
    '16': features_16,  # [N, 256, 16, 16]
    '8': features_8     # [N, 256, 8, 8]
}
```

**How it's created:**
```
Blended Image (LR upscaled + edge, 512×512)
        ↓
    VAE Encoder
        ↓
z_blend (64×64, 4ch)
        ↓
EncoderUNetModelWT (Time-Aware Encoder)
        ↓
Multi-scale features (struct_cond)
```

**Usage in UNET:**
- Injected into **SpatialTransformer** blocks at matching resolutions
- Provides structural guidance alongside text guidance
- **Key feature of StableSR**: Combines semantic (text) + structural (edge/LR) conditioning

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    STABLE DIFFUSION UNET                         │
│                  (UNetModelDualcondV2)                          │
│                                                                  │
│  PRIMARY INPUT:                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │ x_noisy (Noised GT Latent)           │                      │
│  │ Shape: [N, 4, 64, 64]                │                      │
│  │                                       │                      │
│  │ Created from:                         │                      │
│  │ GT Image → VAE → z_gt → Add Noise    │                      │
│  └──────────────────────────────────────┘                      │
│                                                                  │
│  CONDITIONING INPUTS:                                           │
│  ┌──────────────────────────────────────┐                      │
│  │ 1. timesteps: [N]                    │                      │
│  │    - Which diffusion step (0-999)    │                      │
│  │    - Embedded and injected to blocks │                      │
│  └──────────────────────────────────────┘                      │
│                                                                  │
│  ┌──────────────────────────────────────┐                      │
│  │ 2. context: [N, 77, 1024]            │                      │
│  │    - Text embeddings from CLIP       │                      │
│  │    - Used in cross-attention         │                      │
│  │    - Semantic guidance               │                      │
│  └──────────────────────────────────────┘                      │
│                                                                  │
│  ┌──────────────────────────────────────┐                      │
│  │ 3. struct_cond: dict                 │ ⭐ StableSR Feature  │
│  │    {                                 │                      │
│  │      '64': [N, 256, 64, 64],        │                      │
│  │      '32': [N, 256, 32, 32],        │                      │
│  │      '16': [N, 256, 16, 16],        │                      │
│  │      '8':  [N, 256, 8, 8]           │                      │
│  │    }                                 │                      │
│  │    - Multi-scale structural features │                      │
│  │    - From time-aware encoder         │                      │
│  │    - Blended LR+edge information     │                      │
│  └──────────────────────────────────────┘                      │
│                                                                  │
│  OUTPUT:                                                         │
│  ┌──────────────────────────────────────┐                      │
│  │ model_output: [N, 4, 64, 64]         │                      │
│  │                                       │                      │
│  │ Depending on parameterization:       │                      │
│  │ - "eps": predicted noise             │                      │
│  │ - "x0": predicted clean latent       │                      │
│  │ - "v": velocity prediction           │                      │
│  └──────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from Standard Stable Diffusion

### **Standard Stable Diffusion UNET:**
```python
forward(x, timesteps, context)
# x: noisy latent
# timesteps: diffusion step
# context: text embeddings
```

### **StableSR UNET (UNetModelDualcondV2):** ⭐
```python
forward(x, timesteps, context, struct_cond)
# x: noisy latent
# timesteps: diffusion step
# context: text embeddings
# struct_cond: structural features ← ADDITIONAL INPUT!
```

**The key innovation**: Adding `struct_cond` provides structural guidance from blended LR+edge information, enabling better super-resolution quality.

---

## How x_noisy is Created

### **Step-by-Step:**

```python
# 1. Start with GT latent
z_gt = encode_first_stage(gt_image)  # [N, 4, 64, 64]

# 2. Sample random timestep
t = random.randint(0, 999)  # Which noise level

# 3. Generate noise
noise = torch.randn_like(z_gt)  # [N, 4, 64, 64]

# 4. Forward diffusion (add noise)
x_noisy = sqrt(alpha_bar_t) * z_gt + sqrt(1 - alpha_bar_t) * noise

# 5. Pass to UNET
model_output = UNET(x_noisy, t, text_emb, struct_cond)
```

**In code:**
```python
def p_losses(self, x_start, cond, struct_cond, t, t_ori, z_gt, z_blend=None, noise=None):
    noise = torch.randn_like(x_start)  # Random noise
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # Add noise
    model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)  # UNET call
    # ...
```

---

## Summary Table

| Input Parameter | Shape | Description | Source |
|----------------|-------|-------------|--------|
| **x** (x_noisy) | [N, 4, 64, 64] | Noised GT latent | GT → VAE → Add noise |
| **timesteps** | [N] | Diffusion timestep | Random sampling |
| **context** | [N, 77, 1024] | Text embeddings | Text → CLIP encoder |
| **struct_cond** | dict of features | Structural features | Blended(LR+edge) → VAE → Time-aware encoder |

---

## The Answer in One Sentence

**The Stable Diffusion UNET's primary input is `x_noisy` - a noised version of the ground truth latent (4-channel, 64×64), along with timestep embeddings, text conditioning, and structural conditioning from the blended LR+edge input.**

---

## Why This Design?

1. **x_noisy**: The corrupted data that needs to be cleaned (core diffusion task)
2. **timesteps**: Tells UNET how much noise is present
3. **context**: Text provides semantic guidance
4. **struct_cond**: Blended LR+edge provides structural guidance ⭐

This multi-conditional design allows the UNET to generate high-quality super-resolution outputs guided by both semantic (text) and structural (blended LR+edge) information!

