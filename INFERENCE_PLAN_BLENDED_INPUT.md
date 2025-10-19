# Inference Plan: Blended Input (LR 32x32 + Edge 512x512)

## Goal

Create an inference pipeline that:
1. **Input**: LR image (32×32) + Edge map (512×512)
2. **Process**: Upscale LR to 512×512 → Alpha blend with edge map
3. **Output**: High-quality SR image (512×512)

---

## Current vs New Inference Flow

### **Current Inference Flow** (predict.py, sr_val_edge_inference.py)
```
LR Image (any size)
    ↓
Upscale to 512×512 (bicubic)
    ↓
Generate edge map from upscaled LR
    ↓
VAE encode → init_latent
    ↓
Use init_latent as struct_cond (WRONG - should use time-aware encoder!)
    ↓
Sample/Denoise
    ↓
Output 512×512
```

**Issues with current scripts:**
- Don't use time-aware encoder for struct_cond
- Don't support blended input
- Assume LR image is input, not 32×32

---

### **Proposed New Inference Flow** ⭐

```
Input LR 32×32          Input Edge 512×512
    ↓                          ↓
Upscale to 512×512 ────────────┤
(bicubic)                      │
    ↓                          ↓
LR Upscaled 512×512     Edge Map 512×512
    │                          │
    └──────────┬───────────────┘
               ↓
       Alpha Blending
    (α * LR + (1-α) * Edge)
               ↓
    Blended Image 512×512
               ↓
        VAE Encode
               ↓
      z_blend (4ch, 64×64)
               ↓
    Time-Aware Encoder
  (EncoderUNetModelWT)
               ↓
      struct_cond
               ↓
    Diffusion Sampling
  (Start from noise or
   noisy z_blend)
               ↓
       Denoised latent
               ↓
        VAE Decode
               ↓
    Output Image 512×512
```

---

## Detailed Implementation Plan

### **Option 1: Create New Inference Script** (Recommended)

**File**: `scripts/inference_blended_input.py`

**Key Components:**

#### **1. Input Loading**
```python
def load_lr_image(path, size=32):
    """Load LR image at 32×32"""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    # Normalize to [-1, 1]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return 2.0 * image - 1.0

def load_edge_map(path, size=512):
    """Load edge map at 512×512"""
    edge = Image.open(path).convert("RGB")
    edge = edge.resize((size, size), Image.BICUBIC)
    # Normalize to [-1, 1]
    edge = np.array(edge).astype(np.float32) / 255.0
    edge = torch.from_numpy(edge).permute(2, 0, 1).unsqueeze(0)
    return 2.0 * edge - 1.0
```

#### **2. Create Blended Image**
```python
def create_blended_input(lr_image, edge_map, blend_alpha=0.5, output_size=512):
    """
    Create blended image from LR and edge map
    
    Args:
        lr_image: Tensor [1, 3, 32, 32] in [-1, 1]
        edge_map: Tensor [1, 3, 512, 512] in [-1, 1]
        blend_alpha: Blending weight (0=all edge, 1=all LR)
        output_size: Target size (512)
    
    Returns:
        blended_image: Tensor [1, 3, 512, 512] in [-1, 1]
    """
    # Step 1: Upscale LR to output size
    lr_upscaled = F.interpolate(
        lr_image, 
        size=(output_size, output_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Step 2: Convert to [0, 1] for blending
    lr_upscaled_01 = (lr_upscaled + 1.0) / 2.0
    edge_01 = (edge_map + 1.0) / 2.0
    
    # Step 3: Alpha blending (equivalent to cv2.addWeighted)
    # cv2.addWeighted(lr, alpha, edge, 1-alpha, 0) == alpha*lr + (1-alpha)*edge
    blended = blend_alpha * lr_upscaled_01 + (1.0 - blend_alpha) * edge_01
    
    # Step 4: Convert back to [-1, 1]
    blended_image = 2.0 * blended - 1.0
    blended_image = torch.clamp(blended_image, -1.0, 1.0)
    
    return blended_image
```

#### **3. Encode to Latent Space**
```python
def encode_blended_input(model, blended_image):
    """
    Encode blended image to latent space
    
    Args:
        model: StableSR model with VAE encoder
        blended_image: Tensor [1, 3, 512, 512] in [-1, 1]
    
    Returns:
        z_blend: Tensor [1, 4, 64, 64] in latent space
    """
    with torch.no_grad():
        encoder_posterior = model.encode_first_stage(blended_image)
        z_blend = model.get_first_stage_encoding(encoder_posterior)
    return z_blend
```

#### **4. Compute Structural Conditioning**
```python
def compute_struct_cond(model, z_blend, timestep=0):
    """
    Compute structural conditioning from blended latent
    
    Args:
        model: StableSR model with time-aware encoder
        z_blend: Blended input latent [1, 4, 64, 64]
        timestep: Timestep for time-aware encoder (usually 0 for inference)
    
    Returns:
        struct_cond: Dict of multi-scale features
    """
    t_ori = torch.tensor([model.ori_timesteps[timestep]]).long().to(z_blend.device)
    with torch.no_grad():
        struct_cond = model.structcond_stage_model(z_blend, t_ori)
    return struct_cond
```

#### **5. Sampling**
```python
def run_inference(
    model, 
    lr_image_32,      # [1, 3, 32, 32]
    edge_map_512,     # [1, 3, 512, 512]
    blend_alpha=0.5,
    ddpm_steps=200,
    seed=42
):
    """
    Main inference pipeline
    """
    seed_everything(seed)
    device = next(model.parameters()).device
    
    # 1. Create blended image
    blended_image = create_blended_input(lr_image_32, edge_map_512, blend_alpha)
    blended_image = blended_image.to(device)
    
    # 2. Encode to latent space
    z_blend = encode_blended_input(model, blended_image)
    
    # 3. Compute structural conditioning
    struct_cond = compute_struct_cond(model, z_blend, timestep=0)
    
    # 4. Text conditioning (empty or custom prompt)
    text_prompt = ["high quality, detailed"]  # or [""]
    semantic_c = model.cond_stage_model(text_prompt)
    
    # 5. Setup timesteps for sampling
    model.register_schedule(
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3
    )
    model.num_timesteps = 1000
    
    # Create shortened timestep schedule
    use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    
    # 6. Sample
    # Option A: Start from pure noise
    noise = torch.randn_like(z_blend)
    
    # Option B: Start from noisy z_blend (better structure preservation)
    t_start = repeat(torch.tensor([999]), '1 -> b', b=1).to(device).long()
    x_T = model.q_sample(x_start=z_blend, t=t_start, noise=noise)
    
    with torch.no_grad():
        samples, _ = model.sample(
            cond=semantic_c,
            struct_cond=struct_cond,
            batch_size=1,
            timesteps=ddpm_steps,
            time_replace=ddpm_steps,
            x_T=x_T,  # or None to start from pure noise
            return_intermediates=True
        )
    
    # 7. Decode to image space
    x_samples = model.decode_first_stage(samples)
    
    # 8. Post-process and save
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    output_image = Image.fromarray(x_sample.astype(np.uint8))
    
    return output_image, blended_image
```

---

### **Option 2: Modify Existing Scripts**

Update `predict.py` or create variant that:
1. Accepts LR 32×32 and edge 512×512 as separate inputs
2. Creates blended image
3. Computes struct_cond using time-aware encoder (not just raw latent!)

---

## Key Differences from Current Scripts

| Aspect | Current Scripts | New Approach |
|--------|----------------|--------------|
| **LR Input Size** | Any size (upscaled to 512) | Fixed 32×32 |
| **Edge Source** | Generated from LR | Provided as input (512×512) |
| **Blending** | No blending | Alpha blend LR + edge |
| **struct_cond** | Raw init_latent ❌ | Time-aware encoder output ✅ |
| **Encoding** | LR only | Blended image |

---

## Critical Fixes Needed in Current Scripts

### **Issue: predict.py (Line 190)**
```python
# WRONG:
struct_cond=init_latent,  # ❌ Raw latent, not using time-aware encoder!

# CORRECT:
# 1. Create z_blend from blended image
z_blend = model.get_first_stage_encoding(model.encode_first_stage(blended_image))

# 2. Compute struct_cond using time-aware encoder
t_ori = torch.tensor([model.ori_timesteps[0]]).long().to(device)
struct_cond = model.structcond_stage_model(z_blend, t_ori)
```

**This is a critical bug**: Current scripts don't use the time-aware encoder during inference!

---

## Complete Inference Script Template

```python
#!/usr/bin/env python3
"""
Inference with Blended Input (LR 32x32 + Edge 512x512)
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange, repeat

from ldm.util import instantiate_from_config


def load_lr_image(path, size=32):
    """Load and prepare LR image"""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).transpose(2, 0).transpose(0, 1)
    image = image.unsqueeze(0)  # Add batch dimension
    return 2.0 * image - 1.0  # Normalize to [-1, 1]


def load_edge_map(path, size=512):
    """Load and prepare edge map"""
    edge = Image.open(path).convert("RGB")
    edge = edge.resize((size, size), Image.BICUBIC)
    edge = np.array(edge).astype(np.float32) / 255.0
    edge = torch.from_numpy(edge).transpose(2, 0).transpose(0, 1)
    edge = edge.unsqueeze(0)
    return 2.0 * edge - 1.0


def create_blended_input(lr_image, edge_map, blend_alpha=0.5):
    """
    Create blended image using alpha blending
    
    Equivalent to: cv2.addWeighted(lr_up, alpha, edge, 1-alpha, 0)
    """
    # Upscale LR to 512x512
    lr_upscaled = F.interpolate(
        lr_image,
        size=(edge_map.size(-2), edge_map.size(-1)),
        mode='bicubic',
        align_corners=False
    )
    
    # Convert to [0, 1] for blending
    lr_up_01 = (lr_upscaled + 1.0) / 2.0
    edge_01 = (edge_map + 1.0) / 2.0
    
    # Alpha blending
    blended = blend_alpha * lr_up_01 + (1.0 - blend_alpha) * edge_01
    
    # Convert back to [-1, 1]
    blended = 2.0 * blended - 1.0
    return torch.clamp(blended, -1.0, 1.0)


def inference(
    model,
    lr_image_path,
    edge_map_path,
    blend_alpha=0.5,
    lr_size=32,
    output_size=512,
    ddpm_steps=200,
    seed=42,
    text_prompt=""
):
    """
    Run inference with blended input
    """
    seed_everything(seed)
    device = next(model.parameters()).device
    
    # 1. Load inputs
    print("Loading LR image and edge map...")
    lr_image = load_lr_image(lr_image_path, size=lr_size).to(device)
    edge_map = load_edge_map(edge_map_path, size=output_size).to(device)
    
    print(f"LR image shape: {lr_image.shape}")
    print(f"Edge map shape: {edge_map.shape}")
    
    # 2. Create blended image
    print(f"Creating blended image (alpha={blend_alpha})...")
    blended_image = create_blended_input(lr_image, edge_map, blend_alpha)
    print(f"Blended image shape: {blended_image.shape}")
    
    # 3. Encode to latent space
    print("Encoding to latent space...")
    with torch.no_grad():
        z_blend = model.get_first_stage_encoding(
            model.encode_first_stage(blended_image)
        )
    print(f"Blended latent shape: {z_blend.shape}")
    
    # 4. Compute structural conditioning using time-aware encoder
    print("Computing structural conditioning...")
    t_ori = torch.tensor([model.ori_timesteps[0]]).long().to(device)
    with torch.no_grad():
        struct_cond = model.structcond_stage_model(z_blend, t_ori)
    print(f"Struct_cond keys: {struct_cond.keys()}")
    
    # 5. Text conditioning
    print(f"Text prompt: '{text_prompt}'")
    text_list = [text_prompt] if text_prompt else [""]
    with torch.no_grad():
        semantic_c = model.cond_stage_model(text_list)
    
    # 6. Setup timesteps
    print(f"Setting up {ddpm_steps} sampling steps...")
    setup_timesteps(model, ddpm_steps)
    
    # 7. Create initial noise (can start from pure noise or noisy z_blend)
    noise = torch.randn_like(z_blend)
    
    # Option A: Start from pure noise
    # x_T = noise
    
    # Option B: Start from noisy z_blend (better structure preservation)
    t_start = repeat(torch.tensor([999]), '1 -> b', b=1).to(device).long()
    sqrt_alphas_cumprod = model.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod
    x_T = model.q_sample_respace(
        x_start=z_blend,
        t=t_start,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        noise=noise
    )
    
    # 8. Sample/Denoise
    print("Sampling...")
    with torch.no_grad():
        samples, _ = model.sample(
            cond=semantic_c,
            struct_cond=struct_cond,
            batch_size=1,
            timesteps=ddpm_steps,
            time_replace=ddpm_steps,
            x_T=x_T,
            return_intermediates=True
        )
    
    # 9. Decode to image space
    print("Decoding to image space...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
    
    # 10. Post-process
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    output_image = Image.fromarray(x_sample.astype(np.uint8))
    
    return output_image


def setup_timesteps(model, ddpm_steps):
    """Setup timestep schedule for sampling"""
    # Initial schedule
    model.register_schedule(
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3
    )
    
    # Create shortened schedule
    use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()


def space_timesteps(num_timesteps, section_counts):
    """Create timestep schedule"""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {desired_count} steps")
        section_counts = [int(x) for x in section_counts.split(",")]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    
    return set(all_steps)


def load_model_from_config(config, ckpt):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-img", type=str, required=True,
                       help="Path to LR image (32x32)")
    parser.add_argument("--edge-img", type=str, required=True,
                       help="Path to edge map (512x512)")
    parser.add_argument("--outdir", type=str, default="outputs/",
                       help="Output directory")
    parser.add_argument("--config", type=str, 
                       default="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml",
                       help="Config file")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Checkpoint path")
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                       help="Blending weight (0=all edge, 1=all LR)")
    parser.add_argument("--lr-size", type=int, default=32,
                       help="LR image size")
    parser.add_argument("--output-size", type=int, default=512,
                       help="Output image size")
    parser.add_argument("--ddpm-steps", type=int, default=200,
                       help="Number of DDPM steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--text-prompt", type=str, default="",
                       help="Text prompt for conditioning")
    
    args = parser.parse_args()
    
    # Load model
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model.configs = config
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Run inference
    print(f"\n{'='*60}")
    print("Running inference with blended input")
    print(f"{'='*60}")
    print(f"LR image: {args.lr_img}")
    print(f"Edge map: {args.edge_img}")
    print(f"Blend alpha: {args.blend_alpha}")
    print(f"{'='*60}\n")
    
    output_image = inference(
        model,
        args.lr_img,
        args.edge_img,
        blend_alpha=args.blend_alpha,
        ddpm_steps=args.ddpm_steps,
        seed=args.seed,
        text_prompt=args.text_prompt
    )
    
    # Save output
    output_path = os.path.join(args.outdir, "output.png")
    output_image.save(output_path)
    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
```

---

## Usage Example

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/image001.png \
    --edge-img inputs/edges_512x512/edge001.png \
    --outdir outputs/blended_inference/ \
    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
    --ckpt checkpoints/your_model.ckpt \
    --blend-alpha 0.5 \
    --ddpm-steps 200 \
    --text-prompt "high quality, detailed"
```

---

## Batch Processing Script

For processing multiple images:

```python
#!/usr/bin/env python3
"""Batch inference for blended input"""

import os
import glob
from pathlib import Path

def batch_inference(
    lr_dir,        # Directory with LR images (32x32)
    edge_dir,      # Directory with edge maps (512x512)
    output_dir,
    model,
    blend_alpha=0.5,
    **kwargs
):
    """Process all image pairs"""
    lr_images = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    edge_images = sorted(glob.glob(os.path.join(edge_dir, "*.png")))
    
    assert len(lr_images) == len(edge_images), "Mismatch in number of images!"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for lr_path, edge_path in zip(lr_images, edge_images):
        print(f"\nProcessing: {os.path.basename(lr_path)}")
        
        output_image = inference(
            model, lr_path, edge_path, 
            blend_alpha=blend_alpha,
            **kwargs
        )
        
        # Save with same name
        basename = os.path.basename(lr_path)
        output_path = os.path.join(output_dir, basename)
        output_image.save(output_path)
        print(f"Saved: {output_path}")
```

---

## Alternative: If You Have Only LR Image (No Separate Edge)

If you only have LR 32×32 and want to auto-generate edge:

```python
def inference_with_auto_edge(
    model,
    lr_image_path,
    blend_alpha=0.5,
    **kwargs
):
    """Inference when edge map is not provided"""
    from basicsr.utils.edge_utils import EdgeMapGenerator
    
    edge_gen = EdgeMapGenerator()
    
    # Load LR
    lr_image = load_lr_image(lr_image_path, size=32).to(device)
    
    # Upscale LR to 512x512
    lr_upscaled = F.interpolate(lr_image, size=(512, 512), mode='bicubic')
    
    # Generate edge from upscaled LR
    edge_map = edge_gen.generate_from_tensor(
        lr_upscaled,
        input_format='RGB',
        normalize_range='[-1,1]'
    )
    
    # Create blended image
    blended_image = create_blended_input(lr_image, edge_map, blend_alpha)
    
    # Continue with normal inference...
```

---

## Key Implementation Notes

### **1. Critical: Use Time-Aware Encoder for struct_cond**
```python
# WRONG (current scripts):
struct_cond = init_latent  # ❌

# CORRECT:
struct_cond = model.structcond_stage_model(z_blend, t_ori)  # ✅
```

### **2. Blending in PyTorch (not cv2)**
```python
# PyTorch (GPU-efficient):
blended = alpha * lr + (1-alpha) * edge

# Equivalent to cv2:
# cv2.addWeighted(lr, alpha, edge, 1-alpha, 0)
```

### **3. Starting Point Options**
```python
# Option A: Pure noise (more variation)
x_T = torch.randn_like(z_blend)

# Option B: Noisy z_blend (better structure preservation)
x_T = model.q_sample(x_start=z_blend, t=t_max, noise=noise)
```

---

## Summary

### **What to Create:**
1. ✅ New inference script: `scripts/inference_blended_input.py`
2. ✅ Helper functions for loading and blending
3. ✅ Proper struct_cond computation using time-aware encoder

### **Critical Requirements:**
- ⭐ LR input: 32×32 (fixed size)
- ⭐ Edge input: 512×512 (same as output)
- ⭐ Alpha blending before encoding
- ⭐ Use time-aware encoder for struct_cond (not raw latent!)

### **Expected Output:**
- 512×512 high-quality SR image
- Preserves edges from edge map
- Incorporates texture/color from LR

---

Would you like me to create the complete inference script now?

