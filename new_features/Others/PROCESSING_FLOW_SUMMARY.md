# Processing Flow: Input to UNET and Time-Aware Encoder

## Overview

The blended input approach provides structural conditioning to the UNET through a time-aware encoder that processes a combination of upscaled LR image and edge map features.

---

## Complete Data Flow

### **Phase 1: Input Preparation**

```
Ground Truth Image (512×512)
    │
    ├──────────────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            ▼
Degradation Pipeline                                   Canny Edge Generator
(blur, noise, JPEG, etc.)                             (EdgeMapGenerator)
    │                                                            │
    ▼                                                            ▼
LQ Image (512×512)                                    Edge Map (512×512)
[-1, 1] range                                         [-1, 1] range
```

### **Phase 2: Latent Encoding**

```
LQ Image (512×512) ──→ VAE Encoder ──→ z (LR latent, 4ch, 64×64)
                                            │
                                            └──→ Used for compatibility
                                                 (kept in forward pass)

GT Image (512×512) ──→ VAE Encoder ──→ z_gt (GT latent, 4ch, 64×64)
                                            │
                                            └──→ Target for denoising
```

### **Phase 3: Blended Input Creation** ⭐ **NEW**

```
LQ Image (512×512)                      Edge Map (512×512)
[-1, 1] range                           [-1, 1] range
    │                                       │
    ▼                                       │
Downsample to 32×32                         │
(bicubic interpolation)                     │
    │                                       │
    ▼                                       │
LR 32×32                                    │
    │                                       │
    ▼                                       │
Upsample to 512×512                         │
(bicubic interpolation)                     │
    │                                       │
    ▼                                       │
LR Upscaled (512×512)                       │
[-1, 1] range                               │
    │                                       │
    ├───────────────────────────────────────┤
    │                                       │
    ▼                                       ▼
Convert to [0, 1]:                   Convert to [0, 1]:
lr_upscaled_01                       edge_01
    │                                       │
    └───────────────┬───────────────────────┘
                    ▼
            Alpha Blending (PyTorch)
    blended = α * lr_upscaled_01 + (1-α) * edge_01
                    │
                    ▼
            Convert back to [-1, 1]
                    │
                    ▼
        Blended Image (512×512, 3ch)
                    │
                    ▼
              VAE Encoder
                    │
                    ▼
        z_blend (4ch, 64×64) ⭐
```

---

## **Phase 4: Forward Pass Through Networks**

### **A. Training Step Flow**

```python
# 1. Get inputs from batch
x, c, gt, z_blend = get_input(batch, self.first_stage_key)
# x: LR latent (4ch, 64×64)
# c: text conditioning (embeddings)
# gt: GT latent (4ch, 64×64)
# z_blend: Blended input latent (4ch, 64×64) ⭐

# 2. Forward pass
loss = forward(x, c, z_blend, gt)
```

### **B. Inside forward() Method**

```
Timestep Sampling
    │
    ├──→ t: diffusion timestep [0, 1000]
    └──→ t_ori: original timestep for time-aware encoder
    
    
Text Conditioning (c)
    │
    ▼
Text Encoder / CLIP
    │
    ▼
Text Embeddings (for cross-attention in UNET)


z_blend (4ch, 64×64) ──────┐
                            │
t_ori ──────────────────────┤
                            │
                            ▼
                  ╔═══════════════════════════╗
                  ║  structcond_stage_model   ║
                  ║  (Time-Aware Encoder)     ║
                  ║  EncoderUNetModelWT       ║
                  ╚═══════════════════════════╝
                            │
                            ▼
              Multi-scale Features (struct_cond)
              {
                '64': features_64,   # 64×64 resolution
                '32': features_32,   # 32×32 resolution  
                '16': features_16,   # 16×16 resolution
                '8': features_8      # 8×8 resolution
              }


GT Latent (z_gt) ───────────────────┐
    │                                │
    ▼                                │
Add Noise (forward diffusion)       │
    │                                │
    ▼                                │
x_noisy (noised GT latent)          │
    │                                │
    │                                │
    ├────────────────────────────────┴──────────┐
    │                                            │
    │                                            │
    ▼                                            ▼
Text Embeddings (c)              struct_cond (multi-scale features)
(for cross-attention)            (from time-aware encoder)
    │                                            │
    │                                            │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
            ╔═══════════════════════════╗
            ║         UNET              ║
            ║  (UNetModelDualcondV2)    ║
            ║                           ║
            ║  - Cross-attention: c     ║
            ║  - Struct conditioning:   ║
            ║    struct_cond features   ║
            ╚═══════════════════════════╝
                       │
                       ▼
            Predicted Noise / x0
                       │
                       ▼
            Loss Calculation
            (compare with target)
```

---

## **Phase 5: Time-Aware Encoder Details**

### **EncoderUNetModelWT Processing**

```
Input: blend_latent (N, 4, 64, 64)
       timesteps (N,)
    │
    ▼
Timestep Embedding
    │
    ▼
emb = time_embed(timestep_embedding(timesteps))
    │
    │
    ▼
─────────────────────────────────────────────
    Input Blocks (Encoder Path)
─────────────────────────────────────────────
    │
    ├──→ Block 0: Conv 4→256 ch
    │    Resolution: 64×64
    │    │
    │    ├──→ Save features
    │    │
    ├──→ Block 1-2: ResBlocks + Attention
    │    Resolution: 64×64
    │    │
    │    ├──→ Downsample to 32×32
    │    └──→ Save features (for struct_cond['64'])
    │
    ├──→ Block 3-4: ResBlocks + Attention
    │    Resolution: 32×32
    │    │
    │    ├──→ Downsample to 16×16
    │    └──→ Save features (for struct_cond['32'])
    │
    ├──→ Block 5-6: ResBlocks + Attention
    │    Resolution: 16×16
    │    │
    │    ├──→ Downsample to 8×8
    │    └──→ Save features (for struct_cond['16'])
    │
    ▼
─────────────────────────────────────────────
    Middle Block
─────────────────────────────────────────────
    │
    ├──→ ResBlocks + Attention
    │    Resolution: 8×8
    │    │
    │    └──→ Save features (for struct_cond['8'])
    │
    ▼
Feature Transformation
    │
    ├──→ fea_tran[0]: Transform 64×64 features → 256ch
    ├──→ fea_tran[1]: Transform 32×32 features → 256ch
    ├──→ fea_tran[2]: Transform 16×16 features → 256ch
    └──→ fea_tran[3]: Transform 8×8 features → 256ch
    │
    ▼
Output: struct_cond = {
    '64': transformed_features_64,
    '32': transformed_features_32,
    '16': transformed_features_16,
    '8': transformed_features_8
}
```

---

## **Phase 6: UNET Processing**

### **UNetModelDualcondV2 with Dual Conditioning**

```
Input: x_noisy (N, 4, 64, 64) - noised GT latent
       timesteps (N,)
       context (text embeddings) - for cross-attention
       struct_cond (multi-scale features) - from time-aware encoder
    │
    ▼
Timestep Embedding
    │
    ▼
emb = time_embed(timestep_embedding(timesteps))


─────────────────────────────────────────────
    Input Blocks (Encoder Path)
─────────────────────────────────────────────
    │
    For each block:
    │
    ├──→ ResBlock(x, emb)
    │    ├──→ Time conditioning via emb
    │    │
    ├──→ SpatialTransformer
    │    ├──→ Self-attention
    │    ├──→ Cross-attention with text (context)
    │    ├──→ Struct conditioning via struct_cond ⭐
    │    │    (adds multi-scale features from time-aware encoder)
    │    │
    ├──→ Save to skip connections
    │    │
    ├──→ Downsample (if needed)
    │
    ▼
─────────────────────────────────────────────
    Middle Block
─────────────────────────────────────────────
    │
    ├──→ ResBlock + Attention
    │    ├──→ Cross-attention with text
    │    └──→ Struct conditioning ⭐
    │
    ▼
─────────────────────────────────────────────
    Output Blocks (Decoder Path)
─────────────────────────────────────────────
    │
    For each block:
    │
    ├──→ Concatenate skip connection
    │    │
    ├──→ ResBlock(x, emb)
    │    │
    ├──→ SpatialTransformer
    │    ├──→ Self-attention
    │    ├──→ Cross-attention with text
    │    ├──→ Struct conditioning ⭐
    │    │
    ├──→ Upsample (if needed)
    │
    ▼
Output: predicted_noise / predicted_x0
```

---

## **Key Points Summary**

### **1. Blended Input Creation**
- **Purpose**: Combine LR information with edge structure
- **Method**: Alpha blending in image space before encoding
- **Result**: `z_blend` (4ch, 64×64 latent)

### **2. Time-Aware Encoder (structcond_stage_model)**
- **Input**: `z_blend` + `timesteps`
- **Architecture**: EncoderUNetModelWT (encoder-only U-Net)
- **Output**: Multi-scale features (`struct_cond`) at 4 resolutions
- **Role**: Extract structural features from blended input
- **Trainable**: Yes ✓

### **3. UNET (Main Denoising Network)**
- **Input**: Noisy GT latent + timesteps
- **Conditioning**:
  - **Text conditioning** (cross-attention): Semantic guidance
  - **Struct conditioning** (struct_cond): Structural guidance from blended input ⭐
- **Output**: Predicted noise or predicted clean image
- **Role**: Denoise to reconstruct high-quality image

### **4. Information Flow**
```
Blended Image (LR + Edge)
        ↓
    VAE Encode
        ↓
    z_blend ────────────────────────────┐
                                        ↓
                              Time-Aware Encoder
                                        ↓
                                  struct_cond
                                        ↓
GT Latent → Add Noise → x_noisy ──→ UNET ←── Text Conditioning
                                        ↓
                                Predicted Output
```

### **5. Why This Design?**
- **Separation of concerns**: 
  - Time-aware encoder: Extracts structural features
  - UNET: Focuses on denoising with structural guidance
- **Multi-scale features**: Provides guidance at multiple resolutions
- **Flexible conditioning**: Can adjust blend_alpha to control LR vs edge influence
- **End-to-end trainable**: Both networks learn together

---

## **Training vs Inference**

### **Training**
- Input: GT image → Degradation → LQ image
- Edge: Canny edge extracted from GT
- Target: GT latent (z_gt)
- Blended input provides structural guidance for reconstruction

### **Inference**
- Input: Real LR image
- Edge: Canny edge extracted from LR image (or provided)
- Target: High-quality SR output
- Blended input guides the generation process

---

## **Configuration Parameters**

```yaml
blend_alpha: 0.5  # Controls blending ratio
                  # 0.0 = pure edge map
                  # 0.5 = equal blend (default)
                  # 1.0 = pure LR upscaled

lr_size_before_upscale: 32  # LR resolution before upscaling
                            # Smaller = more degraded LR
                            # Larger = better LR quality
```

### **Effect of blend_alpha**
- **α = 0.0**: Pure edge guidance (sharp edges, may miss texture)
- **α = 0.3**: Edge-dominant (good for preserving structure)
- **α = 0.5**: Balanced (default, good starting point)
- **α = 0.7**: LR-dominant (preserves more texture)
- **α = 1.0**: Pure LR guidance (no edge information)

---

## **Advantages of This Approach**

1. ✅ **Complementary Information**
   - LR upscaled: Provides color, texture, rough structure
   - Edge map: Provides sharp boundaries, fine details

2. ✅ **Flexible Control**
   - Adjustable via `blend_alpha` parameter
   - Can tune for different types of images

3. ✅ **Multi-scale Guidance**
   - Time-aware encoder provides features at 4 resolutions
   - UNET receives guidance at appropriate scales

4. ✅ **End-to-End Learning**
   - Time-aware encoder learns to extract useful features
   - UNET learns to use these features effectively

5. ✅ **GPU Efficient**
   - Pure PyTorch operations
   - No CPU-GPU data transfer for blending

---

**This architecture enables the UNET to perform high-quality super-resolution by leveraging both the coarse information from upscaled LR images and the precise structural information from edge maps!** 🎯

