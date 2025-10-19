# Inference Output Examples

## What Files Are Generated?

The inference script **always saves 2 files** by default:

### **1. output.png** (Final SR Result)
- **Size**: 512×512
- **Content**: High-quality super-resolution output
- **What it is**: The final generated image from diffusion sampling

### **2. blended_input.png** (Blended Input) ⭐ **NEW**
- **Size**: 512×512
- **Content**: Alpha-blended LR upscaled + edge map
- **What it is**: The blended image that was used as input to the model
- **Purpose**: Shows what structural information was provided to the model

---

## Output Examples by Mode

### **Single Image Mode** (Default)

**Command:**
```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/image_32x32.png \
    --edge-img inputs/edge_512.png \
    --outdir outputs/single/ \
    --ckpt model.ckpt
```

**Output:**
```
outputs/single/
├── output.png           # 512×512 SR result ⭐
└── blended_input.png    # 512×512 blended image ⭐
```

---

### **Batch Mode**

**Command:**
```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/ \
    --outdir outputs/batch/ \
    --ckpt model.ckpt \
    --batch-mode
```

**Input Structure:**
```
inputs/
├── lr_32x32/
│   ├── cat.png
│   ├── dog.png
│   └── bird.png
└── edges_512/
    ├── cat.png
    ├── dog.png
    └── bird.png
```

**Output:**
```
outputs/batch/
├── cat.png              # SR output ⭐
├── cat_blended.png      # Blended input ⭐
├── dog.png              # SR output ⭐
├── dog_blended.png      # Blended input ⭐
├── bird.png             # SR output ⭐
└── bird_blended.png     # Blended input ⭐
```

---

### **With --save-intermediates**

**Command:**
```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/image_32x32.png \
    --edge-img inputs/edge_512.png \
    --outdir outputs/debug/ \
    --ckpt model.ckpt \
    --save-intermediates
```

**Output:**
```
outputs/debug/
├── output.png           # SR result ⭐
├── blended_input.png    # Blended image ⭐
├── lr_input.png         # LR input (upscaled to 512×512 for viewing)
└── edge_input.png       # Edge map input
```

**Visual Comparison:**
```
lr_input.png      edge_input.png    blended_input.png    output.png
(pixelated        (edges only)      (LR + edge)          (SR result)
 32→512)
```

---

### **With --save-comparison**

**Command:**
```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/image_32x32.png \
    --edge-img inputs/edge_512.png \
    --outdir outputs/comparison/ \
    --ckpt model.ckpt \
    --save-comparison
```

**Output:**
```
outputs/comparison/
├── output.png              # SR result ⭐
├── blended_input.png       # Blended image ⭐
└── comparison_grid.png     # Grid with all 4 stages
```

**comparison_grid.png layout:**
```
┌────────────┬────────────┬────────────┬────────────┐
│  LR Input  │ Edge Map   │  Blended   │   Output   │
│  (32→512)  │ (512×512)  │ (512×512)  │ (512×512)  │
│ (upscaled  │ (edges)    │ (LR+edge)  │ (SR result)│
│  nearest)  │            │            │            │
└────────────┴────────────┴────────────┴────────────┘
```

---

### **With All Options**

**Command:**
```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/image_32x32.png \
    --edge-img inputs/edge_512.png \
    --outdir outputs/full/ \
    --ckpt model.ckpt \
    --save-intermediates \
    --save-comparison \
    --blend-alpha 0.5
```

**Output:**
```
outputs/full/
├── output.png              # SR result ⭐
├── blended_input.png       # Blended image ⭐
├── lr_input.png            # LR visualization
├── edge_input.png          # Edge input
└── comparison_grid.png     # 4-panel grid
```

---

## Why Save blended_input.png?

### **Benefits:**

1. **Transparency** ✅
   - See exactly what was fed to the model
   - Verify blending is working correctly
   - Debug if output doesn't look right

2. **Quality Control** ✅
   - Compare blended input vs output
   - Check if model is improving the input
   - Identify issues (too much edge, too little texture, etc.)

3. **Parameter Tuning** ✅
   - Adjust `blend_alpha` based on blended image quality
   - If blended looks too edgy → increase alpha (more LR)
   - If blended looks too smooth → decrease alpha (more edge)

4. **Documentation** ✅
   - Keep record of input conditions
   - Reproduce results later
   - Share with others for debugging

---

## File Naming Convention

### **Single Image Mode:**
- `output.png` - The SR output
- `blended_input.png` - The blended input

### **Batch Mode:**
- `{original_name}.png` - The SR output
- `{original_name}_blended.png` - The blended input

**Example:**
```
Input: cat.png (LR) + cat.png (edge)
Output: 
  - cat.png (SR result)
  - cat_blended.png (blended input)
```

---

## Typical Output Quality

### **blended_input.png:**
- Pixelated LR upscaled areas
- Sharp edges from edge map
- Visual blend of both

### **output.png:**
- High-quality 512×512 image
- Sharp details preserved from edges
- Realistic texture from LR + model generation
- Should look significantly better than blended_input.png

---

## Comparison Example

```
Input LR (32×32):          Input Edge (512×512):
┌──────────┐               ┌──────────────────┐
│ Pixelated│               │  Sharp edges     │
│ Low res  │               │  No texture      │
│ Blurry   │               │  Black & white   │
└──────────┘               └──────────────────┘
       │                            │
       └────────────┬───────────────┘
                    ▼
           blended_input.png (512×512):
           ┌──────────────────┐
           │ Pixelated texture│
           │ + Sharp edges    │
           │ Still not great  │
           └──────────────────┘
                    │
                    ▼ (Through model)
           output.png (512×512):
           ┌──────────────────┐
           │ Sharp edges ✓    │
           │ Clear texture ✓  │
           │ High quality ✓   │
           └──────────────────┘
```

---

## Quick Visual Check

When you get your outputs, check:

✅ **blended_input.png should show:**
- Combination of LR (pixelated) and edge (sharp lines)
- Visible alpha blending effect
- Proper 512×512 size

✅ **output.png should show:**
- Much better quality than blended_input.png
- Sharp edges preserved from edge map
- Realistic texture (not pixelated like blended input)
- Professional-looking SR result

❌ **If output looks same as blended_input:**
- Model may not be working properly
- Check that model is loaded correctly
- Verify struct_cond is computed using time-aware encoder

---

## Summary

**Always saved:**
- ✅ `output.png` - Your final SR result
- ✅ `blended_input.png` - What was fed to the model

**This gives you complete transparency** about what the model received and what it produced! 🎯

