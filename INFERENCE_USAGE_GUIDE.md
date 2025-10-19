# Inference Usage Guide: Blended Input

## Quick Start

### **Step 1: Prepare Inputs**

You need two inputs:
1. **LR Image**: 32×32 pixels (or will be resized)
2. **Edge Map**: 512×512 pixels (or will be resized)

### **Step 2: Run Inference**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/image.png \
    --edge-img inputs/edges_512/edge.png \
    --outdir outputs/results/ \
    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
    --ckpt checkpoints/your_model.ckpt \
    --blend-alpha 0.5 \
    --ddpm-steps 200
```

### **Step 3: Get Output**

Outputs will be saved to:
- `outputs/results/output.png` - Final SR result (512×512)
- `outputs/results/blended_input.png` - Blended image used as input (512×512)

---

## Complete Usage Examples

### **Example 1: Basic Inference (Single Image)**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/image001.png \
    --edge-img inputs/edges_512/edge001.png \
    --outdir outputs/basic/ \
    --ckpt checkpoints/stablesr_blended.ckpt \
    --blend-alpha 0.5
```

**Output Files:**
- `outputs/basic/output.png` - Final SR result
- `outputs/basic/blended_input.png` - Blended image (LR upscaled + edge)

---

### **Example 2: Batch Processing (Multiple Images)**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/ \
    --outdir outputs/batch_results/ \
    --ckpt checkpoints/stablesr_blended.ckpt \
    --batch-mode \
    --blend-alpha 0.5
```

**Output Files** (for each input):
- `outputs/batch_results/image001.png` - SR output
- `outputs/batch_results/image001_blended.png` - Blended input
- `outputs/batch_results/image002.png` - SR output
- `outputs/batch_results/image002_blended.png` - Blended input
- ...

---

### **Example 3: With Text Conditioning**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/portrait.png \
    --edge-img inputs/edges_512/portrait_edge.png \
    --outdir outputs/text_guided/ \
    --ckpt checkpoints/stablesr_blended.ckpt \
    --text-prompt "high quality, detailed, sharp" \
    --blend-alpha 0.5 \
    --ddpm-steps 200
```

**Result**: Output guided by text prompt

---

### **Example 4: Save Intermediate Images**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/image.png \
    --edge-img inputs/edges_512/edge.png \
    --outdir outputs/debug/ \
    --ckpt checkpoints/stablesr_blended.ckpt \
    --save-intermediates \
    --save-comparison \
    --blend-alpha 0.5
```

**Output Files:**
- `output.png` - Final SR result (always saved)
- `blended_input.png` - Blended image (always saved)
- `lr_input.png` - LR input visualization (with --save-intermediates)
- `edge_input.png` - Edge map input (with --save-intermediates)
- `comparison_grid.png` - All steps in grid (with --save-comparison)

---

### **Example 5: Different Blending Ratios**

```bash
# More edge influence
python scripts/inference_blended_input.py \
    --lr-img inputs/lr.png --edge-img inputs/edge.png \
    --outdir outputs/edge_dominant/ \
    --ckpt checkpoints/model.ckpt \
    --blend-alpha 0.3  # 30% LR, 70% edge

# More LR influence
python scripts/inference_blended_input.py \
    --lr-img inputs/lr.png --edge-img inputs/edge.png \
    --outdir outputs/lr_dominant/ \
    --ckpt checkpoints/model.ckpt \
    --blend-alpha 0.7  # 70% LR, 30% edge
```

---

## Command-Line Arguments

### **Required Arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `--lr-img` | Path to LR image or directory | `inputs/lr_32x32/image.png` |
| `--edge-img` | Path to edge map or directory | `inputs/edges/edge.png` |
| `--ckpt` | Path to model checkpoint | `checkpoints/model.ckpt` |

### **Optional Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `v2-finetune_text_T_512_canny_in.yaml` | Config file path |
| `--outdir` | `outputs/blended_inference/` | Output directory |
| `--blend-alpha` | `0.5` | Blending weight (0-1) |
| `--lr-size` | `32` | LR input size |
| `--output-size` | `512` | Output image size |
| `--ddpm-steps` | `200` | Number of sampling steps |
| `--seed` | `42` | Random seed |
| `--text-prompt` | `""` | Text prompt for conditioning |
| `--start-from-noise` | `False` | Start from pure noise |
| `--colorfix` | `none` | Color correction (none/adain/wavelet) |
| `--batch-mode` | `False` | Process entire directories |
| `--save-intermediates` | `False` | Save intermediate images |
| `--save-comparison` | `False` | Save comparison grid |
| `--verbose` | `False` | Verbose model loading |

---

## Testing Before Real Inference

### **Test 1: Create Dummy Inputs**

```bash
python scripts/test_blended_inference.py --create-test-inputs
```

This creates:
- `test_inputs/test_lr_32x32.png`
- `test_inputs/test_edge_512x512.png`

### **Test 2: Verify Blending Logic**

```bash
python scripts/test_blended_inference.py --test-blending
```

This tests:
- ✓ Upscaling from 32×32 to 512×512
- ✓ Alpha blending
- ✓ Range preservation [-1, 1]

### **Test 3: Run Inference on Test Inputs**

```bash
python scripts/inference_blended_input.py \
    --lr-img test_inputs/test_lr_32x32.png \
    --edge-img test_inputs/test_edge_512x512.png \
    --outdir outputs/test/ \
    --ckpt checkpoints/your_model.ckpt \
    --save-comparison
```

---

## Input Requirements

### **LR Image (32×32)**
- **Format**: PNG, JPG
- **Size**: Exactly 32×32 (or will be resized)
- **Channels**: RGB (3 channels)
- **Quality**: Any - will be upscaled

**How to create**:
```python
from PIL import Image
lr_image = Image.open("original.png").convert("RGB")
lr_image = lr_image.resize((32, 32), Image.BICUBIC)
lr_image.save("lr_32x32.png")
```

### **Edge Map (512×512)**
- **Format**: PNG, JPG
- **Size**: Exactly 512×512 (or will be resized)
- **Channels**: RGB or Grayscale (converted to RGB)
- **Content**: Edges in white/black

**How to create from GT**:
```bash
python scripts/extract_canny_edges.py \
    --input inputs/gt_512/ \
    --output inputs/edges_512/ \
    --low-threshold 100 \
    --high-threshold 200
```

**Or manually with OpenCV**:
```python
import cv2
image = cv2.imread("image_512.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.imwrite("edge_512.png", edges_rgb)
```

---

## Understanding blend_alpha Parameter

```
blend_alpha = 0.0  →  100% Edge, 0% LR
                      (Pure edge guidance - sharp but may lack texture)

blend_alpha = 0.3  →  30% LR, 70% Edge
                      (Edge-dominant - good structure preservation)

blend_alpha = 0.5  →  50% LR, 50% Edge
                      (Balanced - recommended default)

blend_alpha = 0.7  →  70% LR, 30% Edge
                      (LR-dominant - preserves more texture/color)

blend_alpha = 1.0  →  100% LR, 0% Edge
                      (Pure LR guidance - smooth but may lose details)
```

**Recommendation**: Start with 0.5, then adjust based on results

---

## Advanced Options

### **Starting Point Options**

```bash
# Start from pure noise (default, more variation)
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt --start-from-noise

# Start from noisy z_blend (better structure preservation)
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt
```

### **Color Correction**

```bash
# AdaIN color correction (match blended image colors)
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt --colorfix adain

# Wavelet color correction
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt --colorfix wavelet
```

### **Faster Inference**

```bash
# Use fewer steps (faster but lower quality)
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt --ddpm-steps 50
```

---

## Troubleshooting

### **Issue: "Missing keys" when loading checkpoint**

**Solution**: Use `--verbose` to see which keys are missing. If they're from `edge_processor`, this is normal (it was removed).

```bash
python scripts/inference_blended_input.py \
    --lr-img lr.png --edge-img edge.png \
    --ckpt model.ckpt --verbose
```

### **Issue: Output image has wrong colors**

**Solution**: Try color correction

```bash
--colorfix adain  # or --colorfix wavelet
```

### **Issue: Output is too smooth/blurry**

**Solution**: Increase edge influence

```bash
--blend-alpha 0.3  # More edge, less LR
```

### **Issue: Output has too many artifacts**

**Solution**: Increase LR influence or use more sampling steps

```bash
--blend-alpha 0.7  # More LR, less edge
--ddpm-steps 300   # More sampling steps
```

---

## Output Structure

### **Always Saved:**
```
outputs/
├── output.png                  # Main SR result (512×512) ⭐ Always
└── blended_input.png          # Blended image (512×512) ⭐ Always
```

### **With --save-intermediates:**
```
outputs/
├── output.png                  # Main SR result
├── blended_input.png          # Blended image (always saved)
├── lr_input.png               # LR input visualization (upscaled for viewing)
└── edge_input.png             # Edge map input
```

### **With --save-comparison:**
```
outputs/
├── output.png                  # Main SR result
├── blended_input.png          # Blended image (always saved)
└── comparison_grid.png        # Grid: LR | Edge | Blended | Output
```

### **Batch Mode:**
```
outputs/
├── image001.png               # SR output
├── image001_blended.png       # Blended input
├── image002.png               # SR output
├── image002_blended.png       # Blended input
└── ...
```

---

## Batch Processing Tips

### **Organize Inputs**

```
project/
├── inputs/
│   ├── lr_32x32/
│   │   ├── image001.png
│   │   ├── image002.png
│   │   └── image003.png
│   └── edges_512/
│       ├── image001.png  # Same names!
│       ├── image002.png
│       └── image003.png
└── outputs/
    └── results/
```

### **Run Batch**

```bash
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/ \
    --outdir outputs/results/ \
    --ckpt checkpoints/model.ckpt \
    --batch-mode \
    --blend-alpha 0.5
```

---

## Performance Tips

1. **GPU Memory**: Script uses ~4-6GB VRAM for 512×512 output
2. **Speed**: ~5-10 seconds per image with 200 steps (on modern GPU)
3. **Batch Size**: Currently processes one image at a time (can be extended)

---

## Next Steps

1. ✅ Test blending logic:
   ```bash
   python scripts/test_blended_inference.py --test-blending
   ```

2. ✅ Create test inputs:
   ```bash
   python scripts/test_blended_inference.py --create-test-inputs
   ```

3. ✅ Run inference:
   ```bash
   python scripts/inference_blended_input.py \
       --lr-img test_inputs/test_lr_32x32.png \
       --edge-img test_inputs/test_edge_512x512.png \
       --outdir outputs/test/ \
       --ckpt checkpoints/your_model.ckpt \
       --save-comparison
   ```

4. ✅ Tune parameters based on results!

---

**The inference script is ready to use!** 🚀

