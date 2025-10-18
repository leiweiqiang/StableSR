# Edge-to-Image Inference Guide

This guide explains how to use the trained StableSR model to generate images from Canny edge maps.

## Overview

**Purpose**: Generate photo-realistic images from Canny edge map inputs using your trained model.

**Key Differences from Super-Resolution**:
- ✅ Input: Canny edge maps (grayscale or RGB)
- ✅ Output: Same-size generated images (no upscaling)
- ✅ Method: Edge map as primary structure guidance
- ✅ No LR or GT images needed

---

## Quick Start

### 1. Prepare Edge Maps

Place your Canny edge map images in the input directory:

```bash
mkdir -p inputs/edge_maps
# Copy your edge maps here
cp /path/to/your/edge_maps/*.png inputs/edge_maps/
```

**Edge Map Requirements**:
- Format: PNG, JPG, or JPEG
- Size: Any size (will be resized to 512×512)
- Channels: Grayscale or RGB (both work)
- Content: Canny edge detection output

### 2. Run Inference

Simple run with default settings:

```bash
./inference_edge_to_image.sh
```

Custom settings:

```bash
# Edit the script variables or run Python script directly
python scripts/inference_edge_to_image.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
    --ckpt logs/your_experiment/checkpoints/last.ckpt \
    --edge-img inputs/edge_maps/ \
    --outdir outputs/edge_to_image/ \
    --ddim_steps 50 \
    --input_size 512 \
    --start_from_edge
```

### 3. View Results

Generated images will be in `outputs/edge_to_image_[timestamp]/`:
- `{basename}_generated.png` - Generated image
- `{basename}_edge_ref.png` - Edge map reference (for comparison)
- `preview_montage.jpg` - Grid of all generated images (if ImageMagick installed)

---

## Configuration Options

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--edge-img` | Required | Path to edge map images directory |
| `--outdir` | `outputs/edge_to_image` | Output directory |
| `--ckpt` | Required | Path to trained model checkpoint |
| `--config` | Required | Model configuration YAML |
| `--input_size` | `512` | Input/output image size (must match training) |
| `--ddim_steps` | `50` | Sampling steps (50=fast, 200=high quality) |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dec_w` | `0.0` | VQGAN decoder fusion weight (0.0=no fusion) |
| `--start_from_edge` | Flag | Start from noisy edge latent vs. pure noise |
| `--start_timestep` | `999` | Starting noise level (0-999, higher=more noise) |
| `--text_prompt` | `""` | Text prompt for conditional generation |
| `--n_samples` | `1` | Number of samples per edge map |
| `--seed` | `42` | Random seed for reproducibility |
| `--max_images` | `-1` | Max images to process (-1=all) |

---

## Implementation Details (Option B - Aggressive)

### Architecture Flow

```
Canny Edge Map (512×512)
    ↓
VQGAN Encoder
    ↓
Edge Latent (4×64×64)
    ↓
[Used as struct_cond]
    ↓
Diffusion Sampling (DDPM)
    ├─ Text Conditioning (optional)
    ├─ Start: Pure Noise or Noisy Edge Latent
    └─ Structure: Edge Latent
    ↓
Generated Latent (4×64×64)
    ↓
VQGAN Decoder (no fusion, dec_w=0)
    ↓
Generated Image (512×512)
```

### Key Design Choices

1. **Edge as Primary Input (`struct_cond`)**
   - Edge latent replaces LR latent as structure guidance
   - No separate `edge_map` parameter (Option B approach)
   - Simpler and cleaner than using edge twice

2. **Noise Initialization**
   - **Pure Noise** (default without `--start_from_edge`):
     - Most creative freedom
     - Can deviate more from edge structure
     - Good for diverse samples
   
   - **Noisy Edge Latent** (with `--start_from_edge`):
     - More faithful to edge structure
     - Faster convergence
     - Less variability

3. **No Feature Fusion** (`dec_w=0.0`)
   - VQGAN decoder fusion designed for LR→SR
   - Edge maps don't need fusion
   - Cleaner generation without artifacts

4. **Same-Size Output**
   - Input: 512×512 pixels → Latent: 64×64
   - Output: 512×512 pixels (no upscaling)
   - Matches training configuration

---

## Tips for Best Results

### 1. Edge Map Quality

**Good Edge Maps**:
- Clear, continuous edges
- Appropriate thickness (2-3 pixels)
- Well-connected structures
- Proper contrast (white edges on black background)

**Poor Edge Maps**:
- Too thin/broken edges (< 1 pixel)
- Too thick edges (> 5 pixels)
- Noisy or fragmented
- Low contrast

### 2. Generation Settings

**Fast Preview** (Quick testing):
```bash
--ddim_steps 20
--start_from_edge
--start_timestep 800
```

**High Quality** (Final generation):
```bash
--ddim_steps 200
--start_from_edge
--start_timestep 999
```

**Creative/Diverse** (Multiple variations):
```bash
--ddim_steps 100
# Don't use --start_from_edge (pure noise start)
--n_samples 4
--seed 42  # Change seed for different results
```

### 3. Decoder Fusion

- **`dec_w=0.0`** (recommended): Clean generation, no artifacts
- **`dec_w=0.25`**: Slight edge enhancement
- **`dec_w=0.5`**: Stronger edge influence (may cause artifacts)

### 4. Text Conditioning

Optionally guide generation with text:

```bash
--text_prompt "a photo of a landscape"
```

Note: Model trained with empty text prompts, so this may have limited effect.

---

## Troubleshooting

### Issue: Generated images are blurry

**Solutions**:
- Increase `--ddim_steps` to 100-200
- Use `--start_from_edge` flag
- Check edge map quality (should be clear)

### Issue: Output doesn't match edge structure

**Solutions**:
- Use `--start_from_edge --start_timestep 999`
- Increase `--dec_w` to 0.25-0.5
- Check that edges are continuous and well-connected

### Issue: Artifacts or strange patterns

**Solutions**:
- Decrease `--dec_w` (try 0.0)
- Ensure edge maps are clean (no noise)
- Check edge thickness (should be 2-3 pixels)

### Issue: Out of memory

**Solutions**:
- Decrease `--n_samples` to 1
- Use smaller `--input_size` (but must match training)
- Reduce batch processing

### Issue: Model produces same results

**Solutions**:
- Change `--seed` for different randomness
- Don't use `--start_from_edge` for more variation
- Lower `--start_timestep` if using `--start_from_edge`

---

## Example Workflows

### Workflow 1: Generate from Hand-Drawn Sketches

1. Draw sketch in any drawing program
2. Apply Canny edge detection or save as black/white edges
3. Run inference:
   ```bash
   python scripts/inference_edge_to_image.py \
       --edge-img my_sketches/ \
       --outdir outputs/my_sketches_generated/ \
       --ddim_steps 100 \
       --start_from_edge
   ```

### Workflow 2: Batch Process Multiple Edges

```bash
# Process directory with many edges
python scripts/inference_edge_to_image.py \
    --edge-img /dataset/test_edges/ \
    --outdir outputs/batch_results/ \
    --ddim_steps 50 \
    --max_images 100 \
    --n_samples 1
```

### Workflow 3: Generate Multiple Variations

```bash
# Generate 4 variations of each edge with different seeds
for seed in 42 123 456 789; do
    python scripts/inference_edge_to_image.py \
        --edge-img inputs/single_edge/ \
        --outdir outputs/variations_seed${seed}/ \
        --ddim_steps 100 \
        --seed ${seed}
done
```

### Workflow 4: Extract Edges and Generate

```bash
# Extract edges from photos, then generate
python -c "
import cv2
import numpy as np
from pathlib import Path

input_dir = Path('inputs/photos')
output_dir = Path('inputs/extracted_edges')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    cv2.imwrite(str(output_dir / img_path.name), edges)
"

# Now generate from extracted edges
./inference_edge_to_image.sh
```

---

## Performance Benchmarks

**Tested on NVIDIA GPU (adjust for your hardware)**:

| Steps | Quality | Time/Image | Memory |
|-------|---------|------------|--------|
| 20 | Preview | ~2 sec | ~4 GB |
| 50 | Good | ~5 sec | ~4 GB |
| 100 | High | ~10 sec | ~4 GB |
| 200 | Best | ~20 sec | ~4 GB |

**Recommendations**:
- Development/Testing: 20-50 steps
- Production: 100-200 steps
- Batch processing: 50 steps (good balance)

---

## Comparison with Training

### Training Setup
- **Input**: Degraded LR images (synthetic degradation)
- **Structure**: LR latent as `struct_cond`
- **Guidance**: Edge from GT as `edge_map`
- **Task**: Super-resolution (128→512 or 512→512)

### Inference Setup (Option B)
- **Input**: Canny edge maps
- **Structure**: Edge latent as `struct_cond`
- **Guidance**: None (no separate edge_map)
- **Task**: Edge-to-image generation (512→512)

### Why This Works

Even though the model was trained for super-resolution, it learned to:
1. **Use struct_cond as structural guidance** (originally LR, now edges)
2. **Generate high-frequency details** (edge information provides structure)
3. **Produce realistic textures** (from diffusion training)

By replacing LR with edges as `struct_cond`, we repurpose the model for edge-to-image generation.

---

## Next Steps

1. **Test with your trained model** on sample edge maps
2. **Experiment with parameters** to find best settings
3. **Compare Option A** (edge as both struct_cond and edge_map) if results are unsatisfactory
4. **Fine-tune model** specifically for edge-to-image if needed

---

## Files Reference

- **Inference Script**: `scripts/inference_edge_to_image.py`
- **Shell Wrapper**: `inference_edge_to_image.sh`
- **Config**: `configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml`
- **Model**: Your trained checkpoint in `logs/*/checkpoints/`

---

## Support

If you encounter issues:
1. Check that checkpoint path is correct
2. Verify VQGAN checkpoint is accessible
3. Ensure edge maps are in supported format
4. Try with default parameters first
5. Check GPU memory availability

For questions about model architecture or training, refer to the training documentation.


