# Quick Start: Edge-to-Image Generation

## TL;DR - Get Started in 3 Steps

```bash
# 1. Prepare edge maps (or extract from images)
mkdir -p inputs/edge_maps
python scripts/extract_canny_edges.py --input /path/to/images/ --output inputs/edge_maps/ --adaptive

# 2. Run test to verify setup
./test_edge_to_image.sh

# 3. Generate images from your edges
./inference_edge_to_image.sh
```

---

## What This Does

🎨 **Input**: Canny edge maps (black and white edges)  
🖼️ **Output**: Photo-realistic generated images  
📏 **Size**: Same size as input (no upscaling), default 512×512  
🚀 **Method**: Uses your trained StableSR model with edges as structure guidance

---

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `inference_edge_to_image.sh` | Main inference script | Production use |
| `test_edge_to_image.sh` | Quick validation test | First time setup |
| `scripts/inference_edge_to_image.py` | Python implementation | Advanced users |
| `scripts/extract_canny_edges.py` | Edge extraction tool | Prepare test data |
| `EDGE_TO_IMAGE_INFERENCE.md` | Full documentation | Detailed reference |
| `IMPLEMENTATION_SUMMARY.md` | Technical details | Understanding internals |

---

## Common Commands

### Extract Edges from Images

```bash
# Adaptive thresholds (recommended, matches training)
python scripts/extract_canny_edges.py \
    --input /path/to/images/ \
    --output inputs/edge_maps/ \
    --adaptive \
    --save_rgb

# Fixed thresholds
python scripts/extract_canny_edges.py \
    --input /path/to/images/ \
    --output inputs/edge_maps/ \
    --low_threshold 50 \
    --high_threshold 150
```

### Generate Images (Simple)

```bash
# Default settings (fast, good quality)
./inference_edge_to_image.sh
```

### Generate Images (Advanced)

```bash
# High quality
python scripts/inference_edge_to_image.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml \
    --ckpt logs/your_experiment/checkpoints/last.ckpt \
    --edge-img inputs/edge_maps/ \
    --outdir outputs/generated/ \
    --ddim_steps 100 \
    --start_from_edge

# Fast preview
python scripts/inference_edge_to_image.py \
    --edge-img inputs/edge_maps/ \
    --outdir outputs/preview/ \
    --ddim_steps 20

# Multiple variations
python scripts/inference_edge_to_image.py \
    --edge-img inputs/edge_maps/ \
    --outdir outputs/variations/ \
    --ddim_steps 100 \
    --n_samples 4 \
    --seed 42
```

### Test and Compare

```bash
# Automated test with multiple settings
./test_edge_to_image.sh

# Results in: test_edge_to_image_[timestamp]/
#   - edges/               (extracted edges)
#   - generated/           (generated images)
#   - comparison.jpg       (side-by-side comparison)
```

---

## Key Parameters

### Quality Settings

| Goal | DDIM Steps | Start Mode | Time/Image |
|------|-----------|------------|------------|
| **Fast Preview** | 20 | Pure noise | ~2 sec |
| **Balanced** | 50 | From edge | ~5 sec |
| **High Quality** | 100 | From edge | ~10 sec |
| **Best Quality** | 200 | From edge | ~20 sec |

### Parameter Reference

```bash
--ddim_steps 50              # Sampling steps (20=fast, 200=best)
--start_from_edge            # Start from noisy edge (more faithful)
--start_timestep 999         # Noise level (0-999, 999=most noisy)
--dec_w 0.0                  # Decoder fusion (0.0=clean, 0.5=edge-enhanced)
--seed 42                    # Random seed (change for variations)
--n_samples 1                # Samples per edge
--input_size 512             # Image size (must match training)
```

---

## Expected Results

### ✅ What Works Best

- Clear, continuous edges (buildings, objects)
- 2-3 pixel edge thickness
- High contrast (white edges, black background)
- Simple to moderate complexity
- Well-connected structures

### ⚠️ Potential Issues

- Very sparse/broken edges → Add more edge detail
- Too dense edges → Simplify or blur slightly
- Blurry output → Increase `--ddim_steps`
- Ignores edges → Use `--start_from_edge`
- Artifacts → Set `--dec_w 0.0`

---

## Directory Structure

```
StableSR_Canny/
├── inputs/
│   └── edge_maps/              # Put your edge maps here
│       ├── image1_edge.png
│       └── image2_edge.png
├── outputs/
│   └── edge_to_image_*/        # Generated images here
│       ├── image1_generated.png
│       ├── image1_edge_ref.png
│       └── preview_montage.jpg
└── logs/
    └── your_experiment/
        └── checkpoints/
            └── last.ckpt       # Your trained model
```

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "Checkpoint not found" | Update `CHECKPOINT` path in script |
| "No edge maps found" | Check image format (PNG/JPG) and path |
| Blurry results | Increase `--ddim_steps` to 100-200 |
| Out of memory | Reduce `--n_samples` to 1 |
| Same results always | Change `--seed` parameter |

---

## Configuration Locations

Edit these if needed:

1. **Shell script settings**: Edit `inference_edge_to_image.sh` lines 10-30
2. **Model config**: `configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml`
3. **Checkpoint path**: Default searches `logs/*/checkpoints/last.ckpt`
4. **VQGAN path**: Default `/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt`

---

## Example Workflow

```bash
# 1. Extract edges from your images
python scripts/extract_canny_edges.py \
    --input ~/my_photos/ \
    --output inputs/edge_maps/ \
    --adaptive

# 2. Test generation (creates comparison)
./test_edge_to_image.sh

# 3. If results look good, batch process
./inference_edge_to_image.sh

# 4. View results
eog outputs/edge_to_image_*/preview_montage.jpg
```

---

## Performance Tips

- **Fast iteration**: Use 20-50 steps
- **Production**: Use 100-200 steps  
- **Batch processing**: Set `--max_images` to process in chunks
- **Memory limit**: Process 512×512 images use ~4GB VRAM
- **Speed up**: Lower resolution or fewer steps

---

## Next Steps After Testing

1. ✅ **Works well**: Use for your application
2. 🔧 **Needs tuning**: Adjust parameters (see full docs)
3. 🔄 **Try Option A**: If results poor, modify to use edge twice
4. 🎓 **Fine-tune model**: Train specifically for edge-to-image task

---

## Documentation

- **Quick Start**: This file (you are here)
- **User Guide**: `EDGE_TO_IMAGE_INFERENCE.md` (detailed usage)
- **Technical**: `IMPLEMENTATION_SUMMARY.md` (architecture details)

---

## Support Checklist

Before asking for help, verify:
- [ ] Checkpoint path is correct
- [ ] VQGAN checkpoint exists
- [ ] Edge maps are in PNG/JPG format
- [ ] CUDA GPU is available
- [ ] Python packages installed (`pip install -r requirements.txt`)
- [ ] Tested with default parameters first

---

## One-Line Quick Test

```bash
./test_edge_to_image.sh && eog test_edge_to_image_*/comparison.jpg
```

**Expected**: Generates comparison showing edge → generated images in ~30 seconds.

---

Good luck! 🎨✨


