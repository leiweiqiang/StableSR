# Batch Inference: File Matching Guide

## Quick Answer

**Yes, files in the two directories must have the same name** (but can have different extensions).

---

## How File Matching Works

### **Matching Rules:**

1. **Same base name** (excluding extension)
2. Script tries exact name first, then tries `.png` extension
3. Files are matched one-to-one by name

### **Example:**

```
lr_dir/                    edge_dir/
├── cat.png      ←───→    ├── cat.png         ✅ Match
├── dog.jpg      ←───→    ├── dog.png         ✅ Match (different ext OK)
├── bird.png     ←───→    ├── bird.png        ✅ Match
└── fish.png              └── turtle.png      ❌ No match (fish skipped)
```

**Result**: Processes cat, dog, bird (3 images). Skips fish (no matching edge).

---

## Directory Structure Examples

### **✅ Correct Structure (Same Names):**

```
inputs/
├── lr_32x32/
│   ├── image001.png
│   ├── image002.png
│   ├── image003.png
│   ├── portrait_001.jpg
│   └── landscape_001.png
└── edges_512/
    ├── image001.png          # ✅ Matches image001.png
    ├── image002.png          # ✅ Matches image002.png
    ├── image003.png          # ✅ Matches image003.png
    ├── portrait_001.png      # ✅ Matches portrait_001.jpg
    └── landscape_001.png     # ✅ Matches landscape_001.png
```

**Result**: All 5 pairs processed

---

### **❌ Wrong Structure (Different Names):**

```
inputs/
├── lr_32x32/
│   ├── image_001.png
│   ├── image_002.png
│   └── image_003.png
└── edges_512/
    ├── edge_001.png          # ❌ Different name!
    ├── edge_002.png          # ❌ Different name!
    └── edge_003.png          # ❌ Different name!
```

**Result**: No pairs matched, nothing processed!

---

## File Extension Flexibility

### **✅ These Work:**

```
lr_dir/image.png  ←→  edge_dir/image.png   ✅ Both .png
lr_dir/image.jpg  ←→  edge_dir/image.png   ✅ Different extensions OK
lr_dir/image.png  ←→  edge_dir/image.jpg   ✅ Different extensions OK
```

### **Matching Algorithm:**

```python
# For LR image "cat.jpg":
1. Try: edge_dir/cat.jpg
2. If not found, try: edge_dir/cat.png
3. If still not found: Skip this image
```

---

## Auto-Detection Feature ⭐ NEW

### **No Need for --batch-mode Flag!**

The script now **automatically detects** if you're using directories:

```bash
# OLD way (explicit):
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/ \
    --batch-mode                    # Had to specify!

# NEW way (automatic):
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/    # Auto-detects batch mode!
```

**Output:**
```
📁 Detected directory inputs - automatically enabling batch mode
Found 10 LR images and 10 edge maps
Processing images: 100%|████████████| 10/10
```

---

## Renaming Files to Match

### **Option 1: Manual Renaming**

```bash
cd inputs/lr_32x32/
rename 's/lr_//' *.png  # Remove "lr_" prefix

cd ../edges_512/
rename 's/edge_//' *.png  # Remove "edge_" prefix
```

### **Option 2: Python Script**

```python
#!/usr/bin/env python3
# rename_pairs.py
import os
import shutil
import glob

lr_dir = "inputs/lr_old"
edge_dir = "inputs/edges_old"
output_lr = "inputs/lr_32x32"
output_edge = "inputs/edges_512"

os.makedirs(output_lr, exist_ok=True)
os.makedirs(output_edge, exist_ok=True)

lr_files = sorted(glob.glob(f"{lr_dir}/*.png"))
edge_files = sorted(glob.glob(f"{edge_dir}/*.png"))

print(f"Found {len(lr_files)} LR images and {len(edge_files)} edge maps")

for i, (lr, edge) in enumerate(zip(lr_files, edge_files)):
    # Create standardized name
    new_name = f"image_{i:04d}.png"
    
    shutil.copy(lr, os.path.join(output_lr, new_name))
    shutil.copy(edge, os.path.join(output_edge, new_name))
    
    if i < 3:  # Show first 3 examples
        print(f"  {os.path.basename(lr):30s} → {new_name}")
        print(f"  {os.path.basename(edge):30s} → {new_name}")
        print()

print(f"✓ Renamed {len(lr_files)} pairs")
print(f"  LR output: {output_lr}/")
print(f"  Edge output: {output_edge}/")
```

**Usage:**
```bash
python rename_pairs.py
```

**Result:**
```
inputs/
├── lr_32x32/
│   ├── image_0000.png
│   ├── image_0001.png
│   └── image_0002.png
└── edges_512/
    ├── image_0000.png
    ├── image_0001.png
    └── image_0002.png
```

---

## Error Messages Explained

### **Error 1: Is a directory**
```
IsADirectoryError: [Errno 21] Is a directory: '/path/to/dir/'
```

**Cause**: Passed directory without `--batch-mode` (now auto-detected!)

**Solution**: The script now auto-detects and enables batch mode

---

### **Error 2: Missing edge map**
```
Warning: No edge map found for image001.png, skipping...
```

**Cause**: LR image exists but no matching edge map

**Solution**: Ensure edge map with same name exists:
```bash
# Check what's missing:
ls lr_32x32/  # image001.png exists
ls edges_512/ # No image001.png!

# Create or copy edge map with matching name
cp edges_512/some_edge.png edges_512/image001.png
```

---

## Best Practices

### **1. Consistent Naming**
```
✅ GOOD:
lr/image001.png  +  edge/image001.png
lr/image002.png  +  edge/image002.png

❌ BAD:
lr/img_001.png   +  edge/edge_001.png  (different names!)
```

### **2. Use Sequential Numbers**
```
image_0000.png
image_0001.png
image_0002.png
...
```

### **3. Check Pairing Before Processing**

```bash
# List files to verify
ls -1 inputs/lr_32x32/ | sort > lr_list.txt
ls -1 inputs/edges_512/ | sort > edge_list.txt
diff lr_list.txt edge_list.txt  # Should be identical!
```

### **4. Test with Small Batch First**

```bash
# Create test subset
mkdir test_lr test_edges
cp inputs/lr_32x32/image001.png test_lr/
cp inputs/edges_512/image001.png test_edges/

# Test
python scripts/inference_blended_input.py \
    --lr-img test_lr/ \
    --edge-img test_edges/ \
    --outdir outputs/test/ \
    --ckpt model.ckpt

# If successful, run full batch
python scripts/inference_blended_input.py \
    --lr-img inputs/lr_32x32/ \
    --edge-img inputs/edges_512/ \
    --outdir outputs/full/ \
    --ckpt model.ckpt
```

---

## Quick Fix for Your Error

**You ran:**
```bash
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/edges/ \
    # Missing --batch-mode flag!
```

**Solutions:**

### **Solution 1: Add --batch-mode (OLD way)**
```bash
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/edges_512/ \
    --batch-mode  # ← Add this!
    --outdir outputs/ \
    --ckpt model.ckpt
```

### **Solution 2: Just re-run (NEW way - auto-detection)**
```bash
# The updated script will auto-detect directories!
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/edges_512/ \
    --outdir outputs/ \
    --ckpt model.ckpt
    # No --batch-mode needed!
```

---

## Verification Checklist

Before running batch inference:

- [ ] Both directories exist
- [ ] Files have matching names (same base name)
- [ ] Edge maps exist for all LR images
- [ ] File extensions are .png or .jpg

**Quick check script:**
```bash
#!/bin/bash
LR_DIR="inputs/lr_32x32"
EDGE_DIR="inputs/edges_512"

echo "Checking file matching..."
for lr_file in "$LR_DIR"/*.png "$LR_DIR"/*.jpg; do
    basename=$(basename "$lr_file")
    name_no_ext="${basename%.*}"
    
    if [ -f "$EDGE_DIR/$basename" ] || [ -f "$EDGE_DIR/$name_no_ext.png" ]; then
        echo "✓ $basename"
    else
        echo "✗ $basename - NO MATCHING EDGE!"
    fi
done
```

---

## Summary

### **File Matching Requirements:**
- ✅ Same base name (e.g., "image001")
- ✅ Can have different extensions (.png, .jpg)
- ✅ Must exist in both directories

### **Updated Script Features:**
- ✅ Auto-detects directories (no --batch-mode needed!)
- ✅ Clear error messages
- ✅ Validates inputs before processing
- ✅ Skips files without matches (with warning)

**Your command should now work:**
```bash
python scripts/inference_blended_input.py \
    --lr-img /home/tra/jlyi/10_lr_32x32/ \
    --edge-img /path/to/your/edges_512/ \
    --outdir outputs/results/ \
    --ckpt checkpoints/model.ckpt
```

Just make sure the edge directory has files with matching names! 🎯

