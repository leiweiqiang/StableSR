#!/bin/bash

# Quick script to run inference for image 0019
# Based on the requirements:
# (1) 0019 edge 4x w0.0 (epoch 111)
# (2) 0019 noedge 4x w0.0 (epoch 111)
# (3) 0019 edge 8x w0.0 (epoch 111)
# (4) 0019 noedge 8x w0.0 (epoch 111)
# (5) 0019 stablesr 4x w0.5
# (6) 0019 stablesr 8x w0.5

set -e

# Cleanup function to restore files in case of error
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Restore 4x file if backed up
    if [ "$FOUR_X_BAKED" = true ] && [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak" ]; then
        mv /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png
        echo "Restored 4x file"
    fi
    # Restore 8x file if backed up
    if [ "$EIGHT_X_BAKED" = true ] && [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak" ]; then
        mv /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png
        echo "Restored 8x file"
    fi
    # Restore batch script
    if [ -f "batch_inference_blended_tile.sh.bak" ]; then
        cp batch_inference_blended_tile.sh.bak batch_inference_blended_tile.sh
        echo "Restored batch script"
    fi
}

trap cleanup EXIT ERR

echo "=========================================="
echo "Running inference for image 0019"
echo "=========================================="
echo ""
echo "Task 1: 0019 with edge, 4x, w0.0, epoch 111"
echo "Task 2: 0019 no edge, 4x, w0.0, epoch 111"
echo "Task 3: 0019 with edge, 8x, w0.0, epoch 111"
echo "Task 4: 0019 no edge, 8x, w0.0, epoch 111"
echo "Task 5: 0019 stablesr, 4x, w0.5"
echo "Task 6: 0019 stablesr, 8x, w0.5"
echo ""

# Set custom output directory
OUTPUT_BASE_ORIGINAL="outputs"
OUTPUT_BASE_CUSTOM="outputs/0019_inference"

echo "Output directory: $OUTPUT_BASE_CUSTOM"
echo ""

# Configuration for StableSR inference
CONFIG_STABLESR="configs/stableSRNew/v2-finetune_text_T_512.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
STABLESR_CKPT="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"
DEC_W=0.5
DDPM_STEPS=200
SEED=42
N_SAMPLES=1

# Function to run StableSR inference for a single image
run_stablesr_inference() {
    local image_path="$1"
    local upscale="$2"
    local output_dir="$3"
    local scale_name="$4"
    
    echo ""
    echo "=========================================="
    echo "Running StableSR inference for ${scale_name}"
    echo "=========================================="
    
    local image_name=$(basename "$image_path")
    local image_id=$(basename "$image_path" .png)
    
    echo "Processing ${scale_name}: $image_name"
    echo "LR Image: $image_path"
    echo "Output: $output_dir"
    echo "Upscale: $upscale"
    echo "DEC_W: $DEC_W"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Create temporary input directory for this single image
    local temp_input_dir="$output_dir/temp_input"
    mkdir -p "$temp_input_dir"
    
    # Copy the single image to temp directory
    cp "$image_path" "$temp_input_dir/"
    
    # Generate command file
    local command_file="$output_dir/inference_command.txt"
    echo "# StableSR Inference Command for ${scale_name}: $image_name" > "$command_file"
    echo "# Generated on: $(date)" >> "$command_file"
    echo "# LR Image: $image_path" >> "$command_file"
    echo "# Output Directory: $output_dir" >> "$command_file"
    echo "# Upscale: $upscale" >> "$command_file"
    echo "# DEC_W: $DEC_W" >> "$command_file"
    echo "" >> "$command_file"
    echo "python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \\" >> "$command_file"
    echo "    --config \"$CONFIG_STABLESR\" \\" >> "$command_file"
    echo "    --ckpt \"$STABLESR_CKPT\" \\" >> "$command_file"
    echo "    --init-img \"$temp_input_dir\" \\" >> "$command_file"
    echo "    --outdir \"$output_dir\" \\" >> "$command_file"
    echo "    --ddpm_steps \"$DDPM_STEPS\" \\" >> "$command_file"
    echo "    --dec_w \"$DEC_W\" \\" >> "$command_file"
    echo "    --seed \"$SEED\" \\" >> "$command_file"
    echo "    --n_samples \"$N_SAMPLES\" \\" >> "$command_file"
    echo "    --vqgan_ckpt \"$VQGAN_CKPT\" \\" >> "$command_file"
    echo "    --upscale \"$upscale\"" >> "$command_file"
    
    echo "📝 Command saved to: $command_file"
    
    # Activate conda environment and run inference
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate sr
    
    python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
        --config "$CONFIG_STABLESR" \
        --ckpt "$STABLESR_CKPT" \
        --init-img "$temp_input_dir" \
        --outdir "$output_dir" \
        --ddpm_steps "$DDPM_STEPS" \
        --dec_w "$DEC_W" \
        --seed "$SEED" \
        --n_samples "$N_SAMPLES" \
        --vqgan_ckpt "$VQGAN_CKPT" \
        --upscale "$upscale"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed StableSR ${scale_name}: $image_name"
        # Clean up temporary input directory
        rm -rf "$temp_input_dir"
    else
        echo "❌ Failed to process StableSR ${scale_name}: $image_name"
        # Clean up temporary input directory even on failure
        rm -rf "$temp_input_dir"
        return 1
    fi
}

# Check if required StableSR checkpoint exists
if [ ! -f "$STABLESR_CKPT" ]; then
    echo "⚠️  Warning: StableSR checkpoint not found: $STABLESR_CKPT"
    echo "StableSR inference (tasks 5-6) will be skipped"
fi

# Backup and modify batch script to use custom output
cp batch_inference_blended_tile.sh batch_inference_blended_tile.sh.bak
sed -i "s|^OUTPUT_BASE=\"outputs\"|OUTPUT_BASE=\"$OUTPUT_BASE_CUSTOM\"|" batch_inference_blended_tile.sh

# Task 1: Run with edge (NOEDGE=false)
echo ""
echo "=========================================="
echo "Task 1: Running 0019 with edge, 4x"
echo "=========================================="

# Modify NOEDGE to false for edge run (only 4x)
echo "Setting NOEDGE=false for edge processing..."
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

echo "Running: 4x with edge..."

# First, let's ensure 8x LR file doesn't exist temporarily
if [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak
    EIGHT_X_BAKED=true
else
    EIGHT_X_BAKED=false
fi

./batch_inference_blended_tile.sh --only-image 0019

# Restore 8x LR file
if [ "$EIGHT_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png
fi

# Restore NOEDGE=true for noedge runs
echo "Restoring NOEDGE=true for noedge processing..."
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh

echo ""
echo "=========================================="
echo "Task 2: Running 0019 without edge, 4x"
echo "=========================================="
# Temporarily hide 8x file so only 4x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak
    EIGHT_X_BAKED=true
else
    EIGHT_X_BAKED=false
fi

echo "Running: 4x without edge..."
./batch_inference_blended_tile.sh --only-image 0019 --noedge

# Restore 8x LR file
if [ "$EIGHT_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0019.png
fi

echo ""
echo "=========================================="
echo "Task 3: Running 0019 with edge, 8x"
echo "=========================================="

# Set NOEDGE=false for edge processing
echo "Setting NOEDGE=false for edge processing..."
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

# Temporarily hide 4x file so only 8x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak
    FOUR_X_BAKED=true
else
    FOUR_X_BAKED=false
fi

echo "Running: 8x with edge..."
./batch_inference_blended_tile.sh --only-image 0019

# Restore 4x LR file
if [ "$FOUR_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png
fi

# Restore NOEDGE=true for next task
echo "Restoring NOEDGE=true for noedge processing..."
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh

echo ""
echo "=========================================="
echo "Task 4: Running 0019 without edge, 8x"
echo "=========================================="
# Temporarily hide 4x file so only 8x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak
    FOUR_X_BAKED=true
else
    FOUR_X_BAKED=false
fi

echo "Running: 8x without edge..."
./batch_inference_blended_tile.sh --only-image 0019 --noedge

# Restore 4x LR file
if [ "$FOUR_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0019.png
fi

# Task 5: Run StableSR 4x
if [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png" ] && [ -f "$STABLESR_CKPT" ]; then
    OUTPUT_DIR_4X_STABLESR="$OUTPUT_BASE_CUSTOM/4x/0019_4x_stablesr_w0.5"
    
    # Check if output already exists
    if [ -d "$OUTPUT_DIR_4X_STABLESR" ]; then
        echo ""
        echo "⏭️  Task 5: Skipping StableSR 4x (already exists)"
        echo "Directory: $OUTPUT_DIR_4X_STABLESR"
    else
        run_stablesr_inference \
            "/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png" \
            "4" \
            "$OUTPUT_DIR_4X_STABLESR" \
            "4x"
    fi
else
    echo ""
    echo "⚠️  Task 5: Skipping StableSR 4x (input file or checkpoint not found)"
fi

# Task 6: Run StableSR 8x
if [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png" ] && [ -f "$STABLESR_CKPT" ]; then
    OUTPUT_DIR_8X_STABLESR="$OUTPUT_BASE_CUSTOM/8x/0019_8x_stablesr_w0.5"
    
    # Check if output already exists
    if [ -d "$OUTPUT_DIR_8X_STABLESR" ]; then
        echo ""
        echo "⏭️  Task 6: Skipping StableSR 8x (already exists)"
        echo "Directory: $OUTPUT_DIR_8X_STABLESR"
    else
        run_stablesr_inference \
            "/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png" \
            "8" \
            "$OUTPUT_DIR_8X_STABLESR" \
            "8x"
    fi
else
    echo ""
    echo "⚠️  Task 6: Skipping StableSR 8x (input file or checkpoint not found)"
fi

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output location: $OUTPUT_BASE_CUSTOM"
echo ""
echo "Output directories created:"
ls -la "$OUTPUT_BASE_CUSTOM/4x/" 2>/dev/null || echo "No 4x outputs"
ls -la "$OUTPUT_BASE_CUSTOM/8x/" 2>/dev/null || echo "No 8x outputs"

# Restore batch script to original state
if [ -f "batch_inference_blended_tile.sh.bak" ]; then
    cp batch_inference_blended_tile.sh.bak batch_inference_blended_tile.sh
    echo ""
    echo "✅ Batch script restored to original state"
fi
