#!/bin/bash

# Batch Inference Script for 4x and 8x StableSR Tiled Inference
# =============================================================
# This script processes multiple images with different scale factors (4x and 8x)
# using the StableSR tiled inference approach.
#
# Key parameters:
# - DEC_W: Weight for combining VQGAN and Diffusion (default: 0.5)
# - DDPM_STEPS: Number of sampling steps (default: 200)
# - UPSCALE: Scale factor for super-resolution (4x or 8x)
#
# Usage:
#   ./batch_inference_stablesr_tile.sh
#
# Output:
#   Each image will be processed and saved to outputs/4x/ or outputs/8x/
#   Output directory naming: {image_id}_{scale}_stablesr_w{value}
#   Examples:
#     - 0001_4x_stablesr_w0.5 (4x scale, stablesr, w=0.5)
#     - 0001_8x_stablesr_w0.0 (8x scale, stablesr, w=0.0)
#   A command file (inference_command.txt) will be saved in each output directory
#   containing the exact command used for that specific image inference

set -e  # Exit on any error

# Configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoints
STABLESR_CKPT="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"

# Input directories
LR_4X_DIR="/stablesr_dataset/weiqiang/lr_x4_960x540"
LR_8X_DIR="/stablesr_dataset/weiqiang/lr_x8_480x270"

# Output base directory
OUTPUT_BASE="outputs"

# Processing parameters
DDPM_STEPS=200
SEED=42
N_SAMPLES=1
DEC_W=0.5

# Scale factors and upscale parameters
LR_DOWNSCALE_FACTOR_4X=4
LR_DOWNSCALE_FACTOR_8X=8
UPSCALE_4X=4
UPSCALE_8X=8

# Images to skip (already processed)
SKIP_IMAGES=("0001.png" "0013.png" "0031.png" "0044.png")

echo "=========================================="
echo "Batch Inference for 4x and 8x Images (StableSR)"
echo "=========================================="
echo "4x LR Directory: $LR_4X_DIR"
echo "8x LR Directory: $LR_8X_DIR"
echo "StableSR Checkpoint: $STABLESR_CKPT"
echo "Skip Images: ${SKIP_IMAGES[*]}"
echo "=========================================="

# Function to check if image should be skipped
should_skip_image() {
    local image_name="$1"
    for skip_img in "${SKIP_IMAGES[@]}"; do
        if [[ "$image_name" == "$skip_img" ]]; then
            return 0  # Should skip
        fi
    done
    return 1  # Should not skip
}

# Function to process a single image
process_image() {
    local lr_img="$1"
    local output_dir="$2"
    local upscale="$3"
    local scale_name="$4"
    
    local image_name=$(basename "$lr_img")
    local image_id=$(basename "$lr_img" .png)
    
    echo ""
    echo "Processing $scale_name: $image_name"
    echo "LR Image: $lr_img"
    echo "Output: $output_dir"
    echo "Upscale: $upscale"
    echo "----------------------------------------"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Create temporary input directory for this single image
    local temp_input_dir="$output_dir/temp_input"
    mkdir -p "$temp_input_dir"
    
    # Copy the single image to temp directory
    cp "$lr_img" "$temp_input_dir/"
    
    # Generate command file
    local command_file="$output_dir/inference_command.txt"
    echo "# Inference Command for $scale_name: $image_name" > "$command_file"
    echo "# Generated on: $(date)" >> "$command_file"
    echo "# LR Image: $lr_img" >> "$command_file"
    echo "# Output Directory: $output_dir" >> "$command_file"
    echo "# Upscale: $upscale" >> "$command_file"
    echo "# DEC_W: $DEC_W" >> "$command_file"
    echo "" >> "$command_file"
    echo "python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \\" >> "$command_file"
    echo "    --config \"$CONFIG\" \\" >> "$command_file"
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
        --config "$CONFIG" \
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
        echo "✅ Successfully processed $scale_name: $image_name"
        # Clean up temporary input directory
        rm -rf "$temp_input_dir"
    else
        echo "❌ Failed to process $scale_name: $image_name"
        # Clean up temporary input directory even on failure
        rm -rf "$temp_input_dir"
        return 1
    fi
}

# Check if required files and directories exist
echo "Checking required files and directories..."

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$VQGAN_CKPT" ]; then
    echo "❌ Error: VQGAN checkpoint not found: $VQGAN_CKPT"
    exit 1
fi

if [ ! -f "$STABLESR_CKPT" ]; then
    echo "❌ Error: StableSR checkpoint not found: $STABLESR_CKPT"
    exit 1
fi

if [ ! -d "$LR_4X_DIR" ]; then
    echo "❌ Error: 4x LR directory not found: $LR_4X_DIR"
    exit 1
fi

if [ ! -d "$LR_8X_DIR" ]; then
    echo "❌ Error: 8x LR directory not found: $LR_8X_DIR"
    exit 1
fi

echo "✅ All required files and directories found"

# Get list of images to process
echo ""
echo "Scanning for images to process..."

# Get all PNG files from 4x directory
LR_4X_IMAGES=()
for img in "$LR_4X_DIR"/*.png; do
    if [ -f "$img" ]; then
        image_name=$(basename "$img")
        if ! should_skip_image "$image_name"; then
            LR_4X_IMAGES+=("$img")
        else
            echo "⏭️  Skipping 4x: $image_name (already processed)"
        fi
    fi
done

# Get all PNG files from 8x directory
LR_8X_IMAGES=()
for img in "$LR_8X_DIR"/*.png; do
    if [ -f "$img" ]; then
        image_name=$(basename "$img")
        if ! should_skip_image "$image_name"; then
            LR_8X_IMAGES+=("$img")
        else
            echo "⏭️  Skipping 8x: $image_name (already processed)"
        fi
    fi
done

echo "Found ${#LR_4X_IMAGES[@]} 4x images to process"
echo "Found ${#LR_8X_IMAGES[@]} 8x images to process"

if [ ${#LR_4X_IMAGES[@]} -eq 0 ] && [ ${#LR_8X_IMAGES[@]} -eq 0 ]; then
    echo "❌ No images to process (all images are skipped or not found)"
    exit 1
fi

# Process 4x images
if [ ${#LR_4X_IMAGES[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Processing 4x Images (StableSR)"
    echo "=========================================="
    
    for lr_img in "${LR_4X_IMAGES[@]}"; do
        image_name=$(basename "$lr_img")
        image_id=$(basename "$lr_img" .png)
        
        # Build output directory name with identifiers
        output_suffix="4x_stablesr_w${DEC_W}"
        output_dir="$OUTPUT_BASE/4x/${image_id}_${output_suffix}"
        
        process_image "$lr_img" "$output_dir" "$UPSCALE_4X" "4x"
    done
fi

# Process 8x images
if [ ${#LR_8X_IMAGES[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Processing 8x Images (StableSR)"
    echo "=========================================="
    
    for lr_img in "${LR_8X_IMAGES[@]}"; do
        image_name=$(basename "$lr_img")
        image_id=$(basename "$lr_img" .png)
        
        # Build output directory name with identifiers
        output_suffix="8x_stablesr_w${DEC_W}"
        output_dir="$OUTPUT_BASE/8x/${image_id}_${output_suffix}"
        
        process_image "$lr_img" "$output_dir" "$UPSCALE_8X" "8x"
    done
fi

echo ""
echo "=========================================="
echo "Batch Inference Completed!"
echo "=========================================="
echo "4x images processed: ${#LR_4X_IMAGES[@]}"
echo "8x images processed: ${#LR_8X_IMAGES[@]}"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="

# Show output structure
echo ""
echo "Output structure:"
echo "Directory naming: {image_id}_{scale}_stablesr_w{value}"
if [ -d "$OUTPUT_BASE/4x" ]; then
    echo "4x outputs:"
    ls -la "$OUTPUT_BASE/4x" | grep "^d" | awk '{print "  " $9}'
    echo "  Command files: inference_command.txt (in each subdirectory)"
fi

if [ -d "$OUTPUT_BASE/8x" ]; then
    echo "8x outputs:"
    ls -la "$OUTPUT_BASE/8x" | grep "^d" | awk '{print "  " $9}'
    echo "  Command files: inference_command.txt (in each subdirectory)"
fi

echo ""
echo "✅ Batch inference completed successfully!"
