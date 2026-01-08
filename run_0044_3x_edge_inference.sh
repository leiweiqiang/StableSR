#!/bin/bash

# Inference script for image 0044 with 3x_edge configuration
# ===========================================================
# This script runs inference for image 0044 with:
# - 3x scale: edge configuration

# Configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoint
# Using 4x model checkpoint (closest to 3x scale)
# If you have a dedicated 3x model, update this path
CKPT_3X="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"

# Input files
# Using 4x LR images - the script will use --lr-downscale-factor 3 to handle 3x scaling
# If you have dedicated 3x LR images, update this path
LR_3X="/stablesr_dataset/weiqiang/lr_x4_960x540/0044.png"
EDGE="/stablesr_dataset/weiqiang/hr_4k_canny/0044.png"

# Output base directory
OUTPUT_BASE="outputs/0044_inference"

# Processing parameters
BLEND_ALPHA=1.0
BLEND_BETA=1.0
DDPM_STEPS=200
TILE_SIZE=512
TILE_OVERLAP=32
SEED=42
COLORFIX="adain"
VQGAN_TILE_SIZE=1024
VQGAN_TILE_STRIDE=512

echo "Starting inference for image 0044 (3x_edge)..."
echo "=================================="

# Create output directories
mkdir -p "${OUTPUT_BASE}/3x"

# Function to save command to file
save_command() {
    local output_dir="$1"
    local command="$2"
    mkdir -p "$output_dir"
    echo "# Task: $3" > "$output_dir/inference_command.txt"
    echo "# Generated on: $(date)" >> "$output_dir/inference_command.txt"
    echo "$command" >> "$output_dir/inference_command.txt"
}

# 3x Scale - Edge
echo "Processing 3x scale (edge)..."

OUTPUT_DIR_3X_EDGE="${OUTPUT_BASE}/3x/0044_3x_edge_w0.0"
CMD_3X_EDGE="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_3X}\" \
    --edge-img \"${EDGE}\" \
    --outdir \"${OUTPUT_DIR_3X_EDGE}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_3X}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 3 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.0"

save_command "${OUTPUT_DIR_3X_EDGE}" "${CMD_3X_EDGE}" "3x Edge"
eval "${CMD_3X_EDGE}"

echo "=================================="
echo "3x_edge inference completed!"
echo "Results saved to: ${OUTPUT_DIR_3X_EDGE}/"
echo ""

