#!/bin/bash

# Inference script for image 0044 with multiple configurations
# ===========================================================
# This script runs inference for image 0044 with:
# - 4x scale: 3 configurations (stablesr w=0.5, noedge, edge) on epoch 111
# - 8x scale: 3 configurations (stablesr w=0.5, noedge, edge) on epoch 194

# Configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoints
CKPT_4X_E111="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X_E194="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000194.ckpt"

# Input files
LR_4X="/stablesr_dataset/weiqiang/lr_x4_960x540/0044.png"
LR_8X="/stablesr_dataset/weiqiang/lr_x8_480x270/0044.png"
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

echo "Starting inference for image 0044..."
echo "=================================="

# Create output directories
mkdir -p "${OUTPUT_BASE}/4x"
mkdir -p "${OUTPUT_BASE}/8x"

# Function to save command to file
save_command() {
    local output_dir="$1"
    local command="$2"
    mkdir -p "$output_dir"
    echo "# Task: $3" > "$output_dir/inference_command.txt"
    echo "# Generated on: $(date)" >> "$output_dir/inference_command.txt"
    echo "$command" >> "$output_dir/inference_command.txt"
}

# 4x Scale - Epoch 111
echo "Processing 4x scale (epoch 111)..."

# 1. 4x StableSR with w=0.5
echo "  - 4x StableSR w=0.5..."
OUTPUT_DIR_4X_STABLESR="${OUTPUT_BASE}/4x/0044_4x_stablesr_w0.5_e111"
CMD_4X_STABLESR="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_4X}\" \
    --edge-img \"${EDGE}\" \
    --outdir \"${OUTPUT_DIR_4X_STABLESR}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_4X_E111}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 4 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.5"

save_command "${OUTPUT_DIR_4X_STABLESR}" "${CMD_4X_STABLESR}" "4x StableSR w=0.5 on epoch 111"
eval "${CMD_4X_STABLESR}"

# 2. 4x Noedge
echo "  - 4x Noedge..."
OUTPUT_DIR_4X_NOEDGE="${OUTPUT_BASE}/4x/0044_4x_noedge_w0.0_e111"
CMD_4X_NOEDGE="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_4X}\" \
    --noedge \
    --outdir \"${OUTPUT_DIR_4X_NOEDGE}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_4X_E111}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 4 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.0"

save_command "${OUTPUT_DIR_4X_NOEDGE}" "${CMD_4X_NOEDGE}" "4x Noedge on epoch 111"
eval "${CMD_4X_NOEDGE}"

# 3. 4x Edge
echo "  - 4x Edge..."
OUTPUT_DIR_4X_EDGE="${OUTPUT_BASE}/4x/0044_4x_edge_w0.0_e111"
CMD_4X_EDGE="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_4X}\" \
    --edge-img \"${EDGE}\" \
    --outdir \"${OUTPUT_DIR_4X_EDGE}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_4X_E111}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 4 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.0"

save_command "${OUTPUT_DIR_4X_EDGE}" "${CMD_4X_EDGE}" "4x Edge on epoch 111"
eval "${CMD_4X_EDGE}"

# 8x Scale - Epoch 194
echo "Processing 8x scale (epoch 194)..."

# 4. 8x StableSR with w=0.5
echo "  - 8x StableSR w=0.5..."
OUTPUT_DIR_8X_STABLESR="${OUTPUT_BASE}/8x/0044_8x_stablesr_w0.5_e194"
CMD_8X_STABLESR="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_8X}\" \
    --edge-img \"${EDGE}\" \
    --outdir \"${OUTPUT_DIR_8X_STABLESR}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_8X_E194}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 8 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.5"

save_command "${OUTPUT_DIR_8X_STABLESR}" "${CMD_8X_STABLESR}" "8x StableSR w=0.5 on epoch 194"
eval "${CMD_8X_STABLESR}"

# 5. 8x Noedge
echo "  - 8x Noedge..."
OUTPUT_DIR_8X_NOEDGE="${OUTPUT_BASE}/8x/0044_8x_noedge_w0.0_e194"
CMD_8X_NOEDGE="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_8X}\" \
    --noedge \
    --outdir \"${OUTPUT_DIR_8X_NOEDGE}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_8X_E194}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 8 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.0"

save_command "${OUTPUT_DIR_8X_NOEDGE}" "${CMD_8X_NOEDGE}" "8x Noedge on epoch 194"
eval "${CMD_8X_NOEDGE}"

# 6. 8x Edge
echo "  - 8x Edge..."
OUTPUT_DIR_8X_EDGE="${OUTPUT_BASE}/8x/0044_8x_edge_w0.0_e194"
CMD_8X_EDGE="python scripts/inference_blended_input_tile.py \
    --lr-img \"${LR_8X}\" \
    --edge-img \"${EDGE}\" \
    --outdir \"${OUTPUT_DIR_8X_EDGE}\" \
    --config \"${CONFIG}\" \
    --ckpt \"${CKPT_8X_E194}\" \
    --vqgan-ckpt \"${VQGAN_CKPT}\" \
    --blend-alpha ${BLEND_ALPHA} \
    --blend-beta ${BLEND_BETA} \
    --lr-downscale-factor 8 \
    --ddpm-steps ${DDPM_STEPS} \
    --tile-size ${TILE_SIZE} \
    --tile-overlap ${TILE_OVERLAP} \
    --seed ${SEED} \
    --colorfix ${COLORFIX} \
    --vqgan-tile-size ${VQGAN_TILE_SIZE} \
    --vqgan-tile-stride ${VQGAN_TILE_STRIDE} \
    --dec-w 0.0"

save_command "${OUTPUT_DIR_8X_EDGE}" "${CMD_8X_EDGE}" "8x Edge on epoch 194"
eval "${CMD_8X_EDGE}"

echo "=================================="
echo "All inference tasks completed!"
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Output structure:"
echo "4x results:"
echo "  - ${OUTPUT_BASE}/4x/0044_4x_stablesr_w0.5_e111/"
echo "  - ${OUTPUT_BASE}/4x/0044_4x_noedge_w0.0_e111/"
echo "  - ${OUTPUT_BASE}/4x/0044_4x_edge_w0.0_e111/"
echo ""
echo "8x results:"
echo "  - ${OUTPUT_BASE}/8x/0044_8x_stablesr_w0.5_e194/"
echo "  - ${OUTPUT_BASE}/8x/0044_8x_noedge_w0.0_e194/"
echo "  - ${OUTPUT_BASE}/8x/0044_8x_edge_w0.0_e194/"
