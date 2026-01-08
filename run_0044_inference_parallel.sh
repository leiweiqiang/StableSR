#!/bin/bash

# Parallel inference script for image 0044 using multiple GPUs
# This script runs 2 tasks in parallel across 2 GPUs
# ===========================================================
# Tasks:
# - 4x scale: edge configuration on epoch 111
# - 8x scale: edge configuration on epoch 194

set -e

echo "=========================================="
echo "Parallel Inference for image 0044"
echo "=========================================="
echo "Using 2 GPUs for parallel processing"
echo ""

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="outputs/0044_inference_${TIMESTAMP}"
CONFIG_CANNY="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoints
CKPT_4X_E111="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X_E194="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000194.ckpt"

# Input files
LR_4X="/stablesr_dataset/weiqiang/lr_x4_960x540/0044.png"
LR_8X="/stablesr_dataset/weiqiang/lr_x8_480x270/0044.png"
EDGE="/stablesr_dataset/weiqiang/hr_4k_canny/0044.png"

# Inference parameters
DDPM_STEPS=200
TILE_SIZE=512
TILE_OVERLAP=32
VQGAN_TILE_SIZE=1024
VQGAN_TILE_STRIDE=512
SEED=42
BLEND_ALPHA=1.0
BLEND_BETA=1.0
COLORFIX="adain"
DEC_W=0.0

# Function to run a single task on a specific GPU
run_task_on_gpu() {
    local gpu_id=$1
    local task_name=$2
    local output_dir=$3
    local command=$4
    
    echo ""
    echo "=========================================="
    echo "Task: $task_name on GPU $gpu_id"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # Check if already exists
    if [ -d "$output_dir" ] && ls "$output_dir"/*.png 1> /dev/null 2>&1; then
        echo "⏭️  Skipping (already exists): $task_name"
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Save command to file
    echo "# $task_name on GPU $gpu_id" > "$output_dir/inference_command.txt"
    echo "# Generated on: $(date)" >> "$output_dir/inference_command.txt"
    echo "$command" >> "$output_dir/inference_command.txt"
    
    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "$command"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $task_name on GPU $gpu_id"
    else
        echo "❌ Failed: $task_name on GPU $gpu_id"
        return 1
    fi
}

# Create output directories
mkdir -p "$OUTPUT_BASE/4x"
mkdir -p "$OUTPUT_BASE/8x"

# Define all tasks
echo "Defining tasks..."
echo "Input images will be saved in each task's output directory for inspection"

# Task 1: 4x Edge (epoch 111)
TASK1_OUT="$OUTPUT_BASE/4x/0044_4x_edge_w0.0_e111"
TASK1_INPUT_DIR="$TASK1_OUT/input_images"
mkdir -p "$TASK1_INPUT_DIR"
cp "$LR_4X" "$TASK1_INPUT_DIR/lr_4x_0044.png"
cp "$EDGE" "$TASK1_INPUT_DIR/edge_0044.png"
TASK1_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_4X\" \
    --edge-img \"$EDGE\" \
    --outdir \"$TASK1_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_4X_E111\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha $BLEND_ALPHA \
    --blend-beta $BLEND_BETA \
    --lr-downscale-factor 4 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix $COLORFIX \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Task 2: 8x Edge (epoch 194)
TASK2_OUT="$OUTPUT_BASE/8x/0044_8x_edge_w0.0_e194"
TASK2_INPUT_DIR="$TASK2_OUT/input_images"
mkdir -p "$TASK2_INPUT_DIR"
cp "$LR_8X" "$TASK2_INPUT_DIR/lr_8x_0044.png"
cp "$EDGE" "$TASK2_INPUT_DIR/edge_0044.png"
TASK2_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_8X\" \
    --edge-img \"$EDGE\" \
    --outdir \"$TASK2_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_8X_E194\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha $BLEND_ALPHA \
    --blend-beta $BLEND_BETA \
    --lr-downscale-factor 8 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix $COLORFIX \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Run tasks in parallel using different GPUs
echo ""
echo "=========================================="
echo "Starting parallel execution"
echo "=========================================="
echo ""

# Run all tasks in background on different GPUs
run_task_on_gpu 2 "Task 1: 4x Edge (epoch 111)" "$TASK1_OUT" "$TASK1_CMD" &
TASK1_PID=$!

run_task_on_gpu 5 "Task 2: 8x Edge (epoch 194)" "$TASK2_OUT" "$TASK2_CMD" &
TASK2_PID=$!

# Wait for all tasks to complete
echo ""
echo "Waiting for all tasks to complete..."
wait $TASK1_PID
wait $TASK2_PID

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output location: $OUTPUT_BASE"
echo ""
echo "Results:"
echo "4x results:"
ls -la "$OUTPUT_BASE/4x/"
echo ""
echo "8x results:"
ls -la "$OUTPUT_BASE/8x/"
echo ""
echo "GPU allocation summary:"
echo "GPU 2: 4x Edge (epoch 111)"
echo "GPU 5: 8x Edge (epoch 194)"