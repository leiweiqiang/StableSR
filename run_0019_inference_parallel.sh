#!/bin/bash

# Parallel inference script for image 0019 using multiple GPUs
# This script runs 6 tasks in parallel across 8 GPUs

set -e

echo "=========================================="
echo "Parallel Inference for image 0019"
echo "=========================================="
echo "Using multiple GPUs for parallel processing"
echo ""

# Configuration
OUTPUT_BASE_CUSTOM="outputs/0019_inference"
CONFIG_STABLESR="configs/stableSRNew/v2-finetune_text_T_512.yaml"
CONFIG_CANNY="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
STABLESR_CKPT="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"
CKPT_4X="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000111.ckpt"

# Inference parameters
DDPM_STEPS=200
TILE_SIZE=512
TILE_OVERLAP=32
VQGAN_TILE_SIZE=1024
VQGAN_TILE_STRIDE=512
SEED=42
DEC_W=0.0
DEC_W_STABLESR=0.5

# Input files
LR_4X="/stablesr_dataset/weiqiang/lr_x4_960x540/0019.png"
LR_8X="/stablesr_dataset/weiqiang/lr_x8_480x270/0019.png"
EDGE="/stablesr_dataset/weiqiang/hr_4k_canny/0019.png"

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
    
    # Run on specific GPU with conda environment
    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate sr && $command"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $task_name on GPU $gpu_id"
    else
        echo "❌ Failed: $task_name on GPU $gpu_id"
        return 1
    fi
}

# Create output directories
mkdir -p "$OUTPUT_BASE_CUSTOM/4x"
mkdir -p "$OUTPUT_BASE_CUSTOM/8x"

# Define all tasks
echo "Defining tasks..."

# Task 1: Edge 4x
TASK1_OUT="$OUTPUT_BASE_CUSTOM/4x/0019_4x_w0.0_e111"
TASK1_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_4X\" \
    --edge-img \"$EDGE\" \
    --outdir \"$TASK1_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_4X\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha 1.0 \
    --blend-beta 1.0 \
    --lr-downscale-factor 4 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix adain \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Task 2: Noedge 4x
TASK2_OUT="$OUTPUT_BASE_CUSTOM/4x/0019_4x_noedge_w0.0_e111"
TASK2_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_4X\" \
    --noedge \
    --outdir \"$TASK2_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_4X\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha 1.0 \
    --blend-beta 1.0 \
    --lr-downscale-factor 4 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix adain \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Task 3: Edge 8x
TASK3_OUT="$OUTPUT_BASE_CUSTOM/8x/0019_8x_w0.0_e111"
TASK3_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_8X\" \
    --edge-img \"$EDGE\" \
    --outdir \"$TASK3_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_8X\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha 1.0 \
    --blend-beta 1.0 \
    --lr-downscale-factor 8 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix adain \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Task 4: Noedge 8x
TASK4_OUT="$OUTPUT_BASE_CUSTOM/8x/0019_8x_noedge_w0.0_e111"
TASK4_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_8X\" \
    --noedge \
    --outdir \"$TASK4_OUT\" \
    --config \"$CONFIG_CANNY\" \
    --ckpt \"$CKPT_8X\" \
    --vqgan-ckpt \"$VQGAN_CKPT\" \
    --blend-alpha 1.0 \
    --blend-beta 1.0 \
    --lr-downscale-factor 8 \
    --ddpm-steps $DDPM_STEPS \
    --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP \
    --seed $SEED \
    --colorfix adain \
    --vqgan-tile-size $VQGAN_TILE_SIZE \
    --vqgan-tile-stride $VQGAN_TILE_STRIDE \
    --dec-w $DEC_W"

# Task 5: StableSR 4x
TASK5_OUT="$OUTPUT_BASE_CUSTOM/4x/0019_4x_stablesr_w0.5"
TASK5_TEMP="$TASK5_OUT/temp_input"
mkdir -p "$TASK5_TEMP"
cp "$LR_4X" "$TASK5_TEMP/"
TASK5_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_STABLESR\" \
    --ckpt \"$STABLESR_CKPT\" \
    --init-img \"$TASK5_TEMP\" \
    --outdir \"$TASK5_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w $DEC_W_STABLESR \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 4"

# Task 6: StableSR 8x
TASK6_OUT="$OUTPUT_BASE_CUSTOM/8x/0019_8x_stablesr_w0.5"
TASK6_TEMP="$TASK6_OUT/temp_input"
mkdir -p "$TASK6_TEMP"
cp "$LR_8X" "$TASK6_TEMP/"
TASK6_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_STABLESR\" \
    --ckpt \"$STABLESR_CKPT\" \
    --init-img \"$TASK6_TEMP\" \
    --outdir \"$TASK6_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w $DEC_W_STABLESR \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 8"

# Run tasks in parallel using different GPUs
# Note: Conda environment is activated within each background task
echo ""
echo "=========================================="
echo "Starting parallel execution"
echo "=========================================="
echo ""

# Run all tasks in background on different GPUs
run_task_on_gpu 0 "Task 1: Edge 4x" "$TASK1_OUT" "$TASK1_CMD" &
TASK1_PID=$!

run_task_on_gpu 1 "Task 2: Noedge 4x" "$TASK2_OUT" "$TASK2_CMD" &
TASK2_PID=$!

run_task_on_gpu 2 "Task 3: Edge 8x" "$TASK3_OUT" "$TASK3_CMD" &
TASK3_PID=$!

run_task_on_gpu 3 "Task 4: Noedge 8x" "$TASK4_OUT" "$TASK4_CMD" &
TASK4_PID=$!

run_task_on_gpu 4 "Task 5: StableSR 4x" "$TASK5_OUT" "$TASK5_CMD" &
TASK5_PID=$!

run_task_on_gpu 5 "Task 6: StableSR 8x" "$TASK6_OUT" "$TASK6_CMD" &
TASK6_PID=$!

# Wait for all tasks to complete
echo ""
echo "Waiting for all tasks to complete..."
wait $TASK1_PID
wait $TASK2_PID
wait $TASK3_PID
wait $TASK4_PID
wait $TASK5_PID
wait $TASK6_PID

# Cleanup temporary directories
rm -rf "$TASK5_TEMP" "$TASK6_TEMP" 2>/dev/null || true

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output location: $OUTPUT_BASE_CUSTOM"
echo ""
echo "Results:"
ls -la "$OUTPUT_BASE_CUSTOM/4x/"
ls -la "$OUTPUT_BASE_CUSTOM/8x/"
