#!/bin/bash

# Parallel inference script for images 0056 and 0187 using multiple GPUs
# This script runs 12 tasks in parallel across multiple GPUs
# ===========================================================
# Tasks:
# - 0056: 4x scale (stablesr+w0.5, noedge, edge) + 8x scale (stablesr+w0.5, noedge, edge)
# - 0187: 4x scale (stablesr+w0.5, noedge, edge) + 8x scale (stablesr+w0.5, noedge, edge)

set -e

echo "=========================================="
echo "Parallel Inference for images 0056 and 0187"
echo "=========================================="
echo "Using multiple GPUs for parallel processing"
echo ""

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="outputs/0056_0187_inference_${TIMESTAMP}"

# Model configurations
CONFIG_CANNY="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
CONFIG_NOEDGE="configs/stableSRNew/v2-finetune_text_T_512.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoints
CKPT_4X_E111="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X_E194="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000194.ckpt"
CKPT_STABLESR="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"

# Input files base directory
INPUT_BASE="/stablesr_dataset/weiqiang/added_0056_0187"

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
mkdir -p "$OUTPUT_BASE/0056/4x"
mkdir -p "$OUTPUT_BASE/0056/8x"
mkdir -p "$OUTPUT_BASE/0187/4x"
mkdir -p "$OUTPUT_BASE/0187/8x"

# Define all tasks
echo "Defining tasks..."
echo "Input images will be saved in each task's output directory for inspection"

# Image 0056 tasks
LR_0056_4X="$INPUT_BASE/lr_x4_960x540/0056.png"
LR_0056_8X="$INPUT_BASE/lr_x8_480x270/0056.png"
EDGE_0056="$INPUT_BASE/hr_4k_canny/0056.png"

# Image 0187 tasks
LR_0187_4X="$INPUT_BASE/lr_x4_960x540/0187.png"
LR_0187_8X="$INPUT_BASE/lr_x8_480x270/0187.png"
EDGE_0187="$INPUT_BASE/hr_4k_canny/0187.png"

# Task 1: 0056 4x StableSR+w0.5
TASK1_OUT="$OUTPUT_BASE/0056/4x/0056_4x_stablesr_w0.5"
TASK1_TEMP="$TASK1_OUT/temp_input"
mkdir -p "$TASK1_TEMP"
cp "$LR_0056_4X" "$TASK1_TEMP/"
TASK1_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_NOEDGE\" \
    --ckpt \"$CKPT_STABLESR\" \
    --init-img \"$TASK1_TEMP\" \
    --outdir \"$TASK1_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w 0.5 \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 4"

# Task 2: 0056 4x NoEdge
TASK2_OUT="$OUTPUT_BASE/0056/4x/0056_4x_noedge_w0.0"
TASK2_INPUT_DIR="$TASK2_OUT/input_images"
mkdir -p "$TASK2_INPUT_DIR"
cp "$LR_0056_4X" "$TASK2_INPUT_DIR/lr_4x_0056.png"
TASK2_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0056_4X\" \
    --noedge \
    --outdir \"$TASK2_OUT\" \
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
    --dec-w 0.0"

# Task 3: 0056 4x Edge
TASK3_OUT="$OUTPUT_BASE/0056/4x/0056_4x_edge_w0.0"
TASK3_INPUT_DIR="$TASK3_OUT/input_images"
mkdir -p "$TASK3_INPUT_DIR"
cp "$LR_0056_4X" "$TASK3_INPUT_DIR/lr_4x_0056.png"
cp "$EDGE_0056" "$TASK3_INPUT_DIR/edge_0056.png"
TASK3_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0056_4X\" \
    --edge-img \"$EDGE_0056\" \
    --outdir \"$TASK3_OUT\" \
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
    --dec-w 0.0"

# Task 4: 0056 8x StableSR+w0.5
TASK4_OUT="$OUTPUT_BASE/0056/8x/0056_8x_stablesr_w0.5"
TASK4_TEMP="$TASK4_OUT/temp_input"
mkdir -p "$TASK4_TEMP"
cp "$LR_0056_8X" "$TASK4_TEMP/"
TASK4_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_NOEDGE\" \
    --ckpt \"$CKPT_STABLESR\" \
    --init-img \"$TASK4_TEMP\" \
    --outdir \"$TASK4_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w 0.5 \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 8"

# Task 5: 0056 8x NoEdge
TASK5_OUT="$OUTPUT_BASE/0056/8x/0056_8x_noedge_w0.0"
TASK5_INPUT_DIR="$TASK5_OUT/input_images"
mkdir -p "$TASK5_INPUT_DIR"
cp "$LR_0056_8X" "$TASK5_INPUT_DIR/lr_8x_0056.png"
TASK5_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0056_8X\" \
    --noedge \
    --outdir \"$TASK5_OUT\" \
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
    --dec-w 0.0"

# Task 6: 0056 8x Edge
TASK6_OUT="$OUTPUT_BASE/0056/8x/0056_8x_edge_w0.0"
TASK6_INPUT_DIR="$TASK6_OUT/input_images"
mkdir -p "$TASK6_INPUT_DIR"
cp "$LR_0056_8X" "$TASK6_INPUT_DIR/lr_8x_0056.png"
cp "$EDGE_0056" "$TASK6_INPUT_DIR/edge_0056.png"
TASK6_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0056_8X\" \
    --edge-img \"$EDGE_0056\" \
    --outdir \"$TASK6_OUT\" \
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
    --dec-w 0.0"

# Task 7: 0187 4x StableSR+w0.5
TASK7_OUT="$OUTPUT_BASE/0187/4x/0187_4x_stablesr_w0.5"
TASK7_TEMP="$TASK7_OUT/temp_input"
mkdir -p "$TASK7_TEMP"
cp "$LR_0187_4X" "$TASK7_TEMP/"
TASK7_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_NOEDGE\" \
    --ckpt \"$CKPT_STABLESR\" \
    --init-img \"$TASK7_TEMP\" \
    --outdir \"$TASK7_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w 0.5 \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 4"

# Task 8: 0187 4x NoEdge
TASK8_OUT="$OUTPUT_BASE/0187/4x/0187_4x_noedge_w0.0"
TASK8_INPUT_DIR="$TASK8_OUT/input_images"
mkdir -p "$TASK8_INPUT_DIR"
cp "$LR_0187_4X" "$TASK8_INPUT_DIR/lr_4x_0187.png"
TASK8_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0187_4X\" \
    --noedge \
    --outdir \"$TASK8_OUT\" \
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
    --dec-w 0.0"

# Task 9: 0187 4x Edge
TASK9_OUT="$OUTPUT_BASE/0187/4x/0187_4x_edge_w0.0"
TASK9_INPUT_DIR="$TASK9_OUT/input_images"
mkdir -p "$TASK9_INPUT_DIR"
cp "$LR_0187_4X" "$TASK9_INPUT_DIR/lr_4x_0187.png"
cp "$EDGE_0187" "$TASK9_INPUT_DIR/edge_0187.png"
TASK9_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0187_4X\" \
    --edge-img \"$EDGE_0187\" \
    --outdir \"$TASK9_OUT\" \
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
    --dec-w 0.0"

# Task 10: 0187 8x StableSR+w0.5
TASK10_OUT="$OUTPUT_BASE/0187/8x/0187_8x_stablesr_w0.5"
TASK10_TEMP="$TASK10_OUT/temp_input"
mkdir -p "$TASK10_TEMP"
cp "$LR_0187_8X" "$TASK10_TEMP/"
TASK10_CMD="python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
    --config \"$CONFIG_NOEDGE\" \
    --ckpt \"$CKPT_STABLESR\" \
    --init-img \"$TASK10_TEMP\" \
    --outdir \"$TASK10_OUT\" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w 0.5 \
    --seed $SEED \
    --n_samples 1 \
    --vqgan_ckpt \"$VQGAN_CKPT\" \
    --upscale 8"

# Task 11: 0187 8x NoEdge
TASK11_OUT="$OUTPUT_BASE/0187/8x/0187_8x_noedge_w0.0"
TASK11_INPUT_DIR="$TASK11_OUT/input_images"
mkdir -p "$TASK11_INPUT_DIR"
cp "$LR_0187_8X" "$TASK11_INPUT_DIR/lr_8x_0187.png"
TASK11_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0187_8X\" \
    --noedge \
    --outdir \"$TASK11_OUT\" \
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
    --dec-w 0.0"

# Task 12: 0187 8x Edge
TASK12_OUT="$OUTPUT_BASE/0187/8x/0187_8x_edge_w0.0"
TASK12_INPUT_DIR="$TASK12_OUT/input_images"
mkdir -p "$TASK12_INPUT_DIR"
cp "$LR_0187_8X" "$TASK12_INPUT_DIR/lr_8x_0187.png"
cp "$EDGE_0187" "$TASK12_INPUT_DIR/edge_0187.png"
TASK12_CMD="python scripts/inference_blended_input_tile.py \
    --lr-img \"$LR_0187_8X\" \
    --edge-img \"$EDGE_0187\" \
    --outdir \"$TASK12_OUT\" \
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
    --dec-w 0.0"

# Run tasks in parallel using different GPUs
echo ""
echo "=========================================="
echo "Starting parallel execution"
echo "=========================================="
echo ""

# Run all tasks in background on different GPUs
# GPU allocation: 0, 1, 2, 3, 4, 5, 6, 7
run_task_on_gpu 0 "Task 1: 0056 4x StableSR+w0.5" "$TASK1_OUT" "$TASK1_CMD" &
TASK1_PID=$!

run_task_on_gpu 1 "Task 2: 0056 4x NoEdge" "$TASK2_OUT" "$TASK2_CMD" &
TASK2_PID=$!

run_task_on_gpu 2 "Task 3: 0056 4x Edge" "$TASK3_OUT" "$TASK3_CMD" &
TASK3_PID=$!

run_task_on_gpu 3 "Task 4: 0056 8x StableSR+w0.5" "$TASK4_OUT" "$TASK4_CMD" &
TASK4_PID=$!

run_task_on_gpu 4 "Task 5: 0056 8x NoEdge" "$TASK5_OUT" "$TASK5_CMD" &
TASK5_PID=$!

run_task_on_gpu 5 "Task 6: 0056 8x Edge" "$TASK6_OUT" "$TASK6_CMD" &
TASK6_PID=$!

run_task_on_gpu 6 "Task 7: 0187 4x StableSR+w0.5" "$TASK7_OUT" "$TASK7_CMD" &
TASK7_PID=$!

run_task_on_gpu 7 "Task 8: 0187 4x NoEdge" "$TASK8_OUT" "$TASK8_CMD" &
TASK8_PID=$!

# Wait for first batch to complete before starting second batch
wait $TASK1_PID
wait $TASK2_PID
wait $TASK3_PID
wait $TASK4_PID
wait $TASK5_PID
wait $TASK6_PID
wait $TASK7_PID
wait $TASK8_PID

# Start second batch
run_task_on_gpu 0 "Task 9: 0187 4x Edge" "$TASK9_OUT" "$TASK9_CMD" &
TASK9_PID=$!

run_task_on_gpu 1 "Task 10: 0187 8x StableSR+w0.5" "$TASK10_OUT" "$TASK10_CMD" &
TASK10_PID=$!

run_task_on_gpu 2 "Task 11: 0187 8x NoEdge" "$TASK11_OUT" "$TASK11_CMD" &
TASK11_PID=$!

run_task_on_gpu 3 "Task 12: 0187 8x Edge" "$TASK12_OUT" "$TASK12_CMD" &
TASK12_PID=$!

# Wait for all tasks to complete
echo ""
echo "Waiting for all tasks to complete..."
wait $TASK9_PID
wait $TASK10_PID
wait $TASK11_PID
wait $TASK12_PID

# Cleanup temporary directories
rm -rf "$TASK1_TEMP" "$TASK4_TEMP" "$TASK7_TEMP" "$TASK10_TEMP" 2>/dev/null || true

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output location: $OUTPUT_BASE"
echo ""
echo "Results:"
echo "0056 results:"
ls -la "$OUTPUT_BASE/0056/4x/"
ls -la "$OUTPUT_BASE/0056/8x/"
echo ""
echo "0187 results:"
ls -la "$OUTPUT_BASE/0187/4x/"
ls -la "$OUTPUT_BASE/0187/8x/"
echo ""
echo "GPU allocation summary:"
echo "Batch 1:"
echo "GPU 0: 0056 4x StableSR+w0.5"
echo "GPU 1: 0056 4x NoEdge"
echo "GPU 2: 0056 4x Edge"
echo "GPU 3: 0056 8x StableSR+w0.5"
echo "GPU 4: 0056 8x NoEdge"
echo "GPU 5: 0056 8x Edge"
echo "GPU 6: 0187 4x StableSR+w0.5"
echo "GPU 7: 0187 4x NoEdge"
echo "Batch 2:"
echo "GPU 0: 0187 4x Edge"
echo "GPU 1: 0187 8x StableSR+w0.5"
echo "GPU 2: 0187 8x NoEdge"
echo "GPU 3: 0187 8x Edge"
