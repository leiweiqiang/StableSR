#!/bin/bash

# Parallel inference script for 5 people images using StableSR
# This script runs multiple tasks in parallel across multiple GPUs
# ===========================================================
# Tasks:
# - 5 images x 2 scales (4x and 8x) = 10 tasks total
# - Using StableSR inference with w=0.5

set -e

echo "=========================================="
echo "Parallel Inference for 5 People Images (StableSR)"
echo "=========================================="
echo "Using stableSR with w=0.5"
echo ""

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="outputs/5_people_inference_${TIMESTAMP}"
CONFIG_STABLESR="configs/stableSRNew/v2-finetune_text_T_512.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
STABLESR_CKPT="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"

# Input directories
LR_4X_DIR="/stablesr_dataset/weiqiang/5_people/5_lr_x4"
LR_8X_DIR="/stablesr_dataset/weiqiang/5_people/5_lr_x8"

# Inference parameters
DDPM_STEPS=200
SEED=42
N_SAMPLES=1
DEC_W=0.5
UPSCALE_4X=4
UPSCALE_8X=8

# GPU allocation - using multiple GPUs
GPU_IDS=(2 3 4 5 6 7 2 3 4 5)  # 10 tasks distributed across GPUs
TASK_COUNT=0

# Function to run a single task on a specific GPU
run_task_on_gpu() {
    local gpu_id=$1
    local task_name=$2
    local lr_img=$3
    local upscale=$4
    local scale_name=$5
    local output_dir=$6
    
    echo ""
    echo "=========================================="
    echo "Task: $task_name on GPU $gpu_id"
    echo "LR Image: $lr_img"
    echo "Output: $output_dir"
    echo "Scale: ${scale_name}x"
    echo "=========================================="
    
    # Check if already exists
    if [ -d "$output_dir" ] && ls "$output_dir"/*.png 1> /dev/null 2>&1; then
        echo "⏭️  Skipping (already exists): $task_name"
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Create temporary input directory for this single image
    local temp_input_dir="$output_dir/temp_input"
    mkdir -p "$temp_input_dir"
    
    # Copy the image to temp directory
    cp "$lr_img" "$temp_input_dir/"
    
    # Save initial command to file
    local command_file="$output_dir/inference_command.txt"
    echo "# $task_name on GPU $gpu_id" > "$command_file"
    echo "# Generated on: $(date)" >> "$command_file"
    echo "# LR Image: $lr_img" >> "$command_file"
    echo "# Output: $output_dir" >> "$command_file"
    echo "# Scale: ${scale_name}x" >> "$command_file"
    echo "# DEC_W: $DEC_W" >> "$command_file"
    echo "" >> "$command_file"
    echo "CUDA_VISIBLE_DEVICES=$gpu_id python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \\" >> "$command_file"
    echo "    --config \"$CONFIG_STABLESR\" \\" >> "$command_file"
    echo "    --ckpt \"$STABLESR_CKPT\" \\" >> "$command_file"
    echo "    --init-img \"$temp_input_dir\" \\" >> "$command_file"
    echo "    --outdir \"$output_dir\" \\" >> "$command_file"
    echo "    --ddpm_steps $DDPM_STEPS \\" >> "$command_file"
    echo "    --dec_w $DEC_W \\" >> "$command_file"
    echo "    --seed $SEED \\" >> "$command_file"
    echo "    --n_samples $N_SAMPLES \\" >> "$command_file"
    echo "    --vqgan_ckpt \"$VQGAN_CKPT\" \\" >> "$command_file"
    echo "    --upscale $upscale" >> "$command_file"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
        --config "$CONFIG_STABLESR" \
        --ckpt "$STABLESR_CKPT" \
        --init-img "$temp_input_dir" \
        --outdir "$output_dir" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --vqgan_ckpt "$VQGAN_CKPT" \
        --upscale $upscale
    
    local exit_code=$?
    
    # Record end time and calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # Format duration string
    local duration_str=""
    if [ $hours -gt 0 ]; then
        duration_str="${hours}h ${minutes}m ${seconds}s"
    elif [ $minutes -gt 0 ]; then
        duration_str="${minutes}m ${seconds}s"
    else
        duration_str="${seconds}s"
    fi
    
    # Update command file with timing information
    {
        echo ""
        echo "# Completed on: $(date)"
        echo "# Inference duration: $duration_str (${duration} seconds)"
    } >> "$command_file"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $task_name on GPU $gpu_id (Duration: $duration_str)"
        # Clean up temporary input directory
        rm -rf "$temp_input_dir"
    else
        echo "❌ Failed: $task_name on GPU $gpu_id"
        return 1
    fi
}

# Create output directories
mkdir -p "$OUTPUT_BASE/4x"
mkdir -p "$OUTPUT_BASE/8x"

# Check input directories exist
if [ ! -d "$LR_4X_DIR" ]; then
    echo "❌ Error: 4x LR directory not found: $LR_4X_DIR"
    exit 1
fi

if [ ! -d "$LR_8X_DIR" ]; then
    echo "❌ Error: 8x LR directory not found: $LR_8X_DIR"
    exit 1
fi

# Get all images
LR_4X_IMAGES=($(ls "$LR_4X_DIR"/*.png 2>/dev/null))
LR_8X_IMAGES=($(ls "$LR_8X_DIR"/*.png 2>/dev/null))

echo "Found ${#LR_4X_IMAGES[@]} 4x images"
echo "Found ${#LR_8X_IMAGES[@]} 8x images"

if [ ${#LR_4X_IMAGES[@]} -eq 0 ] && [ ${#LR_8X_IMAGES[@]} -eq 0 ]; then
    echo "❌ No images to process"
    exit 1
fi

# Define all tasks
echo ""
echo "Defining tasks..."
TASKS=()
PIDS=()

# Process 4x images
for lr_img in "${LR_4X_IMAGES[@]}"; do
    image_name=$(basename "$lr_img")
    image_id=$(basename "$lr_img" .png)
    output_dir="$OUTPUT_BASE/4x/${image_id}_4x_stablesr_w${DEC_W}"
    gpu_id=${GPU_IDS[$TASK_COUNT]}
    
    TASKS+=("run_task_on_gpu $gpu_id \"Task ${image_id} 4x\" \"$lr_img\" $UPSCALE_4X \"4\" \"$output_dir\"")
    TASK_COUNT=$((TASK_COUNT + 1))
done

# Process 8x images
for lr_img in "${LR_8X_IMAGES[@]}"; do
    image_name=$(basename "$lr_img")
    image_id=$(basename "$lr_img" .png)
    output_dir="$OUTPUT_BASE/8x/${image_id}_8x_stablesr_w${DEC_W}"
    gpu_id=${GPU_IDS[$TASK_COUNT]}
    
    TASKS+=("run_task_on_gpu $gpu_id \"Task ${image_id} 8x\" \"$lr_img\" $UPSCALE_8X \"8\" \"$output_dir\"")
    TASK_COUNT=$((TASK_COUNT + 1))
done

echo "Total tasks: ${#TASKS[@]}"
echo ""

# Run tasks in parallel
echo "=========================================="
echo "Starting parallel execution"
echo "=========================================="
echo ""

# Start all tasks in background
for task_cmd in "${TASKS[@]}"; do
    eval "$task_cmd" &
    PIDS+=($!)
done

# Wait for all tasks to complete
echo ""
echo "Waiting for all tasks to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output location: $OUTPUT_BASE"
echo ""
echo "Results:"
if [ -d "$OUTPUT_BASE/4x" ]; then
    echo "4x results:"
    ls -la "$OUTPUT_BASE/4x/"
fi

if [ -d "$OUTPUT_BASE/8x" ]; then
    echo ""
    echo "8x results:"
    ls -la "$OUTPUT_BASE/8x/"
fi
echo ""
