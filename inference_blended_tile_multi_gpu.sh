#!/bin/bash

# Multi-GPU Blended Input Tiled Inference Script
# ==============================================
# This script demonstrates how to use the multi-GPU inference script
# for faster processing of large images with blended input

# Configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
CKPT="checkpoints/model.ckpt"
VQGAN_CKPT="checkpoints/vqgan_model.ckpt"

# Input/Output paths
LR_IMG="inputs/lr_images/"
EDGE_IMG="inputs/edges_512/"
OUTDIR="outputs/blended_tile_multi_gpu/"

# Processing parameters
BLEND_ALPHA=0.5
BLEND_BETA=0.5
LR_DOWNSCALE_FACTOR=16
DDPM_STEPS=200
TILE_SIZE=1280
TILE_OVERLAP=32
VQGAN_TILE_SIZE=1280
VQGAN_TILE_STRIDE=320

# Multi-GPU settings
GPU_IDS="0,1,2,3"  # Adjust based on your available GPUs
BATCH_SIZE=4

# Text prompt (optional)
TEXT_PROMPT=""

# Color correction
COLORFIX="none"  # Options: none, adain, wavelet

# Sampling options
START_FROM_NOISE=false  # Set to true to start from pure noise

echo "=========================================="
echo "Multi-GPU Blended Input Tiled Inference"
echo "=========================================="
echo "Config: $CONFIG"
echo "Model: $CKPT"
echo "VQGAN: $VQGAN_CKPT"
echo "LR Images: $LR_IMG"
echo "Edge Images: $EDGE_IMG"
echo "Output: $OUTDIR"
echo "GPUs: $GPU_IDS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

# Check if required files exist
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "❌ Error: Model checkpoint not found: $CKPT"
    exit 1
fi

if [ ! -f "$VQGAN_CKPT" ]; then
    echo "❌ Error: VQGAN checkpoint not found: $VQGAN_CKPT"
    exit 1
fi

if [ ! -d "$LR_IMG" ]; then
    echo "❌ Error: LR images directory not found: $LR_IMG"
    exit 1
fi

if [ ! -d "$EDGE_IMG" ]; then
    echo "❌ Error: Edge images directory not found: $EDGE_IMG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTDIR"

# Build command
CMD="python scripts/inference_blended_input_tile_multi_gpu.py"
CMD="$CMD --lr-img $LR_IMG"
CMD="$CMD --edge-img $EDGE_IMG"
CMD="$CMD --outdir $OUTDIR"
CMD="$CMD --config $CONFIG"
CMD="$CMD --ckpt $CKPT"
CMD="$CMD --vqgan-ckpt $VQGAN_CKPT"
CMD="$CMD --blend-alpha $BLEND_ALPHA"
CMD="$CMD --blend-beta $BLEND_BETA"
CMD="$CMD --lr-downscale-factor $LR_DOWNSCALE_FACTOR"
CMD="$CMD --ddpm-steps $DDPM_STEPS"
CMD="$CMD --tile-size $TILE_SIZE"
CMD="$CMD --tile-overlap $TILE_OVERLAP"
CMD="$CMD --vqgan-tile-size $VQGAN_TILE_SIZE"
CMD="$CMD --vqgan-tile-stride $VQGAN_TILE_STRIDE"
CMD="$CMD --multi-gpu"
CMD="$CMD --gpu-ids $GPU_IDS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --batch-mode"
CMD="$CMD --colorfix $COLORFIX"

if [ -n "$TEXT_PROMPT" ]; then
    CMD="$CMD --text-prompt \"$TEXT_PROMPT\""
fi

if [ "$START_FROM_NOISE" = true ]; then
    CMD="$CMD --start-from-noise"
fi

# Add verbose flag for debugging
CMD="$CMD --verbose"

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "Multi-GPU inference completed!"
echo "Results saved to: $OUTDIR"
echo "=========================================="

