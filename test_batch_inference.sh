#!/bin/bash

# Test Script for Batch Inference Configuration
# =============================================
# This script tests the batch inference configuration and shows what will be processed
#
# Key parameters:
# - DEC_W: Weight for combining VQGAN and Diffusion (default: 0.5)
# - BLEND_ALPHA/BETA: Blending weights for LR and edge images (default: 1.0)
# - DDPM_STEPS: Number of sampling steps (default: 200)

echo "=========================================="
echo "Batch Inference Configuration Test"
echo "=========================================="

# Configuration (same as batch_inference_blended_tile.sh)
LR_4X_DIR="/stablesr_dataset/weiqiang/lr_x4_960x540"
LR_8X_DIR="/stablesr_dataset/weiqiang/lr_x8_480x270"
EDGE_DIR="/stablesr_dataset/weiqiang/hr_4k_canny"
OUTPUT_BASE="outputs"

# Images to skip
SKIP_IMAGES=("0001.png" "0019.png")

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

echo "Checking directories and files..."

# Check directories
echo ""
echo "Directory Check:"
if [ -d "$LR_4X_DIR" ]; then
    echo "✅ 4x LR Directory: $LR_4X_DIR"
    echo "   Images: $(ls "$LR_4X_DIR"/*.png 2>/dev/null | wc -l)"
else
    echo "❌ 4x LR Directory not found: $LR_4X_DIR"
fi

if [ -d "$LR_8X_DIR" ]; then
    echo "✅ 8x LR Directory: $LR_8X_DIR"
    echo "   Images: $(ls "$LR_8X_DIR"/*.png 2>/dev/null | wc -l)"
else
    echo "❌ 8x LR Directory not found: $LR_8X_DIR"
fi

if [ -d "$EDGE_DIR" ]; then
    echo "✅ Edge Directory: $EDGE_DIR"
    echo "   Images: $(ls "$EDGE_DIR"/*.png 2>/dev/null | wc -l)"
else
    echo "❌ Edge Directory not found: $EDGE_DIR"
fi

# Check checkpoints
echo ""
echo "Checkpoint Check:"
CKPT_4X="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000194.ckpt"

if [ -f "$CKPT_4X" ]; then
    echo "✅ 4x Checkpoint: $CKPT_4X"
else
    echo "❌ 4x Checkpoint not found: $CKPT_4X"
fi

if [ -f "$CKPT_8X" ]; then
    echo "✅ 8x Checkpoint: $CKPT_8X"
else
    echo "❌ 8x Checkpoint not found: $CKPT_8X"
fi

# Check VQGAN checkpoint
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
if [ -f "$VQGAN_CKPT" ]; then
    echo "✅ VQGAN Checkpoint: $VQGAN_CKPT"
else
    echo "❌ VQGAN Checkpoint not found: $VQGAN_CKPT"
fi

# Check config
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
if [ -f "$CONFIG" ]; then
    echo "✅ Config file: $CONFIG"
else
    echo "❌ Config file not found: $CONFIG"
fi

# Show images to be processed
echo ""
echo "Images to be processed:"

# 4x images
echo ""
echo "4x Images:"
if [ -d "$LR_4X_DIR" ]; then
    for img in "$LR_4X_DIR"/*.png; do
        if [ -f "$img" ]; then
            image_name=$(basename "$img")
            image_id=$(basename "$img" .png)
            edge_img="$EDGE_DIR/$image_name"
            
            if should_skip_image "$image_name"; then
                echo "  ⏭️  $image_name (SKIP - already processed)"
            elif [ -f "$edge_img" ]; then
                echo "  ✅ $image_name -> outputs/4x/${image_id}_4x/"
            else
                echo "  ❌ $image_name (no edge image found)"
            fi
        fi
    done
fi

# 8x images
echo ""
echo "8x Images:"
if [ -d "$LR_8X_DIR" ]; then
    for img in "$LR_8X_DIR"/*.png; do
        if [ -f "$img" ]; then
            image_name=$(basename "$img")
            image_id=$(basename "$img" .png)
            edge_img="$EDGE_DIR/$image_name"
            
            if should_skip_image "$image_name"; then
                echo "  ⏭️  $image_name (SKIP - already processed)"
            elif [ -f "$edge_img" ]; then
                echo "  ✅ $image_name -> outputs/8x/${image_id}_8x/"
            else
                echo "  ❌ $image_name (no edge image found)"
            fi
        fi
    done
fi

# Count total images to process
echo ""
echo "Summary:"
LR_4X_COUNT=0
LR_8X_COUNT=0

if [ -d "$LR_4X_DIR" ]; then
    for img in "$LR_4X_DIR"/*.png; do
        if [ -f "$img" ]; then
            image_name=$(basename "$img")
            edge_img="$EDGE_DIR/$image_name"
            if ! should_skip_image "$image_name" && [ -f "$edge_img" ]; then
                ((LR_4X_COUNT++))
            fi
        fi
    done
fi

if [ -d "$LR_8X_DIR" ]; then
    for img in "$LR_8X_DIR"/*.png; do
        if [ -f "$img" ]; then
            image_name=$(basename "$img")
            edge_img="$EDGE_DIR/$image_name"
            if ! should_skip_image "$image_name" && [ -f "$edge_img" ]; then
                ((LR_8X_COUNT++))
            fi
        fi
    done
fi

echo "  4x images to process: $LR_4X_COUNT"
echo "  8x images to process: $LR_8X_COUNT"
echo "  Total images to process: $((LR_4X_COUNT + LR_8X_COUNT))"

echo ""
echo "=========================================="
echo "Configuration test completed!"
echo "=========================================="

if [ $((LR_4X_COUNT + LR_8X_COUNT)) -gt 0 ]; then
    echo "✅ Ready to run batch inference"
    echo "Run: ./batch_inference_blended_tile.sh"
else
    echo "❌ No images to process"
fi
