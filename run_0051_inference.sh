#!/bin/bash

# Quick script to run inference for image 0051
# Based on the requirements:
# (1) 0051 edge 4x w0.0 (epoch 111)
# (2) 0051 noedge 4x w0.0 (epoch 111) - already exists, will skip
# (3) 0051 edge 8x w0.0 (epoch 111)
# (4) 0051 noedge 8x w0.0 (epoch 111)

set -e

# Cleanup function to restore files in case of error
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Restore 4x file if backed up
    if [ "$FOUR_X_BAKED" = true ] && [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak" ]; then
        mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png
        echo "Restored 4x file"
    fi
    # Restore 8x file if backed up
    if [ "$EIGHT_X_BAKED" = true ] && [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak" ]; then
        mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png
        echo "Restored 8x file"
    fi
    # Restore NOEDGE in batch script
    if [ -f "batch_inference_blended_tile.sh.bak" ]; then
        cp batch_inference_blended_tile.sh.bak batch_inference_blended_tile.sh
        echo "Restored batch script"
    fi
}

trap cleanup EXIT ERR

echo "=========================================="
echo "Running inference for image 0051"
echo "=========================================="
echo ""
echo "Task 1: 0051 with edge, 4x, w0.0, epoch 111"
echo "Task 2: 0051 no edge, 4x, w0.0, epoch 111 (will skip if exists)"
echo "Task 3: 0051 with edge, 8x, w0.0, epoch 111"
echo "Task 4: 0051 no edge, 8x, w0.0, epoch 111"
echo ""

# Task 1: Run with edge (NOEDGE=false)
echo ""
echo "=========================================="
echo "Task 1: Running 0051 with edge, 4x"
echo "=========================================="

# Make a backup
cp batch_inference_blended_tile.sh batch_inference_blended_tile.sh.bak

# Modify NOEDGE to false for edge run (only 4x)
echo "Setting NOEDGE=false for edge processing..."
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

echo "Running: 4x with edge..."
# We need to process only 4x images, not 8x. Let's check what files exist.
# The script will process both 4x and 8x, but we only want 4x for now.
# We'll handle this by skipping the 8x processing manually after 4x is done.

# First, let's ensure 8x LR file doesn't exist temporarily
if [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0051.png" ]; then
    # Temporarily rename it
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak
    EIGHT_X_BAKED=true
else
    EIGHT_X_BAKED=false
fi

./batch_inference_blended_tile.sh --only-image 0051

# Restore 8x LR file
if [ "$EIGHT_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png
fi

# Restore NOEDGE=true for noedge runs
echo "Restoring NOEDGE=true for noedge processing..."
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh

# Restore backup if needed
if ! grep -q "^NOEDGE=true$" batch_inference_blended_tile.sh; then
    echo "Restoring from backup..."
    cp batch_inference_blended_tile.sh.bak batch_inference_blended_tile.sh
fi

echo ""
echo "=========================================="
echo "Task 2: Running 0051 without edge, 4x"
echo "=========================================="
# Temporarily hide 8x file so only 4x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x8_480x270/0051.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak
    EIGHT_X_BAKED=true
else
    EIGHT_X_BAKED=false
fi

echo "Running: 4x without edge..."
./batch_inference_blended_tile.sh --only-image 0051 --noedge

# Restore 8x LR file
if [ "$EIGHT_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png
fi

echo ""
echo "=========================================="
echo "Task 3: Running 0051 with edge, 8x"
echo "=========================================="

# Set NOEDGE=false for edge processing
echo "Setting NOEDGE=false for edge processing..."
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

# Temporarily hide 4x file so only 8x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0051.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak
    FOUR_X_BAKED=true
else
    FOUR_X_BAKED=false
fi

echo "Running: 8x with edge..."
./batch_inference_blended_tile.sh --only-image 0051

# Restore 4x LR file
if [ "$FOUR_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png
fi

# Restore NOEDGE=true for next task
echo "Restoring NOEDGE=true for noedge processing..."
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh

echo ""
echo "=========================================="
echo "Task 4: Running 0051 without edge, 8x"
echo "=========================================="
# Temporarily hide 4x file so only 8x is processed
if [ -f "/stablesr_dataset/weiqiang/lr_x4_960x540/0051.png" ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak
    FOUR_X_BAKED=true
else
    FOUR_X_BAKED=false
fi

echo "Running: 8x without edge..."
./batch_inference_blended_tile.sh --only-image 0051 --noedge

# Restore 4x LR file
if [ "$FOUR_X_BAKED" = true ]; then
    mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png
fi

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Output directories created:"
ls -la outputs/4x/ | grep 0051
ls -la outputs/8x/ | grep 0051
