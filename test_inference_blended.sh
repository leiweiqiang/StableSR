#!/bin/bash

# Quick test script for blended input inference
# This creates test inputs and runs inference

echo "========================================"
echo "Blended Input Inference Test"
echo "========================================"
echo ""

# Step 1: Test blending logic
echo "Step 1: Testing blending logic..."
python scripts/test_blended_inference.py --test-blending
if [ $? -ne 0 ]; then
    echo "✗ Blending test failed!"
    exit 1
fi
echo ""

# Step 2: Create test inputs
echo "Step 2: Creating test inputs..."
python scripts/test_blended_inference.py --create-test-inputs
echo ""

# Step 3: Run inference (requires checkpoint path)
if [ -z "$1" ]; then
    echo "Step 3: Skipping inference (no checkpoint provided)"
    echo ""
    echo "To run full test with inference:"
    echo "  ./test_inference_blended.sh path/to/checkpoint.ckpt"
    echo ""
    echo "Or manually run:"
    echo "  python scripts/inference_blended_input.py \\"
    echo "    --lr-img test_inputs/test_lr_32x32.png \\"
    echo "    --edge-img test_inputs/test_edge_512x512.png \\"
    echo "    --outdir outputs/test/ \\"
    echo "    --ckpt path/to/checkpoint.ckpt \\"
    echo "    --save-comparison \\"
    echo "    --save-intermediates"
else
    CKPT=$1
    echo "Step 3: Running inference with checkpoint: $CKPT"
    python scripts/inference_blended_input.py \
        --lr-img test_inputs/test_lr_32x32.png \
        --edge-img test_inputs/test_edge_512x512.png \
        --outdir outputs/test/ \
        --ckpt "$CKPT" \
        --save-comparison \
        --save-intermediates \
        --blend-alpha 0.5 \
        --ddpm-steps 50
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Inference completed successfully!"
        echo "Check outputs/test/ for results"
    else
        echo ""
        echo "✗ Inference failed!"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"

