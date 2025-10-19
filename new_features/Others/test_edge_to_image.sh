#!/bin/bash
# Quick test script for edge-to-image generation
# Tests the inference pipeline with a small sample

# ========================================
# Configuration
# ========================================

# Test settings
TEST_IMAGE="/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR/0801.png"  # Sample test image
TEST_DIR="test_edge_to_image_$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${TEST_DIR}"

# Model settings
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
CHECKPOINT="logs/*/checkpoints/last.ckpt"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

export CUDA_VISIBLE_DEVICES=0

# ========================================
# Helper Functions
# ========================================

print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# ========================================
# Test Pipeline
# ========================================

print_info "=========================================="
print_info "Edge-to-Image Test Pipeline"
print_info "=========================================="

# Create test directory
mkdir -p "${WORK_DIR}"
print_success "Test directory: ${WORK_DIR}"

# Step 1: Extract edge from test image
print_info ""
print_info "Step 1: Extracting Canny edge from test image..."

if [ ! -f "${TEST_IMAGE}" ]; then
    print_error "Test image not found: ${TEST_IMAGE}"
    print_info "Please update TEST_IMAGE variable with a valid image path"
    exit 1
fi

EDGE_DIR="${WORK_DIR}/edges"
mkdir -p "${EDGE_DIR}"

python scripts/extract_canny_edges.py \
    --input "${TEST_IMAGE}" \
    --output "${EDGE_DIR}" \
    --adaptive \
    --save_rgb \
    2>&1 | grep -v "^$"

if [ $? -eq 0 ]; then
    print_success "Edge extraction completed"
    EDGE_COUNT=$(find "${EDGE_DIR}" -name "*_edge.png" | wc -l)
    print_info "Generated ${EDGE_COUNT} edge map(s)"
else
    print_error "Edge extraction failed"
    exit 1
fi

# Step 2: Generate image from edge
print_info ""
print_info "Step 2: Generating image from edge map..."

OUTPUT_DIR="${WORK_DIR}/generated"

# Find checkpoint
CKPT_FILES=(${CHECKPOINT})
if [ ! -f "${CKPT_FILES[0]}" ]; then
    print_error "Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
CHECKPOINT="${CKPT_FILES[0]}"
print_info "Using checkpoint: ${CHECKPOINT}"

# Run inference with multiple settings for comparison
print_info ""
print_info "Testing different generation settings..."

# Test 1: Fast (pure noise start)
print_info "  Test 1: Fast generation (pure noise, 20 steps)"
python scripts/inference_edge_to_image.py \
    --config "${CONFIG}" \
    --ckpt "${CHECKPOINT}" \
    --vqgan_ckpt "${VQGAN_CKPT}" \
    --edge-img "${EDGE_DIR}" \
    --outdir "${OUTPUT_DIR}/fast_pure_noise" \
    --ddim_steps 20 \
    --input_size 512 \
    --dec_w 0.0 \
    --seed 42 \
    --precision autocast \
    2>&1 | grep -E "(Generating|found|completed|Time elapsed)"

# Test 2: Quality (from edge latent)
print_info "  Test 2: Quality generation (from edge, 50 steps)"
python scripts/inference_edge_to_image.py \
    --config "${CONFIG}" \
    --ckpt "${CHECKPOINT}" \
    --vqgan_ckpt "${VQGAN_CKPT}" \
    --edge-img "${EDGE_DIR}" \
    --outdir "${OUTPUT_DIR}/quality_from_edge" \
    --ddim_steps 50 \
    --input_size 512 \
    --dec_w 0.0 \
    --start_from_edge \
    --start_timestep 999 \
    --seed 42 \
    --precision autocast \
    2>&1 | grep -E "(Generating|found|completed|Time elapsed)"

# Test 3: With decoder fusion
print_info "  Test 3: With decoder fusion (dec_w=0.5, 50 steps)"
python scripts/inference_edge_to_image.py \
    --config "${CONFIG}" \
    --ckpt "${CHECKPOINT}" \
    --vqgan_ckpt "${VQGAN_CKPT}" \
    --edge-img "${EDGE_DIR}" \
    --outdir "${OUTPUT_DIR}/with_fusion" \
    --ddim_steps 50 \
    --input_size 512 \
    --dec_w 0.5 \
    --start_from_edge \
    --start_timestep 999 \
    --seed 42 \
    --precision autocast \
    2>&1 | grep -E "(Generating|found|completed|Time elapsed)"

print_success "All tests completed!"

# Step 3: Create comparison montage
print_info ""
print_info "Step 3: Creating comparison visualization..."

if command -v montage &> /dev/null; then
    COMPARE_OUTPUT="${WORK_DIR}/comparison.jpg"
    
    # Find the generated files
    EDGE_FILE=$(find "${EDGE_DIR}" -name "*_edge.png" | head -1)
    FAST_FILE=$(find "${OUTPUT_DIR}/fast_pure_noise" -name "*_generated.png" | head -1)
    QUALITY_FILE=$(find "${OUTPUT_DIR}/quality_from_edge" -name "*_generated.png" | head -1)
    FUSION_FILE=$(find "${OUTPUT_DIR}/with_fusion" -name "*_generated.png" | head -1)
    
    if [ -f "${EDGE_FILE}" ] && [ -f "${FAST_FILE}" ] && [ -f "${QUALITY_FILE}" ] && [ -f "${FUSION_FILE}" ]; then
        # Create labeled versions
        convert "${EDGE_FILE}" -gravity South -pointsize 20 -annotate +0+5 "Input Edge" "${WORK_DIR}/edge_labeled.png"
        convert "${FAST_FILE}" -gravity South -pointsize 20 -annotate +0+5 "Fast (20 steps)" "${WORK_DIR}/fast_labeled.png"
        convert "${QUALITY_FILE}" -gravity South -pointsize 20 -annotate +0+5 "Quality (50 steps)" "${WORK_DIR}/quality_labeled.png"
        convert "${FUSION_FILE}" -gravity South -pointsize 20 -annotate +0+5 "With Fusion" "${WORK_DIR}/fusion_labeled.png"
        
        # Create montage
        montage \
            "${WORK_DIR}/edge_labeled.png" \
            "${WORK_DIR}/fast_labeled.png" \
            "${WORK_DIR}/quality_labeled.png" \
            "${WORK_DIR}/fusion_labeled.png" \
            -tile 2x2 -geometry 512x512+10+10 -background white \
            "${COMPARE_OUTPUT}"
        
        if [ $? -eq 0 ]; then
            print_success "Comparison saved: ${COMPARE_OUTPUT}"
        fi
        
        # Clean up labeled files
        rm -f "${WORK_DIR}"/*_labeled.png
    else
        print_info "Some generated files not found, skipping comparison"
    fi
else
    print_info "ImageMagick not found, skipping comparison visualization"
fi

# Step 4: Display summary
print_info ""
print_info "=========================================="
print_success "Test Complete!"
print_info "=========================================="
print_info "Test directory: ${WORK_DIR}"
print_info ""
print_info "Generated files:"
print_info "  1. Input edge: ${EDGE_DIR}"
print_info "  2. Fast generation: ${OUTPUT_DIR}/fast_pure_noise"
print_info "  3. Quality generation: ${OUTPUT_DIR}/quality_from_edge"
print_info "  4. With fusion: ${OUTPUT_DIR}/with_fusion"
if [ -f "${COMPARE_OUTPUT}" ]; then
    print_info "  5. Comparison: ${COMPARE_OUTPUT}"
fi
print_info ""
print_info "To view results:"
print_info "  eog ${WORK_DIR}/comparison.jpg"
print_info "  # or"
print_info "  ls -lh ${OUTPUT_DIR}/*/"
print_info "=========================================="


