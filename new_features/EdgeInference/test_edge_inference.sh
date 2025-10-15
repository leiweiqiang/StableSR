#!/bin/bash
################################################################################
# Edge Inference Testing Script
# 
# This script provides various test configurations for edge-enhanced 
# super-resolution inference.
#
# Usage:
#   chmod +x test_edge_inference.sh
#   ./test_edge_inference.sh [test_name]
#
# Available tests:
#   basic       - Basic edge inference test (default)
#   no_edge     - Standard inference without edge processing
#   black_edge  - Inference with black/no edge maps
#   lr_edge     - Edge generation from LR images
#   batch       - Batch processing test
#   quick       - Quick test with 1 image
#
# Author: StableSR_Edge_v3 Team
# Date: 2025-10-15
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_header() {
    echo ""
    echo "================================================================================"
    print_msg "$BLUE" "$1"
    echo "================================================================================"
    echo ""
}

# Activate conda environment
print_header "Activating Conda Environment: sr_infer"
eval "$(conda shell.bash hook)"
conda activate sr_infer

if [ $? -ne 0 ]; then
    print_msg "$RED" "ERROR: Failed to activate conda environment 'sr_infer'"
    print_msg "$YELLOW" "Please ensure the environment exists:"
    echo "  conda create -n sr_infer python=3.8"
    echo "  conda activate sr_infer"
    echo "  pip install -r requirements.txt"
    exit 1
fi

print_msg "$GREEN" "✓ Conda environment 'sr_infer' activated"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"
print_msg "$GREEN" "✓ Changed to project root: $PROJECT_ROOT"

# Default paths (using local test directories)
DEFAULT_LR_DIR="new_features/EdgeInference/lr_images"
DEFAULT_GT_DIR="new_features/EdgeInference/gt_images"
DEFAULT_OUTPUT_DIR="outputs/edge_inference_test"
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
DEFAULT_CKPT="logs/2025-10-14T14-46-40_stablesr_edge_loss_20251014_144637/checkpoints/epoch=000388.ckpt"
DEFAULT_VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Test configuration
TEST_TYPE=${1:-"basic"}

print_header "Test Configuration: $TEST_TYPE"

# Common parameters
COMMON_PARAMS="--config $DEFAULT_CONFIG \
               --ckpt $DEFAULT_CKPT \
               --vqgan_ckpt $DEFAULT_VQGAN_CKPT \
               --ddpm_steps 200 \
               --input_size 512 \
               --seed 42"

# Run test based on type
case "$TEST_TYPE" in
    "basic")
        print_msg "$GREEN" "Running: Basic Edge Inference Test"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing enabled"
        echo "  - Using GT images for edge generation"
        echo "  - Batch size: 1"
        echo "  - DDPM steps: 200"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --gt-img $DEFAULT_GT_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/basic \
            --use_edge_processing \
            --n_samples 1 \
            $COMMON_PARAMS
        ;;
    
    "no_edge")
        print_msg "$GREEN" "Running: Standard Inference (No Edge Processing)"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing disabled"
        echo "  - Standard SR inference"
        echo "  - Batch size: 1"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/no_edge \
            --n_samples 1 \
            $COMMON_PARAMS
        ;;
    
    "black_edge")
        print_msg "$GREEN" "Running: Inference with Black Edge Maps"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing enabled"
        echo "  - Using black/no edge maps"
        echo "  - Tests model behavior with empty edges"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/black_edge \
            --use_edge_processing \
            --use_white_edge \
            --n_samples 1 \
            $COMMON_PARAMS
        ;;
    
    "lr_edge")
        print_msg "$GREEN" "Running: Edge Generation from LR Images"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing enabled"
        echo "  - Generating edges from LR images (not GT)"
        echo "  - ⚠ May cause domain mismatch with training"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/lr_edge \
            --use_edge_processing \
            --n_samples 1 \
            $COMMON_PARAMS
        ;;
    
    "batch")
        print_msg "$GREEN" "Running: Batch Processing Test"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing enabled"
        echo "  - Batch size: 4"
        echo "  - Tests parallel processing"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --gt-img $DEFAULT_GT_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/batch \
            --use_edge_processing \
            --n_samples 4 \
            $COMMON_PARAMS
        ;;
    
    "quick")
        print_msg "$GREEN" "Running: Quick Test (1 Image)"
        print_msg "$YELLOW" "Features:"
        echo "  - Edge processing enabled"
        echo "  - Process only 1 image"
        echo "  - Fast verification test"
        echo ""
        
        python scripts/sr_val_edge_inference.py \
            --init-img $DEFAULT_LR_DIR \
            --gt-img $DEFAULT_GT_DIR \
            --outdir ${DEFAULT_OUTPUT_DIR}/quick \
            --use_edge_processing \
            --max_images 1 \
            --n_samples 1 \
            $COMMON_PARAMS
        ;;
    
    "custom")
        print_msg "$GREEN" "Running: Custom Test Configuration"
        print_msg "$YELLOW" "Edit this script to customize parameters"
        echo ""
        
        # Customize these parameters as needed
        CUSTOM_LR_DIR="your/lr/path"
        CUSTOM_GT_DIR="your/gt/path"
        CUSTOM_OUTPUT="outputs/custom_test"
        CUSTOM_BATCH_SIZE=1
        CUSTOM_DDPM_STEPS=200
        
        python scripts/sr_val_edge_inference.py \
            --init-img $CUSTOM_LR_DIR \
            --gt-img $CUSTOM_GT_DIR \
            --outdir $CUSTOM_OUTPUT \
            --use_edge_processing \
            --n_samples $CUSTOM_BATCH_SIZE \
            --ddpm_steps $CUSTOM_DDPM_STEPS \
            --config $DEFAULT_CONFIG \
            --ckpt $DEFAULT_CKPT \
            --vqgan_ckpt $DEFAULT_VQGAN_CKPT \
            --seed 42
        ;;
    
    "help")
        print_msg "$BLUE" "Available test configurations:"
        echo ""
        echo "  basic       - Basic edge inference with GT-based edge generation (default)"
        echo "  no_edge     - Standard inference without edge processing"
        echo "  black_edge  - Inference with black/no edge maps"
        echo "  lr_edge     - Edge generation from LR images (not recommended)"
        echo "  batch       - Batch processing test (batch_size=4)"
        echo "  quick       - Quick test with only 1 image"
        echo "  custom      - Custom configuration (edit script to customize)"
        echo ""
        echo "Usage:"
        echo "  ./test_edge_inference.sh [test_name]"
        echo ""
        echo "Examples:"
        echo "  ./test_edge_inference.sh basic"
        echo "  ./test_edge_inference.sh quick"
        echo "  ./test_edge_inference.sh batch"
        echo ""
        exit 0
        ;;
    
    *)
        print_msg "$RED" "ERROR: Unknown test type: $TEST_TYPE"
        print_msg "$YELLOW" "Run './test_edge_inference.sh help' to see available tests"
        exit 1
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    print_header "Test Completed Successfully"
    print_msg "$GREEN" "✓ Test '$TEST_TYPE' finished successfully"
    print_msg "$BLUE" "Check output directory for results"
else
    print_header "Test Failed"
    print_msg "$RED" "✗ Test '$TEST_TYPE' failed"
    print_msg "$YELLOW" "Check the error messages above"
    exit 1
fi

echo ""

