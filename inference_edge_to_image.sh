#!/bin/bash
# Edge-to-Image Generation Script
# Generates images from canny edge maps using trained StableSR model
# 从Canny边缘图生成图像

# ========================================
# Configuration
# ========================================

# Model paths
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
CHECKPOINT="logs/*/checkpoints/last.ckpt"  # Auto-match latest checkpoint
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Input/Output paths
EDGE_INPUT="inputs/edge_maps"              # Directory with canny edge map images
OUTPUT_DIR="outputs/edge_to_image_$(date +%Y%m%d_%H%M%S)"  # Output with timestamp

# Generation parameters
DDPM_STEPS=50                # Sampling steps (50=fast, 200=high quality)
INPUT_SIZE=512               # Input and output size (must match training)
DEC_W=0.0                    # Decoder fusion weight (0.0=no fusion, recommended)
SEED=42                      # Random seed
N_SAMPLES=1                  # Number of samples per edge map

# Advanced options
START_FROM_EDGE=true         # Start from noisy edge latent (true) or pure noise (false)
START_TIMESTEP=999           # Starting timestep if START_FROM_EDGE=true (0-999, higher=more noise)
TEXT_PROMPT=""               # Text prompt for generation (empty=unconditional)
MAX_IMAGES=-1                # Max images to process (-1=all)

# GPU setting
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

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# ========================================
# Pre-flight Checks
# ========================================

print_info "=========================================="
print_info "Edge-to-Image Generation"
print_info "=========================================="

# Check config file
if [ ! -f "${CONFIG}" ]; then
    print_error "Config file not found: ${CONFIG}"
    exit 1
fi
print_success "Config: ${CONFIG}"

# Check checkpoint (with wildcard support)
CKPT_FILES=(${CHECKPOINT})
if [ ! -f "${CKPT_FILES[0]}" ]; then
    print_error "Checkpoint not found: ${CHECKPOINT}"
    print_info "Please specify a valid checkpoint path"
    exit 1
fi
CHECKPOINT="${CKPT_FILES[0]}"
print_success "Checkpoint: ${CHECKPOINT}"

# Check VQGAN checkpoint
if [ ! -f "${VQGAN_CKPT}" ]; then
    print_error "VQGAN checkpoint not found: ${VQGAN_CKPT}"
    exit 1
fi
print_success "VQGAN: ${VQGAN_CKPT}"

# Check input directory
if [ ! -d "${EDGE_INPUT}" ]; then
    print_warning "Input directory not found: ${EDGE_INPUT}"
    print_info "Creating input directory..."
    mkdir -p "${EDGE_INPUT}"
    print_info "Please place canny edge map images in: ${EDGE_INPUT}"
    print_info "Supported formats: .png, .jpg, .jpeg"
    exit 1
fi

# Count edge map images
EDGE_COUNT=$(find "${EDGE_INPUT}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
if [ ${EDGE_COUNT} -eq 0 ]; then
    print_error "No edge map images found in ${EDGE_INPUT}"
    print_info "Please add edge map images and try again"
    exit 1
fi
print_success "Found ${EDGE_COUNT} edge map images"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
print_success "Output directory: ${OUTPUT_DIR}"

# ========================================
# Display Configuration
# ========================================

echo ""
print_info "Generation Settings:"
echo "  Input directory: ${EDGE_INPUT}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of edges: ${EDGE_COUNT}"
echo "  Image size: ${INPUT_SIZE}×${INPUT_SIZE}"
echo "  DDPM steps: ${DDPM_STEPS}"
echo "  Decoder fusion: ${DEC_W}"
echo "  Start from edge: ${START_FROM_EDGE}"
if [ "${START_FROM_EDGE}" = "true" ]; then
    echo "    Starting timestep: ${START_TIMESTEP}"
fi
echo "  Text prompt: '${TEXT_PROMPT}'"
echo "  Samples per edge: ${N_SAMPLES}"
echo "  Random seed: ${SEED}"
echo ""

# ========================================
# Build Command
# ========================================

CMD="python scripts/inference_edge_to_image.py"
CMD="${CMD} --config ${CONFIG}"
CMD="${CMD} --ckpt ${CHECKPOINT}"
CMD="${CMD} --vqgan_ckpt ${VQGAN_CKPT}"
CMD="${CMD} --edge-img ${EDGE_INPUT}"
CMD="${CMD} --outdir ${OUTPUT_DIR}"
CMD="${CMD} --ddim_steps ${DDPM_STEPS}"
CMD="${CMD} --input_size ${INPUT_SIZE}"
CMD="${CMD} --dec_w ${DEC_W}"
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --n_samples ${N_SAMPLES}"

if [ "${START_FROM_EDGE}" = "true" ]; then
    CMD="${CMD} --start_from_edge"
    CMD="${CMD} --start_timestep ${START_TIMESTEP}"
fi

if [ -n "${TEXT_PROMPT}" ]; then
    CMD="${CMD} --text_prompt \"${TEXT_PROMPT}\""
fi

if [ ${MAX_IMAGES} -gt 0 ]; then
    CMD="${CMD} --max_images ${MAX_IMAGES}"
fi

CMD="${CMD} --precision autocast"

# ========================================
# Run Generation
# ========================================

print_info "Starting generation..."
echo ""

START_TIME=$(date +%s)

eval ${CMD}

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ========================================
# Display Results
# ========================================

echo ""
print_info "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    print_success "Generation completed successfully!"
    print_info "Time elapsed: ${ELAPSED} seconds"
    
    # Count generated images
    GEN_COUNT=$(find "${OUTPUT_DIR}" -type f -name "*_generated.png" 2>/dev/null | wc -l)
    print_info "Generated images: ${GEN_COUNT}"
    
    if [ ${EDGE_COUNT} -gt 0 ]; then
        AVG_TIME=$(echo "scale=2; ${ELAPSED}/${EDGE_COUNT}" | bc)
        print_info "Average time per edge: ${AVG_TIME} seconds"
    fi
    
    echo ""
    print_info "Output location:"
    print_success "  ${OUTPUT_DIR}"
    echo ""
    
    # Show some sample outputs
    print_info "Sample outputs:"
    find "${OUTPUT_DIR}" -type f -name "*_generated.png" 2>/dev/null | head -5 | while read file; do
        echo "  - $(basename ${file})"
    done
    
    # Create preview montage if ImageMagick is available
    if command -v montage &> /dev/null; then
        print_info "Creating preview montage..."
        MONTAGE_OUTPUT="${OUTPUT_DIR}/preview_montage.jpg"
        montage "${OUTPUT_DIR}"/*_generated.png -tile 4x -geometry 256x256+2+2 "${MONTAGE_OUTPUT}" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "Preview montage: ${MONTAGE_OUTPUT}"
        fi
    fi
    
else
    print_error "Generation failed with exit code: ${EXIT_CODE}"
    print_info "Please check the error messages above"
    exit ${EXIT_CODE}
fi

print_info "=========================================="
echo ""
print_success "Done!"


