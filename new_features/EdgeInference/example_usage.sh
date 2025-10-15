#!/bin/bash
################################################################################
# Edge Inference Usage Examples
# 
# This file contains practical usage examples for edge-enhanced SR inference.
# Copy and modify these examples for your own use cases.
#
# Author: StableSR_Edge_v3 Team
# Date: 2025-10-15
################################################################################

# Activate conda environment
conda activate sr_infer
cd /root/dp/StableSR_Edge_v3

################################################################################
# Example 1: Basic Edge Inference (Recommended)
# - Uses GT images for edge generation
# - Best quality and consistency with training
################################################################################
example_1_basic_edge_inference() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/basic_edge \
        --use_edge_processing \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 200 \
        --n_samples 1 \
        --input_size 512 \
        --seed 42
}

################################################################################
# Example 2: Batch Processing
# - Process multiple images in parallel
# - Faster for large datasets
################################################################################
example_2_batch_processing() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/large_dataset/lr \
        --gt-img inputs/large_dataset/gt \
        --outdir outputs/batch_results \
        --use_edge_processing \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --n_samples 4 \
        --ddpm_steps 200
}

################################################################################
# Example 3: High Quality (More Steps)
# - More DDPM steps for better quality
# - Slower but better results
################################################################################
example_3_high_quality() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/high_quality \
        --use_edge_processing \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 500 \
        --n_samples 1 \
        --colorfix_type adain
}

################################################################################
# Example 4: Fast Inference
# - Fewer steps for faster processing
# - Good for quick tests or large datasets
################################################################################
example_4_fast_inference() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/fast \
        --use_edge_processing \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 100 \
        --n_samples 4
}

################################################################################
# Example 5: No GT Available (LR-based Edge)
# - Use when GT images are not available
# - Edge generated from LR images
# - May have domain mismatch with training
################################################################################
example_5_lr_based_edge() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_only \
        --outdir outputs/lr_edge \
        --use_edge_processing \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 200
}

################################################################################
# Example 6: Standard SR (No Edge, Baseline)
# - Traditional SR without edge processing
# - For comparison with edge-enhanced results
################################################################################
example_6_no_edge_baseline() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --outdir outputs/no_edge \
        --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 200
}

################################################################################
# Example 7: Ablation Study - Black Edge
# - Use black/no edge maps
# - Tests model behavior without edge information
################################################################################
example_7_black_edge_ablation() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --outdir outputs/black_edge \
        --use_edge_processing \
        --use_white_edge \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
        --ddpm_steps 200
}

################################################################################
# Example 8: Color Correction Comparison
# - Test different color correction methods
################################################################################
example_8_color_correction() {
    # AdaIN (used in paper)
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/color_adain \
        --use_edge_processing \
        --colorfix_type adain \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --ddpm_steps 200
    
    # Wavelet
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/color_wavelet \
        --use_edge_processing \
        --colorfix_type wavelet \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --ddpm_steps 200
    
    # No fix
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/color_nofix \
        --use_edge_processing \
        --colorfix_type nofix \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --ddpm_steps 200
}

################################################################################
# Example 9: Process Specific File
# - Process only one specific image
################################################################################
example_9_specific_file() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/specific \
        --use_edge_processing \
        --specific_file "image001.png" \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --ddpm_steps 200
}

################################################################################
# Example 10: Limited Number of Images
# - Process only first N images
################################################################################
example_10_limited_images() {
    python scripts/sr_val_edge_inference.py \
        --init-img inputs/lr_images \
        --gt-img inputs/gt_images \
        --outdir outputs/limited \
        --use_edge_processing \
        --max_images 10 \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
        --ddpm_steps 200
}

################################################################################
# Usage Instructions
################################################################################
show_usage() {
    echo "Edge Inference Usage Examples"
    echo "=============================="
    echo ""
    echo "Available examples:"
    echo "  1. example_1_basic_edge_inference    - Basic edge inference (recommended)"
    echo "  2. example_2_batch_processing        - Batch processing"
    echo "  3. example_3_high_quality            - High quality (more steps)"
    echo "  4. example_4_fast_inference          - Fast inference (fewer steps)"
    echo "  5. example_5_lr_based_edge           - LR-based edge (no GT)"
    echo "  6. example_6_no_edge_baseline        - No edge baseline"
    echo "  7. example_7_black_edge_ablation     - Black edge ablation"
    echo "  8. example_8_color_correction        - Color correction comparison"
    echo "  9. example_9_specific_file           - Process specific file"
    echo " 10. example_10_limited_images         - Process limited images"
    echo ""
    echo "Usage:"
    echo "  1. Edit this file to set your paths"
    echo "  2. Source this file: source example_usage.sh"
    echo "  3. Run example: example_1_basic_edge_inference"
    echo ""
    echo "Or run directly:"
    echo "  bash example_usage.sh [function_name]"
}

################################################################################
# Main
################################################################################
if [ "$1" != "" ]; then
    # Run specific example if provided
    $1
else
    # Show usage if no argument
    show_usage
fi

