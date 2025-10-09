#!/bin/bash
# Automatic inference script for all checkpoints
# This script automatically finds all checkpoint files (excluding last.ckpt) 
# and runs validation on them

# Default parameters (matching your reference command)
LOGS_DIR="logs"
OUTPUT_BASE="~/validation_results"
INIT_IMG="~/nas/test_dataset/128x128_valid_LR"
GT_IMG="~/nas/test_dataset/512x512_White_GT"
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml"
VQGAN_CKPT="~/checkpoints/vqgan_cfw_00011.ckpt"
DDPM_STEPS=200
DEC_W=0.5
SEED=42
N_SAMPLES=1
COLORFIX_TYPE="wavelet"

# Parse command line arguments
DRY_RUN=""
EXP_FILTER=""
SKIP_EXISTING="--skip_existing"  # Default: skip existing results
INCLUDE_LAST=""    # Default: controlled by Python script default (include)
CALCULATE_METRICS=""  # Default: calculate metrics

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --exp-filter)
            EXP_FILTER="--exp_filter $2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip_existing"
            shift
            ;;
        --no-skip-existing|--overwrite)
            SKIP_EXISTING=""
            shift
            ;;
        --include-last)
            INCLUDE_LAST="--include_last"
            shift
            ;;
        --no-include-last|--exclude-last)
            INCLUDE_LAST="--exclude_last"
            shift
            ;;
        --calculate-metrics)
            CALCULATE_METRICS="--calculate_metrics"
            shift
            ;;
        --no-calculate-metrics)
            CALCULATE_METRICS="--no_calculate_metrics"
            shift
            ;;
        --ddpm-steps)
            DDPM_STEPS=$2
            shift 2
            ;;
        --dec-w)
            DEC_W=$2
            shift 2
            ;;
        --config)
            CONFIG=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run              Print commands without executing"
            echo "  --exp-filter FILTER    Only process experiments matching FILTER"
            echo "  --skip-existing        Skip if output directory already exists (DEFAULT)"
            echo "  --no-skip-existing     Force overwrite existing results"
            echo "  --overwrite            Same as --no-skip-existing"
            echo "  --include-last         Also process last.ckpt files"
            echo "  --exclude-last         Exclude last.ckpt files"
            echo "  --calculate-metrics    Calculate PSNR/SSIM/LPIPS metrics (DEFAULT)"
            echo "  --no-calculate-metrics Skip all metric calculations"
            echo "  --ddpm-steps STEPS     Number of DDPM steps (default: 200)"
            echo "  --dec-w WEIGHT         Decoder weight (default: 0.5)"
            echo "  --config CONFIG        Path to config file"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the Python script
python scripts/auto_inference.py \
    --logs_dir "$LOGS_DIR" \
    --output_base "$OUTPUT_BASE" \
    --init_img "$INIT_IMG" \
    --gt_img "$GT_IMG" \
    --config "$CONFIG" \
    --vqgan_ckpt "$VQGAN_CKPT" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w $DEC_W \
    --seed $SEED \
    --n_samples $N_SAMPLES \
    --colorfix_type "$COLORFIX_TYPE" \
    --use_edge_processing \
    $DRY_RUN \
    $EXP_FILTER \
    $SKIP_EXISTING \
    $INCLUDE_LAST \
    $CALCULATE_METRICS

