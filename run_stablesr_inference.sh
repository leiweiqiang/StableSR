#!/bin/bash
# StableSR Inference Script
# This script runs StableSR inference with configurable parameters
# Default setup uses StableSR-Turbo with 4 DDPM steps

# Default parameters (based on StableSR-Turbo reference)
CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
CKPT="/root/checkpoints/stablesr_turbo.ckpt"
VQGAN_CKPT="/root/checkpoints/vqgan_cfw_00011.ckpt"
INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
GT_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"
HR_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"  # HR images for metric comparison
OUTDIR="stablesr_inference/step200"
SUB_FOLDER=""  # Optional subfolder under OUTDIR
DDPM_STEPS=4
DEC_W=0.5
SEED=42
N_SAMPLES=1
COLORFIX_TYPE="wavelet"
CALCULATE_METRICS=true  # Whether to calculate metrics after inference

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --vqgan_ckpt|--vqgan-ckpt)
            VQGAN_CKPT="$2"
            shift 2
            ;;
        --init-img|--init_img)
            INIT_IMG="$2"
            shift 2
            ;;
        --gt-img|--gt_img)
            GT_IMG="$2"
            shift 2
            ;;
        --hr-img|--hr_img)
            HR_IMG="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --sub-folder|--sub_folder)
            SUB_FOLDER="$2"
            shift 2
            ;;
        --calculate-metrics)
            CALCULATE_METRICS=true
            shift
            ;;
        --no-calculate-metrics)
            CALCULATE_METRICS=false
            shift
            ;;
        --ddpm_steps|--ddpm-steps)
            DDPM_STEPS=$2
            shift 2
            ;;
        --dec_w|--dec-w)
            DEC_W=$2
            shift 2
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        --n_samples|--n-samples)
            N_SAMPLES=$2
            shift 2
            ;;
        --colorfix_type|--colorfix-type)
            COLORFIX_TYPE="$2"
            shift 2
            ;;
        --help|-h)
            echo "StableSR Inference Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  --init-img PATH        Path to low-quality input images (default: /mnt/nas_dp/test_dataset/128x128_valid_LR)"
            echo "  --gt-img PATH          Path to ground truth images (default: /mnt/nas_dp/test_dataset/512x512_valid_HR)"
            echo "  --hr-img PATH          Path to HR images for metric comparison (default: same as --gt-img)"
            echo "  --outdir PATH          Output directory for results (default: stablesr_inference/step200)"
            echo "  --sub-folder NAME      Subfolder name under output directory (optional, creates OUTDIR/SUB_FOLDER)"
            echo ""
            echo "Optional arguments:"
            echo "  --config PATH          Config file (default: configs/stableSRNew/v2-finetune_text_T_512.yaml)"
            echo "  --ckpt PATH            Model checkpoint (default: /root/checkpoints/stablesr_turbo.ckpt)"
            echo "  --vqgan-ckpt PATH      VQGAN checkpoint (default: /root/checkpoints/vqgan_cfw_00011.ckpt)"
            echo "  --ddpm-steps N         Number of DDPM steps (default: 4)"
            echo "  --dec-w WEIGHT         Decoder weight (default: 0.5)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --n-samples N          Number of samples (default: 1)"
            echo "  --colorfix-type TYPE   Color fix type: wavelet|adain|nofix (default: wavelet)"
            echo "  --calculate-metrics    Calculate PSNR/SSIM/LPIPS after inference (default)"
            echo "  --no-calculate-metrics Skip metric calculation"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic usage with default settings"
            echo "  $0"
            echo ""
            echo "  # Basic usage with StableSR-Turbo and subfolder"
            echo "  $0 --init-img ./inputs/test_example --outdir ./outputs --sub-folder run1"
            echo ""
            echo "  # Use more DDPM steps for better quality"
            echo "  $0 --init-img ./inputs/test_example --outdir ./outputs --ddpm-steps 200 --sub-folder step200"
            echo ""
            echo "  # Use custom checkpoint without metrics"
            echo "  $0 --init-img ./inputs/test_example --outdir ./outputs --ckpt ./my_model.ckpt --no-calculate-metrics"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Construct actual output directory
if [ -n "$SUB_FOLDER" ]; then
    ACTUAL_OUTDIR="$OUTDIR/$SUB_FOLDER"
else
    ACTUAL_OUTDIR="$OUTDIR"
fi

# Print configuration
echo "========================================="
echo "StableSR Inference Configuration"
echo "========================================="
echo "Config:            $CONFIG"
echo "Checkpoint:        $CKPT"
echo "VQGAN Ckpt:        $VQGAN_CKPT"
echo "Input (LQ):        $INIT_IMG"
echo "GT Images:         $GT_IMG"
echo "HR Images:         $HR_IMG"
echo "Output Base:       $OUTDIR"
if [ -n "$SUB_FOLDER" ]; then
echo "Subfolder:         $SUB_FOLDER"
echo "Actual Output:     $ACTUAL_OUTDIR"
fi
echo "DDPM Steps:        $DDPM_STEPS"
echo "Decoder W:         $DEC_W"
echo "Seed:              $SEED"
echo "N Samples:         $N_SAMPLES"
echo "Color Fix:         $COLORFIX_TYPE"
echo "Calculate Metrics: $CALCULATE_METRICS"
echo "========================================="
echo ""

# Run inference
python scripts/sr_val_ddpm_text_T_vqganfin_old.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --init-img "$INIT_IMG" \
    --outdir "$ACTUAL_OUTDIR" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w $DEC_W \
    --seed $SEED \
    --n_samples $N_SAMPLES \
    --vqgan_ckpt "$VQGAN_CKPT" \
    --colorfix_type "$COLORFIX_TYPE"

# Check inference status
INFERENCE_STATUS=$?
if [ $INFERENCE_STATUS -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "ERROR: Inference failed with exit code $INFERENCE_STATUS"
    echo "========================================="
    exit $INFERENCE_STATUS
fi

echo ""
echo "========================================="
echo "Inference complete! Results saved to: $ACTUAL_OUTDIR"
echo "========================================="

# Calculate metrics if enabled
if [ "$CALCULATE_METRICS" = true ]; then
    echo ""
    echo "========================================="
    echo "Calculating Metrics (PSNR/SSIM/LPIPS)..."
    echo "========================================="
    echo "Output directory: $ACTUAL_OUTDIR"
    echo "HR directory:     $HR_IMG"
    echo ""
    
    python scripts/calculate_metrics_standalone.py \
        --output_dir "$ACTUAL_OUTDIR" \
        --gt_dir "$HR_IMG"
    
    METRICS_STATUS=$?
    if [ $METRICS_STATUS -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Metrics calculation complete!"
        echo "Results saved to:"
        echo "  - $ACTUAL_OUTDIR/metrics.json"
        echo "  - $ACTUAL_OUTDIR/metrics.csv"
        echo "========================================="
    else
        echo ""
        echo "========================================="
        echo "WARNING: Metrics calculation failed with exit code $METRICS_STATUS"
        echo "========================================="
    fi
else
    echo ""
    echo "Metrics calculation skipped (use --calculate-metrics to enable)"
fi

echo ""
echo "========================================="
echo "All tasks complete!"
echo "========================================="

