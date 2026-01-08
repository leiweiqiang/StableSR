#!/bin/bash

# Batch Inference Script for 4x and 8x Blended Input Tiled Inference
# ===================================================================
# This script processes multiple images with different scale factors (4x and 8x)
# using the blended input tiled inference approach.
#
# Key parameters:
# - DEC_W: Weight for combining VQGAN and Diffusion (default: 0.5)
# - BLEND_ALPHA/BETA: Blending weights for LR and edge images (default: 1.0)
# - DDPM_STEPS: Number of sampling steps (default: 200)
# - TILE_SIZE: Size for tiling operation (default: 512)
# - --noedge: Use pure black edge maps instead of loading edge images (default: false)
#
# Usage:
#   ./batch_inference_blended_tile.sh              # Use edge images
#   ./batch_inference_blended_tile.sh --noedge     # Use pure black edge maps
#
# Output:
#   Each image will be processed and saved to outputs/4x/ or outputs/8x/
#   Output directory naming: {image_id}_{scale}_{noedge}_{w=value}_e{epoch_value}
#   Examples:
#     - 0001_4x_w0.5_e111 (4x scale, with edge, w=0.5, epoch=111)
#     - 0001_4x_noedge_w0.5_e111 (4x scale, no edge, w=0.5, epoch=111)
#     - 0001_8x_w0.3_e111 (8x scale, with edge, w=0.3, epoch=111)
#   A command file (inference_command.txt) will be saved in each output directory
#   containing the exact command used for that specific image inference

# Configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# Model checkpoints
CKPT_4X="/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt"
CKPT_8X="/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000111.ckpt"

# Input directories
LR_4X_DIR="/stablesr_dataset/weiqiang/lr_x4_960x540"
LR_8X_DIR="/stablesr_dataset/weiqiang/lr_x8_480x270"
EDGE_DIR="/stablesr_dataset/weiqiang/hr_4k_canny"

# Output base directory
OUTPUT_BASE="outputs"

# Processing parameters
BLEND_ALPHA=1.0
BLEND_BETA=1.0
DDPM_STEPS=200
TILE_SIZE=512
TILE_OVERLAP=32
VQGAN_TILE_SIZE=1024
VQGAN_TILE_STRIDE=512
SEED=42
TEXT_PROMPT=""
COLORFIX="adain"
# DEC_W=0.5
DEC_W=0.0
NOEDGE=true

# Scale factors
LR_DOWNSCALE_FACTOR_4X=4
LR_DOWNSCALE_FACTOR_8X=8

# Images to skip (already processed)
SKIP_IMAGES=("0001.png" "0013.png" "0031.png" "0044.png")

# Optional: process only specific image ID (e.g., --only-image 0051)
ONLY_IMAGE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --noedge)
            NOEDGE=true
            shift
            ;;
        --only-image)
            ONLY_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --noedge           Use pure black edge maps instead of loading edge images"
            echo "  --only-image ID    Process only specific image ID (e.g., --only-image 0051)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "This script processes multiple images with different scale factors (4x and 8x)"
            echo "using the blended input tiled inference approach."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Batch Inference for 4x and 8x Images"
echo "=========================================="
echo "4x LR Directory: $LR_4X_DIR"
echo "8x LR Directory: $LR_8X_DIR"
if [ "$NOEDGE" = true ]; then
    echo "Edge Mode: Pure black edge maps (--noedge)"
else
    echo "Edge Directory: $EDGE_DIR"
fi
echo "4x Checkpoint: $CKPT_4X"
echo "8x Checkpoint: $CKPT_8X"
echo "Skip Images: ${SKIP_IMAGES[*]}"
echo "=========================================="

# Function to check if image should be skipped
should_skip_image() {
    local image_name="$1"
    
    # Check if we should process only specific image
    if [ -n "$ONLY_IMAGE" ]; then
        local image_id=$(basename "$image_name" .png)
        if [[ "$image_id" != "$ONLY_IMAGE" ]]; then
            return 0  # Should skip
        fi
    fi
    
    # Check if image is in skip list
    for skip_img in "${SKIP_IMAGES[@]}"; do
        if [[ "$image_name" == "$skip_img" ]]; then
            return 0  # Should skip
        fi
    done
    return 1  # Should not skip
}

# Function to extract epoch value from checkpoint path
extract_epoch_from_ckpt() {
    local ckpt_path="$1"
    # Try to extract epoch number from different path formats:
    # Format 1: ".../epoch=000111.ckpt"
    local epoch1=$(echo "$ckpt_path" | grep -o 'epoch=[0-9]*' | cut -d'=' -f2 | sed 's/^0*//')
    if [ -n "$epoch1" ]; then
        echo "$epoch1"
        return
    fi
    
    # Format 2: ".../stablesr_000117.ckpt" or similar
    local epoch2=$(echo "$ckpt_path" | grep -o '[^/]*_[0-9]*\.ckpt$' | grep -o '[0-9]*' | sed 's/^0*//')
    if [ -n "$epoch2" ]; then
        echo "$epoch2"
        return
    fi
    
    # If no epoch found, return empty
    echo ""
}

# Function to check and complete existing output directories
complete_existing_directories() {
    echo ""
    echo "=========================================="
    echo "Checking and completing existing directories"
    echo "=========================================="
    
    # Check 4x directories
    if [ -d "$OUTPUT_BASE/4x" ]; then
        echo "Checking 4x directories..."
        for dir in "$OUTPUT_BASE/4x"/*; do
            if [ -d "$dir" ]; then
                dir_name=$(basename "$dir")
                # Check if directory name already contains epoch info and skip stablesr directories
                if [[ ! "$dir_name" =~ _e[0-9]+$ ]] && [[ ! "$dir_name" =~ stablesr ]]; then
                    echo "Found incomplete directory: $dir_name"
                    
                    # Check if inference_command.txt exists
                    command_file="$dir/inference_command.txt"
                    if [ -f "$command_file" ]; then
                        # Extract checkpoint path from command file
                        ckpt_path=$(grep "    --ckpt " "$command_file" | head -1 | sed 's/.*--ckpt "\([^"]*\)".*/\1/')
                        if [ -n "$ckpt_path" ]; then
                            # Extract epoch value
                            epoch_value=$(extract_epoch_from_ckpt "$ckpt_path")
                            if [ -n "$epoch_value" ]; then
                                # Create new directory name with epoch
                                new_dir_name="${dir_name}_e${epoch_value}"
                                new_dir_path="$OUTPUT_BASE/4x/$new_dir_name"
                                
                                if [ ! -d "$new_dir_path" ]; then
                                    echo "  Renaming: $dir_name -> $new_dir_name"
                                    mv "$dir" "$new_dir_path"
                                    echo "  ✅ Completed: $new_dir_name"
                                else
                                    echo "  ⚠️  Target directory already exists: $new_dir_name"
                                fi
                            else
                                echo "  ❌ Could not extract epoch from checkpoint: $ckpt_path"
                            fi
                        else
                            echo "  ❌ Could not find checkpoint path in command file"
                        fi
                    else
                        echo "  ❌ No inference_command.txt found in: $dir_name"
                    fi
                fi
            fi
        done
    fi
    
    # Check 8x directories
    if [ -d "$OUTPUT_BASE/8x" ]; then
        echo "Checking 8x directories..."
        for dir in "$OUTPUT_BASE/8x"/*; do
            if [ -d "$dir" ]; then
                dir_name=$(basename "$dir")
                # Check if directory name already contains epoch info and skip stablesr directories
                if [[ ! "$dir_name" =~ _e[0-9]+$ ]] && [[ ! "$dir_name" =~ stablesr ]]; then
                    echo "Found incomplete directory: $dir_name"
                    
                    # Check if inference_command.txt exists
                    command_file="$dir/inference_command.txt"
                    if [ -f "$command_file" ]; then
                        # Extract checkpoint path from command file
                        ckpt_path=$(grep "    --ckpt " "$command_file" | head -1 | sed 's/.*--ckpt "\([^"]*\)".*/\1/')
                        if [ -n "$ckpt_path" ]; then
                            # Extract epoch value
                            epoch_value=$(extract_epoch_from_ckpt "$ckpt_path")
                            if [ -n "$epoch_value" ]; then
                                # Create new directory name with epoch
                                new_dir_name="${dir_name}_e${epoch_value}"
                                new_dir_path="$OUTPUT_BASE/8x/$new_dir_name"
                                
                                if [ ! -d "$new_dir_path" ]; then
                                    echo "  Renaming: $dir_name -> $new_dir_name"
                                    mv "$dir" "$new_dir_path"
                                    echo "  ✅ Completed: $new_dir_name"
                                else
                                    echo "  ⚠️  Target directory already exists: $new_dir_name"
                                fi
                            else
                                echo "  ❌ Could not extract epoch from checkpoint: $ckpt_path"
                            fi
                        else
                            echo "  ❌ Could not find checkpoint path in command file"
                        fi
                    else
                        echo "  ❌ No inference_command.txt found in: $dir_name"
                    fi
                fi
            fi
        done
    fi
    
    echo "Directory completion check finished."
    echo "=========================================="
}

# Function to process a single image
process_image() {
    local lr_img="$1"
    local edge_img="$2"
    local output_dir="$3"
    local ckpt="$4"
    local lr_downscale_factor="$5"
    local scale_name="$6"
    
    local image_name=$(basename "$lr_img")
    local image_id=$(basename "$lr_img" .png)
    
    # Check if output directory already exists and has results
    if [ -d "$output_dir" ]; then
        # Check if there's an output image in the directory
        local has_output=false
        if ls "$output_dir"/*.png 1> /dev/null 2>&1; then
            has_output=true
        fi
        
        if [ "$has_output" = true ]; then
            echo "⏭️  Skipping $scale_name: $image_name (output directory already exists: $(basename "$output_dir"))"
            return 0
        else
            echo "⚠️  Output directory exists but empty, will regenerate: $(basename "$output_dir")"
            rm -rf "$output_dir"
        fi
    fi
    
    echo ""
    echo "Processing $scale_name: $image_name"
    echo "LR Image: $lr_img"
    echo "Edge Image: $edge_img"
    echo "Output: $output_dir"
    echo "Checkpoint: $ckpt"
    echo "Downscale Factor: $lr_downscale_factor"
    echo "----------------------------------------"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Generate command file
    local command_file="$output_dir/inference_command.txt"
    echo "# Inference Command for $scale_name: $image_name" > "$command_file"
    echo "# Generated on: $(date)" >> "$command_file"
    echo "# LR Image: $lr_img" >> "$command_file"
    if [ "$NOEDGE" = true ]; then
        echo "# Edge Mode: Pure black edge maps (--noedge)" >> "$command_file"
    else
        echo "# Edge Image: $edge_img" >> "$command_file"
    fi
    echo "# Output Directory: $output_dir" >> "$command_file"
    echo "# Checkpoint: $ckpt" >> "$command_file"
    echo "# Downscale Factor: $lr_downscale_factor" >> "$command_file"
    echo "" >> "$command_file"
    echo "python scripts/inference_blended_input_tile.py \\" >> "$command_file"
    echo "    --lr-img \"$lr_img\" \\" >> "$command_file"
    if [ "$NOEDGE" = true ]; then
        echo "    --noedge \\" >> "$command_file"
    else
        echo "    --edge-img \"$edge_img\" \\" >> "$command_file"
    fi
    echo "    --outdir \"$output_dir\" \\" >> "$command_file"
    echo "    --config \"$CONFIG\" \\" >> "$command_file"
    echo "    --ckpt \"$ckpt\" \\" >> "$command_file"
    echo "    --vqgan-ckpt \"$VQGAN_CKPT\" \\" >> "$command_file"
    echo "    --blend-alpha \"$BLEND_ALPHA\" \\" >> "$command_file"
    echo "    --blend-beta \"$BLEND_BETA\" \\" >> "$command_file"
    echo "    --lr-downscale-factor \"$lr_downscale_factor\" \\" >> "$command_file"
    echo "    --ddpm-steps \"$DDPM_STEPS\" \\" >> "$command_file"
    echo "    --tile-size \"$TILE_SIZE\" \\" >> "$command_file"
    echo "    --tile-overlap \"$TILE_OVERLAP\" \\" >> "$command_file"
    echo "    --seed \"$SEED\" \\" >> "$command_file"
    echo "    --text-prompt \"$TEXT_PROMPT\" \\" >> "$command_file"
    echo "    --colorfix \"$COLORFIX\" \\" >> "$command_file"
    echo "    --vqgan-tile-size \"$VQGAN_TILE_SIZE\" \\" >> "$command_file"
    echo "    --vqgan-tile-stride \"$VQGAN_TILE_STRIDE\" \\" >> "$command_file"
    echo "    --dec-w \"$DEC_W\"" >> "$command_file"
    
    echo "📝 Command saved to: $command_file"
    
    # Run inference
    if [ "$NOEDGE" = true ]; then
        python scripts/inference_blended_input_tile.py \
            --lr-img "$lr_img" \
            --noedge \
            --outdir "$output_dir" \
            --config "$CONFIG" \
            --ckpt "$ckpt" \
            --vqgan-ckpt "$VQGAN_CKPT" \
            --blend-alpha "$BLEND_ALPHA" \
            --blend-beta "$BLEND_BETA" \
            --lr-downscale-factor "$lr_downscale_factor" \
            --ddpm-steps "$DDPM_STEPS" \
            --tile-size "$TILE_SIZE" \
            --tile-overlap "$TILE_OVERLAP" \
            --seed "$SEED" \
            --text-prompt "$TEXT_PROMPT" \
            --colorfix "$COLORFIX" \
            --vqgan-tile-size "$VQGAN_TILE_SIZE" \
            --vqgan-tile-stride "$VQGAN_TILE_STRIDE" \
            --dec-w "$DEC_W"
    else
        python scripts/inference_blended_input_tile.py \
            --lr-img "$lr_img" \
            --edge-img "$edge_img" \
            --outdir "$output_dir" \
            --config "$CONFIG" \
            --ckpt "$ckpt" \
            --vqgan-ckpt "$VQGAN_CKPT" \
            --blend-alpha "$BLEND_ALPHA" \
            --blend-beta "$BLEND_BETA" \
            --lr-downscale-factor "$lr_downscale_factor" \
            --ddpm-steps "$DDPM_STEPS" \
            --tile-size "$TILE_SIZE" \
            --tile-overlap "$TILE_OVERLAP" \
            --seed "$SEED" \
            --text-prompt "$TEXT_PROMPT" \
            --colorfix "$COLORFIX" \
            --vqgan-tile-size "$VQGAN_TILE_SIZE" \
            --vqgan-tile-stride "$VQGAN_TILE_STRIDE" \
            --dec-w "$DEC_W"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $scale_name: $image_name"
    else
        echo "❌ Failed to process $scale_name: $image_name"
        return 1
    fi
}

# Check if required files and directories exist
echo "Checking required files and directories..."

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$VQGAN_CKPT" ]; then
    echo "❌ Error: VQGAN checkpoint not found: $VQGAN_CKPT"
    exit 1
fi

if [ ! -f "$CKPT_4X" ]; then
    echo "❌ Error: 4x checkpoint not found: $CKPT_4X"
    exit 1
fi

if [ ! -f "$CKPT_8X" ]; then
    echo "❌ Error: 8x checkpoint not found: $CKPT_8X"
    exit 1
fi

if [ ! -d "$LR_4X_DIR" ]; then
    echo "❌ Error: 4x LR directory not found: $LR_4X_DIR"
    exit 1
fi

if [ ! -d "$LR_8X_DIR" ]; then
    echo "❌ Error: 8x LR directory not found: $LR_8X_DIR"
    exit 1
fi

if [ "$NOEDGE" = false ] && [ ! -d "$EDGE_DIR" ]; then
    echo "❌ Error: Edge directory not found: $EDGE_DIR"
    exit 1
fi

echo "✅ All required files and directories found"

# Complete existing directories that don't have epoch info
complete_existing_directories

# Get list of images to process
echo ""
echo "Scanning for images to process..."

# Get all PNG files from 4x directory
LR_4X_IMAGES=()
for img in "$LR_4X_DIR"/*.png; do
    if [ -f "$img" ]; then
        image_name=$(basename "$img")
        if ! should_skip_image "$image_name"; then
            LR_4X_IMAGES+=("$img")
        else
            echo "⏭️  Skipping 4x: $image_name (already processed)"
        fi
    fi
done

# Get all PNG files from 8x directory
LR_8X_IMAGES=()
for img in "$LR_8X_DIR"/*.png; do
    if [ -f "$img" ]; then
        image_name=$(basename "$img")
        if ! should_skip_image "$image_name"; then
            LR_8X_IMAGES+=("$img")
        else
            echo "⏭️  Skipping 8x: $image_name (already processed)"
        fi
    fi
done

echo "Found ${#LR_4X_IMAGES[@]} 4x images to process"
echo "Found ${#LR_8X_IMAGES[@]} 8x images to process"

if [ ${#LR_4X_IMAGES[@]} -eq 0 ] && [ ${#LR_8X_IMAGES[@]} -eq 0 ]; then
    echo "❌ No images to process (all images are skipped or not found)"
    exit 1
fi

# Process 4x images
if [ ${#LR_4X_IMAGES[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Processing 4x Images"
    echo "=========================================="
    
    for lr_img in "${LR_4X_IMAGES[@]}"; do
        image_name=$(basename "$lr_img")
        image_id=$(basename "$lr_img" .png)
        
        # Build output directory name with identifiers
        output_suffix="4x"
        if [ "$NOEDGE" = true ]; then
            output_suffix="${output_suffix}_noedge"
        fi
        output_suffix="${output_suffix}_w${DEC_W}_e111"
        
        output_dir="$OUTPUT_BASE/4x/${image_id}_${output_suffix}"
        
        if [ "$NOEDGE" = false ]; then
            edge_img="$EDGE_DIR/$image_name"
            if [ ! -f "$edge_img" ]; then
                echo "❌ Warning: Edge image not found for $image_name, skipping..."
                continue
            fi
        else
            edge_img=""  # Not used when --noedge is set
        fi
        
        process_image "$lr_img" "$edge_img" "$output_dir" "$CKPT_4X" "$LR_DOWNSCALE_FACTOR_4X" "4x"
    done
fi

# Process 8x images
if [ ${#LR_8X_IMAGES[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Processing 8x Images"
    echo "=========================================="
    
    for lr_img in "${LR_8X_IMAGES[@]}"; do
        image_name=$(basename "$lr_img")
        image_id=$(basename "$lr_img" .png)
        
        # Build output directory name with identifiers
        output_suffix="8x"
        if [ "$NOEDGE" = true ]; then
            output_suffix="${output_suffix}_noedge"
        fi
        output_suffix="${output_suffix}_w${DEC_W}_e111"
        
        output_dir="$OUTPUT_BASE/8x/${image_id}_${output_suffix}"
        
        if [ "$NOEDGE" = false ]; then
            edge_img="$EDGE_DIR/$image_name"
            if [ ! -f "$edge_img" ]; then
                echo "❌ Warning: Edge image not found for $image_name, skipping..."
                continue
            fi
        else
            edge_img=""  # Not used when --noedge is set
        fi
        
        process_image "$lr_img" "$edge_img" "$output_dir" "$CKPT_8X" "$LR_DOWNSCALE_FACTOR_8X" "8x"
    done
fi

echo ""
echo "=========================================="
echo "Batch Inference Completed!"
echo "=========================================="
echo "4x images processed: ${#LR_4X_IMAGES[@]}"
echo "8x images processed: ${#LR_8X_IMAGES[@]}"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="

# Show output structure
echo ""
echo "Output structure:"
echo "Directory naming: {image_id}_{scale}_{noedge}_{w=value}_e{epoch_value}"
if [ -d "$OUTPUT_BASE/4x" ]; then
    echo "4x outputs:"
    ls -la "$OUTPUT_BASE/4x" | grep "^d" | awk '{print "  " $9}'
    echo "  Command files: inference_command.txt (in each subdirectory)"
fi

if [ -d "$OUTPUT_BASE/8x" ]; then
    echo "8x outputs:"
    ls -la "$OUTPUT_BASE/8x" | grep "^d" | awk '{print "  " $9}'
    echo "  Command files: inference_command.txt (in each subdirectory)"
fi

echo ""
echo "✅ Batch inference completed successfully!"
