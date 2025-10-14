#!/bin/bash
# Enhanced inference script with interactive menu
# This script provides multiple inference modes with parameter persistence

# Configuration file for storing default values
CONFIG_FILE=".inference_defaults.conf"

# Function to load saved defaults
load_defaults() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
    else
        # Initial default parameters
        DEFAULT_CKPT=""
        DEFAULT_LOGS_DIR="logs"
        DEFAULT_OUTPUT_BASE="validation_results"
        # DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
        DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
        DEFAULT_GT_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"
        DEFAULT_MAX_IMAGES="-1"
        DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
        DEFAULT_VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
    fi
}

# Function to save defaults
save_defaults() {
    cat > "$CONFIG_FILE" << EOF
# 保存的推理默认参数
DEFAULT_CKPT="$DEFAULT_CKPT"
DEFAULT_LOGS_DIR="$DEFAULT_LOGS_DIR"
DEFAULT_OUTPUT_BASE="$DEFAULT_OUTPUT_BASE"
DEFAULT_INIT_IMG="$DEFAULT_INIT_IMG"
DEFAULT_GT_IMG="$DEFAULT_GT_IMG"
DEFAULT_MAX_IMAGES="$DEFAULT_MAX_IMAGES"
DEFAULT_CONFIG="$DEFAULT_CONFIG"
DEFAULT_VQGAN_CKPT="$DEFAULT_VQGAN_CKPT"
EOF
    echo "✓ 默认参数已保存"
}

# Other fixed parameters
LOGS_DIR="logs"
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
DDPM_STEPS=200
DEC_W=0.5
SEED=42
N_SAMPLES=1
COLORFIX_TYPE="wavelet"

# Function to display menu
show_menu() {
    clear
    echo "===================================================="
    echo "           StableSR Edge 推理菜单"
    echo "===================================================="
    echo ""
    echo "1. 推理指定目录下全部 checkpoint (edge & no-edge)"
    echo ""
    echo "2. 推理指定 checkpoint 文件 (edge)"
    echo ""
    echo "3. 推理指定 checkpoint 文件 (no-edge)"
    echo ""
    echo "4. 生成推理结果报告 (CSV格式)"
    echo ""
    echo "0. 退出"
    echo ""
    echo "===================================================="
}

# Function to read user input with default value
read_with_default() {
    local prompt="$1"
    local default="$2"
    local result
    
    if [ -n "$default" ]; then
        read -e -p "$prompt [$default]: " result
        result="${result:-$default}"
    else
        read -e -p "$prompt: " result
    fi
    
    echo "$result"
}

# Function for mode 1: Inference all checkpoints
inference_all_checkpoints() {
    echo ""
    echo "=================================================="
    echo "  模式 1: 推理全部 Checkpoints"
    echo "=================================================="
    echo ""
    
    # Ask user for logs directory
    while true; do
        USER_LOGS_DIR=$(read_with_default "请输入 logs 目录路径" "$DEFAULT_LOGS_DIR")
        
        if [ ! -d "$USER_LOGS_DIR" ]; then
            echo "❌ 错误：目录不存在: $USER_LOGS_DIR"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ 目录存在: $USER_LOGS_DIR"
            # Save this as the new default
            DEFAULT_LOGS_DIR="$USER_LOGS_DIR"
            break
        fi
    done
    
    # List available directories in user-specified logs directory
    echo ""
    echo "可用的子目录："
    echo ""
    
    # Get all directories in user-specified directory
    if [ ! -d "$USER_LOGS_DIR" ]; then
        echo "❌ 错误：目录不存在: $USER_LOGS_DIR"
        return
    fi
    
    # Get list of directories (excluding child_runs)
    mapfile -t LOG_DIRS < <(find "$USER_LOGS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "child_runs" -printf "%f\n" | sort)
    
    if [ ${#LOG_DIRS[@]} -eq 0 ]; then
        echo "❌ 错误：目录下没有找到子目录"
        return
    fi
    
    # Display directories with numbers
    for i in "${!LOG_DIRS[@]}"; do
        echo "$((i+1)). ${LOG_DIRS[$i]}"
    done
    echo ""
    
    # Let user select directory
    while true; do
        read -p "请选择目录编号 [1-${#LOG_DIRS[@]}]: " DIR_CHOICE
        
        if [[ "$DIR_CHOICE" =~ ^[0-9]+$ ]] && [ "$DIR_CHOICE" -ge 1 ] && [ "$DIR_CHOICE" -le "${#LOG_DIRS[@]}" ]; then
            break
        else
            echo "❌ 无效选择，请输入 1 到 ${#LOG_DIRS[@]} 之间的数字"
        fi
    done
    
    # Determine target directory
    SELECTED_DIR_NAME="${LOG_DIRS[$((DIR_CHOICE-1))]}"
    TARGET_LOG_DIR="$USER_LOGS_DIR/$SELECTED_DIR_NAME"
    echo "✓ 将处理目录: $SELECTED_DIR_NAME"
    echo ""
    
    # Ask for output directory name
    OUTPUT_BASE=$(read_with_default "请输入保存目录名" "$DEFAULT_OUTPUT_BASE")
    echo "✓ 结果将保存到: $OUTPUT_BASE"
    echo ""
    
    # Save updated defaults
    DEFAULT_OUTPUT_BASE="$OUTPUT_BASE"
    save_defaults
    
    echo "将处理所有 checkpoint"
    echo "包括 edge（真实边缘）、no-edge（黑色边缘）和 dummy-edge（固定边缘）三种模式。"
    echo ""
    
    # Process checkpoints in selected directory
    echo ""
    echo "检查目录下的 checkpoints..."
    CKPT_DIR="$TARGET_LOG_DIR/checkpoints"
    
    if [ ! -d "$CKPT_DIR" ]; then
        echo "❌ 错误：checkpoints 目录不存在: $CKPT_DIR"
        return
    fi
    
    # Find all checkpoint files (excluding last.ckpt)
    # Include both regular files and symbolic links
    mapfile -t CKPT_FILES < <(find "$CKPT_DIR" -name "*.ckpt" \( -type f -o -type l \) ! -name "last.ckpt" | sort)
    
    if [ ${#CKPT_FILES[@]} -eq 0 ]; then
        echo "❌ 错误：没有找到 checkpoint 文件（已排除 last.ckpt）"
        return
    fi
    
    echo "✓ 找到 ${#CKPT_FILES[@]} 个 checkpoint 文件（已排除 last.ckpt）"
    echo ""
    
    # Process each checkpoint for edge mode
    echo "正在运行 EDGE 模式推理..."
    echo ""
    
    EDGE_PROCESSED=0
    EDGE_SKIPPED=0
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            echo "⚠ 跳过无法解析的 checkpoint: $CKPT_BASENAME"
            continue
        fi
        
        # Check if output directory already has images
        OUTPUT_CHECK="$OUTPUT_BASE/$SELECTED_DIR_NAME/edge/epochs_$((10#$EPOCH_NUM))"
        if [ -d "$OUTPUT_CHECK" ]; then
            # Count PNG files in output directory
            PNG_COUNT=$(find "$OUTPUT_CHECK" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
            if [ "$PNG_COUNT" -gt 0 ]; then
                echo "✓ 跳过 epoch=$EPOCH_NUM (已有 $PNG_COUNT 张图片)"
                ((EDGE_SKIPPED++))
                continue
            fi
        fi
        
        echo "→ 处理 epoch=$EPOCH_NUM"
        python scripts/auto_inference.py \
            --ckpt "$CKPT_FILE" \
            --logs_dir "$USER_LOGS_DIR" \
            --output_base "$OUTPUT_BASE" \
            --sub_folder "edge" \
            --init_img "$DEFAULT_INIT_IMG" \
            --gt_img "$DEFAULT_GT_IMG" \
            --config "$CONFIG" \
            --vqgan_ckpt "$VQGAN_CKPT" \
            --ddpm_steps $DDPM_STEPS \
            --dec_w $DEC_W \
            --seed $SEED \
            --n_samples $N_SAMPLES \
            --colorfix_type "$COLORFIX_TYPE" \
            --use_edge_processing \
            --skip_existing
        
        if [ $? -eq 0 ]; then
            ((EDGE_PROCESSED++))
        fi
    done
    
    echo ""
    echo "EDGE 模式统计: 已处理 $EDGE_PROCESSED 个，跳过 $EDGE_SKIPPED 个"
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Process each checkpoint for no-edge mode
    echo "正在运行 NO-EDGE 模式推理（使用黑色边缘图）..."
    echo ""
    
    NO_EDGE_PROCESSED=0
    NO_EDGE_SKIPPED=0
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            echo "⚠ 跳过无法解析的 checkpoint: $CKPT_BASENAME"
            continue
        fi
        
        # Check if output directory already has images
        OUTPUT_CHECK="$OUTPUT_BASE/$SELECTED_DIR_NAME/no_edge/epochs_$((10#$EPOCH_NUM))"
        if [ -d "$OUTPUT_CHECK" ]; then
            # Count PNG files in output directory
            PNG_COUNT=$(find "$OUTPUT_CHECK" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
            if [ "$PNG_COUNT" -gt 0 ]; then
                echo "✓ 跳过 epoch=$EPOCH_NUM (已有 $PNG_COUNT 张图片)"
                ((NO_EDGE_SKIPPED++))
                continue
            fi
        fi
        
        echo "→ 处理 epoch=$EPOCH_NUM"
        python scripts/auto_inference.py \
            --ckpt "$CKPT_FILE" \
            --logs_dir "$USER_LOGS_DIR" \
            --output_base "$OUTPUT_BASE" \
            --sub_folder "no_edge" \
            --init_img "$DEFAULT_INIT_IMG" \
            --gt_img "$DEFAULT_GT_IMG" \
            --config "$CONFIG" \
            --vqgan_ckpt "$VQGAN_CKPT" \
            --ddpm_steps $DDPM_STEPS \
            --dec_w $DEC_W \
            --seed $SEED \
            --n_samples $N_SAMPLES \
            --colorfix_type "$COLORFIX_TYPE" \
            --use_edge_processing \
            --use_white_edge \
            --skip_existing
        
        if [ $? -eq 0 ]; then
            ((NO_EDGE_PROCESSED++))
        fi
    done
    
    echo ""
    echo "NO-EDGE 模式统计: 已处理 $NO_EDGE_PROCESSED 个，跳过 $NO_EDGE_SKIPPED 个"
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Process each checkpoint for dummy-edge mode
    echo "正在运行 DUMMY-EDGE 模式推理（使用固定dummy edge图）..."
    echo ""
    
    DUMMY_EDGE_PROCESSED=0
    DUMMY_EDGE_SKIPPED=0
    DUMMY_EDGE_PATH="/stablesr_dataset/default_edge.png"
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            echo "⚠ 跳过无法解析的 checkpoint: $CKPT_BASENAME"
            continue
        fi
        
        # Check if output directory already has images
        OUTPUT_CHECK="$OUTPUT_BASE/$SELECTED_DIR_NAME/dummy_edge/epochs_$((10#$EPOCH_NUM))"
        if [ -d "$OUTPUT_CHECK" ]; then
            # Count PNG files in output directory
            PNG_COUNT=$(find "$OUTPUT_CHECK" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
            if [ "$PNG_COUNT" -gt 0 ]; then
                echo "✓ 跳过 epoch=$EPOCH_NUM (已有 $PNG_COUNT 张图片)"
                ((DUMMY_EDGE_SKIPPED++))
                continue
            fi
        fi
        
        echo "→ 处理 epoch=$EPOCH_NUM"
        python scripts/auto_inference.py \
            --ckpt "$CKPT_FILE" \
            --logs_dir "$USER_LOGS_DIR" \
            --output_base "$OUTPUT_BASE" \
            --sub_folder "dummy_edge" \
            --init_img "$DEFAULT_INIT_IMG" \
            --gt_img "$DEFAULT_GT_IMG" \
            --config "$CONFIG" \
            --vqgan_ckpt "$VQGAN_CKPT" \
            --ddpm_steps $DDPM_STEPS \
            --dec_w $DEC_W \
            --seed $SEED \
            --n_samples $N_SAMPLES \
            --colorfix_type "$COLORFIX_TYPE" \
            --use_edge_processing \
            --use_dummy_edge \
            --dummy_edge_path "$DUMMY_EDGE_PATH" \
            --skip_existing
        
        if [ $? -eq 0 ]; then
            ((DUMMY_EDGE_PROCESSED++))
        fi
    done
    
    echo ""
    echo "DUMMY-EDGE 模式统计: 已处理 $DUMMY_EDGE_PROCESSED 个，跳过 $DUMMY_EDGE_SKIPPED 个"
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Process with standard StableSR model for comparison
    echo "正在运行标准 STABLESR 模型推理（用于对比）..."
    echo ""
    
    STABLESR_CKPT="/stablesr_dataset/checkpoints/stablesr_000117.ckpt"
    STABLESR_CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
    
    if [ ! -f "$STABLESR_CKPT" ]; then
        echo "⚠ 警告：标准 StableSR checkpoint 不存在: $STABLESR_CKPT"
        echo "跳过 StableSR 模式推理"
        STABLESR_PROCESSED=0
        STABLESR_SKIPPED=0
    else
        # StableSR is a fixed baseline model, only need to run once
        OUTPUT_CHECK="$OUTPUT_BASE/$SELECTED_DIR_NAME/stablesr/baseline"
        
        # Check if StableSR results already exist
        if [ -d "$OUTPUT_CHECK" ]; then
            PNG_COUNT=$(find "$OUTPUT_CHECK" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
            if [ "$PNG_COUNT" -gt 0 ]; then
                echo "✓ StableSR baseline 已存在 ($PNG_COUNT 张图片)，跳过"
                STABLESR_PROCESSED=0
                STABLESR_SKIPPED=1
            else
                echo "→ 运行 StableSR baseline 推理"
                python scripts/auto_inference.py \
                    --ckpt "$STABLESR_CKPT" \
                    --logs_dir "$USER_LOGS_DIR" \
                    --output_base "$OUTPUT_BASE" \
                    --sub_folder "stablesr" \
                    --init_img "$DEFAULT_INIT_IMG" \
                    --gt_img "$DEFAULT_GT_IMG" \
                    --config "$STABLESR_CONFIG" \
                    --vqgan_ckpt "$VQGAN_CKPT" \
                    --ddpm_steps $DDPM_STEPS \
                    --dec_w $DEC_W \
                    --seed $SEED \
                    --n_samples $N_SAMPLES \
                    --colorfix_type "$COLORFIX_TYPE" \
                    --no_edge_processing \
                    --skip_existing \
                    --epoch_override "baseline" \
                    --exp_name_override "$SELECTED_DIR_NAME"
                
                if [ $? -eq 0 ]; then
                    STABLESR_PROCESSED=1
                    STABLESR_SKIPPED=0
                else
                    STABLESR_PROCESSED=0
                    STABLESR_SKIPPED=0
                fi
            fi
        else
            echo "→ 运行 StableSR baseline 推理"
            python scripts/auto_inference.py \
                --ckpt "$STABLESR_CKPT" \
                --logs_dir "$USER_LOGS_DIR" \
                --output_base "$OUTPUT_BASE" \
                --sub_folder "stablesr" \
                --init_img "$DEFAULT_INIT_IMG" \
                --gt_img "$DEFAULT_GT_IMG" \
                --config "$STABLESR_CONFIG" \
                --vqgan_ckpt "$VQGAN_CKPT" \
                --ddpm_steps $DDPM_STEPS \
                --dec_w $DEC_W \
                --seed $SEED \
                --n_samples $N_SAMPLES \
                --colorfix_type "$COLORFIX_TYPE" \
                --no_edge_processing \
                --skip_existing \
                --epoch_override "baseline" \
                --exp_name_override "$SELECTED_DIR_NAME"
            
            if [ $? -eq 0 ]; then
                STABLESR_PROCESSED=1
                STABLESR_SKIPPED=0
            else
                STABLESR_PROCESSED=0
                STABLESR_SKIPPED=0
            fi
        fi
        
        echo ""
        if [ $STABLESR_PROCESSED -eq 1 ]; then
            echo "STABLESR 模式统计: 已完成 baseline 推理"
        elif [ $STABLESR_SKIPPED -eq 1 ]; then
            echo "STABLESR 模式统计: baseline 已存在，跳过"
        else
            echo "STABLESR 模式统计: 推理失败"
        fi
    fi
    
    echo ""
    echo "===================================================="
    echo "  全部 checkpoints 处理完成！"
    echo "===================================================="
    echo ""
    
    # Show statistics
    echo "统计信息："
    echo "  EDGE 模式: 已处理 $EDGE_PROCESSED 个 checkpoints，跳过 $EDGE_SKIPPED 个"
    echo "  NO-EDGE 模式: 已处理 $NO_EDGE_PROCESSED 个 checkpoints，跳过 $NO_EDGE_SKIPPED 个"
    echo "  DUMMY-EDGE 模式: 已处理 $DUMMY_EDGE_PROCESSED 个 checkpoints，跳过 $DUMMY_EDGE_SKIPPED 个"
    if [ -f "$STABLESR_CKPT" ]; then
        if [ $STABLESR_PROCESSED -eq 1 ]; then
            echo "  STABLESR baseline: 已完成"
        elif [ $STABLESR_SKIPPED -eq 1 ]; then
            echo "  STABLESR baseline: 已存在（跳过）"
        else
            echo "  STABLESR baseline: 失败"
        fi
    else
        echo "  STABLESR baseline: 跳过（checkpoint 不存在）"
    fi
    echo "  总计 checkpoints: 已处理 $((EDGE_PROCESSED + NO_EDGE_PROCESSED + DUMMY_EDGE_PROCESSED)) 个，跳过 $((EDGE_SKIPPED + NO_EDGE_SKIPPED + DUMMY_EDGE_SKIPPED)) 个"
    echo ""
    
    echo "✓ 所有推理结果已生成，指标已自动计算"
    echo "结果保存在各子目录的 metrics.json 文件中"
    echo ""
    
    # Generate summary report using the Python script from menu 4
    echo "正在生成推理结果报告..."
    echo ""
    
    # Determine the results path
    RESULTS_PATH="$OUTPUT_BASE/$SELECTED_DIR_NAME"
    
    # Check if Python script exists
    PYTHON_SCRIPT="scripts/generate_metrics_report.py"
    if [ -f "$PYTHON_SCRIPT" ]; then
        echo "正在扫描推理结果目录: $RESULTS_PATH"
        python3 "$PYTHON_SCRIPT" "$RESULTS_PATH"
        
        # Display report location
        DIR_NAME=$(basename "$RESULTS_PATH")
        OUTPUT_REPORT="$RESULTS_PATH/${DIR_NAME}_inference_report.csv"
        
        if [ -f "$OUTPUT_REPORT" ]; then
            # Add footer with timestamp (3 lines)
            echo "" >> "$OUTPUT_REPORT"
            echo "" >> "$OUTPUT_REPORT"
            echo "$(date '+%a %b %d')" >> "$OUTPUT_REPORT"
            echo "$(date '+%T')" >> "$OUTPUT_REPORT"
            echo "$(date '+%Z %Y')" >> "$OUTPUT_REPORT"
            
            echo ""
            echo "===================================================="
            echo "✓ 推理结果报告已生成"
            echo "  报告位置: $OUTPUT_REPORT"
            echo "===================================================="
            echo ""
            
            # Show preview of the report
            echo "报告预览（前10行）："
            head -11 "$OUTPUT_REPORT" | column -t -s ','
            echo ""
            
            echo "报告尾部信息："
            tail -5 "$OUTPUT_REPORT"
            echo ""
        fi
    else
        echo "⚠ 找不到报告生成脚本: $PYTHON_SCRIPT"
        echo "跳过报告生成"
        echo ""
    fi
}

# Function for mode 2: Specific checkpoint with edge
inference_specific_edge() {
    echo ""
    echo "=================================================="
    echo "  模式 2: 推理指定 Checkpoint (Edge 模式)"
    echo "=================================================="
    echo ""
    
    # Get checkpoint path
    while true; do
        CKPT=$(read_with_default "Checkpoint 路径" "$DEFAULT_CKPT")
        
        # Validate checkpoint exists
        if [ ! -f "$CKPT" ]; then
            echo "❌ 错误：Checkpoint 文件不存在: $CKPT"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ Checkpoint 文件存在"
            break
        fi
    done
    
    # Get output directory
    OUTPUT_DIR=$(read_with_default "输出目录" "$DEFAULT_OUTPUT_BASE")
    echo "✓ 输出目录: $OUTPUT_DIR"
    
    # Get init image path
    while true; do
        INIT_IMG=$(read_with_default "输入 LR 图片目录" "$DEFAULT_INIT_IMG")
        
        if [ ! -d "$INIT_IMG" ]; then
            echo "❌ 错误：LR 图片目录不存在: $INIT_IMG"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            IMG_COUNT=$(ls -1 "$INIT_IMG" | wc -l)
            echo "✓ LR 图片目录存在，共 $IMG_COUNT 个文件"
            break
        fi
    done
    
    # Get GT image path
    while true; do
        GT_IMG=$(read_with_default "GT HR 图片目录" "$DEFAULT_GT_IMG")
        
        if [ ! -d "$GT_IMG" ]; then
            echo "❌ 错误：GT 图片目录不存在: $GT_IMG"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            GT_COUNT=$(ls -1 "$GT_IMG" | wc -l)
            echo "✓ GT 图片目录存在，共 $GT_COUNT 个文件"
            break
        fi
    done
    
    # Get config file path
    while true; do
        CONFIG_PATH=$(read_with_default "Config 文件路径" "$DEFAULT_CONFIG")
        
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "❌ 错误：Config 文件不存在: $CONFIG_PATH"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ Config 文件存在"
            break
        fi
    done
    
    # Get VQGAN checkpoint path
    while true; do
        VQGAN_PATH=$(read_with_default "VQGAN Checkpoint 路径" "$DEFAULT_VQGAN_CKPT")
        
        if [ ! -f "$VQGAN_PATH" ]; then
            echo "❌ 错误：VQGAN Checkpoint 文件不存在: $VQGAN_PATH"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ VQGAN Checkpoint 文件存在"
            break
        fi
    done
    
    # Ask if user wants to process specific file
    echo ""
    read -p "是否只推理指定文件? (y/n) [n]: " USE_SPECIFIC_FILE
    USE_SPECIFIC_FILE=${USE_SPECIFIC_FILE:-n}
    
    SPECIFIC_FILE=""
    if [ "$USE_SPECIFIC_FILE" = "y" ] || [ "$USE_SPECIFIC_FILE" = "Y" ]; then
        while true; do
            read -p "输入文件名 (例如: 00001.png): " SPECIFIC_FILE
            
            if [ -z "$SPECIFIC_FILE" ]; then
                echo "❌ 文件名不能为空"
                continue
            fi
            
            if [ ! -f "$INIT_IMG/$SPECIFIC_FILE" ]; then
                echo "❌ 错误：文件不存在: $INIT_IMG/$SPECIFIC_FILE"
                read -p "重新输入? (y/n): " retry
                if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                    return
                fi
            else
                echo "✓ 文件存在: $SPECIFIC_FILE"
                break
            fi
        done
    fi
    
    # Get max images (only if not using specific file)
    if [ -z "$SPECIFIC_FILE" ]; then
        MAX_IMAGES=$(read_with_default "最大推理图片数量 (-1=全部)" "$DEFAULT_MAX_IMAGES")
        echo "✓ 推理图片数量: $MAX_IMAGES"
    else
        MAX_IMAGES=1
        echo "✓ 推理单个文件"
    fi
    
    # Save as new defaults
    DEFAULT_CKPT="$CKPT"
    DEFAULT_OUTPUT_BASE="$OUTPUT_DIR"
    DEFAULT_INIT_IMG="$INIT_IMG"
    DEFAULT_GT_IMG="$GT_IMG"
    DEFAULT_MAX_IMAGES="$MAX_IMAGES"
    DEFAULT_CONFIG="$CONFIG_PATH"
    DEFAULT_VQGAN_CKPT="$VQGAN_PATH"
    save_defaults
    
    echo ""
    echo "正在运行 EDGE 模式推理..."
    echo ""
    
    # Extract experiment name from checkpoint path for output naming
    EXP_NAME=$(basename $(dirname $(dirname "$CKPT")))
    CKPT_NAME=$(basename "$CKPT" .ckpt)
    FINAL_OUTPUT="$OUTPUT_DIR/${EXP_NAME}_${CKPT_NAME}/edge"
    
    # Build command
    CMD="python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
        --config \"$CONFIG_PATH\" \
        --ckpt \"$CKPT\" \
        --init-img \"$INIT_IMG\" \
        --gt-img \"$GT_IMG\" \
        --outdir \"$FINAL_OUTPUT\" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --vqgan_ckpt \"$VQGAN_PATH\" \
        --colorfix_type \"$COLORFIX_TYPE\" \
        --max_images $MAX_IMAGES \
        --use_edge_processing"
    
    # Add specific file if set
    if [ -n "$SPECIFIC_FILE" ]; then
        CMD="$CMD --specific_file \"$SPECIFIC_FILE\""
    fi
    
    # Execute command
    eval $CMD
    
    echo ""
    echo "=================================================="
    echo "  推理完成！"
    echo "  输出位置: $FINAL_OUTPUT"
    echo "=================================================="
    echo ""
    
    # Ask if user wants to calculate metrics
    read -p "是否计算指标 (PSNR/SSIM/LPIPS)? (y/n) [y]: " CALC_METRICS
    CALC_METRICS=${CALC_METRICS:-y}
    
    if [ "$CALC_METRICS" = "y" ] || [ "$CALC_METRICS" = "Y" ]; then
        echo ""
        echo "正在计算指标..."
        python scripts/calculate_metrics_standalone.py \
            --output_dir "$FINAL_OUTPUT" \
            --gt_dir "$GT_IMG" \
            --crop_border 0
        echo ""
        echo "✓ 指标计算完成"
        echo "结果保存在: $FINAL_OUTPUT/metrics.json"
        echo ""
    fi
}

# Function for mode 3: Specific checkpoint without edge
inference_specific_no_edge() {
    echo ""
    echo "=================================================="
    echo "  模式 3: 推理指定 Checkpoint (No-Edge 模式)"
    echo "=================================================="
    echo ""
    
    # Get checkpoint path
    while true; do
        CKPT=$(read_with_default "Checkpoint 路径" "$DEFAULT_CKPT")
        
        # Validate checkpoint exists
        if [ ! -f "$CKPT" ]; then
            echo "❌ 错误：Checkpoint 文件不存在: $CKPT"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ Checkpoint 文件存在"
            break
        fi
    done
    
    # Get output directory
    OUTPUT_DIR=$(read_with_default "输出目录" "$DEFAULT_OUTPUT_BASE")
    echo "✓ 输出目录: $OUTPUT_DIR"
    
    # Get init image path
    while true; do
        INIT_IMG=$(read_with_default "输入 LR 图片目录" "$DEFAULT_INIT_IMG")
        
        if [ ! -d "$INIT_IMG" ]; then
            echo "❌ 错误：LR 图片目录不存在: $INIT_IMG"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            IMG_COUNT=$(ls -1 "$INIT_IMG" | wc -l)
            echo "✓ LR 图片目录存在，共 $IMG_COUNT 个文件"
            break
        fi
    done
    
    # Get GT image path
    while true; do
        GT_IMG=$(read_with_default "GT HR 图片目录" "$DEFAULT_GT_IMG")
        
        if [ ! -d "$GT_IMG" ]; then
            echo "❌ 错误：GT 图片目录不存在: $GT_IMG"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            GT_COUNT=$(ls -1 "$GT_IMG" | wc -l)
            echo "✓ GT 图片目录存在，共 $GT_COUNT 个文件"
            break
        fi
    done
    
    # Get config file path
    while true; do
        CONFIG_PATH=$(read_with_default "Config 文件路径" "$DEFAULT_CONFIG")
        
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "❌ 错误：Config 文件不存在: $CONFIG_PATH"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ Config 文件存在"
            break
        fi
    done
    
    # Get VQGAN checkpoint path
    while true; do
        VQGAN_PATH=$(read_with_default "VQGAN Checkpoint 路径" "$DEFAULT_VQGAN_CKPT")
        
        if [ ! -f "$VQGAN_PATH" ]; then
            echo "❌ 错误：VQGAN Checkpoint 文件不存在: $VQGAN_PATH"
            read -p "重新输入? (y/n): " retry
            if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                return
            fi
        else
            echo "✓ VQGAN Checkpoint 文件存在"
            break
        fi
    done
    
    # Ask if user wants to process specific file
    echo ""
    read -p "是否只推理指定文件? (y/n) [n]: " USE_SPECIFIC_FILE
    USE_SPECIFIC_FILE=${USE_SPECIFIC_FILE:-n}
    
    SPECIFIC_FILE=""
    if [ "$USE_SPECIFIC_FILE" = "y" ] || [ "$USE_SPECIFIC_FILE" = "Y" ]; then
        while true; do
            read -p "输入文件名 (例如: 00001.png): " SPECIFIC_FILE
            
            if [ -z "$SPECIFIC_FILE" ]; then
                echo "❌ 文件名不能为空"
                continue
            fi
            
            if [ ! -f "$INIT_IMG/$SPECIFIC_FILE" ]; then
                echo "❌ 错误：文件不存在: $INIT_IMG/$SPECIFIC_FILE"
                read -p "重新输入? (y/n): " retry
                if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                    return
                fi
            else
                echo "✓ 文件存在: $SPECIFIC_FILE"
                break
            fi
        done
    fi
    
    # Get max images (only if not using specific file)
    if [ -z "$SPECIFIC_FILE" ]; then
        MAX_IMAGES=$(read_with_default "最大推理图片数量 (-1=全部)" "$DEFAULT_MAX_IMAGES")
        echo "✓ 推理图片数量: $MAX_IMAGES"
    else
        MAX_IMAGES=1
        echo "✓ 推理单个文件"
    fi
    
    # Save as new defaults
    DEFAULT_CKPT="$CKPT"
    DEFAULT_OUTPUT_BASE="$OUTPUT_DIR"
    DEFAULT_INIT_IMG="$INIT_IMG"
    DEFAULT_GT_IMG="$GT_IMG"
    DEFAULT_MAX_IMAGES="$MAX_IMAGES"
    DEFAULT_CONFIG="$CONFIG_PATH"
    DEFAULT_VQGAN_CKPT="$VQGAN_PATH"
    save_defaults
    
    echo ""
    echo "正在运行 NO-EDGE 模式推理（使用黑色边缘图）..."
    echo ""
    
    # Extract experiment name from checkpoint path for output naming
    EXP_NAME=$(basename $(dirname $(dirname "$CKPT")))
    CKPT_NAME=$(basename "$CKPT" .ckpt)
    FINAL_OUTPUT="$OUTPUT_DIR/${EXP_NAME}_${CKPT_NAME}/no_edge"
    
    # Build command
    # For no-edge mode, we use --use_edge_processing with --use_white_edge
    # This passes black (all negative ones) edge maps to the model
    CMD="python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
        --config \"$CONFIG_PATH\" \
        --ckpt \"$CKPT\" \
        --init-img \"$INIT_IMG\" \
        --gt-img \"$GT_IMG\" \
        --outdir \"$FINAL_OUTPUT\" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --vqgan_ckpt \"$VQGAN_PATH\" \
        --colorfix_type \"$COLORFIX_TYPE\" \
        --max_images $MAX_IMAGES \
        --use_edge_processing \
        --use_white_edge"
    
    # Add specific file if set
    if [ -n "$SPECIFIC_FILE" ]; then
        CMD="$CMD --specific_file \"$SPECIFIC_FILE\""
    fi
    
    # Execute command
    eval $CMD
    
    echo ""
    echo "=================================================="
    echo "  推理完成！"
    echo "  输出位置: $FINAL_OUTPUT"
    echo "=================================================="
    echo ""
    
    # Ask if user wants to calculate metrics
    read -p "是否计算指标 (PSNR/SSIM/LPIPS)? (y/n) [y]: " CALC_METRICS
    CALC_METRICS=${CALC_METRICS:-y}
    
    if [ "$CALC_METRICS" = "y" ] || [ "$CALC_METRICS" = "Y" ]; then
        echo ""
        echo "正在计算指标..."
        python scripts/calculate_metrics_standalone.py \
            --output_dir "$FINAL_OUTPUT" \
            --gt_dir "$GT_IMG" \
            --crop_border 0
        echo ""
        echo "✓ 指标计算完成"
        echo "结果保存在: $FINAL_OUTPUT/metrics.json"
        echo ""
    fi
}

# Function for mode 4: Generate report
generate_report() {
    echo ""
    echo "=================================================="
    echo "  模式 4: 生成推理结果报告 (CSV格式)"
    echo "=================================================="
    echo ""
    
    # Find latest directory in /logs
    LOGS_BASE_DIR="/root/dp/StableSR_Edge_v2_loss/logs"
    
    if [ -d "$LOGS_BASE_DIR" ]; then
        # Get list of directories (excluding child_runs) sorted by modification time
        LATEST_DIR=$(find "$LOGS_BASE_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "child_runs" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        
        if [ -n "$LATEST_DIR" ]; then
            DIR_NAME=$(basename "$LATEST_DIR")
            echo "检测到 logs 目录中最新的目录:"
            echo "  $DIR_NAME"
            echo ""
            read -p "是否使用此目录? (y/n) [y]: " USE_LATEST
            USE_LATEST=${USE_LATEST:-y}
            
            if [ "$USE_LATEST" = "y" ] || [ "$USE_LATEST" = "Y" ]; then
                RESULTS_PATH="$LATEST_DIR"
                echo "✓ 使用目录: $RESULTS_PATH"
            else
                RESULTS_PATH=""
            fi
        else
            echo "⚠ 未在 logs 目录中找到任何子目录"
            RESULTS_PATH=""
        fi
    else
        echo "⚠ logs 目录不存在: $LOGS_BASE_DIR"
        RESULTS_PATH=""
    fi
    
    # If no path selected, ask user to input manually
    if [ -z "$RESULTS_PATH" ]; then
        echo ""
        while true; do
            read -p "请输入推理结果目录路径: " RESULTS_PATH
            
            if [ -z "$RESULTS_PATH" ]; then
                echo "❌ 错误：路径不能为空"
                read -p "是否返回菜单? (y/n): " return_menu
                if [ "$return_menu" = "y" ] || [ "$return_menu" = "Y" ]; then
                    return
                fi
                continue
            fi
            
            if [ ! -d "$RESULTS_PATH" ]; then
                echo "❌ 错误：目录不存在: $RESULTS_PATH"
                read -p "重新输入? (y/n): " retry
                if [ "$retry" != "y" ] && [ "$retry" != "Y" ]; then
                    return
                fi
            else
                echo "✓ 目录存在: $RESULTS_PATH"
                break
            fi
        done
    fi
    
    echo ""
    echo "正在扫描推理结果目录..."
    echo ""
    
    # Check if Python script exists
    PYTHON_SCRIPT="scripts/generate_metrics_report.py"
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "❌ 错误：找不到报告生成脚本: $PYTHON_SCRIPT"
        return
    fi
    
    # Generate the report using Python script
    echo "正在生成报告..."
    python3 "$PYTHON_SCRIPT" "$RESULTS_PATH"
    
    # Get the new filename format
    DIR_NAME=$(basename "$RESULTS_PATH")
    OUTPUT_REPORT="$RESULTS_PATH/${DIR_NAME}_inference_report.csv"
    
    # Add footer with timestamp (3 lines) if report exists
    if [ -f "$OUTPUT_REPORT" ]; then
        echo "" >> "$OUTPUT_REPORT"
        echo "" >> "$OUTPUT_REPORT"
        echo "$(date '+%a %b %d')" >> "$OUTPUT_REPORT"
        echo "$(date '+%T')" >> "$OUTPUT_REPORT"
        echo "$(date '+%Z %Y')" >> "$OUTPUT_REPORT"
    fi
    
    echo ""
    echo "=================================================="
    echo "  报告生成完成！"
    echo "=================================================="
    echo ""
    echo "✓ 报告已保存到: $OUTPUT_REPORT"
    echo ""
    echo "报告格式说明："
    echo "- 包含 PSNR、SSIM、LPIPS 三种指标"
    echo "- 每种指标包含平均值和10个图片的单独数值"
    echo "- 列包含 StableSR、edge loss 和 Epoch 结果"
    echo "- 报告尾部包含生成时间（三行格式）"
    echo ""
    
    # Show footer preview
    if [ -f "$OUTPUT_REPORT" ]; then
        echo "报告尾部信息："
        tail -5 "$OUTPUT_REPORT"
        echo ""
    fi
}

# Main program
main() {
    # Load saved defaults
    load_defaults
    
    # Main menu - execute once and exit
    show_menu
    read -p "请选择 [0-4]: " choice
    
    case $choice in
        1)
            inference_all_checkpoints
            ;;
        2)
            inference_specific_edge
            ;;
        3)
            inference_specific_no_edge
            ;;
        4)
            generate_report
            ;;
        0)
            echo ""
            echo "退出中..."
            exit 0
            ;;
        *)
            echo ""
            echo "无效选项，请选择 0-4。"
            exit 1
            ;;
    esac
    
    # Exit after completing the selected task
    echo ""
    echo "✓ 任务完成，脚本退出"
    exit 0
}

# Check if script is run with command line arguments (legacy mode)
if [ $# -gt 0 ]; then
    # Legacy mode: support old command line arguments
    echo "传统模式：使用命令行参数运行"
    echo "提示：不带参数运行即可进入交互式菜单"
    echo ""
    
    # Default parameters
    SUB_FOLDER=""
    INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
    GT_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"
    HR_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"
    CKPT=""
    OUTPUT_BASE="validation_results"
    
    # Parse command line arguments
    DRY_RUN=""
    EXP_FILTER=""
    SKIP_EXISTING="--skip_existing"
    INCLUDE_LAST=""
    CALCULATE_METRICS=""
    
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
            --ckpt)
                CKPT=$2
                shift 2
                ;;
            --sub-folder|--sub_folder)
                SUB_FOLDER=$2
                shift 2
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "交互式模式 (新功能):"
                echo "  不带参数运行即可进入交互式菜单"
                echo ""
                echo "传统模式:"
                echo "  使用命令行参数进行批量处理"
                echo ""
                echo "选项:"
                echo "  --ckpt PATH            指定要处理的 checkpoint"
                echo "  --sub-folder NAME      每个实验下的子文件夹名称"
                echo "  --dry-run              仅打印命令不执行"
                echo "  --exp-filter FILTER    只处理匹配 FILTER 的实验"
                echo "  --skip-existing        如果输出目录已存在则跳过 (默认)"
                echo "  --no-skip-existing     强制覆盖已存在的结果"
                echo "  --ddpm-steps STEPS     DDPM 步数 (默认: 200)"
                echo "  --dec-w WEIGHT         解码器权重 (默认: 0.5)"
                echo "  --config CONFIG        配置文件路径"
                echo "  --help                 显示此帮助信息"
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                echo "使用 --help 查看使用信息"
                exit 1
                ;;
        esac
    done
    
    # Build command arguments
    CMD_ARGS=(
        --logs_dir "$LOGS_DIR"
        --output_base "$OUTPUT_BASE"
        --init_img "$INIT_IMG"
        --gt_img "$GT_IMG"
        --config "$CONFIG"
        --vqgan_ckpt "$VQGAN_CKPT"
        --ddpm_steps $DDPM_STEPS
        --dec_w $DEC_W
        --seed $SEED
        --n_samples $N_SAMPLES
        --colorfix_type "$COLORFIX_TYPE"
        --use_edge_processing
    )
    
    # Add optional arguments
    if [ -n "$CKPT" ]; then
        CMD_ARGS+=(--ckpt "$CKPT")
    fi
    if [ -n "$SUB_FOLDER" ]; then
        CMD_ARGS+=(--sub_folder "$SUB_FOLDER")
    fi
    if [ -n "$DRY_RUN" ]; then
        CMD_ARGS+=($DRY_RUN)
    fi
    if [ -n "$EXP_FILTER" ]; then
        CMD_ARGS+=($EXP_FILTER)
    fi
    if [ -n "$SKIP_EXISTING" ]; then
        CMD_ARGS+=($SKIP_EXISTING)
    fi
    if [ -n "$INCLUDE_LAST" ]; then
        CMD_ARGS+=($INCLUDE_LAST)
    fi
    if [ -n "$CALCULATE_METRICS" ]; then
        CMD_ARGS+=($CALCULATE_METRICS)
    fi
    
    # Run the Python script
    python scripts/auto_inference.py "${CMD_ARGS[@]}"
else
    # Interactive mode
    main
fi
