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
        DEFAULT_OUTPUT_BASE="validation_results"
        DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
        DEFAULT_GT_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"
        DEFAULT_MAX_IMAGES="-1"
        DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml"
        DEFAULT_VQGAN_CKPT="/root/checkpoints/vqgan_cfw_00011.ckpt"
    fi
}

# Function to save defaults
save_defaults() {
    cat > "$CONFIG_FILE" << EOF
# 保存的推理默认参数
DEFAULT_CKPT="$DEFAULT_CKPT"
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
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml"
VQGAN_CKPT="/root/checkpoints/vqgan_cfw_00011.ckpt"
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
    echo "1. 推理 logs 目录下全部 checkpoint (edge & no-edge)"
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
    
    # List available directories in logs
    echo "可用的 logs 子目录："
    echo ""
    
    # Get all directories in logs/
    if [ ! -d "$LOGS_DIR" ]; then
        echo "❌ 错误：logs 目录不存在: $LOGS_DIR"
        read -p "按 Enter 返回菜单..."
        return
    fi
    
    # Get list of directories
    mapfile -t LOG_DIRS < <(find "$LOGS_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
    
    if [ ${#LOG_DIRS[@]} -eq 0 ]; then
        echo "❌ 错误：logs 目录下没有找到子目录"
        read -p "按 Enter 返回菜单..."
        return
    fi
    
    # Display directories with numbers
    for i in "${!LOG_DIRS[@]}"; do
        echo "$((i+1)). ${LOG_DIRS[$i]}"
    done
    echo ""
    echo "0. 处理全部目录"
    echo ""
    
    # Let user select directory
    while true; do
        read -p "请选择目录编号 [0-${#LOG_DIRS[@]}] (0=全部): " DIR_CHOICE
        DIR_CHOICE=${DIR_CHOICE:-0}
        
        if [[ "$DIR_CHOICE" =~ ^[0-9]+$ ]] && [ "$DIR_CHOICE" -ge 0 ] && [ "$DIR_CHOICE" -le "${#LOG_DIRS[@]}" ]; then
            break
        else
            echo "❌ 无效选择，请输入 0 到 ${#LOG_DIRS[@]} 之间的数字"
        fi
    done
    
    # Determine target directory
    if [ "$DIR_CHOICE" -eq 0 ]; then
        TARGET_LOG_DIR="$LOGS_DIR"
        SELECTED_DIR_NAME=""
        echo "✓ 将处理全部 logs 目录"
    else
        SELECTED_DIR_NAME="${LOG_DIRS[$((DIR_CHOICE-1))]}"
        TARGET_LOG_DIR="$LOGS_DIR/$SELECTED_DIR_NAME"
        echo "✓ 将处理目录: $SELECTED_DIR_NAME"
    fi
    echo ""
    
    # Ask for output directory name
    read -p "请输入保存目录名 [validation_results]: " OUTPUT_BASE
    OUTPUT_BASE=${OUTPUT_BASE:-validation_results}
    echo "✓ 结果将保存到: $OUTPUT_BASE"
    echo ""
    
    echo "将处理所有 checkpoint"
    echo "包括 edge 和 no-edge 两种模式。"
    echo ""
    
    read -p "按 Enter 继续，或按 Ctrl+C 取消..."
    
    # Check if processing single directory or all directories
    if [ -n "$SELECTED_DIR_NAME" ]; then
        # Single directory mode - process each checkpoint individually
        echo ""
        echo "检查目录下的 checkpoints..."
        CKPT_DIR="$TARGET_LOG_DIR/checkpoints"
        
        if [ ! -d "$CKPT_DIR" ]; then
            echo "❌ 错误：checkpoints 目录不存在: $CKPT_DIR"
            read -p "按 Enter 返回菜单..."
            return
        fi
        
        # Find all checkpoint files
        mapfile -t CKPT_FILES < <(find "$CKPT_DIR" -name "*.ckpt" -type f | sort)
        
        if [ ${#CKPT_FILES[@]} -eq 0 ]; then
            echo "❌ 错误：没有找到 checkpoint 文件"
            read -p "按 Enter 返回菜单..."
            return
        fi
        
        echo "✓ 找到 ${#CKPT_FILES[@]} 个 checkpoint 文件"
        echo ""
        
        # Process each checkpoint for edge mode
        echo "正在运行 EDGE 模式推理..."
        echo ""
        
        for CKPT_FILE in "${CKPT_FILES[@]}"; do
            python scripts/auto_inference.py \
                --ckpt "$CKPT_FILE" \
                --logs_dir "$LOGS_DIR" \
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
        done
        
        echo ""
        echo "=================================================="
        echo ""
        
        # Process each checkpoint for no-edge mode
        echo "正在运行 NO-EDGE 模式推理（使用白色边缘图）..."
        echo ""
        
        for CKPT_FILE in "${CKPT_FILES[@]}"; do
            python scripts/auto_inference.py \
                --ckpt "$CKPT_FILE" \
                --logs_dir "$LOGS_DIR" \
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
        done
    else
        # All directories mode - let auto_inference.py handle discovery
        # First run: Edge mode
        echo ""
        echo "正在运行 EDGE 模式推理..."
        echo ""
        
        python scripts/auto_inference.py \
            --logs_dir "$TARGET_LOG_DIR" \
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
        
        echo ""
        echo "=================================================="
        echo ""
        
        # Second run: No-edge mode (white edge maps)
        echo ""
        echo "正在运行 NO-EDGE 模式推理（使用白色边缘图）..."
        echo ""
        
        python scripts/auto_inference.py \
            --logs_dir "$TARGET_LOG_DIR" \
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
    fi
    
    echo ""
    echo "===================================================="
    echo "  全部 checkpoints 处理完成！"
    echo "===================================================="
    echo ""
    
    echo "✓ 所有推理结果已生成，指标已自动计算"
    echo "结果保存在各子目录的 metrics.json 文件中"
    echo ""
    
    read -p "按 Enter 返回菜单..."
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
    
    read -p "按 Enter 返回菜单..."
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
    echo "正在运行 NO-EDGE 模式推理（使用白色边缘图）..."
    echo ""
    
    # Extract experiment name from checkpoint path for output naming
    EXP_NAME=$(basename $(dirname $(dirname "$CKPT")))
    CKPT_NAME=$(basename "$CKPT" .ckpt)
    FINAL_OUTPUT="$OUTPUT_DIR/${EXP_NAME}_${CKPT_NAME}/no_edge"
    
    # Build command
    # For no-edge mode, we use --use_edge_processing with --use_white_edge
    # This passes white (all ones) edge maps to the model
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
    
    read -p "按 Enter 返回菜单..."
}

# Function for mode 4: Generate report
generate_report() {
    echo ""
    echo "=================================================="
    echo "  模式 4: 生成推理结果报告 (CSV格式)"
    echo "=================================================="
    echo ""
    
    # Get inference results path
    while true; do
        read -p "请输入推理结果目录路径: " RESULTS_PATH
        
        if [ -z "$RESULTS_PATH" ]; then
            echo "❌ 错误：路径不能为空"
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
    
    echo ""
    echo "正在扫描推理结果目录..."
    echo ""
    
    # Check if Python script exists
    PYTHON_SCRIPT="scripts/generate_metrics_report.py"
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "❌ 错误：找不到报告生成脚本: $PYTHON_SCRIPT"
        read -p "按 Enter 返回菜单..."
        return
    fi
    
    # Generate the report using Python script
    echo "正在生成报告..."
    python3 "$PYTHON_SCRIPT" "$RESULTS_PATH"
    
    # Get the new filename format
    DIR_NAME=$(basename "$RESULTS_PATH")
    OUTPUT_REPORT="$RESULTS_PATH/${DIR_NAME}_inference_report.csv"
    
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
    echo ""
    
    read -p "按 Enter 返回菜单..."
}

# Main program
main() {
    # Load saved defaults
    load_defaults
    
    # Main menu loop
    while true; do
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
                sleep 2
                ;;
        esac
    done
}

# Check if script is run with command line arguments (legacy mode)
if [ $# -gt 0 ]; then
    # Legacy mode: support old command line arguments
    echo "传统模式：使用命令行参数运行"
    echo "提示：不带参数运行即可进入交互式菜单"
    echo ""
    
    # Default parameters
    SUB_FOLDER=""
    INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
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
