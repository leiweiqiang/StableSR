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
        DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
        # DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
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

# Other fixed parameters - export them so they can be used in parallel processes
export LOGS_DIR="logs"
export CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
export VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
export DDPM_STEPS=1000
export DEC_W=0
export SEED=42
export N_SAMPLES=1
export COLORFIX_TYPE="wavelet"
export INPUT_SIZE=512  # LR input size - must match training (resize_lq=True resizes LR to GT size)

# Function to display menu
show_menu() {
    clear
    echo "===================================================="
    echo "           StableSR Edge 推理菜单"
    echo "===================================================="
    echo ""
    echo "1. 推理指定目录下全部 checkpoint (edge & no-edge & dummy-edge)"
    echo ""
    echo "2. 推理指定 checkpoint 文件 (edge & no-edge & dummy-edge)"
    echo ""
    echo "3. 生成推理结果报告 (CSV格式)"
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

# Function to process a single checkpoint inference task
process_single_inference() {
    local CKPT_FILE="$1"
    local MODE="$2"
    local USER_LOGS_DIR="$3"
    local OUTPUT_BASE="$4"
    local SELECTED_DIR_NAME="$5"
    local INIT_IMG="$6"
    local GT_IMG="$7"
    local CONFIG="$8"
    local VQGAN_CKPT="$9"
    local ENABLE_METRICS_RECALC="${10}"
    local DUMMY_EDGE_PATH="${11:-}"
    
    # Extract epoch number
    CKPT_BASENAME=$(basename "$CKPT_FILE")
    if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
        EPOCH_NUM="${BASH_REMATCH[1]}"
    else
        echo "⚠ 跳过无法解析的 checkpoint: $CKPT_BASENAME" >&2
        return 1
    fi
    
    # Determine subfolder and flags based on mode
    case "$MODE" in
        edge)
            SUB_FOLDER="edge"
            EDGE_FLAGS="--use_edge_processing"
            ;;
        no_edge)
            SUB_FOLDER="no_edge"
            EDGE_FLAGS="--use_edge_processing --use_white_edge"
            ;;
        dummy_edge)
            SUB_FOLDER="dummy_edge"
            EDGE_FLAGS="--use_edge_processing --use_dummy_edge --dummy_edge_path $DUMMY_EDGE_PATH"
            ;;
        *)
            echo "❌ 错误：未知的模式 $MODE" >&2
            return 1
            ;;
    esac
    
    # Check if output directory already has images
    OUTPUT_CHECK="$OUTPUT_BASE/$SELECTED_DIR_NAME/$SUB_FOLDER/epochs_$((10#$EPOCH_NUM))"
    if [ -d "$OUTPUT_CHECK" ]; then
        PNG_COUNT=$(find "$OUTPUT_CHECK" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
        if [ "$PNG_COUNT" -gt 0 ]; then
            # Images exist, check if metrics need recalculation
            if [ "$ENABLE_METRICS_RECALC" = "true" ]; then
                METRICS_FILE="$OUTPUT_CHECK/metrics.json"
                if [ -f "$METRICS_FILE" ]; then
                    if ! grep -q "edge_overlap" "$METRICS_FILE"; then
                        echo "→ [$MODE] epoch=$EPOCH_NUM 已有图片，正在重新计算指标..." >&2
                        python scripts/recalculate_metrics.py "$OUTPUT_CHECK" "$GT_IMG" > /dev/null 2>&1
                        if [ $? -eq 0 ]; then
                            echo "  ✓ [$MODE] epoch=$EPOCH_NUM 指标计算完成" >&2
                        else
                            echo "  ⚠ [$MODE] epoch=$EPOCH_NUM 指标计算失败" >&2
                        fi
                    fi
                fi
            fi
            echo "✓ [$MODE] 跳过 epoch=$EPOCH_NUM (已有 $PNG_COUNT 张图片)" >&2
            return 0
        fi
    fi
    
    # Run inference
    echo "→ [$MODE] 处理 epoch=$EPOCH_NUM" >&2
    python scripts/auto_inference.py \
        --ckpt "$CKPT_FILE" \
        --logs_dir "$USER_LOGS_DIR" \
        --output_base "$OUTPUT_BASE" \
        --sub_folder "$SUB_FOLDER" \
        --init_img "$INIT_IMG" \
        --gt_img "$GT_IMG" \
        --config "$CONFIG" \
        --vqgan_ckpt "$VQGAN_CKPT" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --colorfix_type "$COLORFIX_TYPE" \
        --input_size $INPUT_SIZE \
        $EDGE_FLAGS \
        --skip_existing 2>&1 | sed "s/^/[$MODE-$EPOCH_NUM] /" >&2
    
    if [ $? -eq 0 ]; then
        echo "✓ [$MODE] epoch=$EPOCH_NUM 完成" >&2
        return 0
    else
        echo "❌ [$MODE] epoch=$EPOCH_NUM 失败" >&2
        return 1
    fi
}

# Export the function so it can be used by parallel processes
export -f process_single_inference

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
    
    # Ask whether to recalculate metrics for existing results
    echo ""
    echo "是否重新计算已有结果的指标？"
    echo "  注意：新推理的结果会自动计算所有指标"
    echo "  此选项仅针对跳过的已有结果"
    echo ""
    read -p "重新计算指标? (y/n) [默认: n]: " RECALC_METRICS
    RECALC_METRICS="${RECALC_METRICS:-n}"
    
    if [ "$RECALC_METRICS" = "y" ] || [ "$RECALC_METRICS" = "Y" ]; then
        echo "✓ 将检查并重新计算缺失的指标（包括 Edge PSNR）"
        ENABLE_METRICS_RECALC=true
    else
        echo "✓ 跳过指标重新计算（仅计算新推理的结果）"
        ENABLE_METRICS_RECALC=false
    fi
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
    
    # Pre-check: count how many checkpoints need inference
    echo "正在检查哪些 checkpoint 需要推理..."
    echo ""
    
    NEW_CKPTS_EDGE=0
    NEW_CKPTS_NO_EDGE=0
    NEW_CKPTS_DUMMY=0
    EXISTING_CKPTS=0
    
    # Arrays to store checkpoint epoch numbers that need inference
    NEW_EDGE_EPOCHS=()
    NEW_NO_EDGE_EPOCHS=()
    NEW_DUMMY_EPOCHS=()
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
            
            # Check edge mode
            OUTPUT_CHECK_EDGE="$OUTPUT_BASE/$SELECTED_DIR_NAME/edge/epochs_$((10#$EPOCH_NUM))"
            if [ -d "$OUTPUT_CHECK_EDGE" ]; then
                PNG_COUNT=$(find "$OUTPUT_CHECK_EDGE" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
                if [ "$PNG_COUNT" -eq 0 ]; then
                    ((NEW_CKPTS_EDGE++))
                    NEW_EDGE_EPOCHS+=("$EPOCH_NUM")
                fi
            else
                ((NEW_CKPTS_EDGE++))
                NEW_EDGE_EPOCHS+=("$EPOCH_NUM")
            fi
            
            # Check no-edge mode
            OUTPUT_CHECK_NO_EDGE="$OUTPUT_BASE/$SELECTED_DIR_NAME/no_edge/epochs_$((10#$EPOCH_NUM))"
            if [ -d "$OUTPUT_CHECK_NO_EDGE" ]; then
                PNG_COUNT=$(find "$OUTPUT_CHECK_NO_EDGE" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
                if [ "$PNG_COUNT" -eq 0 ]; then
                    ((NEW_CKPTS_NO_EDGE++))
                    NEW_NO_EDGE_EPOCHS+=("$EPOCH_NUM")
                fi
            else
                ((NEW_CKPTS_NO_EDGE++))
                NEW_NO_EDGE_EPOCHS+=("$EPOCH_NUM")
            fi
            
            # Check dummy-edge mode
            OUTPUT_CHECK_DUMMY="$OUTPUT_BASE/$SELECTED_DIR_NAME/dummy_edge/epochs_$((10#$EPOCH_NUM))"
            if [ -d "$OUTPUT_CHECK_DUMMY" ]; then
                PNG_COUNT=$(find "$OUTPUT_CHECK_DUMMY" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l)
                if [ "$PNG_COUNT" -eq 0 ]; then
                    ((NEW_CKPTS_DUMMY++))
                    NEW_DUMMY_EPOCHS+=("$EPOCH_NUM")
                fi
            else
                ((NEW_CKPTS_DUMMY++))
                NEW_DUMMY_EPOCHS+=("$EPOCH_NUM")
            fi
        fi
    done
    
    # Create a unique sorted list of all checkpoints that need inference
    declare -A UNIQUE_EPOCHS
    for epoch in "${NEW_EDGE_EPOCHS[@]}" "${NEW_NO_EDGE_EPOCHS[@]}" "${NEW_DUMMY_EPOCHS[@]}"; do
        UNIQUE_EPOCHS[$epoch]=1
    done
    
    # Convert to sorted array
    AVAILABLE_EPOCHS=($(for epoch in "${!UNIQUE_EPOCHS[@]}"; do echo "$epoch"; done | sort))
    
    # Display available checkpoints
    echo "=================================================="
    echo "  未推理的 Checkpoint 列表"
    echo "=================================================="
    echo ""
    
    if [ ${#AVAILABLE_EPOCHS[@]} -eq 0 ]; then
        echo "✓ 没有新的推理任务，所有 checkpoint 结果已存在"
        
        # If no new inference but metrics recalc is enabled, continue
        if [ "$ENABLE_METRICS_RECALC" = true ]; then
            echo "  将继续检查指标..."
        else
            echo "  将直接生成报告..."
        fi
        echo ""
    else
        echo "找到 ${#AVAILABLE_EPOCHS[@]} 个未推理的 checkpoint："
        echo ""
        
        # Display checkpoint list with IDs
        for i in "${!AVAILABLE_EPOCHS[@]}"; do
            epoch="${AVAILABLE_EPOCHS[$i]}"
            id=$((i + 1))
            
            # Check which modes need inference for this epoch
            modes_needed=()
            for check_epoch in "${NEW_EDGE_EPOCHS[@]}"; do
                if [ "$check_epoch" = "$epoch" ]; then
                    modes_needed+=("edge")
                    break
                fi
            done
            for check_epoch in "${NEW_NO_EDGE_EPOCHS[@]}"; do
                if [ "$check_epoch" = "$epoch" ]; then
                    modes_needed+=("no-edge")
                    break
                fi
            done
            for check_epoch in "${NEW_DUMMY_EPOCHS[@]}"; do
                if [ "$check_epoch" = "$epoch" ]; then
                    modes_needed+=("dummy-edge")
                    break
                fi
            done
            
            modes_str=$(IFS=", "; echo "${modes_needed[*]}")
            printf "  [%2d] epoch=%s (需要: %s)\n" "$id" "$epoch" "$modes_str"
        done
        
        echo ""
        echo "=================================================="
        echo ""
        echo "请选择要推理的 checkpoint："
        echo "  - 输入序号，多个序号用逗号分隔，例如：1,3,5"
        echo "  - 输入 'all' 或直接回车推理所有"
        echo "  - 输入 'q' 取消"
        echo ""
        
        while true; do
            read -p "请选择 [all]: " CKPT_SELECTION
            CKPT_SELECTION="${CKPT_SELECTION:-all}"
            
            if [ "$CKPT_SELECTION" = "q" ] || [ "$CKPT_SELECTION" = "Q" ]; then
                echo "✗ 用户取消推理"
                echo ""
                return
            fi
            
            if [ "$CKPT_SELECTION" = "all" ] || [ "$CKPT_SELECTION" = "ALL" ]; then
                # Select all checkpoints
                SELECTED_EPOCHS=("${AVAILABLE_EPOCHS[@]}")
                echo "✓ 将推理所有 ${#SELECTED_EPOCHS[@]} 个 checkpoint"
                break
            fi
            
            # Parse comma-separated IDs
            IFS=',' read -ra SELECTED_IDS <<< "$CKPT_SELECTION"
            SELECTED_EPOCHS=()
            INVALID_SELECTION=false
            
            for id_str in "${SELECTED_IDS[@]}"; do
                # Trim whitespace
                id_str=$(echo "$id_str" | xargs)
                
                # Validate it's a number
                if ! [[ "$id_str" =~ ^[0-9]+$ ]]; then
                    echo "❌ 错误：'$id_str' 不是有效的序号"
                    INVALID_SELECTION=true
                    break
                fi
                
                # Convert to array index (id - 1)
                idx=$((id_str - 1))
                
                # Validate range
                if [ $idx -lt 0 ] || [ $idx -ge ${#AVAILABLE_EPOCHS[@]} ]; then
                    echo "❌ 错误：序号 $id_str 超出范围 (1-${#AVAILABLE_EPOCHS[@]})"
                    INVALID_SELECTION=true
                    break
                fi
                
                # Add epoch to selection
                SELECTED_EPOCHS+=("${AVAILABLE_EPOCHS[$idx]}")
            done
            
            if [ "$INVALID_SELECTION" = true ]; then
                echo "请重新输入"
                echo ""
                continue
            fi
            
            if [ ${#SELECTED_EPOCHS[@]} -eq 0 ]; then
                echo "❌ 错误：未选择任何 checkpoint"
                echo "请重新输入"
                echo ""
                continue
            fi
            
            # Display selected checkpoints
            echo ""
            echo "✓ 已选择 ${#SELECTED_EPOCHS[@]} 个 checkpoint:"
            for epoch in "${SELECTED_EPOCHS[@]}"; do
                echo "    - epoch=$epoch"
            done
            break
        done
        
        echo ""
    fi
    
    # Ask for number of parallel threads
    echo ""
    echo "=================================================="
    echo "  并行处理设置"
    echo "=================================================="
    echo ""
    echo "并行处理可以同时运行多个推理任务，加快处理速度"
    echo "建议根据GPU数量和显存大小设置线程数"
    echo ""
    read -p "请输入并行线程数 [默认: 20]: " NUM_THREADS
    NUM_THREADS="${NUM_THREADS:-20}"
    
    # Validate thread number
    if ! [[ "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ]; then
        echo "❌ 错误：线程数必须是正整数，使用默认值 20"
        NUM_THREADS=20
    fi
    
    echo "✓ 将使用 $NUM_THREADS 个并行线程"
    echo ""
    echo "✓ 开始执行推理..."
    echo ""
    
    # Prepare task list for edge mode
    EDGE_TASK_FILE=$(mktemp)
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            continue
        fi
        
        # Check if this epoch is in the selected list
        EPOCH_SELECTED=false
        if [ ${#SELECTED_EPOCHS[@]} -gt 0 ]; then
            for selected_epoch in "${SELECTED_EPOCHS[@]}"; do
                if [ "$selected_epoch" = "$EPOCH_NUM" ]; then
                    EPOCH_SELECTED=true
                    break
                fi
            done
        else
            # If no selection made (all existing), process all
            EPOCH_SELECTED=true
        fi
        
        if [ "$EPOCH_SELECTED" = true ]; then
            echo "$CKPT_FILE" >> "$EDGE_TASK_FILE"
        fi
    done
    
    # Process edge mode checkpoints in parallel
    echo "正在运行 EDGE 模式推理（并行数：$NUM_THREADS）..."
    echo ""
    
    cat "$EDGE_TASK_FILE" | xargs -P "$NUM_THREADS" -I {} bash -c "process_single_inference '{}' 'edge' '$USER_LOGS_DIR' '$OUTPUT_BASE' '$SELECTED_DIR_NAME' '$DEFAULT_INIT_IMG' '$DEFAULT_GT_IMG' '$CONFIG' '$VQGAN_CKPT' '$ENABLE_METRICS_RECALC'"
    
    rm -f "$EDGE_TASK_FILE"
    
    echo ""
    echo "EDGE 模式处理完成"
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Prepare task list for no-edge mode
    NO_EDGE_TASK_FILE=$(mktemp)
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            continue
        fi
        
        # Check if this epoch is in the selected list
        EPOCH_SELECTED=false
        if [ ${#SELECTED_EPOCHS[@]} -gt 0 ]; then
            for selected_epoch in "${SELECTED_EPOCHS[@]}"; do
                if [ "$selected_epoch" = "$EPOCH_NUM" ]; then
                    EPOCH_SELECTED=true
                    break
                fi
            done
        else
            # If no selection made (all existing), process all
            EPOCH_SELECTED=true
        fi
        
        if [ "$EPOCH_SELECTED" = true ]; then
            echo "$CKPT_FILE" >> "$NO_EDGE_TASK_FILE"
        fi
    done
    
    # Process no-edge mode checkpoints in parallel
    echo "正在运行 NO-EDGE 模式推理（使用黑色边缘图，并行数：$NUM_THREADS）..."
    echo ""
    
    cat "$NO_EDGE_TASK_FILE" | xargs -P "$NUM_THREADS" -I {} bash -c "process_single_inference '{}' 'no_edge' '$USER_LOGS_DIR' '$OUTPUT_BASE' '$SELECTED_DIR_NAME' '$DEFAULT_INIT_IMG' '$DEFAULT_GT_IMG' '$CONFIG' '$VQGAN_CKPT' '$ENABLE_METRICS_RECALC'"
    
    rm -f "$NO_EDGE_TASK_FILE"
    
    echo ""
    echo "NO-EDGE 模式处理完成"
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Prepare task list for dummy-edge mode
    DUMMY_EDGE_TASK_FILE=$(mktemp)
    DUMMY_EDGE_PATH="/stablesr_dataset/default_edge.png"
    
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Extract epoch number from checkpoint filename
        CKPT_BASENAME=$(basename "$CKPT_FILE")
        if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
            EPOCH_NUM="${BASH_REMATCH[1]}"
        else
            continue
        fi
        
        # Check if this epoch is in the selected list
        EPOCH_SELECTED=false
        if [ ${#SELECTED_EPOCHS[@]} -gt 0 ]; then
            for selected_epoch in "${SELECTED_EPOCHS[@]}"; do
                if [ "$selected_epoch" = "$EPOCH_NUM" ]; then
                    EPOCH_SELECTED=true
                    break
                fi
            done
        else
            # If no selection made (all existing), process all
            EPOCH_SELECTED=true
        fi
        
        if [ "$EPOCH_SELECTED" = true ]; then
            echo "$CKPT_FILE" >> "$DUMMY_EDGE_TASK_FILE"
        fi
    done
    
    # Process dummy-edge mode checkpoints in parallel
    echo "正在运行 DUMMY-EDGE 模式推理（使用固定dummy edge图，并行数：$NUM_THREADS）..."
    echo ""
    
    cat "$DUMMY_EDGE_TASK_FILE" | xargs -P "$NUM_THREADS" -I {} bash -c "process_single_inference '{}' 'dummy_edge' '$USER_LOGS_DIR' '$OUTPUT_BASE' '$SELECTED_DIR_NAME' '$DEFAULT_INIT_IMG' '$DEFAULT_GT_IMG' '$CONFIG' '$VQGAN_CKPT' '$ENABLE_METRICS_RECALC' '$DUMMY_EDGE_PATH'"
    
    rm -f "$DUMMY_EDGE_TASK_FILE"
    
    echo ""
    echo "DUMMY-EDGE 模式处理完成"
    
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
                # Images exist, check if Edge PSNR is calculated (if enabled)
                if [ "$ENABLE_METRICS_RECALC" = true ]; then
                    METRICS_FILE="$OUTPUT_CHECK/metrics.json"
                    if [ -f "$METRICS_FILE" ]; then
                        # Check if Edge PSNR exists in metrics.json
                        if ! grep -q "edge_psnr" "$METRICS_FILE"; then
                            echo "→ StableSR baseline 已有图片，但缺少 Edge PSNR，正在计算..."
                            python scripts/recalculate_metrics.py "$OUTPUT_CHECK" "$DEFAULT_GT_IMG" > /dev/null 2>&1
                            if [ $? -eq 0 ]; then
                                echo "  ✓ Edge PSNR 计算完成"
                            else
                                echo "  ⚠ Edge PSNR 计算失败"
                            fi
                        fi
                    fi
                fi
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
                    --input_size $INPUT_SIZE \
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
                --input_size $INPUT_SIZE \
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
    
    # Check and recalculate metrics if needed (only if enabled)
    if [ "$ENABLE_METRICS_RECALC" = true ]; then
        echo "===================================================="
        echo "  批量检查并重新计算缺失的指标"
        echo "===================================================="
        echo ""
        
        # Find all metrics.json files in the results directory
        RESULTS_PATH="$OUTPUT_BASE/$SELECTED_DIR_NAME"
        
        if [ -d "$RESULTS_PATH" ]; then
        echo "扫描目录: $RESULTS_PATH"
        echo ""
        
        # Create temporary files for counting
        TEMP_DIR=$(mktemp -d)
        UPDATED_COUNT_FILE="$TEMP_DIR/updated"
        EXISTING_COUNT_FILE="$TEMP_DIR/existing"
        FAILED_COUNT_FILE="$TEMP_DIR/failed"
        echo "0" > "$UPDATED_COUNT_FILE"
        echo "0" > "$EXISTING_COUNT_FILE"
        echo "0" > "$FAILED_COUNT_FILE"
        
        # Find and process all metrics.json files
        find "$RESULTS_PATH" -name "metrics.json" -type f | while IFS= read -r METRICS_FILE; do
            METRICS_DIR=$(dirname "$METRICS_FILE")
            
            # Display relative path
            REL_PATH="${METRICS_DIR#$RESULTS_PATH/}"
            echo "检查: $REL_PATH"
            
            # Capture output and exit code
            OUTPUT=$(python scripts/recalculate_metrics.py "$METRICS_DIR" "$DEFAULT_GT_IMG" 2>&1)
            EXIT_CODE=$?
            
            # Display output with indentation
            echo "$OUTPUT" | while IFS= read -r line; do
                echo "  $line"
            done
            
            # Update counters based on exit code and output
            if [ $EXIT_CODE -eq 0 ]; then
                if echo "$OUTPUT" | grep -q "所有指标已存在"; then
                    echo $(($(cat "$EXISTING_COUNT_FILE") + 1)) > "$EXISTING_COUNT_FILE"
                else
                    echo $(($(cat "$UPDATED_COUNT_FILE") + 1)) > "$UPDATED_COUNT_FILE"
                fi
            else
                echo $(($(cat "$FAILED_COUNT_FILE") + 1)) > "$FAILED_COUNT_FILE"
            fi
            
            echo ""
        done
        
        # Read final counts
        L2LOSS_UPDATED=$(cat "$UPDATED_COUNT_FILE" 2>/dev/null || echo "0")
        L2LOSS_ALREADY_EXISTS=$(cat "$EXISTING_COUNT_FILE" 2>/dev/null || echo "0")
        L2LOSS_FAILED=$(cat "$FAILED_COUNT_FILE" 2>/dev/null || echo "0")
        
        # Cleanup temp directory
        rm -rf "$TEMP_DIR"
        
        echo "===================================================="
        echo "指标检查完成"
        echo "===================================================="
        echo ""
        
        TOTAL_METRICS=$((L2LOSS_UPDATED + L2LOSS_ALREADY_EXISTS + L2LOSS_FAILED))
        if [ $TOTAL_METRICS -gt 0 ]; then
            echo "统计信息："
            echo "  找到 $TOTAL_METRICS 个 metrics.json 文件"
            if [ $L2LOSS_UPDATED -gt 0 ]; then
                echo "  ✓ 已更新: $L2LOSS_UPDATED 个"
            fi
            if [ $L2LOSS_ALREADY_EXISTS -gt 0 ]; then
                echo "  ✓ 已存在: $L2LOSS_ALREADY_EXISTS 个"
            fi
            if [ $L2LOSS_FAILED -gt 0 ]; then
                echo "  ✗ 失败: $L2LOSS_FAILED 个"
            fi
        else
            echo "未找到 metrics.json 文件"
        fi
        echo ""
        fi  # End of if [ -d "$RESULTS_PATH" ]
    else
        echo "✓ 跳过指标批量检查（用户选择不重新计算）"
        echo ""
    fi  # End of if [ "$ENABLE_METRICS_RECALC" = true ]
    
    # Generate summary report using the Python script from menu 4
    echo "===================================================="
    echo "  正在生成推理结果报告"
    echo "===================================================="
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

# Function for mode 2: Specific checkpoint with all modes
inference_specific_checkpoint() {
    echo ""
    echo "=================================================="
    echo "  模式 2: 推理指定 Checkpoint (全模式)"
    echo "=================================================="
    echo ""
    echo "将使用指定的 checkpoint 运行三种模式："
    echo "  - Edge 模式（真实边缘）"
    echo "  - No-Edge 模式（黑色边缘）"
    echo "  - Dummy-Edge 模式（固定边缘）"
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
    
    # Extract experiment name and epoch from checkpoint path
    EXP_NAME=$(basename $(dirname $(dirname "$CKPT")))
    CKPT_BASENAME=$(basename "$CKPT")
    
    # Extract epoch number
    if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
        EPOCH_NUM="${BASH_REMATCH[1]}"
    else
        EPOCH_NUM="unknown"
    fi
    
    # Base output directory
    BASE_OUTPUT="$OUTPUT_DIR/$EXP_NAME"
    
    echo ""
    echo "=================================================="
    echo "  推理配置"
    echo "=================================================="
    echo "Checkpoint: $CKPT"
    echo "Epoch: $EPOCH_NUM"
    echo "输出目录: $BASE_OUTPUT"
    echo "LR图片: $INIT_IMG"
    echo "GT图片: $GT_IMG"
    if [ -n "$SPECIFIC_FILE" ]; then
        echo "指定文件: $SPECIFIC_FILE"
    else
        echo "推理数量: $MAX_IMAGES 张"
    fi
    echo "=================================================="
    echo ""
    
    # Confirm before running
    read -p "确认开始推理三种模式? (y/n) [y]: " CONFIRM_RUN
    CONFIRM_RUN=${CONFIRM_RUN:-y}
    
    if [ "$CONFIRM_RUN" != "y" ] && [ "$CONFIRM_RUN" != "Y" ]; then
        echo "✗ 用户取消推理"
        return
    fi
    
    # Process EDGE mode
    echo ""
    echo "=================================================="
    echo "  [1/3] EDGE 模式推理"
    echo "=================================================="
    
    python scripts/auto_inference.py \
        --ckpt "$CKPT" \
        --output_base "$BASE_OUTPUT" \
        --sub_folder "edge" \
        --init_img "$INIT_IMG" \
        --gt_img "$GT_IMG" \
        --config "$CONFIG_PATH" \
        --vqgan_ckpt "$VQGAN_PATH" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --colorfix_type "$COLORFIX_TYPE" \
        --input_size $INPUT_SIZE \
        --use_edge_processing \
        --calculate_metrics \
        --epoch_override "$EPOCH_NUM" \
        --exp_name_override "$EXP_NAME"
    
    EDGE_SUCCESS=$?
    
    # Process NO-EDGE mode
    echo ""
    echo "=================================================="
    echo "  [2/3] NO-EDGE 模式推理"
    echo "=================================================="
    
    python scripts/auto_inference.py \
        --ckpt "$CKPT" \
        --output_base "$BASE_OUTPUT" \
        --sub_folder "no_edge" \
        --init_img "$INIT_IMG" \
        --gt_img "$GT_IMG" \
        --config "$CONFIG_PATH" \
        --vqgan_ckpt "$VQGAN_PATH" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --colorfix_type "$COLORFIX_TYPE" \
        --input_size $INPUT_SIZE \
        --use_edge_processing \
        --use_white_edge \
        --calculate_metrics \
        --epoch_override "$EPOCH_NUM" \
        --exp_name_override "$EXP_NAME"
    
    NO_EDGE_SUCCESS=$?
    
    # Process DUMMY-EDGE mode
    echo ""
    echo "=================================================="
    echo "  [3/3] DUMMY-EDGE 模式推理"
    echo "=================================================="
    DUMMY_EDGE_PATH="/stablesr_dataset/default_edge.png"
    
    python scripts/auto_inference.py \
        --ckpt "$CKPT" \
        --output_base "$BASE_OUTPUT" \
        --sub_folder "dummy_edge" \
        --init_img "$INIT_IMG" \
        --gt_img "$GT_IMG" \
        --config "$CONFIG_PATH" \
        --vqgan_ckpt "$VQGAN_PATH" \
        --ddpm_steps $DDPM_STEPS \
        --dec_w $DEC_W \
        --seed $SEED \
        --n_samples $N_SAMPLES \
        --colorfix_type "$COLORFIX_TYPE" \
        --input_size $INPUT_SIZE \
        --use_edge_processing \
        --use_dummy_edge \
        --dummy_edge_path "$DUMMY_EDGE_PATH" \
        --calculate_metrics \
        --epoch_override "$EPOCH_NUM" \
        --exp_name_override "$EXP_NAME"
    
    DUMMY_SUCCESS=$?
    
    # Show summary
    echo ""
    echo "=================================================="
    echo "  全部推理完成！"
    echo "=================================================="
    echo ""
    echo "结果统计："
    if [ $EDGE_SUCCESS -eq 0 ]; then
        echo "  ✓ EDGE 模式: 成功"
        echo "     输出: $BASE_OUTPUT/edge/epochs_$((10#$EPOCH_NUM))"
        echo "     指标: metrics.json, metrics.csv"
    else
        echo "  ✗ EDGE 模式: 失败"
    fi
    
    if [ $NO_EDGE_SUCCESS -eq 0 ]; then
        echo "  ✓ NO-EDGE 模式: 成功"
        echo "     输出: $BASE_OUTPUT/no_edge/epochs_$((10#$EPOCH_NUM))"
        echo "     指标: metrics.json, metrics.csv"
    else
        echo "  ✗ NO-EDGE 模式: 失败"
    fi
    
    if [ $DUMMY_SUCCESS -eq 0 ]; then
        echo "  ✓ DUMMY-EDGE 模式: 成功"
        echo "     输出: $BASE_OUTPUT/dummy_edge/epochs_$((10#$EPOCH_NUM))"
        echo "     指标: metrics.json, metrics.csv"
    else
        echo "  ✗ DUMMY-EDGE 模式: 失败"
    fi
    
    echo ""
    echo "所有指标（PSNR, SSIM, LPIPS, Edge PSNR, Edge Overlap）已自动计算"
    echo ""
    
    # Ask if generate comprehensive report
    read -p "是否生成综合报告? (y/n) [y]: " GEN_REPORT
    GEN_REPORT=${GEN_REPORT:-y}
    
    if [ "$GEN_REPORT" = "y" ] || [ "$GEN_REPORT" = "Y" ]; then
        echo ""
        echo "正在生成综合报告..."
        PYTHON_SCRIPT="scripts/generate_metrics_report.py"
        if [ -f "$PYTHON_SCRIPT" ]; then
            python "$PYTHON_SCRIPT" "$BASE_OUTPUT"
            
            DIR_NAME=$(basename "$BASE_OUTPUT")
            OUTPUT_REPORT="$BASE_OUTPUT/${DIR_NAME}_inference_report.csv"
            if [ -f "$OUTPUT_REPORT" ]; then
                echo "✓ 报告生成成功: $OUTPUT_REPORT"
            fi
        else
            echo "⚠ 报告生成脚本不存在: $PYTHON_SCRIPT"
        fi
    fi
    
    echo ""
}

# Function for mode 3: Generate report
generate_report() {
    echo ""
    echo "=================================================="
    echo "  模式 3: 生成推理结果报告 (CSV格式)"
    echo "=================================================="
    echo ""
    
    # Ask for results directory
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
        
        # Expand tilde and make absolute path
        RESULTS_PATH=$(eval echo "$RESULTS_PATH")
        RESULTS_PATH=$(cd "$RESULTS_PATH" 2>/dev/null && pwd || echo "$RESULTS_PATH")
        
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
        return
    fi
    
    # Generate the report using Python script
    echo "正在生成报告..."
    python "$PYTHON_SCRIPT" "$RESULTS_PATH"
    
    # Get the new filename format
    DIR_NAME=$(basename "$RESULTS_PATH")
    OUTPUT_REPORT="$RESULTS_PATH/${DIR_NAME}_inference_report.csv"
    
    # Add footer with timestamp if report exists
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
    echo "报告包含以下指标："
    echo "  - PSNR（图像质量）"
    echo "  - SSIM（结构相似度）"
    echo "  - LPIPS（感知质量）"
    echo "  - Edge PSNR（边缘质量）"
    echo "  - Edge Overlap（边缘覆盖率）"
    echo ""
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
            inference_specific_checkpoint
            ;;
        3)
            generate_report
            ;;
        0)
            echo ""
            echo "退出中..."
            exit 0
            ;;
        *)
            echo ""
            echo "无效选项，请选择 0-3。"
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
    # INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
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
