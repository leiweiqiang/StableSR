#!/bin/bash
# Automatic Batch Training Script for Edge-enhanced StableSR
# 自动批量训练脚本 - 6种参数组合
# 
# This script will automatically train 6 different configurations:
# lr_downscale_factor: 4, 8, 16
# train_with_edge: true, false

set -e  # Exit on error

# ============================================================
# Activate Conda Environment
# ============================================================
echo "======================================================"
echo "激活 Conda 环境: sr_edge"
echo "======================================================"

# Try to find conda installation
CONDA_BASE=""
if [ -d "$HOME/miniconda" ]; then
    CONDA_BASE="$HOME/miniconda"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/root/miniconda" ]; then
    CONDA_BASE="/root/miniconda"
elif [ -d "/opt/conda" ]; then
    CONDA_BASE="/opt/conda"
elif command -v conda &> /dev/null; then
    # Try to get conda base from conda info
    CONDA_BASE=$(conda info --base 2>/dev/null)
fi

# Initialize conda for this shell
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "找到 Conda 安装: $CONDA_BASE"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate sr_edge
    ACTIVATE_STATUS=$?
elif command -v conda &> /dev/null; then
    echo "使用系统 conda 命令"
    # Try using conda shell hook
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate sr_edge 2>/dev/null
    ACTIVATE_STATUS=$?
    
    # If that didn't work, try source activate (older conda)
    if [ $ACTIVATE_STATUS -ne 0 ]; then
        source activate sr_edge 2>/dev/null
        ACTIVATE_STATUS=$?
    fi
else
    echo "❌ 未找到 conda 安装"
    echo "请确保 conda 已正确安装并添加到 PATH"
    exit 1
fi

# Verify activation
if [ $ACTIVATE_STATUS -eq 0 ] && [[ "$CONDA_DEFAULT_ENV" == "sr_edge" ]]; then
    echo "✓ Conda 环境 'sr_edge' 已激活"
    echo "当前 Python: $(which python)"
    echo "Python 版本: $(python --version 2>&1)"
else
    echo "❌ 无法激活 conda 环境 'sr_edge'"
    echo ""
    echo "请手动激活环境后再运行此脚本："
    echo "  conda activate sr_edge"
    echo "  ./batch_train_canny_in.sh"
    echo ""
    echo "或者确保环境已创建："
    echo "  conda create -n sr_edge python=3.10"
    exit 1
fi
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG_TEMPLATE="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
CONFIG_BACKUP="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml.backup"
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE=2
NUM_NODES=1
ACCUMULATE_GRAD=6

# Backup original config file
echo "======================================================"
echo "自动批量训练脚本启动"
echo "======================================================"
echo "备份原始配置文件..."
cp ${CONFIG_TEMPLATE} ${CONFIG_BACKUP}
echo "✓ 配置文件已备份到: ${CONFIG_BACKUP}"
echo ""

# Define parameter combinations
# 定义参数组合数组
LR_DOWNSCALE_FACTORS=(4 8 16)
TRAIN_WITH_EDGES=("true" "false")

CURRENT_EXPERIMENT=0

echo "======================================================"
echo "训练参数范围"
echo "======================================================"
echo "lr_downscale_factor 参数: ${LR_DOWNSCALE_FACTORS[@]}"
echo "train_with_edge 参数: ${TRAIN_WITH_EDGES[@]}"
echo "可组合成 $((${#LR_DOWNSCALE_FACTORS[@]} * ${#TRAIN_WITH_EDGES[@]})) 种训练任务"
echo "======================================================"
echo ""

# ============================================================
# Function to modify config file
# ============================================================
modify_config() {
    local lr_factor=$1
    local train_edge=$2
    
    echo "修改配置文件参数..."
    # Restore from backup
    cp ${CONFIG_BACKUP} ${CONFIG_TEMPLATE}
    
    # Modify lr_downscale_factor (line 35)
    sed -i "s/lr_downscale_factor: [0-9]*/lr_downscale_factor: ${lr_factor}/" ${CONFIG_TEMPLATE}
    
    # Modify train_with_edge (line 37)
    # Need to handle both True/False and true/false
    sed -i "s/train_with_edge: [Tt]rue/train_with_edge: ${train_edge}/" ${CONFIG_TEMPLATE}
    sed -i "s/train_with_edge: [Ff]alse/train_with_edge: ${train_edge}/" ${CONFIG_TEMPLATE}
    
    echo "✓ 配置已更新:"
    echo "  - lr_downscale_factor: ${lr_factor}"
    echo "  - train_with_edge: ${train_edge}"
    
    # Verify the changes
    echo ""
    echo "验证配置文件修改:"
    grep "lr_downscale_factor:" ${CONFIG_TEMPLATE}
    grep "train_with_edge:" ${CONFIG_TEMPLATE}
    echo ""
}

# ============================================================
# Function to run training
# ============================================================
run_training() {
    local lr_factor=$1
    local train_edge=$2
    local exp_name=$3
    
    echo "======================================================"
    echo "开始训练实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}"
    echo "======================================================"
    echo "实验名称: ${exp_name}"
    echo "参数配置:"
    echo "  - lr_downscale_factor: ${lr_factor}"
    echo "  - train_with_edge: ${train_edge}"
    echo "  - GPU: ${GPU_IDS}"
    echo "  - Batch Size: ${BATCH_SIZE} per GPU"
    echo "  - 梯度累积: ${ACCUMULATE_GRAD} steps"
    echo "======================================================"
    echo ""
    
    # Check critical files
    echo "检查关键文件..."
    if [ ! -f "${CONFIG_TEMPLATE}" ]; then
        echo "❌ 配置文件不存在: ${CONFIG_TEMPLATE}"
        exit 1
    fi
    
    if [ ! -f "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt" ]; then
        echo "❌ VQGAN checkpoint不存在"
        exit 1
    fi
    
    echo "✓ 所有关键文件存在"
    echo ""
    
    # Record start time
    START_TIME=$(date +%s)
    echo "训练开始时间: $(date)"
    echo ""
    
    # Start training (based on lines 83-93 of train_canny_in_t6.sh)
    echo "执行训练命令..."
    python main.py \
        --base ${CONFIG_TEMPLATE} \
        --train \
        --gpus ${GPU_IDS} \
        --logdir logs \
        --name ${exp_name} \
        --scale_lr False \
        --num_nodes ${NUM_NODES}
    
    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "======================================================"
    echo "实验 ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS} 完成！"
    echo "======================================================"
    echo "训练时长: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo "结果保存在: logs/${exp_name}"
    echo "Checkpoints: logs/${exp_name}/checkpoints/"
    echo "======================================================"
    echo ""
}

# ============================================================
# Generate all task combinations
# ============================================================
echo "生成任务列表..."
echo ""

# Arrays to store task configurations
declare -a TASK_LR_FACTORS
declare -a TASK_TRAIN_EDGES
declare -a TASK_NAMES
TASK_INDEX=0

# Generate all combinations
for lr_factor in "${LR_DOWNSCALE_FACTORS[@]}"; do
    for train_edge in "${TRAIN_WITH_EDGES[@]}"; do
        TASK_LR_FACTORS[$TASK_INDEX]=$lr_factor
        TASK_TRAIN_EDGES[$TASK_INDEX]=$train_edge
        TASK_NAMES[$TASK_INDEX]="stablesr_canny_in_${lr_factor}x_${train_edge}"
        TASK_INDEX=$((TASK_INDEX + 1))
    done
done

# ============================================================
# Display task list and get user selection
# ============================================================
echo "======================================================"
echo "可用训练任务列表"
echo "======================================================"
for i in "${!TASK_NAMES[@]}"; do
    task_num=$((i + 1))
    echo "  [${task_num}] lr_downscale_factor=${TASK_LR_FACTORS[$i]}, train_with_edge=${TASK_TRAIN_EDGES[$i]}"
    echo "      -> ${TASK_NAMES[$i]}_YYYYMMDD_HHMMSS"
    echo ""
done
echo "======================================================"
echo ""
echo "请选择要执行的训练任务（支持多选）："
echo "  - 输入任务编号，用空格分隔（例如: 1 3 5）"
echo "  - 输入范围（例如: 1-3）"
echo "  - 输入 'all' 或 'a' 执行全部任务"
echo "  - 输入 'q' 或 'quit' 退出"
echo ""
read -p "请输入选择: " USER_SELECTION

# Parse user selection
declare -a SELECTED_TASKS

if [[ "$USER_SELECTION" == "q" ]] || [[ "$USER_SELECTION" == "quit" ]]; then
    echo "已取消训练任务"
    # Restore original config file before exit
    if [ -f "${CONFIG_BACKUP}" ]; then
        cp ${CONFIG_BACKUP} ${CONFIG_TEMPLATE}
        echo "✓ 配置文件已恢复"
    fi
    exit 0
elif [[ "$USER_SELECTION" == "all" ]] || [[ "$USER_SELECTION" == "a" ]]; then
    # Select all tasks
    for i in "${!TASK_NAMES[@]}"; do
        SELECTED_TASKS+=($i)
    done
else
    # Parse individual numbers and ranges
    for item in $USER_SELECTION; do
        if [[ $item =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # Range format: 1-3
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            for ((j=start; j<=end; j++)); do
                idx=$((j - 1))
                if [ $idx -ge 0 ] && [ $idx -lt ${#TASK_NAMES[@]} ]; then
                    SELECTED_TASKS+=($idx)
                fi
            done
        elif [[ $item =~ ^[0-9]+$ ]]; then
            # Single number
            idx=$((item - 1))
            if [ $idx -ge 0 ] && [ $idx -lt ${#TASK_NAMES[@]} ]; then
                SELECTED_TASKS+=($idx)
            fi
        fi
    done
fi

# Remove duplicates and sort
SELECTED_TASKS=($(echo "${SELECTED_TASKS[@]}" | tr ' ' '\n' | sort -u -n | tr '\n' ' '))

# Check if any tasks selected
if [ ${#SELECTED_TASKS[@]} -eq 0 ]; then
    echo "❌ 未选择任何任务"
    # Restore original config file before exit
    if [ -f "${CONFIG_BACKUP}" ]; then
        cp ${CONFIG_BACKUP} ${CONFIG_TEMPLATE}
        echo "✓ 配置文件已恢复"
    fi
    exit 0
fi

# Display selected tasks
echo ""
echo "======================================================"
echo "已选择的训练任务（共 ${#SELECTED_TASKS[@]} 个）"
echo "======================================================"
for idx in "${SELECTED_TASKS[@]}"; do
    task_num=$((idx + 1))
    echo "  [${task_num}] lr_downscale_factor=${TASK_LR_FACTORS[$idx]}, train_with_edge=${TASK_TRAIN_EDGES[$idx]}"
done
echo "======================================================"
echo ""
read -p "确认开始训练？(y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]] && [[ "$CONFIRM" != "Y" ]]; then
    echo "已取消训练任务"
    # Restore original config file before exit
    if [ -f "${CONFIG_BACKUP}" ]; then
        cp ${CONFIG_BACKUP} ${CONFIG_TEMPLATE}
        echo "✓ 配置文件已恢复"
    fi
    exit 0
fi

# ============================================================
# Main Training Loop
# ============================================================
echo ""
echo "开始批量训练..."
echo ""

# Record overall start time
OVERALL_START_TIME=$(date +%s)

# Execute selected tasks
COMPLETED_COUNT=0
for idx in "${SELECTED_TASKS[@]}"; do
    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    CURRENT_EXPERIMENT=$((idx + 1))
    
    lr_factor=${TASK_LR_FACTORS[$idx]}
    train_edge=${TASK_TRAIN_EDGES[$idx]}
    
    # Generate experiment name with parameters and timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXP_NAME="stablesr_canny_in_${lr_factor}x_${train_edge}_${TIMESTAMP}"
    
    # Modify config file
    modify_config ${lr_factor} ${train_edge}
    
    # Run training
    run_training ${lr_factor} ${train_edge} ${EXP_NAME}
    
    # Wait a bit before next experiment
    if [ ${COMPLETED_COUNT} -lt ${#SELECTED_TASKS[@]} ]; then
        echo "等待10秒后开始下一个实验..."
        sleep 10
        echo ""
    fi
done

# Restore original config file
echo "======================================================"
echo "恢复原始配置文件..."
cp ${CONFIG_BACKUP} ${CONFIG_TEMPLATE}
echo "✓ 配置文件已恢复"
echo ""

# Calculate overall duration
OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINUTES=$(((OVERALL_DURATION % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_DURATION % 60))

# Final summary
echo "======================================================"
echo "🎉 所有批量训练完成！"
echo "======================================================"
echo "完成实验数: ${#SELECTED_TASKS[@]}"
echo "总训练时长: ${OVERALL_HOURS}小时 ${OVERALL_MINUTES}分钟 ${OVERALL_SECONDS}秒"
echo ""
echo "实验结果保存在 logs/ 目录下，格式为:"
echo "  stablesr_canny_in_{factor}x_{true/false}_YYYYMMDD_HHMMSS"
echo ""
echo "已完成的训练任务:"
for idx in "${SELECTED_TASKS[@]}"; do
    echo "  - lr_downscale_factor=${TASK_LR_FACTORS[$idx]}, train_with_edge=${TASK_TRAIN_EDGES[$idx]}"
done
echo "======================================================"

