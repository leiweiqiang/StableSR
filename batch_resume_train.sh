#!/bin/bash
# Batch Resume Training Script for Edge-enhanced StableSR
# 批量恢复训练脚本
# 
# This script will:
# 1. List all resumable training runs
# 2. Allow user to confirm/modify max_steps for each training
# 3. Automatically start next training after current one completes

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
    echo "  ./batch_resume_train.sh"
    exit 1
fi
echo ""

# ============================================================
# Configuration
# ============================================================
LOGS_DIR="logs"
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_NODES=1

# ============================================================
# Function to scan for resumable trainings
# ============================================================
scan_resumable_trainings() {
    echo "======================================================"
    echo "扫描可恢复的训练任务..."
    echo "======================================================"
    
    declare -g -a TRAIN_DIRS
    declare -g -a TRAIN_NAMES
    declare -g -a TRAIN_CHECKPOINTS
    declare -g -a TRAIN_MAX_STEPS
    declare -g -a TRAIN_CONFIGS
    declare -g -a TRAIN_LIGHTNING_CONFIGS
    local idx=0
    local total_dirs=0
    local valid_dirs=0
    
    echo "正在扫描 ${LOGS_DIR} 目录..."
    echo ""
    
    # Scan logs directory
    for dir in ${LOGS_DIR}/*/; do
        if [ -d "${dir}" ]; then
            total_dirs=$((total_dirs + 1))
            dir_name=$(basename "${dir}")
            echo "检查训练目录: ${dir_name}"
            
            # Check if checkpoints directory exists
            checkpoint_dir="${dir}checkpoints"
            last_ckpt="${checkpoint_dir}/last.ckpt"
            
            if [ -d "${checkpoint_dir}" ]; then
                echo "  ✓ 找到 checkpoints 目录"
                
                # List all checkpoint files
                all_ckpts=($(find "${checkpoint_dir}" -name "*.ckpt" -type f 2>/dev/null))
                echo "  📁 发现 ${#all_ckpts[@]} 个 checkpoint 文件:"
                for ckpt in "${all_ckpts[@]}"; do
                    ckpt_name=$(basename "${ckpt}")
                    ckpt_size=$(du -h "${ckpt}" | cut -f1)
                    echo "    - ${ckpt_name} (${ckpt_size})"
                done
                
                # Check if last.ckpt exists
                if [ -f "${last_ckpt}" ]; then
                    echo "  🎯 将使用 checkpoint: last.ckpt"
                    best_ckpt="${last_ckpt}"
                else
                    # Find latest epoch checkpoint
                    latest_epoch=$(find "${checkpoint_dir}" -name "epoch*.ckpt" -type f | sort -V | tail -1)
                    if [ -n "${latest_epoch}" ]; then
                        echo "  🎯 将使用 checkpoint: $(basename "${latest_epoch}")"
                        best_ckpt="${latest_epoch}"
                    else
                        echo "  ❌ 未找到任何 checkpoint 文件"
                        continue
                    fi
                fi
                
                # Try to find config files
                lightning_config=""
                project_config=""
                echo "  🔍 查找配置文件..."
                
                for config in "${dir}configs/"*-lightning.yaml; do
                    if [ -f "${config}" ]; then
                        lightning_config="${config}"
                        echo "    ✓ 找到 lightning 配置: $(basename "${config}")"
                        break
                    fi
                done
                for config in "${dir}configs/"*-project.yaml; do
                    if [ -f "${config}" ]; then
                        project_config="${config}"
                        echo "    ✓ 找到 project 配置: $(basename "${config}")"
                        break
                    fi
                done
                
                # Get current max_steps from lightning config if available
                max_steps="unknown"
                if [ -n "${lightning_config}" ] && [ -f "${lightning_config}" ]; then
                    max_steps=$(grep -E "^\s*max_steps:" "${lightning_config}" | sed -E 's/.*max_steps:\s*([0-9]+).*/\1/')
                    if [ -z "${max_steps}" ]; then
                        max_steps="unknown"
                    else
                        echo "    📊 当前 max_steps: ${max_steps}"
                    fi
                fi
                
                TRAIN_DIRS[$idx]="${dir}"
                TRAIN_NAMES[$idx]="${dir_name}"
                TRAIN_CHECKPOINTS[$idx]="${best_ckpt}"
                TRAIN_MAX_STEPS[$idx]="${max_steps}"
                TRAIN_CONFIGS[$idx]="${project_config}"
                TRAIN_LIGHTNING_CONFIGS[$idx]="${lightning_config}"
                
                valid_dirs=$((valid_dirs + 1))
                idx=$((idx + 1))
                echo "  ✅ 训练任务可恢复"
            else
                echo "  ❌ 未找到 checkpoints 目录"
            fi
            echo ""
        fi
    done
    
    echo "======================================================"
    echo "扫描完成！"
    echo "======================================================"
    echo "总训练目录: ${total_dirs}"
    echo "可恢复训练: ${valid_dirs}"
    echo "✓ 找到 ${#TRAIN_DIRS[@]} 个可恢复的训练任务"
    echo "======================================================"
    echo ""
}

# ============================================================
# Function to display resumable trainings
# ============================================================
display_trainings() {
    echo "======================================================"
    echo "可恢复的训练任务列表"
    echo "======================================================"
    
    if [ ${#TRAIN_DIRS[@]} -eq 0 ]; then
        echo "未找到任何可恢复的训练任务"
        return 1
    fi
    
    for i in "${!TRAIN_NAMES[@]}"; do
        task_num=$((i + 1))
        echo ""
        echo "  [${task_num}] ${TRAIN_NAMES[$i]}"
        echo "      路径: ${TRAIN_DIRS[$i]}"
        echo "      当前 max_steps: ${TRAIN_MAX_STEPS[$i]}"
        echo "      将使用的 checkpoint: $(basename "${TRAIN_CHECKPOINTS[$i]}")"
    done
    
    echo ""
    echo "======================================================"
    echo ""
    return 0
}

# ============================================================
# Function to select trainings
# ============================================================
select_trainings() {
    echo "请选择要恢复的训练任务（支持多选）："
    echo "  - 输入任务编号，用空格分隔（例如: 1 3 5）"
    echo "  - 输入范围（例如: 1-3）"
    echo "  - 输入 'all' 或 'a' 执行全部任务"
    echo "  - 输入 'q' 或 'quit' 退出"
    echo ""
    read -p "请输入选择: " USER_SELECTION
    
    declare -g -a SELECTED_TASKS
    
    if [[ "$USER_SELECTION" == "q" ]] || [[ "$USER_SELECTION" == "quit" ]]; then
        echo "已取消操作"
        exit 0
    elif [[ "$USER_SELECTION" == "all" ]] || [[ "$USER_SELECTION" == "a" ]]; then
        # Select all tasks
        for i in "${!TRAIN_NAMES[@]}"; do
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
                    if [ $idx -ge 0 ] && [ $idx -lt ${#TRAIN_NAMES[@]} ]; then
                        SELECTED_TASKS+=($idx)
                    fi
                done
            elif [[ $item =~ ^[0-9]+$ ]]; then
                # Single number
                idx=$((item - 1))
                if [ $idx -ge 0 ] && [ $idx -lt ${#TRAIN_NAMES[@]} ]; then
                    SELECTED_TASKS+=($idx)
                fi
            fi
        done
    fi
    
    # Remove duplicates while preserving order
    declare -A seen
    declare -a UNIQUE_TASKS
    for task in "${SELECTED_TASKS[@]}"; do
        if [ -z "${seen[$task]}" ]; then
            seen[$task]=1
            UNIQUE_TASKS+=($task)
        fi
    done
    SELECTED_TASKS=("${UNIQUE_TASKS[@]}")
    
    # Check if any tasks selected
    if [ ${#SELECTED_TASKS[@]} -eq 0 ]; then
        echo "❌ 未选择任何任务"
        exit 0
    fi
}

# ============================================================
# Function to confirm max_steps for each training
# ============================================================
confirm_max_steps() {
    declare -g -a CONFIRMED_MAX_STEPS
    
    echo ""
    echo "======================================================"
    echo "确认每个训练的 max_steps"
    echo "======================================================"
    
    for idx in "${SELECTED_TASKS[@]}"; do
        task_num=$((idx + 1))
        current_max_steps="${TRAIN_MAX_STEPS[$idx]}"
        
        echo ""
        echo "训练任务 [${task_num}]: ${TRAIN_NAMES[$idx]}"
        echo "当前 max_steps: ${current_max_steps}"
        echo ""
        
        while true; do
            read -p "输入新的 max_steps (直接回车保持当前值): " new_max_steps
            
            # If empty, keep current value
            if [ -z "${new_max_steps}" ]; then
                if [ "${current_max_steps}" == "unknown" ]; then
                    echo "❌ 当前值未知，必须输入一个有效的 max_steps"
                    continue
                fi
                CONFIRMED_MAX_STEPS[$idx]="${current_max_steps}"
                echo "✓ 保持当前值: ${current_max_steps}"
                break
            fi
            
            # Validate input is a number
            if [[ "${new_max_steps}" =~ ^[0-9]+$ ]]; then
                CONFIRMED_MAX_STEPS[$idx]="${new_max_steps}"
                echo "✓ 设置 max_steps = ${new_max_steps}"
                break
            else
                echo "❌ 无效输入，请输入一个正整数"
            fi
        done
    done
}

# ============================================================
# Function to update lightning config with new max_steps
# ============================================================
update_max_steps() {
    local lightning_config=$1
    local new_max_steps=$2
    
    if [ -f "${lightning_config}" ]; then
        # Create backup
        cp "${lightning_config}" "${lightning_config}.backup"
        
        # Update max_steps
        sed -i "s/max_steps: [0-9]*/max_steps: ${new_max_steps}/" "${lightning_config}"
        
        echo "✓ 已更新 lightning 配置文件: ${lightning_config}"
        echo "  max_steps: ${new_max_steps}"
    else
        echo "⚠️  Lightning 配置文件不存在: ${lightning_config}"
        echo "  无法更新 max_steps"
    fi
}

# ============================================================
# Function to resume training
# ============================================================
resume_training() {
    local idx=$1
    local dir="${TRAIN_DIRS[$idx]}"
    local name="${TRAIN_NAMES[$idx]}"
    local checkpoint="${TRAIN_CHECKPOINTS[$idx]}"
    local project_config="${TRAIN_CONFIGS[$idx]}"
    local lightning_config="${TRAIN_LIGHTNING_CONFIGS[$idx]}"
    local max_steps="${CONFIRMED_MAX_STEPS[$idx]}"
    
    echo ""
    echo "======================================================"
    echo "恢复训练 [${idx}/${#SELECTED_TASKS[@]}]"
    echo "======================================================"
    echo "训练名称: ${name}"
    echo "Checkpoint: ${checkpoint}"
    echo "Project config: ${project_config}"
    echo "Lightning config: ${lightning_config}"
    echo "Max steps: ${max_steps}"
    echo "GPU: ${GPU_IDS}"
    echo "======================================================"
    echo ""
    
    # Check if project config exists
    if [ -z "${project_config}" ] || [ ! -f "${project_config}" ]; then
        echo "❌ 未找到 project 配置文件，无法恢复训练"
        echo "需要的文件: ${dir}configs/*-project.yaml"
        return 1
    fi
    
    # Check checkpoint validity
    echo "🔍 检查 checkpoint 有效性..."
    if [ -f "${checkpoint}" ] && [ -s "${checkpoint}" ] && file "${checkpoint}" | grep -q "Zip archive"; then
        if unzip -t "${checkpoint}" >/dev/null 2>&1; then
            echo "✓ Checkpoint 文件有效"
        else
            echo "❌ Checkpoint 文件损坏: ${checkpoint}"
            
            # If it's last.ckpt, try to find alternative
            if [[ "${checkpoint}" == *"last.ckpt" ]]; then
                echo ""
                echo "🔍 寻找可用的替代 checkpoint..."
                checkpoint_dir="${dir}checkpoints"
                latest_epoch=$(find "${checkpoint_dir}" -name "epoch*.ckpt" -type f | sort -V | tail -1)
                
                if [ -n "${latest_epoch}" ]; then
                    echo "找到最新的 epoch checkpoint: $(basename "${latest_epoch}")"
                    echo ""
                    echo "建议操作："
                    echo "1. 将损坏的 last.ckpt 重命名或删除"
                    echo "2. 将最新的 epoch checkpoint 复制为 last.ckpt"
                    echo ""
                    echo "执行以下命令："
                    echo "  cd ${checkpoint_dir}"
                    echo "  mv last.ckpt last.ckpt.corrupted"
                    echo "  cp $(basename "${latest_epoch}") last.ckpt"
                    echo ""
                    read -p "是否自动执行上述操作？(y/n): " AUTO_FIX
                    
                    if [[ "$AUTO_FIX" == "y" ]] || [[ "$AUTO_FIX" == "Y" ]]; then
                        echo "正在修复 checkpoint..."
                        cd "${checkpoint_dir}"
                        mv last.ckpt last.ckpt.corrupted 2>/dev/null || true
                        cp "$(basename "${latest_epoch}")" last.ckpt
                        checkpoint="${checkpoint_dir}/last.ckpt"
                        echo "✓ 已修复 checkpoint，使用: $(basename "${checkpoint}")"
                        cd - >/dev/null
                    else
                        echo "❌ 无法继续，请手动修复 checkpoint 后重新运行"
                        return 1
                    fi
                else
                    echo "❌ 未找到可用的替代 checkpoint"
                    return 1
                fi
            else
                echo "❌ 无法继续，checkpoint 文件损坏"
                return 1
            fi
        fi
    else
        echo "❌ Checkpoint 文件不存在或格式错误: ${checkpoint}"
        return 1
    fi
    
    # Update max_steps in lightning config if it exists
    if [ -n "${lightning_config}" ] && [ -f "${lightning_config}" ]; then
        update_max_steps "${lightning_config}" "${max_steps}"
    else
        echo "⚠️  未找到 lightning 配置文件，无法更新 max_steps"
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    echo "训练开始时间: $(date)"
    echo ""
    
    # Resume training
    echo "执行 resume 命令..."
    
    python main.py \
        --base "${project_config}" \
        --train \
        --gpus ${GPU_IDS} \
        --resume "${checkpoint}" \
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
    echo "训练完成！"
    echo "======================================================"
    echo "训练时长: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo "======================================================"
    echo ""
}

# ============================================================
# Main Script
# ============================================================
echo "======================================================"
echo "批量恢复训练脚本"
echo "======================================================"
echo ""

# Scan for resumable trainings
scan_resumable_trainings

# Display trainings
if ! display_trainings; then
    echo "没有可恢复的训练任务，退出"
    exit 0
fi

# Select trainings
select_trainings

# Display selected tasks
echo ""
echo "======================================================"
echo "已选择的训练任务（共 ${#SELECTED_TASKS[@]} 个）"
echo "======================================================"
for idx in "${SELECTED_TASKS[@]}"; do
    task_num=$((idx + 1))
    echo "  [${task_num}] ${TRAIN_NAMES[$idx]}"
done
echo "======================================================"
echo ""

# Confirm max_steps for each training
confirm_max_steps

# Display final confirmation
echo ""
echo "======================================================"
echo "最终确认"
echo "======================================================"
for idx in "${SELECTED_TASKS[@]}"; do
    task_num=$((idx + 1))
    echo "  [${task_num}] ${TRAIN_NAMES[$idx]}"
    echo "      max_steps: ${CONFIRMED_MAX_STEPS[$idx]}"
done
echo "======================================================"
echo ""
read -p "确认开始批量恢复训练？(y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]] && [[ "$CONFIRM" != "Y" ]]; then
    echo "已取消操作"
    exit 0
fi

# ============================================================
# Execute trainings
# ============================================================
echo ""
echo "======================================================"
echo "开始批量恢复训练..."
echo "======================================================"
echo ""

# Record overall start time
OVERALL_START_TIME=$(date +%s)

# Execute each selected task
COMPLETED_COUNT=0
for idx in "${SELECTED_TASKS[@]}"; do
    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    
    echo ""
    echo "======================================================"
    echo "任务进度: ${COMPLETED_COUNT}/${#SELECTED_TASKS[@]}"
    echo "======================================================"
    
    resume_training ${idx}
    
    # Wait a bit before next training
    if [ ${COMPLETED_COUNT} -lt ${#SELECTED_TASKS[@]} ]; then
        echo "等待10秒后开始下一个训练..."
        sleep 10
        echo ""
    fi
done

# Calculate overall duration
OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINUTES=$(((OVERALL_DURATION % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_DURATION % 60))

# Final summary
echo ""
echo "======================================================"
echo "🎉 所有批量恢复训练完成！"
echo "======================================================"
echo "完成训练数: ${#SELECTED_TASKS[@]}"
echo "总训练时长: ${OVERALL_HOURS}小时 ${OVERALL_MINUTES}分钟 ${OVERALL_SECONDS}秒"
echo ""
echo "已完成的训练任务:"
for idx in "${SELECTED_TASKS[@]}"; do
    echo "  - ${TRAIN_NAMES[$idx]} (max_steps: ${CONFIRMED_MAX_STEPS[$idx]})"
done
echo "======================================================"
echo ""

