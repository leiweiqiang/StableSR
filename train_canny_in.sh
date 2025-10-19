#!/bin/bash
# Training script for Edge-enhanced StableSR based on Turbo checkpoint
# 基于Turbo checkpoint训练Edge增强模型

# Activate environment
# conda activate sr_edge

# Training configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
GPU_IDS="0,1,2,3,4,5,6,7"  # 使用的GPU ID，根据你的资源调整
BATCH_SIZE=2       # 每GPU的batch size
NUM_NODES=1        # 节点数
ACCUMULATE_GRAD=6  # 梯度累积步数

# Resume from checkpoint (optional)
# 设置此变量以从checkpoint恢复训练
# 可以是checkpoint文件路径 (.ckpt) 或者包含checkpoints目录的日志目录
# 例如: RESUME_PATH="logs/stablesr_canny_in_20231015_123456/checkpoints/epoch-000010.ckpt"
# 或者: RESUME_PATH="logs/stablesr_canny_in_20231015_123456"
RESUME_PATH="logs/2025-10-19T11-05-15_stablesr_canny_in_20251019_110512/checkpoints/epoch=000053.ckpt"
# Experiment name with timestamp
EXP_NAME="stablesr_canny_in_$(date +%Y%m%d_%H%M%S)"

echo "======================================================"
echo "开始在T6(1x Nvidia 4090)训练Edge增强模型（Loss+Edge）"
echo "======================================================"
echo "实验名称: ${EXP_NAME}"
echo "配置文件: ${CONFIG}"
echo "GPU数量: $(echo ${GPU_IDS} | tr ',' '\n' | wc -l)"
echo "批次大小: ${BATCH_SIZE} per GPU"
echo "梯度累积: ${ACCUMULATE_GRAD} steps"
if [ -n "${RESUME_PATH}" ]; then
    echo "恢复训练: ${RESUME_PATH}"
fi
echo "======================================================"

# 检查关键文件
echo "检查关键文件..."
if [ ! -f "${CONFIG}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG}"
    exit 1
fi

if [ ! -f "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt" ]; then
    echo "❌ VQGAN checkpoint不存在"
    exit 1
fi

# 检查resume路径（如果指定）
if [ -n "${RESUME_PATH}" ]; then
    if [ ! -e "${RESUME_PATH}" ]; then
        echo "❌ Resume路径不存在: ${RESUME_PATH}"
        exit 1
    fi
    if [ -f "${RESUME_PATH}" ]; then
        echo "✓ 将从checkpoint文件恢复: ${RESUME_PATH}"
    elif [ -d "${RESUME_PATH}" ]; then
        LAST_CKPT="${RESUME_PATH}/checkpoints/last.ckpt"
        if [ -f "${LAST_CKPT}" ]; then
            echo "✓ 将从最新checkpoint恢复: ${LAST_CKPT}"
        else
            echo "❌ 在目录中未找到 checkpoints/last.ckpt: ${RESUME_PATH}"
            exit 1
        fi
    fi
fi

echo "✓ 所有关键文件存在"
echo ""

# Training command
if [ -n "${RESUME_PATH}" ]; then
    # Resume training from checkpoint
    echo "从checkpoint恢复训练..."
    python main.py \
        --base ${CONFIG} \
        --train \
        --gpus ${GPU_IDS} \
        --resume ${RESUME_PATH} \
        --scale_lr False \
        --num_nodes ${NUM_NODES}
else
    # Start new training
    echo "开始新训练..."
    python main.py \
        --base ${CONFIG} \
        --train \
        --gpus ${GPU_IDS} \
        --logdir logs \
        --name ${EXP_NAME} \
        --scale_lr False \
        --num_nodes ${NUM_NODES}
fi

echo ""
echo "================================================"
echo "训练完成！"
echo "查看日志: logs/${EXP_NAME}"
echo "Checkpoints: logs/${EXP_NAME}/checkpoints/"
echo "================================================"

