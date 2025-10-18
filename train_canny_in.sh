#!/bin/bash
# Training script for Edge-enhanced StableSR based on Turbo checkpoint
# 基于Turbo checkpoint训练Edge增强模型

cd ~/pd/StableSR_Canny

# Activate environment
# conda activate sr_edge

# Training configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_canny_in.yaml"
GPU_IDS="0,"  # 使用的GPU ID，根据你的资源调整
BATCH_SIZE=2       # 每GPU的batch size
NUM_NODES=1        # 节点数
ACCUMULATE_GRAD=6  # 梯度累积步数

# Experiment name with timestamp
EXP_NAME="stablesr_canny_in_$(date +%Y%m%d_%H%M%S)"

echo "======================================================"
echo "开始在T5(1x Nvidia 4090)训练Edge增强模型（Loss+Edge）"
echo "======================================================"
echo "实验名称: ${EXP_NAME}"
echo "配置文件: ${CONFIG}"
echo "GPU数量: $(echo ${GPU_IDS} | tr ',' '\n' | wc -l)"
echo "批次大小: ${BATCH_SIZE} per GPU"
echo "梯度累积: ${ACCUMULATE_GRAD} steps"
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

echo "✓ 所有关键文件存在"
echo ""

# Training command
python main.py \
    --base ${CONFIG} \
    --train \
    --gpus ${GPU_IDS} \
    --logdir logs \
    --name ${EXP_NAME} \
    --scale_lr False \
    --num_nodes ${NUM_NODES}

echo ""
echo "================================================"
echo "训练完成！"
echo "查看日志: logs/${EXP_NAME}"
echo "Checkpoints: logs/${EXP_NAME}/checkpoints/"
echo "================================================"

