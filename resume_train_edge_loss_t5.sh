#!/bin/bash
# Resume training script for Edge-enhanced StableSR
# 恢复Edge增强模型训练脚本

cd /root/dp/StableSR_Edge_v2_edge_loss

# Activate environment
source /root/miniconda/bin/activate sr_edge

# Training configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml"
GPU_IDS="0,"  # 使用的GPU ID，根据你的资源调整
NUM_NODES=1   # 节点数

# ============================================
# 重要: 设置要恢复的checkpoint路径
# ============================================
# 选项1: 指定具体的checkpoint路径
RESUME_CKPT="logs/2025-10-08T16-29-22_stablesr_edge_loss_20251008_162920/checkpoints/last.ckpt"

# 选项2: 或者指定训练目录（会自动查找最新的checkpoint）
# RESUME_DIR="logs/2025-10-08T16-29-22_stablesr_edge_loss_20251008_162920"

echo "================================================"
echo "恢复训练Edge增强模型（Loss+Edge）"
echo "================================================"
echo "配置文件: ${CONFIG}"
echo "恢复路径: ${RESUME_CKPT}"
echo "GPU数量: $(echo ${GPU_IDS} | tr ',' '\n' | wc -l)"
echo "================================================"

# 检查关键文件
echo "检查关键文件..."
if [ ! -f "${CONFIG}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG}"
    exit 1
fi

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "❌ Checkpoint不存在: ${RESUME_CKPT}"
    echo ""
    echo "可用的训练目录："
    ls -lt logs/ | head -10
    exit 1
fi

echo "✓ 所有关键文件存在"
echo ""

# Resume training command
# 注意: 使用 --resume 而不是 --finetune_from
python main.py \
    --base ${CONFIG} \
    --train \
    --gpus ${GPU_IDS} \
    --resume ${RESUME_CKPT} \
    --scale_lr False \
    --num_nodes ${NUM_NODES} \
    --check_val_every_n_epoch 1

echo ""
echo "================================================"
echo "训练恢复完成！"
echo "================================================"

