#!/bin/bash
# Training script for Edge-enhanced StableSR based on Turbo checkpoint
# 基于Turbo checkpoint训练Edge增强模型

cd /root/dp/StableSR_Edge_v2_loss

# Activate environment
source /root/miniconda/bin/activate sr_edge

# Training configuration
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml"
GPU_IDS="0,1,2,3,4,5,6,7"  # 使用的GPU ID，根据你的资源调整
BATCH_SIZE=2       # 每GPU的batch size
NUM_NODES=1        # 节点数
ACCUMULATE_GRAD=6  # 梯度累积步数

# Ask if resuming from checkpoint
echo "================================================"
echo "训练模式选择"
echo "================================================"
echo "1. 从 Turbo checkpoint 开始新训练"
echo "2. 从已有 checkpoint 恢复训练 (Resume)"
echo "================================================"
read -p "请选择 [1-2]: " TRAIN_MODE

RESUME_CHECKPOINT=""
if [ "$TRAIN_MODE" = "2" ]; then
    echo ""
    echo "可用的实验目录："
    echo ""
    
    # List available experiment directories
    if [ -d "logs" ]; then
        EXPERIMENTS=($(ls -dt logs/*/checkpoints 2>/dev/null | sed 's|/checkpoints||'))
        
        if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
            echo "❌ 没有找到可恢复的实验目录"
            exit 1
        fi
        
        for i in "${!EXPERIMENTS[@]}"; do
            EXP_DIR=$(basename "${EXPERIMENTS[$i]}")
            echo "$((i+1)). $EXP_DIR"
        done
        echo ""
        
        read -p "请选择实验编号 [1-${#EXPERIMENTS[@]}]: " EXP_CHOICE
        
        if [[ ! "$EXP_CHOICE" =~ ^[0-9]+$ ]] || [ "$EXP_CHOICE" -lt 1 ] || [ "$EXP_CHOICE" -gt ${#EXPERIMENTS[@]} ]; then
            echo "❌ 无效选择"
            exit 1
        fi
        
        SELECTED_EXP="${EXPERIMENTS[$((EXP_CHOICE-1))]}"
        CKPT_DIR="$SELECTED_EXP/checkpoints"
        
        echo ""
        echo "可用的 checkpoint："
        echo ""
        
        # List available checkpoints
        CHECKPOINTS=($(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null))
        
        if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
            echo "❌ 没有找到 checkpoint 文件"
            exit 1
        fi
        
        for i in "${!CHECKPOINTS[@]}"; do
            CKPT_NAME=$(basename "${CHECKPOINTS[$i]}")
            echo "$((i+1)). $CKPT_NAME"
        done
        echo ""
        
        read -p "请选择 checkpoint 编号 [1-${#CHECKPOINTS[@]}]: " CKPT_CHOICE
        
        if [[ ! "$CKPT_CHOICE" =~ ^[0-9]+$ ]] || [ "$CKPT_CHOICE" -lt 1 ] || [ "$CKPT_CHOICE" -gt ${#CHECKPOINTS[@]} ]; then
            echo "❌ 无效选择"
            exit 1
        fi
        
        RESUME_CHECKPOINT="${CHECKPOINTS[$((CKPT_CHOICE-1))]}"
        EXP_NAME=$(basename "$SELECTED_EXP")
        
        echo ""
        echo "✓ 将从以下 checkpoint 恢复训练:"
        echo "  $RESUME_CHECKPOINT"
        echo ""
    else
        echo "❌ logs 目录不存在"
        exit 1
    fi
else
    # New training with timestamp
    EXP_NAME="stablesr_edge_loss_$(date +%Y%m%d_%H%M%S)"
fi

echo "================================================"
echo "开始训练Edge增强模型（Loss+Edge）"
echo "================================================"
echo "实验名称: ${EXP_NAME}"
echo "配置文件: ${CONFIG}"
echo "GPU数量: $(echo ${GPU_IDS} | tr ',' '\n' | wc -l)"
echo "批次大小: ${BATCH_SIZE} per GPU"
echo "梯度累积: ${ACCUMULATE_GRAD} steps"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "恢复训练: ${RESUME_CHECKPOINT}"
fi
echo "================================================"

# 检查关键文件
echo "检查关键文件..."
if [ ! -f "${CONFIG}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG}"
    exit 1
fi

if [ ! -f "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt" ]; then
    echo "❌ Turbo checkpoint不存在"
    exit 1
fi

if [ ! -f "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt" ]; then
    echo "❌ VQGAN checkpoint不存在"
    exit 1
fi

echo "✓ 所有关键文件存在"
echo ""

# Training command
if [ -n "$RESUME_CHECKPOINT" ]; then
    # Resume from checkpoint
    python main.py \
        --base ${CONFIG} \
        --train \
        --gpus ${GPU_IDS} \
        --logdir logs \
        --name ${EXP_NAME} \
        --scale_lr False \
        --num_nodes ${NUM_NODES} \
        --check_val_every_n_epoch 1 \
        --resume_from_checkpoint ${RESUME_CHECKPOINT}
else
    # Start new training from Turbo checkpoint
    python main.py \
        --base ${CONFIG} \
        --train \
        --gpus ${GPU_IDS} \
        --logdir logs \
        --name ${EXP_NAME} \
        --scale_lr False \
        --num_nodes ${NUM_NODES} \
        --check_val_every_n_epoch 1 \
        --finetune_from /stablesr_dataset/checkpoints/stablesr_turbo.ckpt
fi

echo ""
echo "================================================"
echo "训练完成！"
echo "查看日志: logs/${EXP_NAME}"
echo "Checkpoints: logs/${EXP_NAME}/checkpoints/"
echo "================================================"

