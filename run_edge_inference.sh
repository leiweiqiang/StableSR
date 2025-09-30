#!/bin/bash

# StableSR Edge-enhanced Inference Script
# 使用edge_map增强的StableSR推理脚本

# 设置基本参数
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml"
CKPT="/home/tra/stablesr_dataset/ckpt/v2-1_512-ema-pruned.ckpt"
VQGAN_CKPT="/home/tra/stablesr_dataset/ckpt/vqgan_cfw_00011.ckpt"

# 输入输出路径
INIT_IMG="/home/tra/stablesr_dataset/weiql_0920/paired/LQ"
EDGE_IMG="/home/tra/stablesr_dataset/weiql_0920/paired/EdgeMap"
OUTDIR="output/results_with_edge"

# 推理参数
DDPM_STEPS=4
DEC_W=0.5
SEED=42
N_SAMPLES=1
COLORFIX_TYPE="adain"

echo "=========================================="
echo "StableSR Edge-enhanced Inference"
echo "=========================================="
echo "Config: $CONFIG"
echo "Model: $CKPT"
echo "VQGAN: $VQGAN_CKPT"
echo "Input: $INIT_IMG"
echo "Edge: $EDGE_IMG"
echo "Output: $OUTDIR"
echo "DDPM Steps: $DDPM_STEPS"
echo "=========================================="

# 检查输入目录是否存在
if [ ! -d "$INIT_IMG" ]; then
    echo "Error: Input directory $INIT_IMG does not exist!"
    exit 1
fi

# 检查edge目录是否存在
if [ ! -d "$EDGE_IMG" ]; then
    echo "Warning: Edge directory $EDGE_IMG does not exist!"
    echo "Will use zero tensors for edge maps."
    EDGE_IMG=""
fi

# 创建输出目录
mkdir -p "$OUTDIR"

# 运行推理
python scripts/sr_val_ddpm_text_T_vqganfin_with_edge.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --init-img "$INIT_IMG" \
    --edge-img "$EDGE_IMG" \
    --outdir "$OUTDIR" \
    --ddpm_steps "$DDPM_STEPS" \
    --dec_w "$DEC_W" \
    --seed "$SEED" \
    --n_samples "$N_SAMPLES" \
    --vqgan_ckpt "$VQGAN_CKPT" \
    --colorfix_type "$COLORFIX_TYPE"

echo "=========================================="
echo "Inference completed!"
echo "Results saved to: $OUTDIR"
echo "=========================================="
