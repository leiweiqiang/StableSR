#!/bin/bash
# Quick inference test - 快速推理测试
# 使用最少的步数快速验证模型是否正常工作

cd ~/pd/StableSR_Edge_v3

# 简单配置
INPUT="${1:-inputs/test_images}"  # 第一个参数作为输入，或使用默认值
OUTPUT="outputs/quick_test_$(date +%Y%m%d_%H%M)"
CHECKPOINT="${2:-logs/stablesr_edge_loss_*/checkpoints/last.ckpt}"  # 第二个参数作为checkpoint

# 查找最新的checkpoint
CKPT_FILES=(${CHECKPOINT})
CHECKPOINT="${CKPT_FILES[0]}"

echo "=========================================="
echo "快速推理测试"
echo "=========================================="
echo "输入: ${INPUT}"
echo "输出: ${OUTPUT}"
echo "模型: ${CHECKPOINT}"
echo "=========================================="
echo ""

# 检查
if [ ! -d "${INPUT}" ]; then
    echo "❌ 输入目录不存在: ${INPUT}"
    echo "用法: ./quick_test.sh <input_dir> [checkpoint]"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Checkpoint不存在: ${CHECKPOINT}"
    exit 1
fi

mkdir -p ${OUTPUT}

# 快速推理（使用较少的DDIM步数）
export CUDA_VISIBLE_DEVICES=0

python scripts/sr_val_ddim_text_T_negativeprompt.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
    --ckpt ${CHECKPOINT} \
    --init-img ${INPUT} \
    --outdir ${OUTPUT} \
    --ddim_steps 25 \
    --ddim_eta 1.0 \
    --n_samples 1 \
    --input_size 512 \
    --colorfix_type adain \
    --seed 42 \
    --scale 1.0 \
    --precision autocast

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 测试完成！"
    echo "输出目录: ${OUTPUT}"
    echo "=========================================="
    ls -lh ${OUTPUT}/*.png 2>/dev/null | head -5
else
    echo ""
    echo "❌ 测试失败"
    exit 1
fi

