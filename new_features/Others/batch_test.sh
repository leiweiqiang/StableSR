#!/bin/bash
# Batch inference test - 批量推理测试不同配置
# 测试不同的DDIM步数和参数组合

cd ~/pd/StableSR_Edge_v3

# 配置
INPUT_DIR="${1:-inputs/test_images}"
BASE_OUTPUT="outputs/batch_test_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${2:-logs/stablesr_edge_loss_*/checkpoints/last.ckpt}"

# 查找checkpoint
CKPT_FILES=(${CHECKPOINT})
CHECKPOINT="${CKPT_FILES[0]}"

echo "=========================================="
echo "批量推理测试 - 测试不同配置"
echo "=========================================="
echo "输入目录: ${INPUT_DIR}"
echo "基础输出: ${BASE_OUTPUT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="
echo ""

if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Checkpoint不存在: ${CHECKPOINT}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# 测试配置数组
declare -a CONFIGS=(
    "steps25_eta0.0:25:0.0:快速确定性"
    "steps25_eta1.0:25:1.0:快速随机"
    "steps50_eta0.0:50:0.0:中速确定性"
    "steps50_eta1.0:50:1.0:中速随机"
    "steps100_eta0.0:100:0.0:高质量确定性"
    "steps200_eta1.0:200:1.0:最高质量"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # 解析配置
    IFS=':' read -r NAME STEPS ETA DESC <<< "$config"
    OUTPUT="${BASE_OUTPUT}/${NAME}"
    
    echo ""
    echo "=========================================="
    echo "测试 ${CURRENT}/${TOTAL}: ${DESC}"
    echo "配置: steps=${STEPS}, eta=${ETA}"
    echo "输出: ${OUTPUT}"
    echo "=========================================="
    
    mkdir -p ${OUTPUT}
    
    START=$(date +%s)
    
    python scripts/sr_val_ddim_text_T_negativeprompt.py \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
        --ckpt ${CHECKPOINT} \
        --init-img ${INPUT_DIR} \
        --outdir ${OUTPUT} \
        --ddim_steps ${STEPS} \
        --ddim_eta ${ETA} \
        --n_samples 1 \
        --input_size 512 \
        --colorfix_type adain \
        --seed 42 \
        --scale 1.0 \
        --precision autocast
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        IMG_COUNT=$(ls -1 ${OUTPUT}/*.png 2>/dev/null | wc -l)
        echo "✓ 完成 - 耗时: ${ELAPSED}秒, 生成: ${IMG_COUNT}张图像"
        echo "${NAME},${STEPS},${ETA},${ELAPSED},${IMG_COUNT},成功" >> ${BASE_OUTPUT}/results.csv
    else
        echo "❌ 失败"
        echo "${NAME},${STEPS},${ETA},${ELAPSED},0,失败" >> ${BASE_OUTPUT}/results.csv
    fi
done

echo ""
echo "=========================================="
echo "批量测试完成！"
echo "=========================================="
echo "输出目录: ${BASE_OUTPUT}"
echo ""
echo "性能对比（results.csv）:"
cat ${BASE_OUTPUT}/results.csv
echo ""
echo "可以使用以下命令查看结果:"
echo "  ls -R ${BASE_OUTPUT}/"
echo "=========================================="

