#!/bin/bash
# Test script to verify lr_input resolution fix

echo "======================================================"
echo "测试 lr_input 分辨率修复"
echo "======================================================"
echo ""

# 检查是否有测试图像
TEST_LR_DIR="/mnt/nas_dp/test_dataset/128x128_valid_LR"
TEST_GT_DIR="/mnt/nas_dp/test_dataset/512x512_valid_HR"

if [ ! -d "$TEST_LR_DIR" ]; then
    echo "❌ 错误: 测试LR目录不存在: $TEST_LR_DIR"
    exit 1
fi

if [ ! -d "$TEST_GT_DIR" ]; then
    echo "❌ 错误: 测试GT目录不存在: $TEST_GT_DIR"
    exit 1
fi

# 获取第一个测试图像
FIRST_LR_IMG=$(ls "$TEST_LR_DIR" | head -n 1)
if [ -z "$FIRST_LR_IMG" ]; then
    echo "❌ 错误: LR目录中没有图像"
    exit 1
fi

echo "测试图像: $FIRST_LR_IMG"

# 获取原始LR图像的分辨率
LR_RESOLUTION=$(identify -format "%wx%h" "$TEST_LR_DIR/$FIRST_LR_IMG" 2>/dev/null)
if [ -z "$LR_RESOLUTION" ]; then
    # 使用Python获取分辨率
    LR_RESOLUTION=$(python3 -c "from PIL import Image; img=Image.open('$TEST_LR_DIR/$FIRST_LR_IMG'); print(f'{img.width}x{img.height}')")
fi

echo "原始LR分辨率: $LR_RESOLUTION"
echo ""

# 查找最新的checkpoint
LATEST_CKPT=$(find logs -name "*.ckpt" -type f | grep -E "epoch=|last.ckpt" | sort -r | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ 错误: 未找到checkpoint文件"
    exit 1
fi

echo "使用checkpoint: $LATEST_CKPT"
echo ""

# 创建临时输出目录
TEST_OUTPUT="test_lr_input_fix_$(date +%Y%m%d_%H%M%S)"
echo "输出目录: $TEST_OUTPUT"
echo ""

# 运行推理（只处理1张图像）
echo "运行推理测试..."
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
    --ckpt "$LATEST_CKPT" \
    --init-img "$TEST_LR_DIR" \
    --gt-img "$TEST_GT_DIR" \
    --outdir "$TEST_OUTPUT" \
    --ddpm_steps 25 \
    --use_edge_processing \
    --vqgan_ckpt /stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt \
    --max_images 1

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 推理失败"
    exit 1
fi

echo ""
echo "======================================================"
echo "验证 lr_input 分辨率"
echo "======================================================"
echo ""

# 检查lr_input目录
if [ ! -d "$TEST_OUTPUT/lr_input" ]; then
    echo "❌ 错误: lr_input目录不存在"
    exit 1
fi

# 获取保存的lr_input图像
SAVED_LR=$(ls "$TEST_OUTPUT/lr_input" | head -n 1)
if [ -z "$SAVED_LR" ]; then
    echo "❌ 错误: lr_input目录中没有图像"
    exit 1
fi

# 获取保存的LR图像分辨率
SAVED_LR_RESOLUTION=$(python3 -c "from PIL import Image; img=Image.open('$TEST_OUTPUT/lr_input/$SAVED_LR'); print(f'{img.width}x{img.height}')")

echo "原始LR分辨率:   $LR_RESOLUTION"
echo "保存的LR分辨率: $SAVED_LR_RESOLUTION"
echo ""

if [ "$LR_RESOLUTION" = "$SAVED_LR_RESOLUTION" ]; then
    echo "✅ 成功! lr_input 分辨率正确"
    echo ""
    echo "测试通过!"
    exit 0
else
    echo "❌ 失败! lr_input 分辨率不匹配"
    echo ""
    echo "预期: $LR_RESOLUTION"
    echo "实际: $SAVED_LR_RESOLUTION"
    exit 1
fi







