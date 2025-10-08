#!/bin/bash
# Validation script for Edge-enhanced StableSR models
# 边缘增强StableSR模型验证脚本

# 使用说明：
# bash valid_edge_turbo.sh <模型路径> <验证图片路径> <输出路径>
# 例如:
# bash valid_edge_turbo.sh logs/2025-10-07T02-28-22_stablesr_edge_8_channels/checkpoints/epoch=000030.ckpt 128x128_valid_LR outputs

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "错误: 需要3个参数"
    echo "用法: bash valid_edge_turbo.sh <模型路径> <验证图片路径> <输出路径>"
    echo "示例: bash valid_edge_turbo.sh logs/model/checkpoints/epoch=000030.ckpt 128x128_valid_LR outputs"
    exit 1
fi

# 获取参数
MODEL_PATH="$1"
VAL_IMG_PATH="$2"
OUTPUT_BASE="$3"

# 切换到项目目录
cd /root/dp/StableSR_Edge_v2

# 激活环境
source /root/miniconda/bin/activate sr_edge

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查验证图片目录是否存在
if [ ! -d "$VAL_IMG_PATH" ]; then
    echo "❌ 验证图片目录不存在: $VAL_IMG_PATH"
    exit 1
fi

# 从模型路径提取模型名称
# 例如: logs/2025-10-07T02-28-22_stablesr_edge_8_channels/checkpoints/epoch=000030.ckpt
# 提取: 2025-10-07T02-28-22_stablesr_edge_8_channels_epoch=000030
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_DIR=$(dirname "$MODEL_DIR")  # 去掉 checkpoints 目录
MODEL_NAME=$(basename "$MODEL_DIR")
CHECKPOINT_NAME=$(basename "$MODEL_PATH" .ckpt)
OUTPUT_SUBDIR="${MODEL_NAME}_${CHECKPOINT_NAME}"

# 创建输出目录
OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}"
mkdir -p "$OUTPUT_DIR"

# 配置文件路径
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml"

# VQGAN checkpoint路径
VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"

# 检查VQGAN checkpoint
if [ ! -f "$VQGAN_CKPT" ]; then
    echo "⚠ VQGAN checkpoint不存在: $VQGAN_CKPT"
    echo "尝试使用备用路径..."
    VQGAN_CKPT="models/ldm/stable-diffusion-v1/epoch=000011.ckpt"
    if [ ! -f "$VQGAN_CKPT" ]; then
        echo "❌ 备用VQGAN checkpoint也不存在: $VQGAN_CKPT"
        exit 1
    fi
fi

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "⚠ 配置文件不存在: $CONFIG"
    echo "尝试使用备用配置文件..."
    CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "❌ 备用配置文件也不存在: $CONFIG"
        exit 1
    fi
fi

echo "================================================"
echo "开始验证Edge增强模型"
echo "================================================"
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "检查点名称: $CHECKPOINT_NAME"
echo "验证图片: $VAL_IMG_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG"
echo "VQGAN检查点: $VQGAN_CKPT"
echo "================================================"

# 推理参数
DDPM_STEPS=200
DEC_W=0.5
COLORFIX_TYPE="adain"
N_SAMPLES=1
SEED=42

echo ""
echo "推理参数:"
echo "  DDPM步数: $DDPM_STEPS"
echo "  解码器权重: $DEC_W"
echo "  颜色修正: $COLORFIX_TYPE"
echo "  样本数: $N_SAMPLES"
echo "  随机种子: $SEED"
echo "================================================"
echo ""

# 运行验证推理
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --use_edge_processing \
    --config "$CONFIG" \
    --ckpt "$MODEL_PATH" \
    --vqgan_ckpt "$VQGAN_CKPT" \
    --init-img "$VAL_IMG_PATH" \
    --outdir "$OUTPUT_DIR" \
    --ddpm_steps $DDPM_STEPS \
    --dec_w $DEC_W \
    --colorfix_type $COLORFIX_TYPE \
    --n_samples $N_SAMPLES \
    --seed $SEED

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ 验证完成！"
    echo "================================================"
    echo "输出目录: $OUTPUT_DIR"
    echo "模型信息:"
    echo "  - 模型名称: $MODEL_NAME"
    echo "  - 检查点: $CHECKPOINT_NAME"
    echo "  - 验证图片数: $(ls -1 "$VAL_IMG_PATH" | wc -l)"
    echo "  - 输出图片数: $(ls -1 "$OUTPUT_DIR" | wc -l)"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "❌ 验证失败！"
    echo "================================================"
    exit 1
fi
