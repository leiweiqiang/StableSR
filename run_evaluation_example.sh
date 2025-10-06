#!/bin/bash

# TraReport评估示例脚本
# 使用真实数据运行超分辨率模型评估

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sr_edge

# 设置路径变量
GT_DIR="/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR"
VAL_DIR="/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR"  # 你需要准备低分辨率版本
MODEL_PATH="./weights/stablesr_000117.ckpt"  # 或者你的模型路径
CONFIG_PATH="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"

# 检查路径是否存在
echo "检查路径..."
if [ ! -d "$GT_DIR" ]; then
    echo "错误: GT目录不存在: $GT_DIR"
    echo "请检查DIV2K数据集路径"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请检查模型路径"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p ./evaluation_results

# 运行评估
echo "开始运行TraReport评估..."
echo "GT目录: $GT_DIR"
echo "Val目录: $VAL_DIR"
echo "模型路径: $MODEL_PATH"
echo "配置文件: $CONFIG_PATH"

python run_tra_report.py \
    --gt_dir "$GT_DIR" \
    --val_dir "$VAL_DIR" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output "./evaluation_results/tra_report_results.json" \
    --ddpm_steps 200 \
    --upscale 4.0 \
    --colorfix_type "adain" \
    --seed 42

echo "评估完成！结果保存在 ./evaluation_results/tra_report_results.json"
