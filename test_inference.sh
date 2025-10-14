#!/bin/bash
# Inference test script for StableSR Edge-enhanced model
# 推理测试脚本

cd ~/pd/StableSR_Edge_v3

# ========================================
# 配置参数
# ========================================

# 模型配置和权重
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
# CHECKPOINT="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"  # 使用预训练模型
CHECKPOINT="logs/stablesr_edge_loss_20250114_*/checkpoints/last.ckpt"  # 使用训练的模型（自动匹配最新）

# 输入输出路径
INPUT_DIR="inputs/test_images"         # 低分辨率输入图像目录
OUTPUT_DIR="outputs/inference_test_$(date +%Y%m%d_%H%M%S)"  # 输出目录（带时间戳）

# 采样参数
DDIM_STEPS=50          # DDIM采样步数（50=快速，200=高质量）
DDIM_ETA=1.0           # DDIM eta参数（0.0=确定性，1.0=随机性）
GUIDANCE_SCALE=1.0     # 无条件引导尺度
INPUT_SIZE=512         # 输入图像大小

# 其他参数
SEED=42                # 随机种子
N_SAMPLES=1            # 每张图生成的样本数
COLORFIX_TYPE="adain"  # 颜色修正类型: adain, wavelet, nofix
DEC_W=0.5              # VQGAN和Diffusion结合权重

# GPU设置
export CUDA_VISIBLE_DEVICES=0

# ========================================
# 函数定义
# ========================================

# 颜色输出函数
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# ========================================
# 检查环境和文件
# ========================================

print_info "=========================================="
print_info "StableSR Edge推理测试"
print_info "=========================================="

# 检查配置文件
if [ ! -f "${CONFIG}" ]; then
    print_error "配置文件不存在: ${CONFIG}"
    exit 1
fi
print_success "配置文件: ${CONFIG}"

# 检查checkpoint（支持通配符）
CKPT_FILES=(${CHECKPOINT})
if [ ! -f "${CKPT_FILES[0]}" ]; then
    print_error "Checkpoint不存在: ${CHECKPOINT}"
    print_info "提示: 请指定正确的checkpoint路径"
    exit 1
fi
CHECKPOINT="${CKPT_FILES[0]}"  # 使用第一个匹配的文件
print_success "Checkpoint: ${CHECKPOINT}"

# 检查输入目录
if [ ! -d "${INPUT_DIR}" ]; then
    print_warning "输入目录不存在: ${INPUT_DIR}"
    print_info "创建输入目录和示例目录结构..."
    mkdir -p "${INPUT_DIR}"
    print_info "请将低分辨率测试图像放入: ${INPUT_DIR}"
    print_info "支持的格式: .png, .jpg, .jpeg"
fi

# 统计输入图像数量
IMG_COUNT=$(find "${INPUT_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
if [ ${IMG_COUNT} -eq 0 ]; then
    print_error "未在 ${INPUT_DIR} 中找到图像文件"
    print_info "请添加测试图像后重试"
    exit 1
fi
print_success "找到 ${IMG_COUNT} 张输入图像"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
print_success "输出目录: ${OUTPUT_DIR}"

# ========================================
# 显示测试配置
# ========================================

echo ""
print_info "测试配置："
echo "  - 输入目录: ${INPUT_DIR}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 图像数量: ${IMG_COUNT}"
echo "  - DDIM步数: ${DDIM_STEPS}"
echo "  - 采样参数: eta=${DDIM_ETA}, scale=${GUIDANCE_SCALE}"
echo "  - 颜色修正: ${COLORFIX_TYPE}"
echo "  - 输入尺寸: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "  - 随机种子: ${SEED}"
echo ""

# ========================================
# 选择推理脚本
# ========================================

# 根据配置选择合适的推理脚本
INFERENCE_SCRIPT="scripts/sr_val_ddim_text_T_negativeprompt.py"

if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    print_error "推理脚本不存在: ${INFERENCE_SCRIPT}"
    exit 1
fi

print_info "使用推理脚本: ${INFERENCE_SCRIPT}"

# ========================================
# 运行推理
# ========================================

print_info "开始推理..."
echo ""

START_TIME=$(date +%s)

python ${INFERENCE_SCRIPT} \
    --config ${CONFIG} \
    --ckpt ${CHECKPOINT} \
    --init-img ${INPUT_DIR} \
    --outdir ${OUTPUT_DIR} \
    --ddim_steps ${DDIM_STEPS} \
    --ddim_eta ${DDIM_ETA} \
    --n_samples ${N_SAMPLES} \
    --input_size ${INPUT_SIZE} \
    --dec_w ${DEC_W} \
    --colorfix_type ${COLORFIX_TYPE} \
    --seed ${SEED} \
    --scale ${GUIDANCE_SCALE} \
    --precision autocast

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ========================================
# 显示结果
# ========================================

echo ""
print_info "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    print_success "推理完成！"
    print_info "耗时: ${ELAPSED} 秒"
    print_info "平均速度: $(echo "scale=2; ${ELAPSED}/${IMG_COUNT}" | bc) 秒/图"
    echo ""
    print_info "输出文件位置:"
    print_success "  ${OUTPUT_DIR}"
    echo ""
    
    # 统计输出图像
    OUTPUT_COUNT=$(find "${OUTPUT_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | wc -l)
    print_info "生成图像数量: ${OUTPUT_COUNT}"
    
    # 显示部分输出文件
    print_info "输出示例:"
    find "${OUTPUT_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | head -5 | while read file; do
        echo "  - $(basename ${file})"
    done
    
    # 生成对比HTML（可选）
    if command -v montage &> /dev/null; then
        print_info "正在生成预览拼图..."
        montage "${OUTPUT_DIR}"/*.png -tile 4x -geometry 256x256+2+2 "${OUTPUT_DIR}/preview.jpg" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "预览拼图: ${OUTPUT_DIR}/preview.jpg"
        fi
    fi
    
else
    print_error "推理失败！退出码: ${EXIT_CODE}"
    print_info "请检查上方错误信息"
    exit ${EXIT_CODE}
fi

print_info "=========================================="
echo ""

# ========================================
# 快速质量评估（可选）
# ========================================

if command -v python &> /dev/null && [ ${EXIT_CODE} -eq 0 ]; then
    print_info "提示: 可以使用以下命令进行质量评估:"
    echo "  python -c \"from basicsr.metrics import calculate_niqe; print('NIQE:', calculate_niqe('${OUTPUT_DIR}/xxx.png', crop_border=0))\""
fi

echo ""
print_success "测试完成！"

