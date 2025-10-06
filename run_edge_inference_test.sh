#!/bin/bash
# StableSR Edge Map 推理测试启动脚本

set -e  # 遇到错误时退出

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "StableSR Edge Map 推理测试"
echo "=========================="
echo "项目根目录: $PROJECT_ROOT"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请确保Python已安装"
    exit 1
fi

# 检查必要的文件
CONFIG_FILE="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "请确保配置文件路径正确"
    exit 1
fi

echo "✓ 找到配置文件: $CONFIG_FILE"

# 默认模型路径（需要用户修改）
DEFAULT_CKPT="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"

# 函数：显示帮助信息
show_help() {
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  quick                    - 运行快速测试（使用合成图像）"
    echo "  test <image_path>        - 测试指定图像"
    echo "  compare <image_path>     - 对比测试（使用/不使用edge检测）"
    echo "  --ckpt <path>            - 指定模型检查点路径"
    echo "  --steps <number>         - 指定DDPM采样步数（默认20）"
    echo "  --output <dir>           - 指定输出目录（默认inference_output）"
    echo "  --help                   - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 quick                                    # 快速测试"
    echo "  $0 test input.jpg                          # 测试单张图像"
    echo "  $0 compare input.jpg                       # 对比测试"
    echo "  $0 test input.jpg --ckpt /path/to/model.ckpt  # 指定模型路径"
    echo "  $0 test input.jpg --steps 30               # 指定采样步数"
    echo ""
}

# 函数：运行快速测试
run_quick_test() {
    echo ""
    echo "运行快速测试..."
    echo "=================="
    
    cd "$PROJECT_ROOT"
    python quick_edge_test.py
    
    if [ $? -eq 0 ]; then
        echo "✓ 快速测试完成"
        echo "结果保存在: quick_test_output/"
    else
        echo "❌ 快速测试失败"
        return 1
    fi
}

# 函数：测试单张图像
run_image_test() {
    local image_path="$1"
    local ckpt_path="$2"
    local steps="$3"
    local output_dir="$4"
    
    if [ ! -f "$image_path" ]; then
        echo "❌ 图像文件不存在: $image_path"
        return 1
    fi
    
    echo ""
    echo "测试图像: $image_path"
    echo "模型检查点: $ckpt_path"
    echo "DDPM步数: $steps"
    echo "输出目录: $output_dir"
    echo "=================="
    
    cd "$PROJECT_ROOT"
    python test_edge_inference.py \
        --config "$CONFIG_FILE" \
        --ckpt "$ckpt_path" \
        --input "$image_path" \
        --steps "$steps" \
        --output "$output_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ 图像测试完成"
        echo "结果保存在: $output_dir/"
    else
        echo "❌ 图像测试失败"
        return 1
    fi
}

# 函数：运行对比测试
run_compare_test() {
    local image_path="$1"
    local ckpt_path="$2"
    local steps="$3"
    local output_dir="$4"
    
    if [ ! -f "$image_path" ]; then
        echo "❌ 图像文件不存在: $image_path"
        return 1
    fi
    
    echo ""
    echo "对比测试: $image_path"
    echo "模型检查点: $ckpt_path"
    echo "DDPM步数: $steps"
    echo "输出目录: $output_dir"
    echo "=================="
    
    cd "$PROJECT_ROOT"
    python test_edge_inference.py \
        --config "$CONFIG_FILE" \
        --ckpt "$ckpt_path" \
        --input "$image_path" \
        --compare \
        --steps "$steps" \
        --output "$output_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ 对比测试完成"
        echo "结果保存在: $output_dir/"
    else
        echo "❌ 对比测试失败"
        return 1
    fi
}

# 解析命令行参数
CKPT_PATH="$DEFAULT_CKPT"
STEPS=20
OUTPUT_DIR="inference_output"
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        quick)
            COMMAND="quick"
            shift
            ;;
        test)
            COMMAND="test"
            IMAGE_PATH="$2"
            shift 2
            ;;
        compare)
            COMMAND="compare"
            IMAGE_PATH="$2"
            shift 2
            ;;
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查模型检查点
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 模型检查点不存在: $CKPT_PATH"
    echo "请使用 --ckpt 参数指定正确的模型路径"
    echo "或修改脚本中的 DEFAULT_CKPT 变量"
    exit 1
fi

echo "✓ 找到模型检查点: $CKPT_PATH"

# 执行命令
case $COMMAND in
    quick)
        run_quick_test
        ;;
    test)
        if [ -z "$IMAGE_PATH" ]; then
            echo "❌ 请指定图像路径"
            show_help
            exit 1
        fi
        run_image_test "$IMAGE_PATH" "$CKPT_PATH" "$STEPS" "$OUTPUT_DIR"
        ;;
    compare)
        if [ -z "$IMAGE_PATH" ]; then
            echo "❌ 请指定图像路径"
            show_help
            exit 1
        fi
        run_compare_test "$IMAGE_PATH" "$CKPT_PATH" "$STEPS" "$OUTPUT_DIR"
        ;;
    "")
        echo "❌ 请指定要执行的命令"
        show_help
        exit 1
        ;;
    *)
        echo "❌ 未知命令: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo ""
echo "🎉 测试完成！"
