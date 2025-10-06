#!/bin/bash
# Edge Map测试运行脚本
# 自动激活conda环境并运行edge map测试

set -e  # 遇到错误时退出

# 配置
CONDA_ENV="sr_edge"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Edge Map测试运行脚本"
echo "===================="
echo "项目根目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo "目标conda环境: $CONDA_ENV"

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ conda命令未找到，请确保conda已安装并添加到PATH"
    exit 1
fi

# 检查conda环境是否存在
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "❌ conda环境 '$CONDA_ENV' 不存在"
    echo "请先创建环境:"
    echo "  conda create -n $CONDA_ENV python=3.8"
    echo "  conda activate $CONDA_ENV"
    echo "  pip install torch torchvision opencv-python matplotlib pillow numpy"
    exit 1
fi

echo "✓ 找到conda环境: $CONDA_ENV"

# 函数：运行测试脚本
run_test() {
    local test_type="$1"
    local script_name="$2"
    local extra_args="$3"
    
    echo ""
    echo "运行 $test_type 测试..."
    echo "脚本: $script_name"
    echo "参数: $extra_args"
    echo "----------------------------------------"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 使用conda run在指定环境中运行脚本
    conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/$script_name" $extra_args
    
    if [ $? -eq 0 ]; then
        echo "✓ $test_type 测试完成"
    else
        echo "❌ $test_type 测试失败"
        return 1
    fi
}

# 主菜单
show_menu() {
    echo ""
    echo "请选择要运行的测试:"
    echo "1) 快速测试 (合成图像 + edge处理器)"
    echo "2) 综合测试 (所有功能)"
    echo "3) 真实图像测试"
    echo "4) 性能测试"
    echo "5) 自定义测试"
    echo "6) 运行所有测试"
    echo "0) 退出"
    echo ""
    read -p "请输入选择 (0-6): " choice
}

# 处理用户选择
handle_choice() {
    case $choice in
        1)
            run_test "快速测试" "test_edge_map_quick.py" "--synthetic --test_processor --output_dir quick_test_results"
            ;;
        2)
            run_test "综合测试" "test_edge_map_comprehensive.py" "--output_dir comprehensive_test_results"
            ;;
        3)
            read -p "请输入图像路径或目录: " input_path
            if [ -z "$input_path" ]; then
                echo "❌ 未提供输入路径"
                return 1
            fi
            
            if [ -f "$input_path" ]; then
                run_test "真实图像测试" "test_edge_map_real_images.py" "--input_image \"$input_path\" --output_dir real_test_results"
            elif [ -d "$input_path" ]; then
                run_test "真实图像测试" "test_edge_map_real_images.py" "--input_dir \"$input_path\" --output_dir real_test_results"
            else
                echo "❌ 路径不存在: $input_path"
                return 1
            fi
            ;;
        4)
            run_test "性能测试" "test_edge_map_performance.py" "--output_dir performance_test_results"
            ;;
        5)
            echo "可用的测试脚本:"
            echo "  - test_edge_map_quick.py"
            echo "  - test_edge_map_comprehensive.py"
            echo "  - test_edge_map_real_images.py"
            echo "  - test_edge_map_performance.py"
            echo ""
            read -p "请输入脚本名称: " script_name
            read -p "请输入额外参数 (可选): " extra_args
            
            if [ -f "$SCRIPT_DIR/$script_name" ]; then
                run_test "自定义测试" "$script_name" "$extra_args"
            else
                echo "❌ 脚本不存在: $script_name"
                return 1
            fi
            ;;
        6)
            echo "运行所有测试..."
            run_test "快速测试" "test_edge_map_quick.py" "--synthetic --test_processor --output_dir all_tests/quick"
            run_test "综合测试" "test_edge_map_comprehensive.py" "--output_dir all_tests/comprehensive"
            run_test "性能测试" "test_edge_map_performance.py" "--output_dir all_tests/performance"
            echo "🎉 所有测试完成!"
            ;;
        0)
            echo "退出"
            exit 0
            ;;
        *)
            echo "❌ 无效选择: $choice"
            return 1
            ;;
    esac
}

# 检查命令行参数
if [ $# -gt 0 ]; then
    # 如果有命令行参数，直接运行指定测试
    case "$1" in
        "quick")
            run_test "快速测试" "test_edge_map_quick.py" "--synthetic --test_processor --output_dir quick_test_results"
            ;;
        "comprehensive")
            run_test "综合测试" "test_edge_map_comprehensive.py" "--output_dir comprehensive_test_results"
            ;;
        "performance")
            run_test "性能测试" "test_edge_map_performance.py" "--output_dir performance_test_results"
            ;;
        "real")
            if [ -z "$2" ]; then
                echo "❌ 真实图像测试需要指定图像路径"
                echo "用法: $0 real <image_path_or_directory>"
                exit 1
            fi
            
            if [ -f "$2" ]; then
                run_test "真实图像测试" "test_edge_map_real_images.py" "--input_image \"$2\" --output_dir real_test_results"
            elif [ -d "$2" ]; then
                run_test "真实图像测试" "test_edge_map_real_images.py" "--input_dir \"$2\" --output_dir real_test_results"
            else
                echo "❌ 路径不存在: $2"
                exit 1
            fi
            ;;
        *)
            echo "用法: $0 [quick|comprehensive|performance|real <path>]"
            echo "或者不带参数运行交互式菜单"
            exit 1
            ;;
    esac
else
    # 交互式菜单
    while true; do
        show_menu
        handle_choice
        echo ""
        read -p "按回车键继续，或输入 'q' 退出: " continue_choice
        if [ "$continue_choice" = "q" ]; then
            break
        fi
    done
fi

echo "测试完成!"
