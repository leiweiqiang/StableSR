#!/bin/bash
# EdgeMapGenerator 测试脚本

echo "运行 EdgeMapGenerator 测试..."
echo "================================"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 初始化conda（如果需要）
if [ -f "/root/miniconda/etc/profile.d/conda.sh" ]; then
    source /root/miniconda/etc/profile.d/conda.sh
fi

# 激活环境并运行测试
conda activate sr_edge && python "${SCRIPT_DIR}/test_edge_generator.py"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✓ 测试完成！"
    echo "查看生成的edge map: ${SCRIPT_DIR}/test_edge_output.png"
    echo "查看文档: ${SCRIPT_DIR}/README.md"
else
    echo ""
    echo "================================"
    echo "✗ 测试失败，退出码: $exit_code"
fi

exit $exit_code

