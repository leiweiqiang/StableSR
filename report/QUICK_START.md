# Edge Map测试快速开始指南

## 🚀 快速开始

### 1. 环境检查
首先检查你的环境是否准备就绪：

```bash
python check_environment.py
```

### 2. 激活conda环境
```bash
conda activate sr_edge
```

### 3. 运行测试

#### 方法1: 使用bash脚本（推荐）
```bash
# 交互式菜单
./run_edge_test.sh

# 或直接运行快速测试
./run_edge_test.sh quick
```

#### 方法2: 使用Python调试脚本
```bash
# 快速测试
python debug_edge_test.py --test_type quick

# 综合测试
python debug_edge_test.py --test_type comprehensive
```

#### 方法3: 直接运行测试脚本
```bash
# 激活环境后直接运行
conda activate sr_edge
python test_edge_map_quick.py --synthetic --test_processor
```

## 📋 测试类型说明

### 快速测试 (`quick`)
- 使用合成图像
- 测试基本edge map生成
- 测试edge处理器（如果可用）
- 运行时间：< 1分钟

### 综合测试 (`comprehensive`)
- 包含所有测试功能
- 性能基准测试
- 可视化功能
- 运行时间：5-10分钟

### 真实图像测试 (`real`)
- 需要提供真实图像
- 多种边缘检测方法对比
- 详细分析报告
- 运行时间：取决于图像数量

### 性能测试 (`performance`)
- 详细的性能基准测试
- 内存使用分析
- 不同配置下的性能对比
- 运行时间：10-15分钟

## 🔧 常见问题

### Q: conda环境不存在怎么办？
```bash
# 创建环境
conda create -n sr_edge python=3.8

# 激活环境
conda activate sr_edge

# 安装依赖
pip install torch torchvision opencv-python matplotlib pillow numpy
```

### Q: 如何测试真实图像？
```bash
# 单张图像
./run_edge_test.sh real /path/to/your/image.jpg

# 图像目录
python debug_edge_test.py --test_type real --input_dir /path/to/images
```

### Q: 如何查看测试结果？
测试结果会保存在输出目录中，通常包括：
- `edge_maps/` - 生成的edge map图像
- `comparisons/` - 对比图
- `statistics/` - 分析报告
- `performance_results.txt` - 性能数据

### Q: 测试失败怎么办？
1. 检查环境：`python check_environment.py`
2. 查看错误信息
3. 确保conda环境已激活
4. 检查依赖包是否安装完整

## 📊 预期结果

### 成功运行的标志
- ✅ 看到 "测试完成" 或 "All tests passed" 消息
- ✅ 输出目录中有结果文件
- ✅ 没有错误信息

### 输出文件示例
```
output_directory/
├── edge_maps/
│   ├── synthetic_original.png
│   ├── synthetic_edge_canny.png
│   └── synthetic_edge_sobel.png
├── comparisons/
│   └── synthetic_comparison.png
└── statistics/
    └── summary_report.txt
```

## 🎯 下一步

测试成功后，你可以：
1. 查看生成的edge map图像
2. 分析性能报告
3. 尝试不同的参数配置
4. 测试你自己的图像数据

## 📞 获取帮助

如果遇到问题：
1. 查看 `README_edge_test.md` 获取详细说明
2. 运行 `python check_environment.py` 检查环境
3. 查看错误日志和输出信息
4. 确保按照本指南的步骤操作
