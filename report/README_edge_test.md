# Edge Map测试脚本使用说明

本项目提供了多个用于测试edge map功能的脚本，帮助验证StableSR Edge处理功能的正确性和性能。

## 环境准备

### 激活conda环境
在运行任何测试脚本之前，请先激活conda环境：

```bash
conda activate sr_edge
```

### 环境检查
运行环境检查脚本确保所有依赖都已正确安装：

```bash
python check_environment.py
```

### 快速开始
使用提供的调试脚本，自动激活conda环境并运行测试：

```bash
# 使用Python调试脚本
python debug_edge_test.py --test_type quick

# 使用bash脚本（推荐）
./run_edge_test.sh quick
```

## 脚本概览

### 1. `test_edge_map_comprehensive.py` - 综合测试脚本
**功能最全面的测试脚本，包含所有测试功能**

**主要功能:**
- Edge map生成测试（合成图像）
- Edge处理器测试
- 特征融合测试
- 真实图像测试
- 性能基准测试
- 可视化功能

**使用方法:**
```bash
# 运行所有测试
python test_edge_map_comprehensive.py

# 运行特定测试
python test_edge_map_comprehensive.py --test_type generation
python test_edge_map_comprehensive.py --test_type processor
python test_edge_map_comprehensive.py --test_type fusion
python test_edge_map_comprehensive.py --test_type real --image_dir /path/to/images
python test_edge_map_comprehensive.py --test_type performance
python test_edge_map_comprehensive.py --test_type visualize

# 指定输出目录
python test_edge_map_comprehensive.py --output_dir my_test_results

# 指定计算设备
python test_edge_map_comprehensive.py --device cuda
```

### 2. `test_edge_map_quick.py` - 快速测试脚本
**轻量级测试脚本，用于快速验证基本功能**

**主要功能:**
- Edge map生成测试
- 多种边缘检测方法对比
- 简单的edge处理器测试
- 可视化对比

**使用方法:**
```bash
# 使用合成图像测试
python test_edge_map_quick.py --synthetic

# 使用真实图像测试
python test_edge_map_quick.py --input_image /path/to/image.jpg

# 测试edge处理器
python test_edge_map_quick.py --synthetic --test_processor

# 指定输出目录
python test_edge_map_quick.py --synthetic --output_dir quick_test_results
```

### 3. `test_edge_map_real_images.py` - 真实图像测试脚本
**专门用于测试真实图像的edge map生成和分析**

**主要功能:**
- 真实图像edge map生成
- 多种边缘检测方法（Canny、Sobel、Laplacian等）
- 详细的edge map分析
- 批量图像处理
- 统计分析报告

**使用方法:**
```bash
# 测试单张图像
python test_edge_map_real_images.py --input_image /path/to/image.jpg

# 测试图像目录
python test_edge_map_real_images.py --input_dir /path/to/images

# 限制测试图像数量
python test_edge_map_real_images.py --input_dir /path/to/images --max_images 10

# 指定边缘检测方法
python test_edge_map_real_images.py --input_image /path/to/image.jpg --methods canny sobel laplacian

# 指定输出目录
python test_edge_map_real_images.py --input_dir /path/to/images --output_dir real_test_results
```

### 4. `debug_edge_test.py` - 调试脚本
**自动激活conda环境并运行测试**

**主要功能:**
- 自动激活conda环境
- 支持多种测试类型
- 错误处理和检查
- 灵活的参数配置

**使用方法:**
```bash
# 快速测试
python debug_edge_test.py --test_type quick

# 综合测试
python debug_edge_test.py --test_type comprehensive

# 真实图像测试
python debug_edge_test.py --test_type real --input_image /path/to/image.jpg

# 性能测试
python debug_edge_test.py --test_type performance
```

### 5. `run_edge_test.sh` - Bash运行脚本
**交互式bash脚本，自动管理conda环境**

**主要功能:**
- 自动检查conda环境
- 交互式菜单
- 命令行参数支持
- 错误处理

**使用方法:**
```bash
# 交互式菜单
./run_edge_test.sh

# 直接运行指定测试
./run_edge_test.sh quick
./run_edge_test.sh comprehensive
./run_edge_test.sh performance
./run_edge_test.sh real /path/to/image.jpg
```

### 6. `check_environment.py` - 环境检查脚本
**检查运行测试所需的环境和依赖**

**主要功能:**
- Python版本检查
- conda环境检查
- 依赖包检查
- CUDA可用性检查
- 项目结构检查

**使用方法:**
```bash
python check_environment.py
```

## 边缘检测方法说明

### 1. Canny边缘检测
- **方法**: `canny`
- **特点**: 最常用的边缘检测方法，效果好，噪声抑制能力强
- **参数**: 自动计算阈值

### 2. Otsu Canny边缘检测
- **方法**: `canny_otsu`
- **特点**: 使用Otsu方法自动确定最优阈值
- **优势**: 自适应阈值，适合不同对比度的图像

### 3. Sobel边缘检测
- **方法**: `sobel`
- **特点**: 基于梯度的边缘检测
- **优势**: 计算速度快，对噪声有一定抗性

### 4. Laplacian边缘检测
- **方法**: `laplacian`
- **特点**: 基于二阶导数的边缘检测
- **优势**: 对边缘定位准确

### 5. Scharr边缘检测
- **方法**: `scharr`
- **特点**: Sobel的改进版本
- **优势**: 更好的旋转不变性

## 输出结果说明

### 文件结构
```
output_directory/
├── edge_maps/              # Edge map图像文件
│   ├── image_original.png
│   ├── image_edge_canny.png
│   ├── image_edge_sobel.png
│   └── ...
├── comparisons/            # 对比图
│   ├── image_comparison.png
│   └── ...
├── statistics/             # 统计分析
│   ├── edge_analysis_results.json
│   ├── summary_report.txt
│   └── performance_results.txt
└── edge_maps_visualization.png  # 可视化结果
```

### 分析指标
- **边缘像素比例**: 边缘像素占总像素的百分比
- **连通组件数**: 边缘的连通区域数量
- **边缘总长度**: 所有边缘轮廓的总长度
- **边缘强度**: 边缘像素的平均强度值
- **处理时间**: 各种操作的耗时统计

## 性能要求

### 硬件要求
- **GPU**: 推荐使用CUDA兼容的GPU（用于edge处理器测试）
- **内存**: 至少4GB RAM
- **存储**: 根据测试图像数量，预留足够存储空间

### 软件依赖
```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install pillow
pip install numpy
```

## 常见问题

### 1. 模块导入错误
```
ImportError: No module named 'ldm.modules.diffusionmodules.edge_processor'
```
**解决方案**: 确保在项目根目录运行脚本，或检查edge处理模块是否正确安装。

### 2. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 减少batch size
- 使用CPU模式: `--device cpu`
- 减少测试图像数量

### 3. 图像读取失败
```
cv2.error: OpenCV(4.x.x) /path/to/file: error: (-215:Assertion failed)
```
**解决方案**: 
- 检查图像文件是否存在
- 确认图像格式是否支持
- 检查文件权限

### 4. 输出目录权限问题
```
PermissionError: [Errno 13] Permission denied
```
**解决方案**: 
- 检查输出目录的写入权限
- 使用其他输出目录
- 以管理员权限运行

## 测试建议

### 1. 开发阶段测试
```bash
# 快速验证基本功能
python test_edge_map_quick.py --synthetic --test_processor
```

### 2. 功能验证测试
```bash
# 测试真实图像处理
python test_edge_map_real_images.py --input_image test_image.jpg
```

### 3. 性能基准测试
```bash
# 全面性能测试
python test_edge_map_comprehensive.py --test_type performance
```

### 4. 批量图像测试
```bash
# 测试大量图像
python test_edge_map_real_images.py --input_dir /path/to/dataset --max_images 100
```

## 结果解读

### 1. Edge Map质量评估
- **边缘连续性**: 好的edge map应该有连续的边缘线条
- **噪声抑制**: 应该有效抑制图像噪声
- **细节保留**: 应该保留重要的图像细节

### 2. 性能指标
- **处理速度**: 不同方法的处理时间对比
- **内存使用**: GPU/CPU内存占用情况
- **准确性**: 边缘检测的准确性评估

### 3. 统计分析
- **边缘密度**: 不同图像的边缘密度分布
- **方法对比**: 各种边缘检测方法的效果对比
- **适用场景**: 不同方法适用的图像类型

## 扩展使用

### 1. 自定义边缘检测方法
可以在脚本中添加新的边缘检测方法，参考现有方法的实现。

### 2. 批量处理优化
对于大量图像，可以修改脚本使用多进程或GPU并行处理。

### 3. 结果后处理
可以基于测试结果进行进一步的分析和可视化。

## 联系支持

如果遇到问题或有改进建议，请：
1. 检查本文档的常见问题部分
2. 查看脚本的错误输出信息
3. 确认环境和依赖是否正确安装
4. 提供详细的错误信息和复现步骤
