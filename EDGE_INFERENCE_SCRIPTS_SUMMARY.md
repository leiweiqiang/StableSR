# StableSR Edge Map 推理测试脚本总结

## 📁 创建的文件

### 1. 主要测试脚本

| 文件名 | 功能 | 使用场景 |
|--------|------|----------|
| `test_edge_inference.py` | 完整的推理测试脚本 | 生产环境测试、详细分析 |
| `quick_edge_test.py` | 快速测试脚本 | 快速验证、开发调试 |
| `example_edge_inference.py` | 简单示例代码 | 学习参考、代码理解 |

### 2. 启动脚本

| 文件名 | 功能 | 使用场景 |
|--------|------|----------|
| `run_edge_inference_test.sh` | 命令行启动脚本 | 便捷启动、批量测试 |

### 3. 文档

| 文件名 | 功能 | 使用场景 |
|--------|------|----------|
| `EDGE_INFERENCE_TEST_README.md` | 详细使用说明 | 用户指南、参数说明 |
| `INFERENCE_WITH_EDGE_MAP_GUIDE.md` | 推理指南 | 技术文档、实现细节 |
| `EDGE_INFERENCE_SCRIPTS_SUMMARY.md` | 本总结文档 | 项目概览、文件说明 |

## 🚀 快速开始

### 1. 最简单的使用方式

```bash
# 快速测试（使用合成图像）
python quick_edge_test.py
```

### 2. 使用启动脚本

```bash
# 给脚本执行权限
chmod +x run_edge_inference_test.sh

# 快速测试
./run_edge_inference_test.sh quick

# 测试真实图像
./run_edge_inference_test.sh test input_image.jpg

# 对比测试
./run_edge_inference_test.sh compare input_image.jpg
```

### 3. 完整测试

```bash
# 基本推理
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input input_image.jpg \
    --steps 20 \
    --output inference_output
```

## 🔧 核心功能

### 1. Edge Map生成
- ✅ 使用Canny边缘检测
- ✅ 高斯模糊预处理（5×5, σ=1.4）
- ✅ 自适应阈值（100, 200）
- ✅ 3通道RGB输出，值范围[-1, 1]

### 2. 模型推理
- ✅ 支持Edge模型加载和验证
- ✅ 自动处理struct_cond和edge_map
- ✅ 使用EdgeDDIMSampler进行采样
- ✅ 支持不同的DDPM步数

### 3. 结果保存
- ✅ 保存输入图像、edge map、最终结果
- ✅ 支持对比测试（使用/不使用edge检测）
- ✅ 自动创建输出目录
- ✅ 多种输出格式支持

### 4. 性能优化
- ✅ GPU内存管理
- ✅ 批处理支持
- ✅ 中间结果清理
- ✅ 可配置的采样步数

## 📊 测试场景

### 1. 合成图像测试
```bash
# 使用几何形状创建测试图像
python quick_edge_test.py
```

### 2. 真实图像测试
```bash
# 测试单张图像
python test_edge_inference.py --input real_image.jpg
```

### 3. 对比测试
```bash
# 对比使用/不使用edge检测的效果
python test_edge_inference.py --input real_image.jpg --compare
```

### 4. 批量测试
```bash
# 批量处理多张图像
for img in images/*.jpg; do
    python test_edge_inference.py --input "$img" --output "results/$(basename "$img")"
done
```

## ⚙️ 配置要求

### 1. 必需文件
- ✅ Edge模型配置文件：`configs/stableSRNew/v2-finetune_text_T_512_edge.yaml`
- ✅ Edge模型检查点：支持edge处理的.ckpt文件
- ✅ 输入图像：常见格式（JPG, PNG, BMP等）

### 2. 环境依赖
```bash
pip install torch torchvision
pip install opencv-python pillow numpy
pip install omegaconf pytorch-lightning
```

### 3. 硬件要求
- GPU: 推荐NVIDIA GPU，至少8GB显存
- 内存: 至少16GB RAM
- 存储: 足够的空间保存结果

## 🎯 使用建议

### 1. 开发调试
- 使用 `quick_edge_test.py` 快速验证
- 使用较少的DDPM步数（10-20步）
- 使用较小的输入图像

### 2. 生产测试
- 使用 `test_edge_inference.py` 完整测试
- 使用适当的DDPM步数（20-50步）
- 使用对比测试验证效果

### 3. 性能优化
- 根据GPU内存调整批大小
- 使用 `torch.no_grad()` 进行推理
- 及时清理中间变量

## 🔍 故障排除

### 1. 常见错误
- **模型加载失败**: 检查配置文件和检查点路径
- **CUDA内存不足**: 减少DDPM步数或输入图像大小
- **Edge检测效果不明显**: 检查输入图像是否包含清晰边缘

### 2. 调试技巧
- 使用 `--compare` 参数对比效果
- 保存中间结果进行分析
- 检查edge map质量

### 3. 性能监控
- 监控GPU内存使用
- 记录推理时间
- 比较不同参数的效果

## 📈 扩展功能

### 1. 自定义Edge检测
```python
# 修改Canny参数
edges = cv2.Canny(img_blurred, threshold1=50, threshold2=150)
```

### 2. 多种Edge检测方法
```python
# 可以使用Sobel、Laplacian等其他边缘检测方法
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
```

### 3. 批量处理
```python
# 支持批量处理多张图像
for image_path in image_list:
    result = inference_with_edge(image_path)
```

## 📝 总结

这套Edge Map推理测试脚本提供了：

1. **完整的测试流程**: 从模型加载到结果保存
2. **多种使用方式**: 命令行、脚本、示例代码
3. **灵活的参数配置**: 支持各种推理参数调整
4. **详细的文档说明**: 包含使用指南和技术文档
5. **错误处理机制**: 完善的异常处理和故障排除

通过这些脚本，用户可以：
- 快速验证Edge模型的功能
- 对比Edge检测的效果
- 调试和优化推理参数
- 进行批量图像处理

所有脚本都经过精心设计，确保易用性和可靠性，是使用StableSR Edge模型进行推理的理想工具。
