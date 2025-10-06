# StableSR Edge Map 推理测试脚本使用说明

## 概述

本项目提供了两个用于测试StableSR Edge模型推理功能的脚本：

1. **`test_edge_inference.py`** - 完整的推理测试脚本
2. **`quick_edge_test.py`** - 快速测试脚本

## 脚本功能

### 1. 完整测试脚本 (`test_edge_inference.py`)

#### 主要功能
- ✅ 加载Edge模型并验证edge处理支持
- ✅ 从输入图像生成edge map
- ✅ 执行完整的推理流程
- ✅ 对比使用/不使用edge检测的效果
- ✅ 支持合成图像测试
- ✅ 保存中间结果和最终输出

#### 使用方法

```bash
# 基本推理测试
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input input_image.jpg \
    --steps 20 \
    --output inference_output

# 对比测试（使用/不使用edge检测）
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input input_image.jpg \
    --compare \
    --output comparison_output

# 使用合成图像测试
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --synthetic \
    --output synthetic_test

# 禁用edge检测
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input input_image.jpg \
    --no-edge \
    --output no_edge_test
```

#### 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--config` | 配置文件路径 | ✅ |
| `--ckpt` | 模型检查点路径 | ✅ |
| `--input` | 输入图像路径 | ❌ |
| `--output` | 输出目录 | ❌ |
| `--steps` | DDPM采样步数 | ❌ |
| `--caption` | 文本描述 | ❌ |
| `--seed` | 随机种子 | ❌ |
| `--compare` | 对比测试 | ❌ |
| `--synthetic` | 使用合成图像 | ❌ |
| `--no-edge` | 禁用edge检测 | ❌ |

### 2. 快速测试脚本 (`quick_edge_test.py`)

#### 主要功能
- ✅ 简化的edge模型推理测试
- ✅ 自动创建合成测试图像
- ✅ 生成edge map并保存
- ✅ 执行推理并保存结果

#### 使用方法

```bash
# 直接运行快速测试
python quick_edge_test.py
```

#### 配置修改

在运行前，需要修改脚本中的路径：

```python
# 修改配置文件路径
config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"

# 修改模型检查点路径
ckpt_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
```

## 输出结果

### 完整测试脚本输出

```
inference_output/
├── input.png              # 输入图像
├── upscaled_input.png     # 上采样后的输入
├── edge_map.png          # 生成的edge map
└── result_edge_20steps.png  # 最终结果

comparison_output/
├── input.png
├── without_edge/         # 不使用edge的结果
│   ├── upscaled_input.png
│   └── result_edge_20steps.png
├── with_edge/            # 使用edge的结果
│   ├── upscaled_input.png
│   ├── edge_map.png
│   └── result_edge_20steps.png
├── result_without_edge.png
└── result_with_edge.png
```

### 快速测试脚本输出

```
quick_test_output/
├── input.png      # 输入图像
├── edge_map.png   # edge map
└── result.png     # 超分辨率结果
```

## 环境要求

### 依赖包
```bash
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install numpy
pip install omegaconf
pip install pytorch-lightning
```

### 硬件要求
- GPU: 推荐NVIDIA GPU，至少8GB显存
- CPU: 支持CUDA的CPU
- 内存: 至少16GB RAM

## 常见问题

### 1. 模型加载失败

**问题**: `模型加载失败: FileNotFoundError`

**解决**: 
- 检查配置文件路径是否正确
- 检查模型检查点路径是否存在
- 确保模型是支持edge处理的版本

### 2. CUDA内存不足

**问题**: `CUDA out of memory`

**解决**:
- 减少DDPM步数: `--steps 10`
- 使用更小的输入图像
- 减少批处理大小

### 3. Edge检测效果不明显

**问题**: Edge检测效果不明显

**解决**:
- 检查输入图像是否包含清晰的边缘
- 调整Canny参数（在代码中修改threshold1和threshold2）
- 使用对比测试查看差异

### 4. 推理速度慢

**问题**: 推理速度过慢

**解决**:
- 减少DDPM步数: `--steps 10`
- 使用更小的输入图像
- 确保使用GPU推理

## 性能优化建议

### 1. 推理速度优化
```bash
# 使用较少的DDPM步数
--steps 10

# 使用较小的输入图像
# 在代码中修改target_size参数
```

### 2. 内存优化
```python
# 使用torch.no_grad()进行推理
with torch.no_grad():
    samples, _ = sampler.sample(...)

# 及时清理中间变量
del intermediate_tensors
torch.cuda.empty_cache()
```

### 3. 质量优化
```bash
# 使用更多的DDPM步数
--steps 50

# 使用高质量输入图像
# 确保输入图像分辨率足够高
```

## 示例用法

### 测试真实图像
```bash
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input examples/test_image.jpg \
    --steps 20 \
    --output test_results
```

### 对比测试
```bash
python test_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/edge_model.ckpt \
    --input examples/test_image.jpg \
    --compare \
    --steps 20 \
    --output comparison_results
```

### 批量测试
```bash
# 创建批量测试脚本
for img in examples/*.jpg; do
    python test_edge_inference.py \
        --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
        --ckpt /path/to/edge_model.ckpt \
        --input "$img" \
        --steps 20 \
        --output "results/$(basename "$img" .jpg)"
done
```

## 注意事项

1. **模型兼容性**: 确保使用的模型检查点支持edge处理
2. **配置文件**: 必须使用edge版本的配置文件
3. **输入格式**: 支持常见的图像格式（JPG, PNG, BMP等）
4. **输出目录**: 脚本会自动创建输出目录
5. **内存管理**: 大图像可能需要更多GPU内存

## 技术支持

如果遇到问题，请检查：

1. 模型和配置文件是否正确
2. 输入图像格式是否支持
3. GPU内存是否充足
4. 依赖包是否完整安装

更多详细信息请参考项目文档或联系开发团队。
