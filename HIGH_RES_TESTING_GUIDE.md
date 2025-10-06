# 高分辨率Edge模型测试指南

本指南介绍如何使用Edge模型进行高分辨率图像超分辨率测试。

## 🎯 测试更大分辨率输出的方法

### 方法1: 使用高分辨率测试脚本

```bash
# 激活环境
conda activate sr_edge

# 测试合成图像到1024分辨率（最短边）
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --target_size 1024 \
    --steps 50 \
    --output test_output_1024

# 测试到精确尺寸 1920x1080
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --target_h 1080 \
    --target_w 1920 \
    --steps 50 \
    --output test_output_2k

# 测试真实图像
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input /path/to/your/image.jpg \
    --target_size 1536 \
    --steps 50 \
    --output test_output_real
```

### 方法2: 多分辨率批量测试

```bash
# 测试多个分辨率
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --multi_res \
    --steps 50
```

这将测试以下分辨率：
- 512 (标准)
- 768 (中等高分辨率)
- 1024 (高分辨率)
- 1536 (超高分辨率)
- 1920x1080 (2K分辨率)
- 2560x1440 (2.5K分辨率)

### 方法3: 修改原始测试脚本

如果您想修改原始测试脚本，可以：

1. **修改固定目标尺寸**:
```python
# 在 test_edge_inference.py 中修改第164行
upscaled_image = self.upscale_image(input_image, target_size=1024)  # 改为1024或其他值
```

2. **添加命令行参数**:
```python
# 在 argparse 部分添加
parser.add_argument("--target_size", type=int, default=512, help="目标尺寸")
```

## 📊 分辨率对应关系

| 输入尺寸 | 目标尺寸 | 输出尺寸 | 说明 | 分类目录 |
|---------|---------|---------|------|----------|
| 510×339 | 512 | ~770×512 | 标准测试 | HD |
| 510×339 | 768 | ~1155×768 | 1.5倍放大 | HD |
| 510×339 | 1024 | ~1540×1024 | 2倍放大 | HD |
| 510×339 | 1536 | ~2310×1536 | 3倍放大 | 2K |
| 510×339 | 2048 | ~3080×2048 | 4倍放大 | 2K |
| 510×339 | 4096 | ~6160×4096 | 8倍放大 | 8K |
| 510×339 | 1920×1080 | 1920×1080 | 精确2K输出 | 2K |
| 510×339 | 2560×1440 | 2560×1440 | 精确2.5K输出 | 2K |
| 510×339 | 3840×2160 | 3840×2160 | 精确4K输出 | 4K |
| 510×339 | 7680×4320 | 7680×4320 | 精确8K输出 | 8K |

## ⚡ 性能优化建议

### 1. 采样步数调整
- **512-768分辨率**: 20-30步
- **1024-1536分辨率**: 30-50步  
- **2K+分辨率**: 50-100步

### 2. 内存管理
```bash
# 如果遇到内存不足，可以：
# 1. 减少batch_size
# 2. 使用更少的采样步数
# 3. 分块处理大图像
# 4. 使用内存优化版本
```

### 3. 质量vs速度权衡
```bash
# 快速测试（较低质量）
--steps 20

# 平衡模式
--steps 50

# 高质量模式
--steps 100
```

### 4. 内存优化版本
```bash
# 使用内存优化版本（推荐用于高分辨率测试）
python test_edge_inference_memory_optimized.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --safe_test \
    --steps 30

# 安全分辨率测试（自动停止在内存不足的分辨率）
python test_edge_inference_memory_optimized.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_image.jpg \
    --safe_test \
    --steps 30
```

### 5. 超内存优化版本（支持8K分辨率）
```bash
# 测试8K分辨率（7680×4320）
python test_edge_inference_ultra_optimized.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_image.jpg \
    --target_h 4320 \
    --target_w 7680 \
    --steps 30

# 测试8K分辨率（4096最短边）
python test_edge_inference_ultra_optimized.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_image.jpg \
    --target_size 4096 \
    --steps 30

# 超高分辨率批量测试（包含8K）
python test_edge_inference_ultra_optimized.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_image.jpg \
    --ultra_test \
    --steps 30
```

## 🔧 自定义分辨率

### 使用精确尺寸
```bash
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --target_h 1440 \
    --target_w 2560 \
    --steps 50
```

### 修改多分辨率测试列表
在脚本中修改 `resolutions` 列表：
```python
resolutions = [
    512,   # 标准尺寸
    1024,  # 高分辨率
    2048,  # 4K分辨率
    (1080, 1920),  # 2K
    (1440, 2560),  # 2.5K
    (2160, 3840),  # 4K
]
```

## 📁 输出文件结构（分目录保存）

### 目录分类规则
- **HD**: 分辨率 ≤ 1024 或 ≤ 1920×1080
- **2K**: 1024 < 分辨率 ≤ 2048 或 1920×1080 < 尺寸 ≤ 2560×1440
- **4K**: 2048 < 分辨率 ≤ 4096 或 2560×1440 < 尺寸 ≤ 3840×2160
- **8K**: 分辨率 > 4096 或 尺寸 > 3840×2160

### 目录结构
```
test_output/
├── HD/                           # HD分辨率结果
│   ├── input_images/             # 输入图像目录
│   │   ├── input_original.png
│   │   └── input_res_hd.png
│   ├── edge_maps/                # Edge map目录
│   │   └── edge_map.png
│   ├── res_512_ultra_optimized/
│   │   ├── processed_input.png
│   │   ├── edge_map.png
│   │   └── result_ultra_optimized.png
│   ├── res_768_ultra_optimized/
│   └── res_1024_ultra_optimized/
├── 2K/                           # 2K分辨率结果
│   ├── input_images/             # 输入图像目录
│   │   ├── input_original.png
│   │   └── input_res_2k.png
│   ├── edge_maps/                # Edge map目录
│   │   └── edge_map.png
│   ├── res_1536_ultra_optimized/
│   ├── res_2048_ultra_optimized/
│   ├── res_1920x1080_ultra_optimized/
│   └── res_2560x1440_ultra_optimized/
├── 4K/                           # 4K分辨率结果
│   ├── input_images/             # 输入图像目录
│   │   ├── input_original.png
│   │   └── input_res_4k.png
│   ├── edge_maps/                # Edge map目录
│   │   └── edge_map.png
│   ├── res_3840x2160_ultra_optimized/
│   └── res_4096_ultra_optimized/
└── 8K/                           # 8K分辨率结果
    ├── input_images/             # 输入图像目录
    │   ├── input_original.png
    │   └── input_res_8k.png
    ├── edge_maps/                # Edge map目录
    │   └── edge_map.png
    ├── res_4096_ultra_optimized/
    └── res_7680x4320_ultra_optimized/
```

### 文件说明
- **输入图像目录** (`input_images/`):
  - `input_original.png`: 原始输入图像
  - `input_res_[hd/2k/4k/8k].png`: 按分辨率分类的输入图像
- **Edge map目录** (`edge_maps/`):
  - `edge_map.png`: 生成的边缘图
- **结果目录** (`res_*_ultra_optimized/`):
  - `processed_input.png`: 分块处理后的输入图像
  - `edge_map.png`: Edge map副本
  - `result_ultra_optimized.png`: 最终超分辨率结果

## 🎯 实际使用示例

### 示例1: 将510×339图像超分辨率到2K
```bash
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_510x339_image.jpg \
    --target_h 1080 \
    --target_w 1920 \
    --steps 50 \
    --output test_2k_output
```

### 示例2: 批量测试不同分辨率
```bash
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --input your_image.jpg \
    --multi_res \
    --steps 30
```

## ⚠️ 注意事项

1. **内存使用**: 更高分辨率需要更多GPU内存
2. **处理时间**: 分辨率越高，处理时间越长
3. **质量权衡**: 极高分辨率可能不会显著提升视觉质量
4. **模型限制**: 模型在训练时可能没有见过极高分辨率的样本

## 🚀 快速开始

```bash
# 最简单的测试
python test_edge_inference_high_res.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --synthetic \
    --target_size 1024 \
    --steps 30
```
