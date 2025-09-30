# StableSR Edge-enhanced Inference Guide

## 概述

本指南介绍如何在StableSR推理时使用edge_map数据来增强超分辨率效果。Edge_map提供了额外的边缘信息，可以帮助模型生成更清晰、更准确的超分辨率图像。

## 文件结构

```
StableSR/
├── scripts/
│   ├── sr_val_ddpm_text_T_vqganfin_old.py          # 原始推理脚本
│   └── sr_val_ddpm_text_T_vqganfin_with_edge.py    # 支持edge_map的推理脚本
├── configs/stableSRNew/
│   └── v2-finetune_text_T_512_with_edge.yaml       # 支持edge的配置文件
├── ldm/
│   ├── models/diffusion/ddpm_with_edge.py           # 支持edge的模型
│   └── modules/edge_processor.py                    # Edge处理器
├── run_edge_inference.sh                            # 推理脚本
└── EDGE_INFERENCE_README.md                         # 本说明文档
```

## 使用方法

### 1. 基本使用

```bash
# 使用edge_map进行推理
python scripts/sr_val_ddpm_text_T_vqganfin_with_edge.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml \
    --ckpt /path/to/your/edge_trained_model.ckpt \
    --init-img /path/to/input/images \
    --edge-img /path/to/edge/maps \
    --outdir /path/to/output \
    --ddpm_steps 4 \
    --dec_w 0.5 \
    --seed 42 \
    --n_samples 1 \
    --vqgan_ckpt /path/to/vqgan.ckpt \
    --colorfix_type adain
```

### 2. 使用便捷脚本

```bash
# 修改run_edge_inference.sh中的路径，然后运行
./run_edge_inference.sh
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `v2-finetune_text_T_512_with_edge.yaml` |
| `--ckpt` | 模型权重路径 | 需要指定 |
| `--init-img` | 输入图像目录 | 需要指定 |
| `--edge-img` | Edge map目录 | 可选，不提供时使用零张量 |
| `--outdir` | 输出目录 | `outputs/user_upload` |
| `--ddpm_steps` | DDPM采样步数 | `4` |
| `--dec_w` | VQGAN和Diffusion权重 | `0.5` |
| `--seed` | 随机种子 | `42` |
| `--n_samples` | 批次大小 | `1` |
| `--vqgan_ckpt` | VQGAN模型路径 | 需要指定 |
| `--colorfix_type` | 颜色校正类型 | `adain` |

## 目录结构要求

### 输入目录结构
```
input_images/
├── image1.png
├── image2.jpg
└── image3.png

edge_maps/
├── image1.png    # 对应的edge map
├── image2.png    # 对应的edge map
└── image3.png    # 对应的edge map
```

### Edge Map要求
- **格式**: 灰度图像 (推荐PNG格式)
- **尺寸**: 512x512像素 (会自动调整)
- **命名**: 与输入图像同名
- **内容**: 边缘检测结果，白色表示边缘，黑色表示非边缘

## Edge处理流程

1. **Edge图像加载**: 512x512x1 灰度图像
2. **Edge处理**: EdgeProcessor → 64x64x4 特征张量
3. **特征融合**: 与diffusion latent融合 (64x64x4 + 64x64x4 = 64x64x8)
4. **超分辨率**: 使用融合特征进行super-resolution

## 模型要求

### 配置文件
必须使用支持edge的配置文件：
- `configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml`

### 模型权重
需要使用训练时支持edge_map的模型权重，包含：
- `edge_processor` 权重
- `edge_fusion` 权重
- 支持8通道输入的U-Net权重

## 兼容性

### 向后兼容
- 如果不提供edge_map目录，脚本会自动使用零张量
- 可以使用标准配置文件，但会失去edge增强效果

### 模型检测
脚本会自动检测模型是否支持edge处理：
- 支持edge: 使用edge增强流程
- 不支持edge: 使用标准流程

## 示例

### 完整示例
```bash
python scripts/sr_val_ddpm_text_T_vqganfin_with_edge.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml \
    --ckpt /home/tra/stablesr_dataset/ckpt/v2-1_512-ema-pruned.ckpt \
    --init-img /home/tra/stablesr_dataset/weiql_0920/paired/LQ \
    --edge-img /home/tra/stablesr_dataset/weiql_0920/paired/EdgeMap \
    --outdir output/results_with_edge \
    --ddpm_steps 4 \
    --dec_w 0.5 \
    --seed 42 \
    --n_samples 1 \
    --vqgan_ckpt /home/tra/stablesr_dataset/ckpt/vqgan_cfw_00011.ckpt \
    --colorfix_type adain
```

### 无Edge Map示例
```bash
python scripts/sr_val_ddpm_text_T_vqganfin_with_edge.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml \
    --ckpt /home/tra/stablesr_dataset/ckpt/v2-1_512-ema-pruned.ckpt \
    --init-img /home/tra/stablesr_dataset/weiql_0920/paired/LQ \
    --outdir output/results_without_edge \
    --ddpm_steps 4 \
    --dec_w 0.5 \
    --seed 42 \
    --n_samples 1 \
    --vqgan_ckpt /home/tra/stablesr_dataset/ckpt/vqgan_cfw_00011.ckpt \
    --colorfix_type adain
```

## 故障排除

### 常见问题

1. **配置文件错误**
   ```
   WARNING: 建议使用带有edge支持的配置文件
   ```
   - 解决：使用 `v2-finetune_text_T_512_with_edge.yaml`

2. **Edge map未找到**
   ```
   Edge map not found at /path/to/edge.png, using zero tensor
   ```
   - 解决：检查edge map路径和文件名

3. **模型不支持edge**
   ```
   Using standard model (no edge support)
   ```
   - 解决：使用支持edge的模型权重

4. **内存不足**
   - 解决：减少 `--n_samples` 参数

### 性能优化

1. **减少采样步数**: 使用较少的 `--ddpm_steps`
2. **调整批次大小**: 根据GPU内存调整 `--n_samples`
3. **使用混合精度**: 默认使用 `autocast`

## 技术细节

### Edge Processor架构
```
输入: 512x512x1 edge图像
├── First Stage: 3x3 conv layers (1→128→256→256)
└── Second Stage: 4x4 conv layers with stride=2 (256→128→64→4)
输出: 64x64x4 特征张量
```

### 特征融合
```
SD Latent: 64x64x4
Edge Features: 64x64x4
├── Concatenation: 64x64x8
└── Fusion Conv: 3x3 conv + BatchNorm + ReLU
输出: 64x64x8 融合特征
```

## 更新日志

- **v1.0**: 初始版本，支持edge_map推理
- 支持自动检测模型edge能力
- 支持向后兼容
- 提供便捷脚本和详细文档
