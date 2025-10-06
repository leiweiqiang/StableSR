# StableSR_ScaleLR 使用说明

## 概述

`StableSR_ScaleLR` 是一个基于 StableSR 的图像超分辨率处理类，参考了 `scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py` 的实现逻辑。该类可以将低分辨率图像进行超分辨率处理，并按照指定的目录结构保存结果。

## 功能特性

- 支持单张图像或批量图像处理
- 自动创建输出目录结构（RES、LR、HQ）
- 支持瓦片处理大尺寸图像
- 支持多种颜色修正方法（adain、wavelet、nofix）
- 可配置的采样步数和处理参数
- 支持命令行和编程接口两种使用方式

## 目录结构

处理完成后，输出目录将包含以下子目录：

```
OUT_DIR/
├── RES/          # 超分辨率处理结果
├── LR/           # 原始低分辨率图像
└── HQ/           # 高质量参考图像（可选）
```

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install omegaconf
pip install pytorch-lightning
pip install tqdm
pip install einops
pip install pillow
pip install opencv-python
pip install numpy
```

## 使用方法

### 1. 编程接口

```python
from stable_sr_scale_lr import StableSR_ScaleLR

# 创建处理器实例
processor = StableSR_ScaleLR(
    config_path="configs/stableSRNew/v2-finetune_text_T_512.yaml",
    ckpt_path="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt",
    vqgan_ckpt_path="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt",
    ddpm_steps=200,
    dec_w=0.5,
    colorfix_type="adain"
)

# 处理图像
processor.process_images(
    input_path="path/to/input/images",
    out_dir="path/to/output",
    hq_path="path/to/hq/images"  # 可选
)
```

### 2. 命令行接口

```bash
python stable_sr_scale_lr.py \
    --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
    --ckpt /stablesr_dataset/checkpoints/stablesr_turbo.ckpt \
    --vqgan_ckpt /stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt \
    --input_path path/to/input/images \
    --out_dir path/to/output \
    --hq_path path/to/hq/images \
    --ddpm_steps 200 \
    --dec_w 0.5 \
    --colorfix_type adain
```

## 参数说明

### 必需参数

- `config_path`: 配置文件路径
- `ckpt_path`: 模型检查点路径（默认：/stablesr_dataset/checkpoints/stablesr_turbo.ckpt）
- `vqgan_ckpt_path`: VQGAN模型检查点路径（默认：/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt）
- `input_path`: 输入图像路径（文件或目录）
- `out_dir`: 输出目录

### 可选参数

- `hq_path`: 高质量图像路径（可选）
- `ddpm_steps`: DDPM采样步数（默认：200）
- `dec_w`: VQGAN和Diffusion结合权重（默认：0.5）
- `colorfix_type`: 颜色修正类型（默认：adain）
  - `adain`: 自适应实例归一化
  - `wavelet`: 小波重构
  - `nofix`: 不进行颜色修正
- `input_size`: 输入尺寸（默认：512）
- `upscale`: 上采样倍数（默认：4.0）
- `tile_overlap`: 瓦片重叠大小（默认：32）
- `vqgantile_stride`: VQGAN瓦片步长（默认：1000）
- `vqgantile_size`: VQGAN瓦片大小（默认：1280）
- `seed`: 随机种子（默认：42）
- `precision`: 精度类型（默认：autocast）

## 使用示例

### 基本使用

```python
# 处理单张图像
processor.process_images(
    input_path="input.jpg",
    out_dir="output"
)

# 处理图像目录
processor.process_images(
    input_path="input_images/",
    out_dir="output"
)

# 包含高质量参考图像
processor.process_images(
    input_path="input_images/",
    out_dir="output",
    hq_path="hq_images/"
)
```

### 高级配置

```python
processor = StableSR_ScaleLR(
    config_path="configs/stableSRNew/v2-finetune_text_T_512.yaml",
    ckpt_path="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt",
    vqgan_ckpt_path="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt",
    ddpm_steps=500,  # 更多采样步数，质量更好但速度更慢
    dec_w=0.7,       # 调整VQGAN权重
    colorfix_type="wavelet",  # 使用小波颜色修正
    upscale=2.0,     # 2倍上采样
    input_size=1024  # 更大的输入尺寸
)
```

## 注意事项

1. **GPU内存**: 处理大图像时可能需要大量GPU内存，建议根据显存大小调整 `vqgantile_size` 和 `vqgantile_stride` 参数。

2. **处理时间**: 更多的 `ddpm_steps` 会提高质量但增加处理时间。

3. **图像格式**: 支持常见的图像格式（JPG、PNG等），输出格式为PNG。

4. **路径设置**: 确保所有路径都是正确的，特别是模型检查点路径。

5. **依赖检查**: 确保所有必要的依赖都已正确安装。

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小 `vqgantile_size` 或 `vqgantile_stride`
2. **模型加载失败**: 检查检查点路径和配置文件路径
3. **图像处理失败**: 检查输入图像格式和路径

### 调试模式

可以通过设置 `verbose=True` 来获取更详细的调试信息：

```python
model = load_model_from_config(config, ckpt, verbose=True)
```

## 许可证

请参考项目根目录的 LICENSE 文件。
