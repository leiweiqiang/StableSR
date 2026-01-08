# 批量推理脚本使用说明

## 概述

`batch_inference_blended_tile.sh` 脚本用于批量处理4x和8x图片的混合输入推理，支持跳过已处理的图片。

## 文件说明

- `batch_inference_blended_tile.sh` - 主批量推理脚本
- `test_batch_inference.sh` - 配置测试脚本
- `BATCH_INFERENCE_README.md` - 本说明文档

## 配置参数

### 输入目录
- **4x LR图片**: `/stablesr_dataset/weiqiang/lr_x4_960x540`
- **8x LR图片**: `/stablesr_dataset/weiqiang/lr_x8_480x270`
- **边缘图片**: `/stablesr_dataset/weiqiang/hr_4k_canny`

### 模型检查点
- **4x模型**: `/root/dp/StableSR_Canny/logs/2025-10-21T11-44-00_stablesr_canny_in_4x_true_20251021_114357/checkpoints/epoch=000111.ckpt`
- **8x模型**: `/root/dp/StableSR_Canny/logs/2025-10-21T03-44-51_stablesr_canny_in_20251021_034447/checkpoints/epoch=000194.ckpt`
- **VQGAN模型**: `/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt`

### 处理参数
- **4x下采样因子**: 4
- **8x下采样因子**: 8
- **混合参数**: alpha=1.0, beta=1.0
- **推理步数**: 200
- **Tile大小**: 512
- **颜色校正**: AdaIN
- **VQGAN权重**: dec-w=0.5

### 跳过图片
- `0001.png` (已处理)
- `0019.png` (已处理)

## 输出格式

输出目录按照以下格式命名：
- **4x图片**: `outputs/4x/{image_id}_4x/`
- **8x图片**: `outputs/8x/{image_id}_8x/`

例如：
- `outputs/4x/0013_4x/`
- `outputs/8x/0013_8x/`

## 使用方法

### 1. 测试配置
```bash
# 检查配置和文件是否存在
./test_batch_inference.sh
```

### 2. 运行批量推理
```bash
# 执行批量推理
./batch_inference_blended_tile.sh
```

## 脚本功能

### 主要功能
1. **自动扫描图片**: 扫描指定目录中的所有PNG图片
2. **跳过已处理**: 自动跳过指定的已处理图片
3. **分别处理**: 4x和8x图片使用不同的模型和参数
4. **错误处理**: 单个图片失败不影响其他图片处理
5. **进度显示**: 显示处理进度和结果

### 处理流程
1. 检查所有必需的文件和目录
2. 扫描4x和8x图片目录
3. 过滤掉已处理的图片
4. 逐个处理4x图片
5. 逐个处理8x图片
6. 显示处理结果和输出结构

## 预期处理的图片

根据目录扫描，预期处理以下图片：

### 4x图片 (跳过0001.png, 0019.png)
- `0013.png` → `outputs/4x/0013_4x/`
- `0031.png` → `outputs/4x/0031_4x/`
- `0044.png` → `outputs/4x/0044_4x/`
- `0051.png` → `outputs/4x/0051_4x/`

### 8x图片 (跳过0001.png, 0019.png)
- `0013.png` → `outputs/8x/0013_8x/`
- `0031.png` → `outputs/8x/0031_8x/`
- `0044.png` → `outputs/8x/0044_8x/`
- `0051.png` → `outputs/8x/0051_8x/`

## 输出文件

每个图片处理完成后，会在对应的输出目录中生成：
- `output.png` - 主要输出图片
- `blended_input.png` - 混合输入图片
- `lr_upscaled.png` - LR上采样图片
- `edge_input.png` - 边缘输入图片

## 故障排除

### 常见问题
1. **文件不存在**: 检查所有路径是否正确
2. **权限问题**: 确保脚本有执行权限
3. **内存不足**: 调整tile大小参数
4. **GPU内存不足**: 减小tile大小或使用CPU模式

### 调试方法
```bash
# 检查文件是否存在
ls -la /stablesr_dataset/weiqiang/lr_x4_960x540/
ls -la /stablesr_dataset/weiqiang/lr_x8_480x270/
ls -la /stablesr_dataset/weiqiang/hr_4k_canny/

# 检查检查点文件
ls -la /root/dp/StableSR_Canny/logs/*/checkpoints/

# 运行测试脚本
./test_batch_inference.sh
```

## 性能估计

根据图片大小和处理参数：
- **4x图片**: 每张约2-5分钟
- **8x图片**: 每张约3-7分钟
- **总处理时间**: 约20-50分钟（4张4x + 4张8x）

## 注意事项

1. 确保有足够的磁盘空间存储输出结果
2. 处理过程中不要中断脚本
3. 可以随时查看输出目录确认处理进度
4. 如果某个图片处理失败，可以单独重新处理

## 自定义配置

如需修改配置，编辑 `batch_inference_blended_tile.sh` 文件中的以下变量：
- `SKIP_IMAGES`: 要跳过的图片列表
- `BLEND_ALPHA`, `BLEND_BETA`: 混合参数
- `DDPM_STEPS`: 推理步数
- `TILE_SIZE`: Tile大小
- `DEC_W`: VQGAN权重 (默认0.5)
- 其他处理参数
