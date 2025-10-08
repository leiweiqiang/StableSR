# StableSR Edge版本GT图像集成说明

## 概述

已成功为StableSR Edge版本的推理脚本添加了`--gt-img`参数，允许从Ground Truth (GT)图像生成edge map，而不是从低分辨率(LR)输入图像生成。这可以提供更准确和详细的边缘信息，从而提升超分辨率质量。

## 修改内容

### 1. 新增参数

在`scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py`中添加了新的命令行参数：

```bash
--gt-img /path/to/gt/images
```

- **参数名**: `--gt-img`
- **类型**: 字符串
- **默认值**: `None`
- **说明**: GT图像目录路径，用于生成edge map

### 2. 新增函数

#### `generate_edge_map_from_gt(gt_image_path, target_size, device)`

从GT图像文件生成edge map的新函数：

```python
def generate_edge_map_from_gt(gt_image_path, target_size, device):
    """
    Generate edge map from ground truth image file
    
    Args:
        gt_image_path: Path to ground truth image
        target_size: Target size for the edge map (H, W)
        device: Device to place the tensor on
        
    Returns:
        edge_map: Edge map tensor [1, 3, H, W], values in [-1, 1]
    """
```

**功能特点**:
- 自动加载GT图像并调整到目标尺寸
- 支持多种图像格式 (.png, .jpg, .jpeg, .bmp, .tiff)
- 使用与原始函数相同的Canny边缘检测算法
- 输出格式与原始函数完全兼容

### 3. 修改的推理逻辑

Edge map生成逻辑现在支持两种模式：

1. **GT模式** (当提供`--gt-img`参数时):
   - 根据LR图像文件名查找对应的GT图像
   - 从GT图像生成edge map
   - 如果GT图像不存在，自动回退到LR模式

2. **LR模式** (原始行为，当未提供`--gt-img`参数时):
   - 从LR输入图像生成edge map
   - 保持原有的行为不变

## 使用方法

### 基本用法

```bash
# 使用GT图像生成edge map
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --init-img /path/to/lr/images \
    --gt-img /path/to/gt/images \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/model.ckpt \
    --outdir /path/to/output

# 使用LR图像生成edge map (原始行为)
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --init-img /path/to/lr/images \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt /path/to/model.ckpt \
    --outdir /path/to/output
```

### 文件命名要求

- LR图像和GT图像必须具有相同的基础文件名
- 支持不同的文件扩展名
- 例如: `image_001.png` (LR) 对应 `image_001.png` (GT)

### 支持的图像格式

GT图像支持以下格式：
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`

## 技术细节

### Edge Map生成流程

1. **文件匹配**: 根据LR图像文件名查找对应的GT图像
2. **图像加载**: 加载GT图像并转换为RGB格式
3. **尺寸调整**: 将GT图像调整到与LR图像相同的尺寸
4. **边缘检测**: 应用Canny边缘检测算法
5. **格式转换**: 转换为模型所需的tensor格式

### 错误处理

- 如果GT图像不存在，自动回退到使用LR图像生成edge map
- 支持多种文件扩展名的自动检测
- 提供详细的警告信息

### 性能优化

- GT图像只在需要时加载，避免不必要的内存占用
- 使用高效的图像处理库 (OpenCV, PIL)
- 支持GPU加速的tensor操作

## 优势

1. **更准确的边缘信息**: GT图像包含更多细节，生成的edge map更准确
2. **向后兼容**: 不提供`--gt-img`参数时保持原有行为
3. **自动回退**: GT图像不存在时自动使用LR图像
4. **灵活的文件格式**: 支持多种图像格式
5. **错误处理**: 完善的错误处理和警告机制

## 注意事项

1. **文件组织**: 确保LR和GT图像目录中的文件名对应关系正确
2. **内存使用**: GT图像通常比LR图像大，可能增加内存使用
3. **处理时间**: 从GT图像生成edge map可能需要更多时间
4. **图像质量**: GT图像的质量直接影响edge map的质量

## 测试验证

修改已通过以下测试：
- ✅ Edge map生成函数测试
- ✅ 参数解析测试
- ✅ 文件格式兼容性测试
- ✅ 错误处理测试
- ✅ 形状和数值范围验证

## 示例输出

运行时会显示以下信息：

```
>>>>>>>>>>edge processing>>>>>>>>>>>
Edge processing enabled - using edge-enhanced model
Using GT images from: /path/to/gt/images
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

或

```
>>>>>>>>>>edge processing>>>>>>>>>>>
Edge processing enabled - using edge-enhanced model
Using LR images for edge map generation
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

