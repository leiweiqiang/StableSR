# Tile-Based Edge Processing 完全指南

## 📋 概述

本指南详细说明如何在处理大于512x512的图片时，正确地从GT（Ground Truth）图片中提取tile并生成对应的edge map。

## 🎯 核心问题

**问题**: 当使用tile-based处理时，如何从GT图片中提取正确的区域来生成edge map？

**答案**: 需要考虑三个坐标空间之间的映射关系：
1. **原始LR空间** - 输入的低分辨率图片原始尺寸
2. **上采样LR空间** - LR图片经过bicubic上采样后的尺寸
3. **GT空间** - Ground Truth高分辨率图片的尺寸

## 📐 坐标空间映射关系

### 图解说明

```
原始LR图片 (128x128)
    ↓ bicubic upsample (4x)
上采样LR图片 (512x512)  ← tile坐标在这个空间中
    ↓ 对应关系
GT图片 (512x512 或更大)  ← 需要提取对应的tile
```

### 处理流程

```
1. 读取LR图片: (H_lr_orig, W_lr_orig)
   例如: 128x128

2. 上采样LR图片: (H_lr_up, W_lr_up)
   例如: 512x512
   scale = max(input_size/min(H,W), upscale_factor)

3. Tile切分在上采样空间进行:
   tile坐标: (h_start, h_end, w_start, w_end)
   例如: (0, 512, 0, 512) 或 (256, 768, 256, 768)

4. GT图片尺寸: (H_gt, W_gt)
   例如: 512x512 (与LR同名，但分辨率可能不同)
```

## 🔢 坐标转换公式

### 关键函数：`extract_tile_from_gt`

```python
def extract_tile_from_gt(gt_image_tensor, tile_coords, original_lr_size, upsampled_lr_size):
    """
    从GT图片中提取对应的tile
    
    Args:
        gt_image_tensor: GT图片 [1, 3, H_gt, W_gt]
        tile_coords: 在upsampled LR空间中的tile坐标 (h_start, h_end, w_start, w_end)
        original_lr_size: 原始LR尺寸 (H_lr_orig, W_lr_orig)
        upsampled_lr_size: 上采样后LR尺寸 (H_lr_up, W_lr_up)
    
    Returns:
        gt_tile: 对应的GT tile [1, 3, tile_h, tile_w]
    """
```

### 转换步骤

#### Step 1: 计算缩放因子

```python
scale_h = H_gt / H_lr_up
scale_w = W_gt / W_lr_up
```

**说明**: 这个缩放因子将"上采样LR空间"的坐标映射到"GT空间"

#### Step 2: 计算GT tile坐标

```python
h_start_gt = int(h_start * scale_h)
h_end_gt = int(h_end * scale_h)
w_start_gt = int(w_start * scale_w)
w_end_gt = int(w_end * scale_w)
```

#### Step 3: 边界检查

```python
h_start_gt = max(0, min(h_start_gt, H_gt))
h_end_gt = max(0, min(h_end_gt, H_gt))
w_start_gt = max(0, min(w_start_gt, W_gt))
w_end_gt = max(0, min(w_end_gt, W_gt))
```

#### Step 4: 提取tile

```python
gt_tile = gt_image_tensor[:, :, h_start_gt:h_end_gt, w_start_gt:w_end_gt]
```

#### Step 5: Resize到目标尺寸

```python
tile_h = h_end - h_start
tile_w = w_end - w_start

if gt_tile.shape[2] != tile_h or gt_tile.shape[3] != tile_w:
    gt_tile = F.interpolate(gt_tile, size=(tile_h, tile_w), mode='bicubic')
```

## 📊 实际案例分析

### 案例1: LR和GT尺寸相同

```
输入:
- LR原始: 128x128
- LR上采样: 512x512
- GT: 512x512
- Tile在上采样空间: (0, 256, 0, 256)

计算:
- scale_h = 512 / 512 = 1.0
- scale_w = 512 / 512 = 1.0
- GT tile坐标: (0, 256, 0, 256)

结果:
- 提取GT的 [0:256, 0:256] 区域
```

### 案例2: GT尺寸是上采样LR的2倍

```
输入:
- LR原始: 128x128
- LR上采样: 512x512
- GT: 1024x1024
- Tile在上采样空间: (0, 256, 0, 256)

计算:
- scale_h = 1024 / 512 = 2.0
- scale_w = 1024 / 512 = 2.0
- GT tile坐标: (0, 512, 0, 512)

结果:
- 提取GT的 [0:512, 0:512] 区域
- Resize到 256x256 以匹配tile尺寸
```

### 案例3: GT尺寸小于上采样LR

```
输入:
- LR原始: 64x64
- LR上采样: 512x512
- GT: 256x256
- Tile在上采样空间: (0, 256, 0, 256)

计算:
- scale_h = 256 / 512 = 0.5
- scale_w = 256 / 512 = 0.5
- GT tile坐标: (0, 128, 0, 128)

结果:
- 提取GT的 [0:128, 0:128] 区域
- Resize到 256x256 以匹配tile尺寸
```

### 案例4: 大图片分块处理

```
输入:
- LR原始: 256x256
- LR上采样: 2048x2048 (upscale=8x)
- GT: 2048x2048
- vqgantile_size: 1280
- vqgantile_stride: 1000

Tile划分:
- Tile 1: (0, 1280, 0, 1280)
- Tile 2: (0, 1280, 1000, 2280) -> (0, 1280, 1000, 2048)
- Tile 3: (1000, 2280, 0, 1280) -> (1000, 2048, 0, 1280)
- Tile 4: (1000, 2280, 1000, 2280) -> (1000, 2048, 1000, 2048)

对于Tile 1:
- scale = 2048 / 2048 = 1.0
- GT tile坐标: (0, 1280, 0, 1280)
- 提取GT的 [0:1280, 0:1280] 区域

对于Tile 2:
- GT tile坐标: (0, 1280, 1000, 2048)
- 实际提取尺寸: 1280x1048
- Resize到: 1280x1048 (保持实际tile尺寸)
```

## 🔧 实现类：ImageSpliterWithEdge

### 核心特性

```python
class ImageSpliterWithEdge(ImageSpliterTh):
    def __init__(self, im, pch_size, stride, sf=1, gt_image=None, original_lr_size=None):
        """
        Args:
            im: 上采样后的LR图片 [B, C, H_up, W_up]
            pch_size: tile尺寸 (像素)
            stride: tile步长 (像素)
            sf: 缩放因子 (通常为1)
            gt_image: GT图片 [1, 3, H_gt, W_gt] (可选)
            original_lr_size: 原始LR尺寸 (H_orig, W_orig)
        """
```

### 使用方法

```python
# 创建spliter
im_spliter = ImageSpliterWithEdge(
    im=upsampled_lr_image,          # [1, 3, 2048, 2048]
    pch_size=1280,                   # tile尺寸
    stride=1000,                     # tile步长
    sf=1,                            # 缩放因子
    gt_image=gt_image_tensor,        # [1, 3, 2048, 2048]
    original_lr_size=(256, 256)      # 原始LR尺寸
)

# 遍历tiles
for lr_tile, (h_start, h_end, w_start, w_end) in im_spliter:
    # 获取对应的edge map
    edge_map = im_spliter.get_edge_map_for_current_tile(
        (h_start, h_end, w_start, w_end)
    )
    
    # 使用lr_tile和edge_map进行处理
    # ...
```

## 📝 重要注意事项

### 1. 坐标空间一致性

⚠️ **关键**: Tile坐标必须在同一个坐标空间中使用：
- `ImageSpliterTh.__next__()` 返回的坐标已经经过`sf`缩放
- 在我们的场景中，`sf=1`，所以坐标直接在像素空间
- GT tile的坐标需要从这个空间映射到GT空间

### 2. Edge Map尺寸匹配

✅ **必须**: Edge map的尺寸必须与LR tile完全一致
- LR tile尺寸: `(h_end - h_start, w_end - w_start)`
- Edge map尺寸: 必须相同
- 如果GT tile尺寸不匹配，使用`F.interpolate`调整

### 3. 边界处理

🔒 **安全**: 始终进行边界检查
```python
h_start_gt = max(0, min(h_start_gt, H_gt))
h_end_gt = max(0, min(h_end_gt, H_gt))
```

### 4. 高斯权重融合

🎨 **重要**: Tile拼接使用高斯权重，边缘区域平滑过渡
```python
im_spliter.update_gaussian(x_samples, index_infos)
```

## 🚀 使用示例

### 基本用法（使用GT edge map）

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./128x128_valid_LR \
    --gt-img ./512x512_valid_HR \
    --outdir ./results_tile_edge \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 200 \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt
```

### 不使用GT（从LR生成edge map）

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./input_images \
    --outdir ./results_tile_edge \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --input_size 512 \
    --ddpm_steps 200
```

### 超大图片处理

```bash
# 处理4K图片 (3840x2160)
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./4k_images \
    --gt-img ./4k_gt_images \
    --outdir ./4k_results \
    --use_edge_processing \
    --vqgantile_size 1536 \
    --vqgantile_stride 1200 \
    --tile_overlap 48 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 200 \
    --colorfix_type adain
```

## 🔍 调试技巧

### 1. 打印tile信息

脚本会自动打印每个tile的处理信息：
```
Processing tile 1/4: [0:1280, 0:1280]
  Edge map range: [-1.000, 1.000], mean: -0.234
```

### 2. 保存中间结果

修改代码保存edge map以验证：
```python
# 在生成edge map后添加
edge_map_vis = (edge_map + 1.0) / 2.0 * 255.0
edge_map_vis = edge_map_vis[0].cpu().numpy().transpose(1, 2, 0)
cv2.imwrite(f"edge_map_tile_{tile_count}.png", edge_map_vis)
```

### 3. 验证坐标映射

```python
print(f"LR tile coords: ({h_start}, {h_end}, {w_start}, {w_end})")
print(f"GT tile coords: ({h_start_gt}, {h_end_gt}, {w_start_gt}, {w_end_gt})")
print(f"Scale factors: h={scale_h:.3f}, w={scale_w:.3f}")
```

## 📊 性能考虑

### 内存使用

| 配置 | VQGAN Tile | Diffusion Tile | 显存需求 |
|------|-----------|----------------|---------|
| 小图 (≤512) | 无需分块 | 64 (latent) | ~8GB |
| 中图 (512-1280) | 无需分块 | 64 (latent) | ~12GB |
| 大图 (1280-2048) | 1280 | 64 (latent) | ~16GB |
| 超大图 (>2048) | 1280 | 64 (latent) | ~20GB |

### 优化建议

1. **减小tile尺寸**: 降低`vqgantile_size`以减少内存
2. **减少重叠**: 降低`tile_overlap`加快速度（可能影响质量）
3. **调整步数**: 降低`ddpm_steps`加快速度（可能影响质量）

## 🎓 技术细节

### 为什么不直接在原始LR空间计算？

❌ **错误方法**:
```python
# 在原始LR空间计算tile坐标
scale = H_gt / H_lr_orig  # 例如: 512 / 128 = 4
```

**问题**: 处理过程是在上采样后的空间进行的，tile切分也在这个空间。如果在原始空间计算，会导致tile不对应。

✅ **正确方法**:
```python
# 在上采样LR空间计算
scale = H_gt / H_lr_up  # 例如: 512 / 512 = 1
```

### Tile重叠的作用

1. **VQGAN级别** (`vqgantile_stride < vqgantile_size`):
   - 产生重叠区域
   - 使用高斯权重融合
   - 消除块效应

2. **Diffusion级别** (`tile_overlap`):
   - 在latent空间中重叠
   - 提高一致性
   - 减少伪影

### 两层Tile的协同

```
外层Tile (VQGAN): 1280x1280, stride=1000
    ├─ Tile 1: 处理整个图片块
    │   └─ 内层Tile (Diffusion): 64x64 (latent), overlap=32
    │       ├─ 处理latent空间的子块
    │       └─ 保证扩散过程的一致性
    └─ Tile 2: ...
```

## 🔗 相关文件

- **实现脚本**: `scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py`
- **工具类**: `scripts/util_image.py` (ImageSpliterTh)
- **Edge模型**: `ldm/models/diffusion/ddpm_with_edge.py`
- **标准版本**: `scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py` (不支持tile)

## 📚 参考资源

1. **ImageSpliterTh**: 基础tile切分类
2. **高斯权重融合**: 平滑tile拼接
3. **sample_canvas**: 支持tile的采样方法
4. **Edge处理**: Canny边缘检测 + UNet融合

---

**作者**: AI Assistant  
**日期**: 2025-10-08  
**版本**: 1.0

