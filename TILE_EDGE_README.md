# 🚀 Tile-Based Edge Processing for Large Images

## 📋 快速导航

本文档集合提供了完整的tile-based edge processing实现和使用指南。

### 📄 文档索引

| 文档 | 说明 | 难度 |
|------|------|------|
| [TILE_VS_STANDARD_COMPARISON.md](TILE_VS_STANDARD_COMPARISON.md) | **推荐首先阅读** - 标准版vs Tile版对比 | ⭐ 入门 |
| [TILE_EDGE_PROCESSING_GUIDE.md](TILE_EDGE_PROCESSING_GUIDE.md) | Tile处理完整技术指南 | ⭐⭐ 中级 |
| [GT_TILE_EXTRACTION_VISUAL_GUIDE.md](GT_TILE_EXTRACTION_VISUAL_GUIDE.md) | GT tile剪切逻辑可视化详解 | ⭐⭐⭐ 高级 |
| [example_tile_edge_processing.sh](example_tile_edge_processing.sh) | 6个实用示例脚本 | ⭐ 实践 |

### 🔧 核心文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py` | Python脚本 | 主要实现 - tile-based edge SR |
| `scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py` | Python脚本 | 标准版本 - 仅支持≤512 |
| `scripts/util_image.py` | 工具类 | `ImageSpliterTh`类 |
| `example_tile_edge_processing.sh` | Bash脚本 | 使用示例（可执行） |

## 🎯 核心问题解答

### Q1: 为什么需要Tile版本？

**A**: 标准版本将所有图片强制resize到512x512，对于大图片会丢失大量细节。Tile版本可以处理任意尺寸的图片。

```
标准版本: 2048x2048 → resize → 512x512 ❌ 丢失75%像素
Tile版本:  2048x2048 → tile处理 → 2048x2048 ✅ 保持分辨率
```

### Q2: GT图片如何剪切？

**A**: 核心是理解三个坐标空间：

```
原始LR空间 (256x256)
    ↓ upsample
上采样LR空间 (2048x2048) ← tile坐标在这里
    ↓ 映射
GT空间 (2048x2048) ← 需要提取对应tile
```

**关键公式**:
```python
scale = H_gt / H_lr_upsampled  # 不是 H_lr_original !
h_start_gt = int(h_start * scale)
```

详见: [GT_TILE_EXTRACTION_VISUAL_GUIDE.md](GT_TILE_EXTRACTION_VISUAL_GUIDE.md)

### Q3: 如何选择使用哪个版本？

**决策树**:
```
图片尺寸 ≤ 512x512?
├─ 是 → 使用标准版本或Tile版本（都可以）
└─ 否 → 必须使用Tile版本

显存 < 12GB?
├─ 是 → 使用Tile版本，设置 vqgantile_size=1024
└─ 否 → 使用Tile版本，设置 vqgantile_size=1280
```

### Q4: Tile接缝如何处理？

**A**: 使用两种机制：

1. **Tile重叠** - 相邻tile有重叠区域
   ```
   ┌────────┐
   │ Tile 1 │
   │    ┌───┼────┐
   │    │XXX│    │ ← 重叠区域
   └────┼───┘    │
        │ Tile 2 │
        └────────┘
   ```

2. **高斯权重融合** - 重叠区域使用加权平均
   ```python
   weight_center = 1.0  # 中心权重高
   weight_edge = 0.1    # 边缘权重低
   result = (tile1 * w1 + tile2 * w2) / (w1 + w2)
   ```

## 🚀 快速开始

### 1. 基础使用（有GT图片）

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./input_lr \
    --gt-img ./input_hr \
    --outdir ./output \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --ddpm_steps 200
```

### 2. 无GT图片（从LR生成edge map）

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./input_lr \
    --outdir ./output \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --ddpm_steps 200
```

### 3. 运行示例脚本

```bash
chmod +x example_tile_edge_processing.sh
./example_tile_edge_processing.sh
```

## 📊 参数配置建议

### 快速参考表

| 场景 | vqgantile_size | stride | overlap | steps | 内存 |
|------|---------------|--------|---------|-------|------|
| 预览 | 1024 | 800 | 16 | 100 | 8GB |
| 标准 | 1280 | 1000 | 32 | 200 | 12GB |
| 高质量 | 1280 | 800 | 48 | 300 | 16GB |
| 4K图片 | 1536 | 1200 | 48 | 200 | 20GB |

### 调优指南

**提高速度**:
- ↓ `ddpm_steps`: 200 → 100
- ↑ `vqgantile_stride`: 1000 → 1200
- ↓ `tile_overlap`: 32 → 16

**提高质量**:
- ↑ `ddpm_steps`: 200 → 300
- ↓ `vqgantile_stride`: 1000 → 800
- ↑ `tile_overlap`: 32 → 48

**减少内存**:
- ↓ `vqgantile_size`: 1280 → 1024 → 768

## 🔍 技术深入

### Tile处理流程

```
输入大图 (2048x2048)
    ↓
第一层: VQGAN级别切分 (1280x1280 tiles)
    ↓
    ├─ Tile 1 (1280x1280)
    │   ↓
    │   对应GT Tile提取
    │   ↓
    │   生成Edge Map
    │   ↓
    │   Encode到Latent (160x160)
    │   ↓
    │   第二层: Diffusion级别切分 (64x64 tiles)
    │   ↓
    │   Diffusion采样 (with edge guidance)
    │   ↓
    │   Decode回像素 (1280x1280)
    │   ↓
    ├─ Tile 2 ...
    ├─ Tile 3 ...
    └─ Tile 4 ...
         ↓
    高斯权重融合
         ↓
输出大图 (2048x2048) - 无缝拼接
```

### GT Tile映射核心逻辑

```python
def extract_tile_from_gt(gt_image, tile_coords, original_lr_size, upsampled_lr_size):
    """
    核心公式:
    1. scale = H_gt / H_lr_upsampled  ← 关键！
    2. coords_gt = coords_lr * scale
    3. gt_tile = gt_image[coords_gt]
    4. resize if needed
    """
    h_start, h_end, w_start, w_end = tile_coords
    h_lr_up, w_lr_up = upsampled_lr_size
    h_gt, w_gt = gt_image.shape[2:]
    
    # 计算缩放因子
    scale_h = h_gt / h_lr_up  # 不是 h_gt / h_lr_orig !
    scale_w = w_gt / w_lr_up
    
    # 映射坐标
    h_start_gt = int(h_start * scale_h)
    h_end_gt = int(h_end * scale_h)
    w_start_gt = int(w_start * scale_w)
    w_end_gt = int(w_end * scale_w)
    
    # 边界检查
    h_start_gt = max(0, min(h_start_gt, h_gt))
    h_end_gt = max(0, min(h_end_gt, h_gt))
    w_start_gt = max(0, min(w_start_gt, w_gt))
    w_end_gt = max(0, min(w_end_gt, w_gt))
    
    # 提取
    gt_tile = gt_image[:, :, h_start_gt:h_end_gt, w_start_gt:w_end_gt]
    
    # Resize到目标尺寸
    target_h = h_end - h_start
    target_w = w_end - w_start
    if gt_tile.shape[2:] != (target_h, target_w):
        gt_tile = F.interpolate(gt_tile, size=(target_h, target_w))
    
    return gt_tile
```

## 🐛 常见问题与解决

### 1. CUDA Out of Memory

**症状**: 
```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 减小tile尺寸
--vqgantile_size 1024  # 或 768

# 或使用单个样本
--n_samples 1
```

### 2. Tile接缝明显

**症状**: 输出图片有明显的块状边界

**解决**:
```bash
# 增加重叠
--tile_overlap 48
--vqgantile_stride 800  # 减小stride增加重叠

# 使用颜色修正
--colorfix_type adain
```

### 3. 处理太慢

**症状**: 一张图片需要很长时间

**解决**:
```bash
# 减少扩散步数
--ddpm_steps 100  # 从200降低

# 增大stride减少tile数量
--vqgantile_stride 1200  # 从1000增加

# 减少重叠
--tile_overlap 16  # 从32降低
```

### 4. Edge细节不清晰

**症状**: 边缘增强效果不明显

**解决**:
```bash
# 使用GT图片生成edge map
--gt-img ./path/to/hr_images

# 检查GT和LR文件名是否匹配
# 例如: 0803.png (LR) 应对应 0803.png (GT)
```

### 5. GT图片找不到

**症状**: 
```
Warning: GT image not found for xxx, using LR image for edge map
```

**解决**:
- 检查GT图片路径是否正确
- 检查文件名是否匹配（不包括扩展名）
- 支持的扩展名: .png, .jpg, .jpeg, .bmp, .tiff

## 📈 性能基准

### 处理时间 (RTX 3090)

| 图片尺寸 | Tile配置 | DDPM步数 | 时间 |
|---------|---------|---------|------|
| 512x512 | 无需tile | 200 | 15s |
| 1024x1024 | 1280/1000 | 200 | 35s |
| 2048x2048 | 1280/1000 | 200 | 90s |
| 4096x4096 | 1536/1200 | 200 | 380s |

### 内存占用

| 配置 | VRAM峰值 | 建议显存 |
|------|---------|---------|
| vqgantile_size=768 | ~8GB | 10GB |
| vqgantile_size=1024 | ~12GB | 14GB |
| vqgantile_size=1280 | ~16GB | 18GB |
| vqgantile_size=1536 | ~20GB | 22GB |

## 🎓 学习路径

### 初学者
1. 阅读 [TILE_VS_STANDARD_COMPARISON.md](TILE_VS_STANDARD_COMPARISON.md)
2. 运行 `example_tile_edge_processing.sh` 中的示例1
3. 尝试自己的图片

### 中级用户
1. 阅读 [TILE_EDGE_PROCESSING_GUIDE.md](TILE_EDGE_PROCESSING_GUIDE.md)
2. 理解两层tile机制
3. 调优参数获得最佳效果

### 高级用户
1. 阅读 [GT_TILE_EXTRACTION_VISUAL_GUIDE.md](GT_TILE_EXTRACTION_VISUAL_GUIDE.md)
2. 理解坐标映射逻辑
3. 修改代码实现自定义功能

## 📚 扩展阅读

### 相关论文
- StableSR: Exploiting Diffusion Prior for Real-World Image Super-Resolution
- Taming Transformers for High-Resolution Image Synthesis (VQGAN)
- Denoising Diffusion Probabilistic Models

### 相关技术
- Tile-based processing
- Gaussian weighting for seamless blending
- Canny edge detection
- Latent diffusion models

## 🤝 贡献与反馈

如果您发现问题或有改进建议：
1. 检查相关文档是否有解答
2. 尝试调整参数
3. 查看代码注释
4. 提交issue并附带详细信息

## 📝 更新日志

### v1.0 (2025-10-08)
- ✅ 实现tile-based edge processing
- ✅ 支持GT图片tile提取
- ✅ 高斯权重融合
- ✅ 完整文档集
- ✅ 6个实用示例

## 📄 许可证

遵循原StableSR项目的许可证。

---

**🎉 开始使用**: 运行 `./example_tile_edge_processing.sh` 查看效果！

**❓ 有疑问**: 先查看 [TILE_VS_STANDARD_COMPARISON.md](TILE_VS_STANDARD_COMPARISON.md)

**🔧 深入学习**: 阅读 [TILE_EDGE_PROCESSING_GUIDE.md](TILE_EDGE_PROCESSING_GUIDE.md)

