# 🏗️ Tile-Based Edge SR 系统架构

## 📐 系统整体架构

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Tile-Based Edge Super-Resolution System                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌────────┐       ┌──────────┐      ┌──────────┐
   │ Input  │       │ GT Image │      │  Output  │
   │  LR    │       │(Optional)│      │   HR     │
   └────────┘       └──────────┘      └──────────┘
        │                 │                 ▲
        └─────────────────┼─────────────────┘
                          │
              ┌───────────┴───────────┐
              │   Main Processing     │
              │      Pipeline         │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐     ┌──────────┐
   │ Pre-    │      │  Tile    │     │  Post-   │
   │ Process │      │Processing│     │ Process  │
   └─────────┘      └──────────┘     └──────────┘
```

## 🔄 详细处理流程

### Phase 1: 预处理阶段

```
┌──────────────────────────────────────────────────────┐
│  PHASE 1: Pre-processing                             │
└──────────────────────────────────────────────────────┘

输入 LR 图片 (任意尺寸)
    │
    ├─ 读取图片: read_image()
    │   └─ 归一化到 [-1, 1]
    │
    ├─ 计算上采样比例
    │   upsample_scale = max(input_size/min(H,W), upscale)
    │   
    ├─ 上采样 LR 图片
    │   F.interpolate(mode='bicubic')
    │   原始: 256x256 → 上采样: 2048x2048
    │
    ├─ 填充到32的倍数
    │   if not (H%32==0 and W%32==0):
    │       pad to multiple of 32
    │
    └─ 加载 GT 图片 (如果提供)
        read_image(gt_path)
        
Output: upsampled_lr, gt_image, original_size
```

### Phase 2: Tile切分阶段

```
┌──────────────────────────────────────────────────────┐
│  PHASE 2: Tile Splitting                             │
└──────────────────────────────────────────────────────┘

检查图片尺寸
    │
    ├─ 如果 H > vqgantile_size or W > vqgantile_size
    │   │
    │   └─ 创建 ImageSpliterWithEdge
    │       │
    │       ├─ 参数:
    │       │   - pch_size: 1280 (tile尺寸)
    │       │   - stride: 1000 (tile步长)
    │       │   - sf: 1 (缩放因子)
    │       │   - gt_image: GT图片张量
    │       │   - original_lr_size: 原始尺寸
    │       │
    │       └─ 计算tile布局:
    │           图片 2048x2048
    │           ┌─────────┬─────────┐
    │           │ Tile 1  │ Tile 2  │
    │           │1280x1280│1280x1048│
    │           ├─────────┼─────────┤
    │           │ Tile 3  │ Tile 4  │
    │           │1048x1280│1048x1048│
    │           └─────────┴─────────┘
    │           重叠区域: 280像素
    │
    └─ 否则: 整图处理，无需tile

Output: tile_iterator or full_image
```

### Phase 3: Tile处理循环

```
┌──────────────────────────────────────────────────────┐
│  PHASE 3: Tile Processing Loop                       │
└──────────────────────────────────────────────────────┘

对每个 tile:
    │
    ├─ 3.1: 提取 LR tile
    │   lr_tile = im_lq[h_start:h_end, w_start:w_end]
    │   尺寸: 1280x1280 (或边界tile的实际尺寸)
    │
    ├─ 3.2: 获取对应的 Edge Map
    │   │
    │   ├─ 如果有 GT 图片:
    │   │   │
    │   │   ├─ 提取 GT tile
    │   │   │   scale = H_gt / H_lr_upsampled
    │   │   │   coords_gt = coords_lr * scale
    │   │   │   gt_tile = gt_image[coords_gt]
    │   │   │   resize if needed
    │   │   │
    │   │   └─ 从 GT tile 生成 edge map
    │   │       Canny(gt_tile) → edge_map
    │   │
    │   └─ 否则: 从 LR tile 生成
    │       Canny(lr_tile) → edge_map
    │
    ├─ 3.3: Encode 到 Latent 空间
    │   lr_tile (1280x1280)
    │       ↓ VQGAN Encoder
    │   latent (160x160)  # 8x downsampling
    │
    ├─ 3.4: Diffusion 采样 (带edge guidance)
    │   │
    │   ├─ 输入:
    │   │   - latent: 160x160
    │   │   - edge_map: 1280x1280
    │   │   - semantic_c: text embedding
    │   │
    │   ├─ 第二层 Tile 处理 (在 latent 空间)
    │   │   latent (160x160)
    │   │   ┌────────┬────────┬────────┐
    │   │   │ 64x64  │ 64x64  │ 64x64  │
    │   │   ├────────┼────────┼────────┤
    │   │   │ 64x64  │ 64x64  │ 64x64  │
    │   │   └────────┴────────┴────────┘
    │   │   重叠: 32像素 (latent空间)
    │   │
    │   └─ Diffusion 步骤:
    │       for t in timesteps (200 steps):
    │           noise_pred = UNet(x_t, t, edge_map, semantic_c)
    │           x_{t-1} = denoise(x_t, noise_pred)
    │
    ├─ 3.5: Decode 回像素空间
    │   latent (160x160)
    │       ↓ VQGAN Decoder (with LR features)
    │   hr_tile (1280x1280)
    │
    ├─ 3.6: 颜色修正 (可选)
    │   if colorfix_type == 'adain':
    │       hr_tile = AdaIN(hr_tile, lr_tile)
    │   elif colorfix_type == 'wavelet':
    │       hr_tile = wavelet_reconstruction(hr_tile, lr_tile)
    │
    └─ 3.7: 高斯权重累积
        spliter.update_gaussian(hr_tile, coords)
        
        累积矩阵:
        result[coords] += hr_tile * gaussian_weight
        count[coords] += gaussian_weight

循环结束
```

### Phase 4: 后处理阶段

```
┌──────────────────────────────────────────────────────┐
│  PHASE 4: Post-processing                            │
└──────────────────────────────────────────────────────┘

├─ 4.1: 融合所有 tiles
│   final_result = spliter.gather()
│   result = result / count  # 归一化
│   
├─ 4.2: 去除填充
│   if flag_pad:
│       result = result[:ori_h, :ori_w]
│
├─ 4.3: 最终缩放 (如果需要)
│   if upsample_scale > target_upscale:
│       result = F.interpolate(result, target_size)
│
├─ 4.4: 转换到 [0, 255]
│   result = (result + 1.0) / 2.0 * 255.0
│
└─ 4.5: 保存结果
    Image.save(output_path)
```

## 🎯 关键组件详解

### 1. ImageSpliterWithEdge 类

```python
class ImageSpliterWithEdge(ImageSpliterTh):
    """
    扩展的图片分割器，支持edge map处理
    
    继承自: ImageSpliterTh
    新增功能: GT tile提取和edge map生成
    """
    
    ┌─────────────────────────────────────┐
    │  ImageSpliterWithEdge               │
    ├─────────────────────────────────────┤
    │  属性:                              │
    │  - im_ori: 原始图片                │
    │  - gt_image: GT图片                │
    │  - original_lr_size: 原始LR尺寸    │
    │  - upsampled_lr_size: 上采样尺寸   │
    │  - im_res: 累积结果                │
    │  - pixel_count: 像素计数           │
    │  - weight: 高斯权重矩阵            │
    ├─────────────────────────────────────┤
    │  方法:                              │
    │  - __init__(): 初始化               │
    │  - __next__(): 获取下一个tile      │
    │  - get_edge_map_for_current_tile() │
    │  - update_gaussian(): 高斯权重更新 │
    │  - gather(): 收集最终结果          │
    └─────────────────────────────────────┘
```

### 2. Edge Map 生成流程

```
GT Tile (1280x1280, RGB)
    │
    ├─ 1. 转换到灰度
    │   cv2.cvtColor(RGB2GRAY)
    │   → gray_image (1280x1280, 单通道)
    │
    ├─ 2. 高斯模糊
    │   cv2.GaussianBlur(kernel=5x5, sigma=1.4)
    │   → blurred_image
    │
    ├─ 3. Canny边缘检测
    │   cv2.Canny(threshold1=100, threshold2=200)
    │   → edges (1280x1280, 二值图)
    │
    ├─ 4. 转换为3通道
    │   cv2.cvtColor(GRAY2RGB)
    │   → edges_3ch (1280x1280x3)
    │
    └─ 5. 归一化到 [-1, 1]
        (edges_3ch / 127.5) - 1.0
        → edge_map (1280x1280x3, [-1,1])

输出: edge_map tensor [1, 3, 1280, 1280]
```

### 3. 高斯权重融合

```
高斯权重生成:
┌──────────────────────────────────────┐
│  _gaussian_weights(w, h)             │
├──────────────────────────────────────┤
│  var = 0.01                          │
│  for x in range(w):                  │
│      prob_x = exp(-(x-mid)²/var)     │
│  for y in range(h):                  │
│      prob_y = exp(-(y-mid)²/var)     │
│  weights = outer(prob_y, prob_x)     │
└──────────────────────────────────────┘

权重矩阵可视化 (1280x1280):
     ┌─────────────────────────────┐
  0.1│ .  .  .  .  .  .  .  .  .  │
  0.3│ .  .  *  *  *  *  *  .  .  │
  0.5│ .  *  *  *  *  *  *  *  .  │
  0.7│ .  *  *  *  █  █  *  *  .  │
  0.9│ .  *  *  █  █  █  █  *  .  │
  1.0│ .  *  *  █  █  █  █  *  .  │  ← 中心
  0.9│ .  *  *  █  █  █  █  *  .  │
  0.7│ .  *  *  *  █  █  *  *  .  │
  0.5│ .  *  *  *  *  *  *  *  .  │
  0.3│ .  .  *  *  *  *  *  .  .  │
  0.1│ .  .  .  .  .  .  .  .  .  │
     └─────────────────────────────┘

融合公式:
result[coords] = Σ(tile_i * weight_i) / Σ(weight_i)
```

## 🔀 数据流图

### 单个Tile的完整数据流

```
                    ┌──────────┐
                    │ GT Image │
                    └─────┬────┘
                          │
                          ▼
                 extract_tile_from_gt()
                          │
┌──────────┐              ▼              ┌──────────┐
│ LR Tile  │      ┌──────────────┐       │ Edge Map │
│1280x1280 │      │   GT Tile    │       │1280x1280 │
└────┬─────┘      │  1280x1280   │       └────┬─────┘
     │            └──────┬───────┘            │
     │                   │                    │
     │                   ▼                    │
     │          generate_edge_map()           │
     │                   │                    │
     │                   └────────────────────┘
     │                            │
     ├────────────────────────────┤
     │                            │
     ▼                            ▼
┌─────────────────────────────────────────┐
│         VQGAN Encoder                   │
│  Input: 1280x1280x3                     │
│  Output: 160x160x4 (latent)             │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Latent (init)  │
        │   160x160x4    │
        └────────┬───────┘
                 │
                 ├──── Add noise ───┐
                 │                  │
                 ▼                  ▼
        ┌────────────────┐   ┌──────────┐
        │  Noisy Latent  │   │  Edge    │
        │   160x160x4    │   │  Map     │
        └────────┬───────┘   └────┬─────┘
                 │                │
                 │                │
                 ▼                ▼
┌────────────────────────────────────────────┐
│      Diffusion Sampling (200 steps)        │
│  - UNet prediction with edge guidance      │
│  - Tile-based processing (64x64 in latent) │
│  - Progressive denoising                   │
└────────────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Clean Latent  │
        │   160x160x4    │
        └────────┬───────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│         VQGAN Decoder                      │
│  Input: 160x160x4 (latent)                 │
│  Output: 1280x1280x3 (HR tile)             │
│  With: LR features for better detail       │
└────────────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │    HR Tile     │
        │   1280x1280    │
        └────────┬───────┘
                 │
                 ▼
          Color Correction
                 │
                 ▼
      Gaussian Weight Accumulation
                 │
                 ▼
           ┌──────────┐
           │  Result  │
           └──────────┘
```

## 💾 内存占用分析

### 各阶段内存占用

```
                 Memory Usage (GB)
                 0    4    8    12   16   20
Phase 1: Load   ├────┤
Phase 2: Split  ├────┤
Phase 3: Tile 1 ├────────────┤          ← Peak
  - Encode      ├────────┤
  - Diffusion   ├────────────┤          ← Peak
  - Decode      ├────────┤
Phase 3: Tile 2 ├────────────┤
  ...
Phase 4: Merge  ├────────┤
Phase 5: Save   ├────┤

总结:
- 基础占用: ~4GB (模型)
- Tile处理: +8-12GB (取决于tile_size)
- 峰值: Diffusion采样阶段
```

### 参数对内存的影响

```
vqgantile_size = 768:
├─ LR tile: 768x768x3 = ~1.7MB
├─ Latent: 96x96x4 = ~0.14MB
└─ 峰值内存: ~8GB

vqgantile_size = 1280:
├─ LR tile: 1280x1280x3 = ~4.7MB
├─ Latent: 160x160x4 = ~0.39MB
└─ 峰值内存: ~16GB

vqgantile_size = 1536:
├─ LR tile: 1536x1536x3 = ~6.8MB
├─ Latent: 192x192x4 = ~0.56MB
└─ 峰值内存: ~20GB
```

## ⚙️ 模型架构

### 完整模型栈

```
┌───────────────────────────────────────────────────┐
│  LatentDiffusionSRTextWTWithEdge                  │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────────────────────────────────────┐   │
│  │  Text Encoder (CLIP)                     │   │
│  │  - Input: text prompt                    │   │
│  │  - Output: semantic embedding            │   │
│  └──────────────────────────────────────────┘   │
│                                                   │
│  ┌──────────────────────────────────────────┐   │
│  │  First Stage (VQGAN)                     │   │
│  │  ├─ Encoder: Image → Latent              │   │
│  │  │  - 8x downsampling                    │   │
│  │  │  - 4 channels                         │   │
│  │  └─ Decoder: Latent → Image              │   │
│  │     - With LR feature fusion             │   │
│  │     - Residual connections               │   │
│  └──────────────────────────────────────────┘   │
│                                                   │
│  ┌──────────────────────────────────────────┐   │
│  │  UNet (with Edge Processing)             │   │
│  │  ├─ Input channels: 4 (latent) + 3 (edge)│   │
│  │  ├─ Cross-attention with text            │   │
│  │  ├─ Self-attention layers                │   │
│  │  ├─ ResNet blocks                        │   │
│  │  └─ Output: Noise prediction             │   │
│  └──────────────────────────────────────────┘   │
│                                                   │
│  ┌──────────────────────────────────────────┐   │
│  │  Edge Processor                          │   │
│  │  ├─ Edge encoding                        │   │
│  │  ├─ Feature extraction                   │   │
│  │  └─ Fusion with latent                   │   │
│  └──────────────────────────────────────────┘   │
│                                                   │
└───────────────────────────────────────────────────┘
```

## 🎛️ 配置空间

### 可调参数及其影响

```
参数维度:

1. Tile尺寸维度
   vqgantile_size ─┬─ 768  (小) → 低内存, 更多tile
                   ├─ 1024 (中) → 平衡
                   ├─ 1280 (大) → 高质量, 高内存
                   └─ 1536 (超大) → 超大图片

2. 重叠维度
   vqgantile_stride ─┬─ 1200 (小重叠) → 快, 可能有接缝
                     ├─ 1000 (中重叠) → 平衡
                     └─ 800  (大重叠) → 慢, 无接缝

3. 精细度维度
   tile_overlap ─┬─ 16 (小) → 快
                 ├─ 32 (中) → 平衡
                 └─ 48 (大) → 高质量

4. 质量维度
   ddpm_steps ─┬─ 100 (快速)
               ├─ 200 (标准)
               └─ 300 (高质量)

5. 颜色维度
   colorfix_type ─┬─ nofix (无)
                  ├─ wavelet (好)
                  └─ adain (最好)
```

## 🚀 性能优化策略

### 优化金字塔

```
                    最高质量
                       ▲
                       │
        ┌──────────────┴──────────────┐
        │  tile_size=1280, stride=800 │
        │  overlap=48, steps=300      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  tile_size=1280, stride=1000│  ← 推荐
        │  overlap=32, steps=200      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  tile_size=1024, stride=1000│
        │  overlap=16, steps=100      │
        └──────────────┬──────────────┘
                       │
                    最快速度
```

## 📊 决策树

### 参数选择决策流程

```
                  开始
                   │
         ┌─────────┴─────────┐
         │  显存多少？        │
         └─────────┬─────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
        <12GB    12-16GB   >16GB
         │         │         │
         ▼         ▼         ▼
      size=768  size=1024 size=1280
         │         │         │
         └─────────┼─────────┘
                   │
         ┌─────────┴─────────┐
         │  追求什么？        │
         └─────────┬─────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
        速度      平衡      质量
         │         │         │
         ▼         ▼         ▼
    steps=100 steps=200 steps=300
    stride↑  stride=1k stride↓
    overlap↓ overlap=32 overlap↑
         │         │         │
         └─────────┼─────────┘
                   │
                 运行！
```

---

## 🔗 快速链接

- **入门**: [TILE_EDGE_README.md](TILE_EDGE_README.md)
- **对比**: [TILE_VS_STANDARD_COMPARISON.md](TILE_VS_STANDARD_COMPARISON.md)
- **技术**: [TILE_EDGE_PROCESSING_GUIDE.md](TILE_EDGE_PROCESSING_GUIDE.md)
- **GT剪切**: [GT_TILE_EXTRACTION_VISUAL_GUIDE.md](GT_TILE_EXTRACTION_VISUAL_GUIDE.md)
- **示例**: [example_tile_edge_processing.sh](example_tile_edge_processing.sh)

---

**版本**: 1.0  
**日期**: 2025-10-08  
**状态**: ✅ 生产就绪

