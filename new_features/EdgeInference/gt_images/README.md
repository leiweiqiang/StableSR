# GT (Ground Truth) Test Images

## 📁 目录说明

这个目录用于存放**高分辨率(GT)测试图像**，即Ground Truth原始高质量图像。

---

## 📋 使用说明

### 1. 放置图像

将你的GT测试图像放在这个目录中。支持的格式：
- `.png` (推荐)
- `.jpg` / `.jpeg`
- `.bmp`
- `.tiff`

### 2. 分辨率建议

- **推荐分辨率**: 2048×2048 或更高
- **对应LR**: LR图像应为GT的1/4（如GT=2048×2048，LR=512×512）

### 3. 文件命名

- 文件名应与对应的LR图像保持一致
- 示例：
  - GT: `image001.png` (在此目录)
  - LR: `image001.png` (在lr_images目录)

---

## 🎯 GT图像的作用

### 1. Edge Map生成（主要用途）⭐

GT图像用于生成高质量的edge map：
```python
# 训练时的edge生成（来自GT）
edge_map = generate_edge_map_from_gt(gt_image)
```

**为什么使用GT生成edge？**
- ✅ 与训练保持一致（训练时用GT生成edge）
- ✅ Edge质量更高（原始分辨率，细节丰富）
- ✅ 避免domain mismatch

### 2. 结果对比

GT图像作为参考标准，用于：
- 评估超分辨率质量
- 计算PSNR、SSIM等指标
- 视觉对比

---

## 🧪 测试示例

### 快速测试
```bash
# 确保已放置测试图像
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference
ls lr_images/  # 查看LR图像
ls gt_images/  # 查看GT图像

# 运行测试（使用GT生成edge）
./test_edge_inference.sh quick
```

### GT-based Edge推理
```bash
python ../../scripts/sr_val_edge_inference.py \
    --init-img lr_images \
    --gt-img gt_images \
    --outdir outputs/test_results \
    --use_edge_processing \
    --config ../../configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt ../../models/your_model.ckpt
```

---

## 📊 图像要求

### GT图像特点
- 高分辨率原始图像
- 无压缩损失（建议PNG格式）
- 清晰、细节丰富
- 用于edge生成和结果评估

### 示例尺寸对应
| GT尺寸 | LR尺寸 | 下采样倍数 |
|--------|--------|------------|
| 512×512 | 128×128 | 4x |
| 1024×1024 | 256×256 | 4x |
| 2048×2048 | 512×512 | 4x |

---

## 💡 提示

1. **高质量**: GT应该是无损或高质量图像
2. **配对**: 确保每个GT图像都有对应的LR图像
3. **命名**: GT和LR使用相同的文件名
4. **分辨率**: GT应该是LR的4倍大小
5. **必要性**: 强烈推荐提供GT用于edge生成

---

## ⚠️ 注意事项

### 如果没有GT图像

可以从LR生成edge（不推荐）：
```bash
./test_edge_inference.sh lr_edge
```

但这会导致：
- ⚠️ Domain mismatch（训练用GT，推理用LR）
- ⚠️ Edge质量较低
- ⚠️ SR效果可能下降

---

## 🔗 相关目录

- **LR图像**: `../lr_images/` - 对应的低分辨率输入图像
- **输出结果**: 运行测试后会在项目根目录的`outputs/`下生成

---

**创建日期**: 2025-10-15  
**用途**: Edge推理测试数据 - GT原始图像（用于edge生成）

