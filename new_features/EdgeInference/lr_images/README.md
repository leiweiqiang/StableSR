# LR (Low Resolution) Test Images

## 📁 目录说明

这个目录用于存放**低分辨率(LR)测试图像**。

---

## 📋 使用说明

### 1. 放置图像

将你的LR测试图像放在这个目录中。支持的格式：
- `.png` (推荐)
- `.jpg` / `.jpeg`
- `.bmp`
- `.tiff`

### 2. 分辨率建议

- **推荐分辨率**: 512×512 或更小
- **对应GT**: GT图像应为LR的4倍（如LR=512×512，GT=2048×2048）

### 3. 文件命名

- 文件名应与对应的GT图像保持一致
- 示例：
  - LR: `image001.png` 
  - GT: `image001.png` (在gt_images目录)

---

## 🧪 测试示例

### 快速测试
```bash
# 确保已放置测试图像
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference
ls lr_images/  # 查看LR图像
ls gt_images/  # 查看GT图像

# 运行测试
./test_edge_inference.sh quick
```

### 本地测试配置

测试脚本会自动使用本地目录：
```bash
DEFAULT_LR_DIR="new_features/EdgeInference/lr_images"
DEFAULT_GT_DIR="new_features/EdgeInference/gt_images"
```

---

## 📊 图像要求

### LR图像特点
- 分辨率较低（通常是GT的1/4）
- 可能包含模糊、噪声等退化
- 用作超分辨率的输入

### 示例尺寸对应
| LR尺寸 | GT尺寸 | 放大倍数 |
|--------|--------|----------|
| 128×128 | 512×512 | 4x |
| 256×256 | 1024×1024 | 4x |
| 512×512 | 2048×2048 | 4x |

---

## 💡 提示

1. **配对**: 确保每个LR图像都有对应的GT图像
2. **命名**: LR和GT使用相同的文件名（扩展名可不同）
3. **质量**: LR图像应该是从GT下采样得到的
4. **数量**: 至少放置1张图像用于快速测试

---

## 🔗 相关目录

- **GT图像**: `../gt_images/` - 对应的高分辨率图像
- **输出结果**: 运行测试后会在项目根目录的`outputs/`下生成

---

**创建日期**: 2025-10-15  
**用途**: Edge推理测试数据 - LR输入图像

