# 测试数据说明

## 📁 测试数据目录结构

```
new_features/EdgeInference/
├── lr_images/              # LR (低分辨率) 测试图像
│   ├── README.md
│   └── [你的LR图像]
│
├── gt_images/              # GT (Ground Truth) 测试图像  
│   ├── README.md
│   └── [你的GT图像]
│
└── outputs/                # 测试输出结果（自动生成）
    └── [测试结果]
```

---

## 🎯 快速开始

### 步骤1: 准备测试图像

```bash
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference

# 1. 放置LR图像到lr_images/
cp your/lr/images/*.png lr_images/

# 2. 放置对应的GT图像到gt_images/
cp your/gt/images/*.png gt_images/

# 3. 验证文件
ls lr_images/
ls gt_images/
```

### 步骤2: 运行测试

```bash
# 快速测试（处理1张图）
./test_edge_inference.sh quick

# 完整测试
./test_edge_inference.sh basic
```

### 步骤3: 查看结果

```bash
# 查看输出（在项目根目录）
ls ../../outputs/edge_inference_test/quick/
```

---

## 📊 图像要求

### LR图像（输入）
- **位置**: `lr_images/`
- **分辨率**: 推荐512×512或更小
- **格式**: PNG, JPG, BMP, TIFF
- **用途**: 超分辨率的输入

### GT图像（参考）
- **位置**: `gt_images/`
- **分辨率**: LR的4倍（如LR=512×512，GT=2048×2048）
- **格式**: PNG（推荐无损）
- **用途**: 
  - ⭐ 生成高质量edge map
  - 结果质量评估

---

## 💡 图像对应关系

### 文件命名规则

LR和GT图像必须文件名一致（扩展名可不同）：

```
lr_images/
├── image001.png          ←→  gt_images/image001.png
├── image002.jpg          ←→  gt_images/image002.png
└── test_scene.png        ←→  gt_images/test_scene.png
```

### 分辨率对应

| 场景 | LR尺寸 | GT尺寸 | 倍数 |
|------|--------|--------|------|
| 小图测试 | 128×128 | 512×512 | 4x |
| 标准测试 | 256×256 | 1024×1024 | 4x |
| 推荐配置 | 512×512 | 2048×2048 | 4x |

---

## 🧪 测试配置

### 默认路径（test_edge_inference.sh）

脚本已配置使用本地测试目录：

```bash
DEFAULT_LR_DIR="new_features/EdgeInference/lr_images"
DEFAULT_GT_DIR="new_features/EdgeInference/gt_images"
DEFAULT_OUTPUT_DIR="outputs/edge_inference_test"
```

### 可用测试

```bash
# 1. 快速测试（1张图）
./test_edge_inference.sh quick

# 2. 基础测试（所有图）
./test_edge_inference.sh basic

# 3. 批处理测试
./test_edge_inference.sh batch

# 4. 无edge对比
./test_edge_inference.sh no_edge

# 5. 消融实验
./test_edge_inference.sh black_edge

# 6. LR-based edge
./test_edge_inference.sh lr_edge
```

---

## 📂 输出结构

测试运行后，结果会保存到项目根目录的`outputs/`：

```
outputs/edge_inference_test/
├── quick/                      # quick测试的输出
│   ├── image001_edge.png       # SR结果 ⭐
│   ├── edge_maps/              # Edge可视化
│   │   └── image001_edge.png
│   ├── lr_input/               # 原始LR输入
│   │   └── image001.png
│   ├── gt_hr/                  # GT图像（复制）
│   │   └── image001.png
│   └── edge_inference_*.log    # 详细日志
│
├── basic/                      # basic测试的输出
└── ...
```

---

## 🎨 准备测试数据的方法

### 方法1: 使用现有数据集

如果你有现成的LR-GT配对数据：

```bash
# 直接复制
cp /path/to/dataset/lr/*.png lr_images/
cp /path/to/dataset/gt/*.png gt_images/
```

### 方法2: 从GT生成LR

如果只有GT图像，可以生成LR：

```python
from PIL import Image
import os

gt_dir = "gt_images"
lr_dir = "lr_images"
scale = 4  # 下采样倍数

os.makedirs(lr_dir, exist_ok=True)

for gt_file in os.listdir(gt_dir):
    if gt_file.endswith(('.png', '.jpg', '.jpeg')):
        gt_path = os.path.join(gt_dir, gt_file)
        gt_img = Image.open(gt_path)
        
        # 下采样生成LR
        w, h = gt_img.size
        lr_img = gt_img.resize((w//scale, h//scale), Image.BICUBIC)
        
        # 保存LR
        lr_path = os.path.join(lr_dir, gt_file)
        lr_img.save(lr_path)
        print(f"Generated LR: {lr_path}")
```

### 方法3: 下载测试数据

常用测试数据集：
- Set5, Set14 (标准SR测试集)
- Urban100, BSD100
- DIV2K (高质量配对数据)

---

## ✅ 数据准备检查清单

运行测试前，确认：

- [ ] LR图像已放入 `lr_images/`
- [ ] GT图像已放入 `gt_images/`
- [ ] LR和GT文件名一一对应
- [ ] GT分辨率 = LR分辨率 × 4
- [ ] 至少有1张配对图像用于测试
- [ ] 图像格式正确（PNG/JPG等）

验证命令：
```bash
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference

echo "=== LR图像 ==="
ls -lh lr_images/*.{png,jpg,jpeg} 2>/dev/null | wc -l

echo "=== GT图像 ==="
ls -lh gt_images/*.{png,jpg,jpeg} 2>/dev/null | wc -l

echo "=== 准备就绪！==="
```

---

## 🔧 常见问题

### Q1: 必须提供GT图像吗？

**A**: 强烈推荐！
- ✅ 有GT: 使用GT生成edge（与训练一致，效果最佳）
- ⚠️ 无GT: 使用LR生成edge（domain mismatch，效果可能下降）

### Q2: LR和GT的分辨率必须是4倍吗？

**A**: 推荐4倍关系，这是项目的默认设置。如果比例不同，可能需要调整配置。

### Q3: 文件格式有要求吗？

**A**: 
- **推荐**: PNG（无损）
- **可用**: JPG, JPEG, BMP, TIFF
- **注意**: JPG有压缩损失，可能影响edge质量

### Q4: 需要多少张测试图像？

**A**:
- **快速验证**: 1-3张
- **完整测试**: 10-50张
- **性能评估**: 100+张

---

## 📖 相关文档

- [LR图像说明](lr_images/README.md)
- [GT图像说明](gt_images/README.md)
- [快速开始指南](QUICK_START.md)
- [完整使用文档](README.md)

---

## 🎉 开始测试

数据准备好后，运行：

```bash
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference
./test_edge_inference.sh quick
```

祝测试顺利！ 🚀

---

**创建日期**: 2025-10-15  
**用途**: EdgeInference测试数据准备和说明

