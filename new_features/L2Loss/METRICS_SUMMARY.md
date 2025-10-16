# StableSR Edge 指标系统总结

## 📊 支持的指标（共5个）

| # | 指标名称 | 度量对象 | 值域 | 方向 | 单位 | 说明 |
|---|---------|---------|------|------|------|------|
| 1 | **PSNR** | 整体图像 | [0, ∞) | ↑ 越大越好 | dB | 峰值信噪比 |
| 2 | **SSIM** | 图像结构 | [0, 1] | ↑ 越大越好 | 无 | 结构相似度 |
| 3 | **LPIPS** | 感知质量 | [0, ∞) | ↓ 越小越好 | 无 | 感知距离 |
| 4 | **Edge PSNR** | 边缘质量 | [0, ∞) | ↑ 越大越好 | dB | 边缘精确度 |
| 5 | **Edge Overlap** | 边缘覆盖 | [0, 1] | ↑ 越大越好 | 无 | 边缘召回率 |

---

## 🎯 指标作用

### 整体质量指标

**PSNR + SSIM + LPIPS** → 评估整体图像质量
- PSNR：像素级误差
- SSIM：结构相似性
- LPIPS：人眼感知质量

### 边缘质量指标

**Edge PSNR + Edge Overlap** → 评估边缘重建质量
- Edge PSNR：边缘位置和强度的精确度
- Edge Overlap：捕获了多少GT边缘（召回率）

---

## 📈 典型值参考

### PSNR（图像）
- > 30 dB：优秀
- 25-30 dB：好
- 20-25 dB：一般
- < 20 dB：较差

### SSIM
- > 0.9：优秀
- 0.8-0.9：好
- 0.7-0.8：一般
- < 0.7：较差

### LPIPS
- < 0.1：优秀
- 0.1-0.2：好
- 0.2-0.4：一般
- > 0.4：较差

### Edge PSNR
- > 35 dB：优秀
- 30-35 dB：好
- 25-30 dB：一般
- < 25 dB：较差

### Edge Overlap
- > 0.9：优秀
- 0.8-0.9：好
- 0.7-0.8：一般
- < 0.7：较差

---

## 💡 指标组合解读

### 组合1：整体好 + 边缘好

```
PSNR: 28 dB ✓
SSIM: 0.88 ✓
Edge PSNR: 33 dB ✓
Edge Overlap: 0.89 ✓

→ 模型表现优秀，整体和边缘都很好
```

### 组合2：整体好 + 边缘一般

```
PSNR: 27 dB ✓
SSIM: 0.86 ✓
Edge PSNR: 24 dB ⚠
Edge Overlap: 0.68 ⚠

→ 整体质量不错，但边缘重建需要改进
```

### 组合3：整体一般 + 边缘好

```
PSNR: 22 dB ⚠
SSIM: 0.75 ⚠
Edge PSNR: 31 dB ✓
Edge Overlap: 0.86 ✓

→ 边缘保持得好，但整体质量一般
→ 可能纹理/颜色重建不足
```

---

## 📁 输出文件格式

### metrics.json
```json
{
  "images": [
    {
      "image_name": "0801.png",
      "psnr": 24.5379,           // Image PSNR
      "ssim": 0.7759,            // SSIM
      "lpips": 0.2655,           // LPIPS
      "edge_psnr": 29.0891,      // Edge PSNR
      "edge_overlap": 0.8523     // Edge Overlap
    }
  ],
  "average_psnr": 21.0714,
  "average_ssim": 0.5853,
  "average_lpips": 0.3036,
  "average_edge_psnr": 26.1234,
  "average_edge_overlap": 0.7891,
  "total_images": 10
}
```

### metrics.csv
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB),Edge Overlap
0801.png,24.5379,0.7759,0.2655,29.0891,0.8523
0802.png,25.7358,0.6455,0.2015,27.3456,0.7654
...
Average,21.0714,0.5853,0.3036,26.1234,0.7891
```

### 综合报告 CSV

```csv
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,...
,,,dummy edge,edge,no edge,...
PSNR,Average,20.92,20.26,20.34,20.28,...
SSIM,Average,0.5955,0.5406,0.5461,0.5453,...
LPIPS,Average,0.2935,0.3373,0.3366,0.3344,...
Edge PSNR,Average,26.12,25.34,26.78,25.91,...
Edge Overlap,Average,0.79,0.73,0.81,0.75,...
```

---

## 🚀 使用方法

### 自动计算（推荐）

```bash
./run_auto_inference.sh
# 选择：1
# 4次回车

# 所有5个指标自动计算！
```

### 手动计算

```python
from basicsr.metrics.edge_overlap import calculate_edge_overlap
import cv2

gen_img = cv2.imread('generated.png')
gt_img = cv2.imread('ground_truth.png')

overlap = calculate_edge_overlap(gen_img, gt_img)
print(f"Edge Overlap: {overlap:.4f}")
```

---

## 📖 相关文档

### Edge Overlap 文档
- `new_features/L2Loss/EDGE_OVERLAP_README.md` ⭐

### 其他指标文档
- `new_features/L2Loss/EDGE_PSNR_QUICKREF.md` - Edge PSNR
- `new_features/L2Loss/USER_GUIDE.md` - 用户指南
- `new_features/L2Loss/COMPLETE_CHANGELOG.md` - 完整更新日志

---

## ✅ 完整功能

### 核心实现
- [x] EdgePSNRCalculator - 边缘 PSNR
- [x] EdgeOverlapCalculator - 边缘重叠率
- [x] 5个完整指标支持

### 系统集成
- [x] auto_inference.py - 自动计算
- [x] generate_metrics_report.py - 报告生成
- [x] recalculate脚本 - 补充计算
- [x] run_auto_inference.sh - 智能检查

### 智能功能
- [x] 可选重新计算
- [x] 推理前确认
- [x] 智能跳过
- [x] CSV自动排序
- [x] 批量检查

---

**🎉 现在支持5个完整指标，全方位评估超分辨率质量！**

**立即使用**：
```bash
conda activate sr_infer
./run_auto_inference.sh
```

所有指标自动计算并保存！✨

