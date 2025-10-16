# Edge Overlap 指标说明

## 📅 添加时间：2025-10-16

---

## 🎯 指标说明

**Edge Overlap（边缘重叠率）**是用于评估生成图片与GT图片边缘匹配程度的指标。

### 定义

```
Edge Overlap = 交集(生成边缘 ∩ GT边缘) / GT边缘总数
```

### 计算步骤

1. **生成 edge maps**：使用 EdgeMapGenerator 从两张图片生成边缘图
2. **二值化**：将边缘图转换为二值图像（边缘=1，背景=0）
3. **计算交集**：找到两张边缘图的重叠部分
4. **计算比率**：交集面积 / GT边缘总面积

### 公式

```python
gen_edge_bin = (gen_edge > 0.5).astype(uint8)  # 生成图片的二值边缘
gt_edge_bin = (gt_edge > 0.5).astype(uint8)    # GT图片的二值边缘

intersection = gen_edge_bin AND gt_edge_bin     # 交集
overlap = sum(intersection) / sum(gt_edge_bin)  # 重叠率
```

---

## 📊 指标特性

| 属性 | 说明 |
|-----|------|
| **值域** | [0, 1] |
| **单位** | 无（比率） |
| **方向** | ↑ **越大越好** |
| **含义** | 生成边缘覆盖了多少GT边缘 |

### 值的解读

| Edge Overlap | 含义 | 质量评价 |
|-------------|------|---------|
| 1.0 | 完全覆盖GT所有边缘 | 完美 ⭐⭐⭐⭐⭐ |
| 0.9 - 1.0 | 覆盖了90%以上的GT边缘 | 优秀 ⭐⭐⭐⭐⭐ |
| 0.8 - 0.9 | 覆盖了80-90%的GT边缘 | 很好 ⭐⭐⭐⭐ |
| 0.7 - 0.8 | 覆盖了70-80%的GT边缘 | 好 ⭐⭐⭐ |
| 0.6 - 0.7 | 覆盖了60-70%的GT边缘 | 一般 ⭐⭐ |
| < 0.6 | 覆盖不足60%的GT边缘 | 较差 ⭐ |

---

## 🔄 与其他指标的关系

| 指标 | 度量内容 | 值域 | 方向 | 单位 |
|-----|---------|------|------|------|
| Image PSNR | 整体图像质量 | [0, ∞) | ↑ 越大越好 | dB |
| SSIM | 结构相似度 | [0, 1] | ↑ 越大越好 | 无 |
| LPIPS | 感知相似度 | [0, ∞) | ↓ 越小越好 | 无 |
| **Edge PSNR** | 边缘质量 | [0, ∞) | ↑ 越大越好 | dB |
| **Edge Overlap** | 边缘覆盖率 | [0, 1] | ↑ **越大越好** | 无 |

### Edge PSNR vs Edge Overlap

两者都评估边缘，但侧重点不同：

- **Edge PSNR**：边缘的**精确度**（边缘位置和强度的误差）
- **Edge Overlap**：边缘的**召回率**（捕获了多少GT边缘）

**理想情况**：
- Edge PSNR 高：生成的边缘位置准确
- Edge Overlap 高：生成的边缘覆盖全面

**典型组合解读**：
```
组合1: Edge PSNR 高 + Edge Overlap 高
  → 边缘质量优秀（准确且全面）✓✓

组合2: Edge PSNR 高 + Edge Overlap 低
  → 生成的边缘准确，但遗漏了部分GT边缘
  → 可能过于保守

组合3: Edge PSNR 低 + Edge Overlap 高
  → 覆盖了大部分边缘，但位置不够准确
  → 可能有误检

组合4: Edge PSNR 低 + Edge Overlap 低
  → 边缘质量差（需要改进）✗✗
```

---

## 💻 使用方法

### 自动计算（推荐）

```bash
# 运行推理，自动计算所有指标
python scripts/auto_inference.py \
    --ckpt checkpoint.ckpt \
    --init_img lr_images/ \
    --gt_img gt_images/ \
    --calculate_metrics
```

### 在代码中使用

```python
from basicsr.metrics.edge_overlap import calculate_edge_overlap
import cv2

# 读取图片
gen_img = cv2.imread('generated.png')
gt_img = cv2.imread('ground_truth.png')

# 计算 Edge Overlap
overlap = calculate_edge_overlap(gen_img, gt_img)

print(f"Edge Overlap: {overlap:.4f}")
# 值越大越好，1.0 表示完全覆盖
```

### API 参考

```python
from basicsr.metrics.edge_overlap import EdgeOverlapCalculator

# 初始化
calc = EdgeOverlapCalculator()

# 从 numpy 数组
overlap = calc.calculate_from_arrays(img1, img2, input_format='BGR')

# 从文件
overlap = calc.calculate_from_files('path1.png', 'path2.png')

# 从 tensor
overlap = calc.calculate_from_tensors(tensor1, tensor2, normalize_range='[-1,1]')

# 便捷调用
overlap = calc(img1, img2)
```

---

## 📁 输出格式

### metrics.json

```json
{
  "images": [
    {
      "image_name": "0801.png",
      "psnr": 24.5379,
      "ssim": 0.7759,
      "lpips": 0.2655,
      "edge_psnr": 29.0891,
      "edge_overlap": 0.8523     ← 新增
    }
  ],
  "average_psnr": 21.0714,
  "average_ssim": 0.5853,
  "average_lpips": 0.3036,
  "average_edge_psnr": 26.1234,
  "average_edge_overlap": 0.7891  ← 新增
}
```

### metrics.csv

```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB),Edge Overlap
0801.png,24.5379,0.7759,0.2655,29.0891,0.8523
Average,21.0714,0.5853,0.3036,26.1234,0.7891
```

### 综合报告 CSV

现在包含5个指标块：
1. PSNR
2. SSIM
3. LPIPS
4. Edge PSNR
5. **Edge Overlap** ← 新增

---

## 🎨 使用示例

### 示例1：评估边缘质量

```
结果：
  Edge PSNR: 29.09 dB    ← 边缘精确度
  Edge Overlap: 0.85     ← 边缘覆盖率

解读：
  ✓ Edge PSNR 29 dB：边缘质量好
  ✓ Edge Overlap 0.85：覆盖了85%的GT边缘
  结论：边缘重建效果很好
```

### 示例2：对比不同模型

```
模型A:
  Edge PSNR: 30.5 dB
  Edge Overlap: 0.82

模型B:
  Edge PSNR: 28.3 dB
  Edge Overlap: 0.89

分析：
  - 模型A 边缘更精确（PSNR更高）
  - 模型B 边缘更全面（Overlap更高）
  - 根据应用选择：精确性 vs 完整性
```

### 示例3：诊断问题

```
情况1：Edge PSNR 高，Edge Overlap 低
  → 可能原因：模型过于保守，只生成确定的边缘
  → 改进方向：增加边缘检测的敏感度

情况2：Edge PSNR 低，Edge Overlap 高
  → 可能原因：模型生成了很多边缘，但位置不准
  → 改进方向：提高边缘定位精度

情况3：两者都低
  → 边缘重建能力不足
  → 改进方向：增强边缘增强模块
```

---

## ⚙️ 技术细节

### Edge Map 生成

使用 `EdgeMapGenerator` （与训练一致）：
- Canny 边缘检测
- 阈值：100, 200
- 输出：灰度边缘图，值域 [0, 1]

### 二值化阈值

```python
edge_bin = (edge > 0.5).astype(uint8)
```

- 阈值 0.5：edge map 中大于 0.5 的为边缘
- 边缘图值域是 [0, 1]，边缘处接近1

### 分母说明

```python
overlap = sum(intersection) / sum(gt_edge_bin)
```

**分母是 GT 边缘总数**，而不是生成边缘总数。

这样设计的原因：
- 评估的是"召回率"（覆盖了多少GT边缘）
- 而不是"精确率"（生成的边缘有多少是对的）

### 边界情况

```python
if gt_edge_count == 0:
    return 0.0  # GT没有边缘
```

如果GT图片没有边缘（如纯色图），返回 0。

---

## 📈 实际应用

### 评估模型性能

```
模型训练进度：
  Epoch 27:  Edge Overlap = 0.72  ← 初期
  Epoch 55:  Edge Overlap = 0.78  ← 改进
  Epoch 83:  Edge Overlap = 0.85  ← 继续改进
  Epoch 111: Edge Overlap = 0.87  ← 趋于稳定

结论：模型在边缘捕获能力上持续改进
```

### 对比不同设置

```
Edge 模式:      Edge Overlap = 0.87  ← 最好
No-Edge 模式:   Edge Overlap = 0.73
Dummy-Edge模式: Edge Overlap = 0.71

结论：使用真实边缘能更好地捕获GT边缘
```

### 与 SSIM 对比

```
图片A:
  SSIM: 0.85
  Edge Overlap: 0.90
  → 整体结构和边缘都很好

图片B:
  SSIM: 0.82
  Edge Overlap: 0.65
  → 整体结构还行，但边缘捕获不足
```

---

## ✅ 优势

1. **互补性强**：与 Edge PSNR 互补
   - PSNR 评估精确度
   - Overlap 评估完整性

2. **直观易懂**：
   - 0.85 = 覆盖了 85% 的GT边缘
   - 百分比形式，易于理解

3. **实用性高**：
   - 诊断边缘检测问题
   - 优化模型参数
   - 对比不同方法

4. **计算高效**：
   - 与 Edge PSNR 共用 edge map
   - 开销极小（几乎不增加时间）

---

## 🔧 自定义参数

```python
from basicsr.metrics.edge_overlap import EdgeOverlapCalculator

# 使用自定义的边缘检测参数
calc = EdgeOverlapCalculator(
    gaussian_kernel_size=(7, 7),      # 更大的高斯核
    gaussian_sigma=2.0,                # 更强的模糊
    canny_threshold_lower_factor=0.5,  # 更敏感的阈值
    canny_threshold_upper_factor=1.5
)

overlap = calc.calculate_from_files('gen.png', 'gt.png')
```

---

## 📚 完整指标总结

现在系统支持 **6 个指标**：

| 指标 | 度量对象 | 值域 | 方向 | 单位 |
|-----|---------|------|------|------|
| 1. PSNR | 图像整体 | [0, ∞) | ↑ | dB |
| 2. SSIM | 图像结构 | [0, 1] | ↑ | 无 |
| 3. LPIPS | 图像感知 | [0, ∞) | ↓ | 无 |
| 4. Edge PSNR | 边缘质量 | [0, ∞) | ↑ | dB |
| 5. **Edge Overlap** | **边缘覆盖** | **[0, 1]** | **↑** | **无** |

全方位评估图像质量！

---

## 🎯 输出示例

### 终端输出

```
============================================================
Metrics Summary:
  Total images: 10
  Average PSNR: 21.0714 dB
  Average SSIM: 0.5853
  Average LPIPS: 0.3036
  Average Edge PSNR: 26.1234 dB
  Average Edge Overlap: 0.7891        ← 新增
============================================================
```

### CSV 文件

```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB),Edge Overlap
0801.png,24.5379,0.7759,0.2655,29.0891,0.8523
0802.png,25.7358,0.6455,0.2015,27.3456,0.7654
Average,21.0714,0.5853,0.3036,26.1234,0.7891
```

---

## ⚠️ 注意事项

### 1. 与 Precision/Recall 的区别

**Edge Overlap 类似召回率 (Recall)**：
```
Overlap = 捕获的GT边缘 / GT边缘总数 = Recall
```

如果需要精确率 (Precision)，公式应该是：
```
Precision = 捕获的GT边缘 / 生成边缘总数
```

当前实现是 **Recall 导向**。

### 2. 二值化阈值

当前使用 0.5 作为二值化阈值：
```python
edge_bin = (edge > 0.5)
```

可以根据需要调整，但建议保持一致。

### 3. 空图片处理

如果GT图片没有边缘（如纯色），返回 0.0。

---

## 📖 技术实现

### 核心文件

- **实现**：`basicsr/metrics/edge_overlap.py`
- **类名**：`EdgeOverlapCalculator`
- **函数**：`calculate_edge_overlap()`

### 集成位置

- `scripts/auto_inference.py` - 自动计算
- `scripts/generate_metrics_report.py` - 报告生成
- `scripts/recalculate_edge_l2_loss.py` - 重新计算

### 依赖

- `EdgeMapGenerator`：生成边缘图
- NumPy：数组运算
- OpenCV：图像读取

---

## 🚀 快速使用

### 最简单的方式

```bash
./run_auto_inference.sh
# 选择：1
# 4次回车完成

# Edge Overlap 自动计算并保存！
```

### 查看结果

```bash
# 查看 JSON
cat validation_results/.../metrics.json | grep edge_overlap

# 查看 CSV
cat validation_results/.../metrics.csv | head -1

# 应该看到 Edge Overlap 列
```

---

## ✅ 总结

### 新增内容

- [x] EdgeOverlapCalculator 类
- [x] 集成到 auto_inference.py
- [x] 集成到 generate_metrics_report.py
- [x] 集成到 recalculate脚本
- [x] 更新 run_auto_inference.sh
- [x] 完整文档

### 现有指标

现在系统完整支持 **5 个指标**：
1. PSNR（图像）
2. SSIM
3. LPIPS
4. Edge PSNR（边缘精确度）
5. **Edge Overlap**（边缘覆盖率）← 新增

**全方位评估图像和边缘质量！** 🎉

---

**更多信息**：查看 `new_features/L2Loss/` 目录中的其他文档

