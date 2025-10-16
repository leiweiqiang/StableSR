# Edge L2 Loss 改为 Edge PSNR 迁移说明

## 📅 更新日期：2025-10-16

## 🎯 改动内容

将原来的 **Edge L2 Loss (MSE)** 指标改为 **Edge PSNR** 指标。

### 主要区别

| 项目 | Edge L2 Loss (之前) | Edge PSNR (现在) |
|-----|-------------------|-----------------|
| **度量方式** | MSE (均方误差) | PSNR (峰值信噪比) |
| **计算公式** | `mean((edge1 - edge2)^2)` | `10 * log10(1.0 / MSE)` |
| **值域** | [0, 1] | [0, ∞) dB |
| **解释** | 越小越好 | **越大越好** ⭐ |
| **单位** | 无 | dB (分贝) |
| **格式** | 0.001234 (6位小数) | 25.4321 dB (4位小数) |

### 为什么改为 PSNR？

1. **更直观**：PSNR 是标准的图像质量指标，与图像PSNR一致
2. **易于比较**：可以直接与图像PSNR对比，理解边缘质量
3. **行业标准**：PSNR 是图像处理领域的通用指标
4. **值越大越好**：与SSIM方向一致，更符合直觉

## 📝 修改的文件

### 1. `basicsr/metrics/edge_l2_loss.py`

**主要改动**：
- 类名：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- 函数名：`calculate_edge_l2_loss()` → `calculate_edge_psnr()`
- 计算逻辑：添加PSNR计算 `10 * log10(1.0 / MSE)`
- 返回值：从MSE值改为PSNR值（dB）
- **向后兼容**：保留了 `EdgeL2LossCalculator` 作为别名

```python
# 新的计算逻辑
mse = np.mean((gen_edge - gt_edge) ** 2)
if mse == 0:
    return float('inf')  # 完全相同
psnr = 10 * np.log10(1.0 / mse)
return float(psnr)
```

### 2. `scripts/auto_inference.py`

**改动内容**：
- 导入：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- 变量名：`edge_l2_calculator` → `edge_psnr_calculator`
- 字段名：`edge_l2_loss` → `edge_psnr`
- 平均值字段：`average_edge_l2_loss` → `average_edge_psnr`
- CSV表头：`'Edge L2 Loss'` → `'Edge PSNR (dB)'`
- 格式化：`.6f` → `.4f` (4位小数，符合PSNR惯例)
- 输出：显示 "dB" 单位

### 3. `scripts/generate_metrics_report.py`

**改动内容**：
- 字段名：`edge_l2_loss` → `edge_psnr`
- 平均值字段：`average_edge_l2_loss` → `average_edge_psnr`
- 指标名称：`'Edge L2 Loss'` → `'Edge PSNR'`
- metrics_types：更新列表中的指标名称

### 4. `scripts/recalculate_edge_l2_loss.py`

**改动内容**：
- 导入：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- 函数名：`check_metrics_has_edge_l2()` → `check_metrics_has_edge_psnr()`
- 函数名：`recalculate_edge_l2_loss()` → `recalculate_edge_psnr()`
- 字段名：所有 `edge_l2_loss` → `edge_psnr`
- 输出格式：`.6f` → `.4f dB`
- 提示文本：更新所有用户可见的文本

### 5. `run_auto_inference.sh`

**改动内容**：
- 检查字段：`edge_l2_loss` → `edge_psnr`
- 提示文本：`L2Loss` → `Edge PSNR`
- 标题：`Edge L2 Loss 指标` → `Edge PSNR 指标`
- 所有用户可见的文本都已更新

## 📊 输出格式对比

### metrics.json

**之前**：
```json
{
  "images": [
    {
      "image_name": "0801.png",
      "psnr": 24.5379,
      "ssim": 0.7759,
      "lpips": 0.2655,
      "edge_l2_loss": 0.001234
    }
  ],
  "average_edge_l2_loss": 0.002456
}
```

**现在**：
```json
{
  "images": [
    {
      "image_name": "0801.png",
      "psnr": 24.5379,
      "ssim": 0.7759,
      "lpips": 0.2655,
      "edge_psnr": 29.0891
    }
  ],
  "average_edge_psnr": 26.1234
}
```

### metrics.csv

**之前**：
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
Average,21.0714,0.5853,0.3036,0.002456
```

**现在**：
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
0801.png,24.5379,0.7759,0.2655,29.0891
Average,21.0714,0.5853,0.3036,26.1234
```

### 终端输出

**之前**：
```
Metrics Summary:
  Average PSNR: 21.0714 dB
  Average SSIM: 0.5853
  Average LPIPS: 0.3036
  Average Edge L2 Loss: 0.002456
```

**现在**：
```
Metrics Summary:
  Average PSNR: 21.0714 dB
  Average SSIM: 0.5853
  Average LPIPS: 0.3036
  Average Edge PSNR: 26.1234 dB
```

## 🔄 向后兼容性

为了保证向后兼容，在 `edge_l2_loss.py` 中添加了别名：

```python
# 向后兼容的别名
EdgeL2LossCalculator = EdgePSNRCalculator
default_edge_l2_calculator = default_edge_psnr_calculator
calculate_edge_l2_loss = calculate_edge_psnr
```

这意味着：
- ✅ 旧代码仍然可以使用 `EdgeL2LossCalculator`
- ✅ 但建议新代码使用 `EdgePSNRCalculator`

## 📈 PSNR值的解释

### Edge PSNR 范围参考

| Edge PSNR (dB) | 边缘质量 | 说明 |
|---------------|---------|------|
| > 40 dB | 优秀 ⭐⭐⭐⭐⭐ | 边缘几乎完美匹配 |
| 35-40 dB | 很好 ⭐⭐⭐⭐ | 边缘质量很高 |
| 30-35 dB | 好 ⭐⭐⭐ | 边缘质量良好 |
| 25-30 dB | 一般 ⭐⭐ | 边缘有一定差异 |
| 20-25 dB | 较差 ⭐ | 边缘差异明显 |
| < 20 dB | 差 | 边缘质量不佳 |

### L2 Loss 到 PSNR 的转换

```python
# 如果之前的 Edge L2 Loss = 0.001234
# 则现在的 Edge PSNR = 10 * log10(1.0 / 0.001234)
#                     = 10 * log10(810.37)
#                     ≈ 29.09 dB

# 快速转换表
L2 Loss    →  Edge PSNR
0.0001     →  40.00 dB  (优秀)
0.001      →  30.00 dB  (好)
0.01       →  20.00 dB  (一般)
0.1        →  10.00 dB  (差)
```

## 🎯 使用建议

### 1. 评估模型性能

现在可以这样评估：
```
Image PSNR: 24.54 dB  ← 整体图像质量
Edge PSNR:  29.09 dB  ← 边缘质量
```

如果 Edge PSNR > Image PSNR：边缘保持得比整体图像好（通常情况）  
如果 Edge PSNR ≈ Image PSNR：边缘和整体质量相当  
如果 Edge PSNR < Image PSNR：边缘质量不如整体（需要改进）

### 2. 比较不同模型

```
Model A: Image PSNR 24.5 dB, Edge PSNR 29.1 dB (Δ = +4.6 dB)
Model B: Image PSNR 24.8 dB, Edge PSNR 28.2 dB (Δ = +3.4 dB)
```

- Model A 的边缘增强更明显
- 两个指标都要看才能全面评估

### 3. 优化方向

- 目标：提高 Edge PSNR
- 方法：使用边缘增强技术、edge loss等
- 监控：Edge PSNR 提升的同时确保 Image PSNR 不降低

## ⚠️ 注意事项

### 1. 指标方向改变

**重要**：Edge PSNR 是**越大越好**，与之前的 L2 Loss（越小越好）相反！

- ✅ 正确：Edge PSNR 30 dB > 25 dB，质量更好
- ✗ 错误：不要误认为越小越好

### 2. 旧数据需要重新计算

如果有旧的 metrics.json 文件（包含 edge_l2_loss），需要重新计算：

```bash
# 方法1：运行 run_auto_inference.sh 选项1
# 会自动检查并重新计算

# 方法2：手动重新计算
python scripts/recalculate_edge_l2_loss.py \
    validation_results/exp/edge/epochs_27 \
    /path/to/gt_images \
    --force
```

### 3. 脚本名称未改变

注意：`recalculate_edge_l2_loss.py` 脚本名称保持不变（避免破坏现有脚本），但内部已改为计算 Edge PSNR。

### 4. CSV表头改变

新生成的 CSV 文件表头为 `Edge PSNR (dB)`，与旧文件 `Edge L2 Loss` 不兼容。  
合并新旧数据时需要注意列名。

## 🔍 验证改动

### 检查单个文件

```bash
# 查看 metrics.json
cat metrics.json | grep edge_psnr

# 应该看到
"edge_psnr": 29.0891
"average_edge_psnr": 26.1234
```

### 检查 CSV

```bash
head -1 metrics.csv
# 应该看到
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
```

### 运行测试

```bash
# 测试计算器
python -c "
from basicsr.metrics.edge_l2_loss import EdgePSNRCalculator
import numpy as np

calc = EdgePSNRCalculator()
img1 = np.zeros((256, 256, 3), dtype=np.uint8)
img2 = np.zeros((256, 256, 3), dtype=np.uint8)

psnr = calc.calculate_from_arrays(img1, img2)
print(f'Edge PSNR: {psnr:.4f} dB')
# 完全相同的图片应该返回 inf
"
```

## 📚 相关指标对比

| 指标 | 度量内容 | 值域 | 方向 | 单位 |
|-----|---------|------|------|------|
| **Image PSNR** | 整体图像质量 | [0, ∞) | 越大越好 ↑ | dB |
| **SSIM** | 结构相似度 | [0, 1] | 越大越好 ↑ | 无 |
| **LPIPS** | 感知相似度 | [0, ∞) | 越小越好 ↓ | 无 |
| **Edge PSNR** | 边缘质量 | [0, ∞) | **越大越好 ↑** | **dB** |

## 🎉 改进总结

### 优势
1. ✅ 更符合行业标准（PSNR是标准指标）
2. ✅ 更直观易懂（与图像PSNR一致）
3. ✅ 便于比较（可与图像PSNR直接对比）
4. ✅ 方向一致（与SSIM都是越大越好）

### 兼容性
1. ✅ 保留了向后兼容的别名
2. ✅ 自动重新计算旧数据
3. ✅ 脚本名称未改变
4. ✅ 完整的迁移文档

---

**✅ 所有改动已完成，可以正常使用！**

现在运行推理时会自动计算 Edge PSNR 指标，并以 dB 为单位显示。

