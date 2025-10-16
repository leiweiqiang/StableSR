# Edge PSNR 快速参考

## 一句话总结
**Edge PSNR** 指标用于评估超分辨率图像与GT图像之间的**边缘相似度**，单位为 dB，**值越大越好**。

## 快速使用

### 自动计算（推荐）
```bash
python scripts/auto_inference.py \
    --ckpt checkpoint.ckpt \
    --init_img lr_images/ \
    --gt_img gt_images/ \
    --calculate_metrics
```

### 在代码中使用
```python
from basicsr.metrics.edge_l2_loss import calculate_edge_psnr
import cv2

gen_img = cv2.imread('generated.png')
gt_img = cv2.imread('ground_truth.png')

psnr = calculate_edge_psnr(gen_img, gt_img)
print(f"Edge PSNR: {psnr:.4f} dB")
```

## 如何解读

| Edge PSNR | 质量评价 |
|-----------|---------|
| > 40 dB | 优秀 ⭐⭐⭐⭐⭐ |
| 35-40 dB | 很好 ⭐⭐⭐⭐ |
| 30-35 dB | 好 ⭐⭐⭐ |
| 25-30 dB | 一般 ⭐⭐ |
| 20-25 dB | 较差 ⭐ |
| < 20 dB | 差 |

**重要**：值越大越好！↑

## 计算原理

```
生成图片 → Edge Map → 
                       ├→ MSE → PSNR = 10*log10(1/MSE)
GT图片   → Edge Map →
```

- Edge Map: 使用Canny边缘检测
- MSE: 均方误差
- PSNR: 峰值信噪比，单位 dB

## 与 Image PSNR 的关系

```
Image PSNR: 24.54 dB  ← 整体图像质量
Edge PSNR:  29.09 dB  ← 边缘质量

Edge PSNR > Image PSNR: 边缘保持得好 ✓
```

## 输出示例

### metrics.csv
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
0801.png,24.5379,0.7759,0.2655,29.0891
Average,21.0714,0.5853,0.3036,26.1234
```

### 终端输出
```
Metrics Summary:
  Average PSNR: 21.0714 dB
  Average Edge PSNR: 26.1234 dB  ← 新增
```

## 常见问题

**Q: Edge PSNR 和 Image PSNR 有什么区别？**  
A: Image PSNR 评估整体图像，Edge PSNR 专门评估边缘质量。

**Q: 为什么 Edge PSNR 通常比 Image PSNR 高？**  
A: 边缘是稀疏的，大部分区域（无边缘）的误差为0，平均MSE较小，因此PSNR较高。

**Q: 值越大越好还是越小越好？**  
A: **越大越好！** PSNR 是 Peak Signal-to-Noise Ratio，值越大表示信噪比越高，质量越好。

**Q: 和之前的 Edge L2 Loss 有什么关系？**  
A: Edge PSNR = 10 * log10(1.0 / Edge_L2_Loss)，是同一个 MSE 的不同表示方式。

## API 快速参考

```python
from basicsr.metrics.edge_l2_loss import EdgePSNRCalculator

# 初始化
calc = EdgePSNRCalculator()

# 从numpy数组
psnr = calc.calculate_from_arrays(img1, img2, input_format='BGR')

# 从文件
psnr = calc.calculate_from_files('path1.png', 'path2.png')

# 从tensor
psnr = calc.calculate_from_tensors(tensor1, tensor2, normalize_range='[-1,1]')

# 便捷调用
psnr = calc(img1, img2)
```

## 更多信息

- 📖 完整文档: `new_features/L2Loss/EDGE_L2_LOSS_README.md`
- 🔄 迁移说明: `EDGE_L2_TO_PSNR_MIGRATION.md`
- 💻 核心代码: `basicsr/metrics/edge_l2_loss.py`

