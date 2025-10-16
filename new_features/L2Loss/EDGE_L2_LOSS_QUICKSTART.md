# Edge L2 Loss - 快速开始

## 一句话总结
新增了**Edge L2 Loss (MSE)**指标，用于评估超分辨率图像与GT图像之间的边缘相似度。

## 使用方法

### 方法1: 在推理时自动计算（推荐）

```bash
python scripts/auto_inference.py \
    --ckpt path/to/checkpoint.ckpt \
    --init_img path/to/lr_images \
    --gt_img path/to/gt_images \
    --calculate_metrics
```

结果自动保存到：
- `metrics.json` - 包含 `edge_l2_loss` 和 `average_edge_l2_loss`
- `metrics.csv` - 增加了 "Edge L2 Loss" 列

### 方法2: 在Python代码中使用

```python
from basicsr.metrics.edge_l2_loss import calculate_edge_l2_loss
import cv2

# 读取图片
gen_img = cv2.imread('generated.png')
gt_img = cv2.imread('ground_truth.png')

# 计算Edge L2 Loss
loss = calculate_edge_l2_loss(gen_img, gt_img)

print(f"Edge L2 Loss: {loss:.6f}")
# 值越小越好，< 0.001 表示边缘非常相似
```

### 方法3: 生成综合报告

```bash
python scripts/generate_metrics_report.py \
    validation_results/ \
    -o comprehensive_report.csv
```

CSV报告将包含4个指标：PSNR, SSIM, LPIPS, Edge L2 Loss

## 如何解读结果

| Edge L2 Loss 值 | 含义 |
|----------------|------|
| 0.0 | 边缘完全相同 |
| < 0.001 | 边缘非常相似 ✓ |
| 0.001 - 0.01 | 边缘相似 |
| 0.01 - 0.05 | 有一定差异 |
| > 0.05 | 差异较大 ✗ |

## 技术原理

1. 使用Canny边缘检测从两张图片生成edge map
2. 计算两张edge map之间的均方误差(MSE)
3. 结果归一化到[0, 1]范围

## 更多信息

- 📖 完整文档: `EDGE_L2_LOSS_README.md`
- 📝 实现总结: `EDGE_L2_LOSS_SUMMARY.md`
- 🧪 测试脚本: `scripts/test_edge_l2_loss.py`
- 💻 核心代码: `basicsr/metrics/edge_l2_loss.py`

## API 快速参考

```python
from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator

# 初始化
calc = EdgeL2LossCalculator()

# 从numpy数组 (cv2读取的图片)
loss = calc.calculate_from_arrays(img1, img2, input_format='BGR')

# 从文件路径
loss = calc.calculate_from_files('path1.png', 'path2.png')

# 从PyTorch tensor
loss = calc.calculate_from_tensors(tensor1, tensor2, normalize_range='[-1,1]')

# 便捷调用（自动检测类型）
loss = calc(img1, img2)
```

## 示例输出

### metrics.csv
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
0802.png,25.7358,0.6455,0.2015,0.001567
Average,21.0714,0.5853,0.3036,0.002456
```

### Terminal 输出
```
============================================================
Metrics Summary:
  Total images: 10
  Average PSNR: 21.0714 dB
  Average SSIM: 0.5853
  Average LPIPS: 0.3036
  Average Edge L2 Loss: 0.002456
============================================================
```

## 常见问题

**Q: 会影响现有功能吗？**  
A: 不会，完全向后兼容，只是新增了一个指标。

**Q: 计算慢吗？**  
A: 很快，每张512x512图片约10-50ms。

**Q: 必须计算吗？**  
A: 不必须，如果计算失败会自动跳过，不影响其他指标。

**Q: 和LPIPS有什么区别？**  
A: LPIPS度量整体感知质量，Edge L2 Loss专注边缘保真度。

## 开始使用

现在就可以运行推理脚本，Edge L2 Loss会自动计算并保存到结果中！

```bash
# 立即试用
python scripts/auto_inference.py \
    --ckpt your_checkpoint.ckpt \
    --init_img your_lr_images/ \
    --gt_img your_gt_images/ \
    --calculate_metrics
```

