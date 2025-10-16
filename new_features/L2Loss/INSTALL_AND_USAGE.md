# Edge L2 Loss - 安装与使用指南

## 📦 安装状态

Edge L2 Loss 功能已完全集成到项目中，无需额外安装。

## 📂 文件位置

### 核心代码
- **主要实现**: `basicsr/metrics/edge_l2_loss.py`
  - `EdgeL2LossCalculator` 类
  - `calculate_edge_l2_loss()` 便捷函数

### 系统集成
- **推理脚本**: `scripts/auto_inference.py` (已集成)
- **报告生成**: `scripts/generate_metrics_report.py` (已集成)

### 文档和测试（本目录）
- 📖 `README.md` - 目录说明和快速导航
- 📖 `EDGE_L2_LOSS_QUICKSTART.md` - 快速入门 ⭐
- 📖 `EDGE_L2_LOSS_README.md` - 完整文档
- 📖 `EDGE_L2_LOSS_SUMMARY.md` - 实现总结
- 🧪 `test_edge_l2_loss.py` - 测试脚本

## 🚀 快速使用

### 1. 自动计算（推荐）

在运行推理时，Edge L2 Loss 会自动计算：

```bash
python scripts/auto_inference.py \
    --ckpt path/to/checkpoint.ckpt \
    --init_img path/to/lr_images \
    --gt_img path/to/gt_images \
    --calculate_metrics
```

结果会自动保存到：
- `metrics.json` - 包含 `edge_l2_loss` 字段
- `metrics.csv` - 包含 "Edge L2 Loss" 列

### 2. 在 Python 代码中使用

```python
from basicsr.metrics.edge_l2_loss import calculate_edge_l2_loss
import cv2

# 读取图片
gen_img = cv2.imread('generated_image.png')
gt_img = cv2.imread('ground_truth.png')

# 计算 Edge L2 Loss
loss = calculate_edge_l2_loss(gen_img, gt_img)

print(f"Edge L2 Loss: {loss:.6f}")
# 值 < 0.001 表示边缘非常相似
```

### 3. 生成综合报告

```bash
python scripts/generate_metrics_report.py \
    validation_results/ \
    --output comprehensive_report.csv
```

## 🧪 运行测试

```bash
# 确保在正确的 conda 环境中
conda activate sr_infer  # 或你的环境名

# 从项目根目录运行测试
cd /root/dp/StableSR_Edge_v3
python new_features/L2Loss/test_edge_l2_loss.py
```

预期输出：
```
============================================================
EdgeL2LossCalculator Test Suite
============================================================

Test 1: Calculate Edge L2 Loss from numpy arrays
✓ Edge L2 Loss between img1 and img2: 0.000234
...

============================================================
Test Results: 5/5 passed
✓ All tests passed!
============================================================
```

## 📊 输出格式

### metrics.json
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
  "average_psnr": 21.0714,
  "average_ssim": 0.5853,
  "average_lpips": 0.3036,
  "average_edge_l2_loss": 0.002456,
  "total_images": 10
}
```

### metrics.csv
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
0802.png,25.7358,0.6455,0.2015,0.001567
Average,21.0714,0.5853,0.3036,0.002456
```

### 综合报告 CSV
会在现有的 PSNR、SSIM、LPIPS 后增加 **Edge L2 Loss** 指标块。

## 💡 结果解读

| Edge L2 Loss | 含义 | 质量评价 |
|-------------|------|---------|
| 0.0 | 边缘完全相同 | 完美 |
| < 0.001 | 边缘非常相似 | 优秀 ⭐ |
| 0.001 - 0.01 | 边缘相似 | 良好 |
| 0.01 - 0.05 | 有一定差异 | 一般 |
| > 0.05 | 差异较大 | 需改进 |

**注意**: 值越小越好

## 🔧 高级用法

### 自定义 Edge 检测参数

```python
from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator

# 创建自定义参数的计算器
calculator = EdgeL2LossCalculator(
    gaussian_kernel_size=(7, 7),      # 更大的高斯核
    gaussian_sigma=2.0,                # 更强的模糊
    canny_threshold_lower_factor=0.5,  # 更敏感的边缘检测
    canny_threshold_upper_factor=1.5
)

# 使用自定义计算器
loss = calculator.calculate_from_files('gen.png', 'gt.png')
```

### 批量处理 PyTorch Tensors

```python
import torch
from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator

calculator = EdgeL2LossCalculator()

# 批量tensor (B, C, H, W)，范围 [-1, 1]
gen_batch = torch.randn(4, 3, 512, 512) * 2 - 1
gt_batch = torch.randn(4, 3, 512, 512) * 2 - 1

# 计算整个batch的平均loss
loss = calculator.calculate_from_tensors(
    gen_batch, 
    gt_batch, 
    normalize_range='[-1,1]'
)
```

## ❓ 常见问题

**Q: 需要单独安装吗？**  
A: 不需要，已经集成到项目中。

**Q: 会影响原有功能吗？**  
A: 不会，完全向后兼容。

**Q: 计算速度慢吗？**  
A: 很快，512x512 图片约 10-50ms。

**Q: 必须使用吗？**  
A: 不必须，如果计算失败会自动跳过。

**Q: 和 LPIPS 有什么区别？**  
A: LPIPS 度量整体感知质量，Edge L2 Loss 专注边缘保真度。

## 📚 更多文档

- 🚀 快速入门: `EDGE_L2_LOSS_QUICKSTART.md`
- 📖 完整文档: `EDGE_L2_LOSS_README.md`
- 📝 实现总结: `EDGE_L2_LOSS_SUMMARY.md`
- 📂 目录说明: `README.md`

## 🐛 问题反馈

如遇到问题，请检查：
1. ✅ conda 环境是否正确激活
2. ✅ 依赖包是否安装（cv2, numpy, torch）
3. ✅ 图片路径是否正确
4. ✅ 查看错误日志

---

**准备就绪！现在就可以使用 Edge L2 Loss 功能了！** 🎉

