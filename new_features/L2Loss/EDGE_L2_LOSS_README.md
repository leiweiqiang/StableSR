# Edge L2 Loss (MSE) 指标实现文档

## 概述

本文档描述了新增的 **Edge L2 Loss (MSE)** 指标的实现，该指标用于评估超分辨率图像与真值图像之间的边缘保真度。

## 需求分析

### 目标
在CSV报告中增加一个 L2 loss (MSE) between images 的指标，用于量化生成图像和GT图像的边缘相似度。

### 输入
1. **第一张图片**：推理结果图片
2. **第二张图片**：Ground Truth (GT) 图片

### 处理流程
1. 使用 `EdgeMapGenerator` (来自 `basicsr/utils/edge_utils.py`) 从两张输入图片生成边缘图
2. 计算两张边缘图之间的 Mean Squared Error (MSE)
3. 将结果保存到 metrics.json 和 metrics.csv 文件中

### 技术细节
- **Edge Map 生成**：使用 Canny 边缘检测算法（阈值：100, 200）
- **L2 Loss 计算**：MSE = mean((edge_map1 - edge_map2)^2)
- **值域**：Edge maps 归一化到 [0, 1]，因此 MSE 范围也是 [0, 1]
- **解释**：值越小表示边缘越相似，0 表示完全相同

## 实现的文件

### 1. `basicsr/metrics/edge_l2_loss.py` (新建)

**核心类：`EdgeL2LossCalculator`**

这是计算 Edge L2 Loss 的主要类，支持多种输入格式。

#### 主要方法

```python
# 1. 从numpy数组计算 (适用于cv2读取的图片)
loss = calculator.calculate_from_arrays(gen_img, gt_img, input_format='BGR')

# 2. 从文件路径计算
loss = calculator.calculate_from_files(gen_img_path, gt_img_path)

# 3. 从PyTorch tensor计算
loss = calculator.calculate_from_tensors(gen_tensor, gt_tensor, normalize_range='[-1,1]')

# 4. 便捷调用（自动检测输入类型）
loss = calculator(gen_img, gt_img)
```

#### 初始化参数

```python
calculator = EdgeL2LossCalculator(
    gaussian_kernel_size=(5, 5),        # 高斯模糊核大小
    gaussian_sigma=1.4,                  # 高斯模糊标准差
    canny_threshold_lower_factor=0.7,    # Canny下阈值因子
    canny_threshold_upper_factor=1.3,    # Canny上阈值因子
    morph_kernel_size=(3, 3),            # 形态学操作核大小
    morph_kernel_shape=cv2.MORPH_ELLIPSE,# 形态学核形状
    device='cuda'                        # 设备类型
)
```

#### 便捷函数

```python
from basicsr.metrics.edge_l2_loss import calculate_edge_l2_loss

# 使用默认参数快速计算
loss = calculate_edge_l2_loss(gen_img, gt_img)
```

### 2. `scripts/auto_inference.py` (修改)

**修改内容：**

1. **导入新模块**（第32行）：
   ```python
   from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator
   ```

2. **初始化计算器**（第60行）：
   ```python
   edge_l2_calculator = EdgeL2LossCalculator()
   ```

3. **添加到metrics字典**（第183行）：
   ```python
   'average_edge_l2_loss': 0.0
   ```

4. **计算Edge L2 Loss**（第256-265行）：
   ```python
   # Calculate Edge L2 Loss
   img_edge_l2 = 0.0
   try:
       img_edge_l2 = edge_l2_calculator.calculate_from_arrays(
           gen_img, gt_img, input_format='BGR'
       )
   except Exception as e:
       print(f"Warning: Edge L2 Loss calculation failed for {img_base_name}: {e}")
       img_edge_l2 = -1.0  # Mark as failed
   ```

5. **保存到metrics**（第272行）：
   ```python
   'edge_l2_loss': float(img_edge_l2)
   ```

6. **计算平均值**（第290行）：
   ```python
   metrics['average_edge_l2_loss'] = total_edge_l2 / count
   ```

7. **更新CSV输出**（第314-334行）：
   - 添加 "Edge L2 Loss" 列到CSV表头
   - 为每张图片和平均值写入 Edge L2 Loss 数据

### 3. `scripts/generate_metrics_report.py` (修改)

**修改内容：**

1. **收集Edge L2 Loss平均值**（第95行）：
   ```python
   avg_edge_l2 = data.get('average_edge_l2_loss')
   if avg_edge_l2 is not None:
       metrics_data['Edge L2 Loss'][column_name] = {'Average': avg_edge_l2}
   ```

2. **收集每张图片的Edge L2 Loss**（第117, 140-145行）：
   ```python
   edge_l2_val = img_data.get('edge_l2_loss')
   
   if edge_l2_val is not None:
       if 'Edge L2 Loss' not in metrics_data:
           metrics_data['Edge L2 Loss'] = {}
       if column_name not in metrics_data['Edge L2 Loss']:
           metrics_data['Edge L2 Loss'][column_name] = {}
       metrics_data['Edge L2 Loss'][column_name][img_name] = edge_l2_val
   ```

3. **添加到报告输出**（第267行）：
   ```python
   metric_types = ["PSNR", "SSIM", "LPIPS", "Edge L2 Loss"]
   ```

## 使用方法

### 1. 基本使用（在Python脚本中）

```python
from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator
import cv2

# 初始化计算器
calculator = EdgeL2LossCalculator()

# 读取图片
gen_img = cv2.imread('path/to/generated_image.png')
gt_img = cv2.imread('path/to/ground_truth_image.png')

# 计算Edge L2 Loss
loss = calculator.calculate_from_arrays(gen_img, gt_img, input_format='BGR')

print(f"Edge L2 Loss: {loss:.6f}")
```

### 2. 在推理脚本中自动计算

运行 `auto_inference.py` 时，Edge L2 Loss 会自动计算并保存：

```bash
python scripts/auto_inference.py \
    --ckpt path/to/checkpoint.ckpt \
    --init_img path/to/lr_images \
    --gt_img path/to/gt_images \
    --calculate_metrics
```

生成的 `metrics.json` 文件将包含：
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

生成的 `metrics.csv` 文件格式：
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
0802.png,25.7358,0.6455,0.2015,0.001567
...
Average,21.0714,0.5853,0.3036,0.002456
```

### 3. 生成综合报告

使用 `generate_metrics_report.py` 生成包含所有实验的综合CSV报告：

```bash
python scripts/generate_metrics_report.py \
    path/to/validation_results \
    --output comprehensive_report.csv
```

生成的报告将包含 Edge L2 Loss 作为一个独立的指标块，格式与PSNR、SSIM、LPIPS类似。

## 输出示例

### metrics.csv 示例

```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
0802.png,25.7358,0.6455,0.2015,0.001567
0803.png,24.6927,0.8546,0.2245,0.001890
...
Average,21.0714,0.5853,0.3036,0.002456
```

### 综合报告 CSV 示例

```csv
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,...
,,,dummy edge,edge,no edge,...
PSNR,Average,20.9199,20.2555,20.3397,20.2841,...
,0801.png,23.5642,21.4285,21.5756,21.3972,...
,0802.png,25.3789,24.0150,24.3193,24.4175,...

SSIM,Average,0.5955,0.5406,0.5461,0.5453,...
,0801.png,0.7656,0.5730,0.5915,0.5767,...

LPIPS,Average,0.2935,0.3373,0.3366,0.3344,...
,0801.png,0.2527,0.4066,0.3931,0.4015,...

Edge L2 Loss,Average,0.002456,0.003123,0.002987,0.003456,...
,0801.png,0.001234,0.002345,0.002123,0.002678,...
```

## 测试

### 运行测试脚本

```bash
# 确保环境已激活
conda activate your_env_name

# 运行测试
python scripts/test_edge_l2_loss.py
```

测试脚本将验证：
1. 从numpy数组计算
2. 从文件计算
3. 从PyTorch tensor计算
4. 便捷调用方法
5. 不同尺寸图片的处理

### 预期测试输出

```
============================================================
EdgeL2LossCalculator Test Suite
============================================================

============================================================
Test 1: Calculate Edge L2 Loss from numpy arrays
============================================================
✓ Edge L2 Loss between img1 and img2: 0.000234
✓ Edge L2 Loss for identical images: 0.000000
  (应该接近0)

...

============================================================
Test Results: 5/5 passed
✓ All tests passed!
============================================================
```

## 技术说明

### Edge Map 生成过程

1. **输入**：RGB/BGR 图片，值域 [0, 255]
2. **转灰度**：使用 OpenCV 的 cvtColor
3. **高斯模糊**：核大小 (5, 5)，sigma = 1.4
4. **Canny边缘检测**：阈值 threshold1=100, threshold2=200
5. **归一化**：输出范围 [0, 1]

### MSE 计算

```python
edge1 = generate_edge_map(img1)  # [0, 1]
edge2 = generate_edge_map(img2)  # [0, 1]
mse = np.mean((edge1 - edge2) ** 2)  # [0, 1]
```

### 值的解释

- **0.0**: 两张图片的边缘完全相同
- **< 0.001**: 边缘非常相似（优秀）
- **0.001 - 0.01**: 边缘相似（良好）
- **0.01 - 0.05**: 边缘有一定差异（一般）
- **> 0.05**: 边缘差异较大（需要改进）

注意：这些阈值是参考值，实际解释需要根据具体应用场景调整。

## 与其他指标的关系

| 指标 | 度量内容 | 值域 | 越小越好 |
|------|---------|------|---------|
| PSNR | 像素级差异 | [0, ∞) | ✗ (越大越好) |
| SSIM | 结构相似度 | [0, 1] | ✗ (越大越好) |
| LPIPS | 感知相似度 | [0, 1] | ✓ |
| **Edge L2 Loss** | **边缘相似度** | **[0, 1]** | **✓** |

## 常见问题

### Q1: Edge L2 Loss 和 LPIPS 有什么区别？

**A1**: 
- **LPIPS** 度量整体感知相似度，基于深度神经网络特征
- **Edge L2 Loss** 专门度量边缘保真度，基于Canny边缘检测

两者互补：LPIPS关注整体感知质量，Edge L2 Loss关注边缘细节。

### Q2: 为什么我的 Edge L2 Loss 值很小？

**A2**: 很小的值（如 < 0.001）通常是好事，表示边缘保真度很高。如果值为0或接近0，可能是：
- 两张图片完全相同
- 两张图片都没有明显边缘（均匀区域）

### Q3: 如何调整 Edge 检测参数？

**A3**: 可以在初始化 `EdgeL2LossCalculator` 时传入自定义参数：

```python
calculator = EdgeL2LossCalculator(
    gaussian_kernel_size=(7, 7),  # 更大的模糊核
    gaussian_sigma=2.0,            # 更强的模糊
    canny_threshold_lower_factor=0.5,  # 更敏感的边缘检测
    canny_threshold_upper_factor=1.5   # 更宽松的阈值范围
)
```

### Q4: 计算 Edge L2 Loss 会影响推理速度吗？

**A4**: 会有一定影响，但很小：
- Canny边缘检测是高效的
- 对于 512x512 图片，额外时间约 10-50ms
- 相比完整的SR推理（数秒），影响可忽略

### Q5: Edge L2 Loss 支持批处理吗？

**A5**: 支持！使用 `calculate_from_tensors` 方法时，可以传入batch tensor：

```python
# gen_tensor: (B, C, H, W)
# gt_tensor: (B, C, H, W)
loss = calculator.calculate_from_tensors(gen_tensor, gt_tensor)
# 返回整个batch的平均loss
```

## 未来改进方向

1. **支持更多边缘检测算法**：
   - Sobel
   - Laplacian
   - HED (Holistically-Nested Edge Detection)

2. **权重边缘损失**：
   - 根据边缘强度加权
   - 重点关注显著边缘

3. **多尺度边缘损失**：
   - 在多个分辨率上计算
   - 综合评估不同尺度的边缘

4. **GPU加速**：
   - 将Canny边缘检测移至GPU
   - 使用PyTorch实现完整流程

## 参考

- **EdgeMapGenerator**: `basicsr/utils/edge_utils.py`
- **Canny边缘检测**: OpenCV文档
- **MSE**: Mean Squared Error

## 版本历史

- **v1.0** (2025-10-16): 初始实现
  - 基础 Edge L2 Loss 计算
  - 集成到 auto_inference.py
  - 集成到 generate_metrics_report.py
  - 支持 numpy/file/tensor 输入

## 联系

如有问题或建议，请联系项目维护者。

