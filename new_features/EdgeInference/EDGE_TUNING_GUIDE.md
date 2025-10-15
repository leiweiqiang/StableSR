# Edge Map参数调优指南

## 🎯 问题描述

推理生成的edge map可能太稀疏，边缘效果不明显。这通常是因为Canny边缘检测的阈值设置过高。

---

## ✅ 已修复

**修改位置**: `scripts/sr_val_edge_inference.py`

**修改内容**:
```python
# 原始参数（边缘太少）
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.7,
    canny_threshold_upper_factor=1.3
)

# 改进参数（边缘更丰富）⭐
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.4,  # 从0.7降到0.4
    canny_threshold_upper_factor=0.9   # 从1.3降到0.9
)
```

**效果**:
- ❌ 原参数: 1.72%边缘像素，阈值(109,204)
- ✅ 新参数: 2.40%边缘像素，阈值(62,141)
- 📈 **提升**: 40%更多边缘信息

---

## 📊 实际测试结果

### 测试图像: 0803.png (512×512)

| 参数 | 阈值 | 边缘像素 | 占比 | 文件大小 | 效果 |
|------|------|----------|------|----------|------|
| **原始** (0.7/1.3) | (109,204) | 4,497 | 1.72% | 6.1KB | ❌ 太少 |
| **改进** (0.4/0.9) | (62,141) | 6,301 | 2.40% | 7.9KB | ✅ 合适 |
| **更敏感** (0.3/0.7) | (47,110) | ~8,000 | ~3.0% | ~9KB | ⚠️ 可能过多 |

---

## 🔧 如何调整参数

### 1. 边缘太少（推荐降低阈值）

**症状**: Edge map几乎全黑，只有很少的白线

**解决**: 降低阈值因子
```python
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.3,  # 更低
    canny_threshold_upper_factor=0.7   # 更低
)
```

### 2. 边缘太多（提高阈值）

**症状**: Edge map噪点很多，边缘过于密集

**解决**: 提高阈值因子
```python
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.5,  # 更高
    canny_threshold_upper_factor=1.1   # 更高
)
```

### 3. 使用固定阈值

如果自适应阈值效果不好，可以修改`basicsr/utils/edge_utils.py`使用固定阈值：

```python
# 在 generate_from_numpy 方法中，替换自适应阈值计算：

# 原来的自适应阈值
median = np.median(img_blurred)
lower_thresh = int(max(0, self.canny_threshold_lower_factor * median))
upper_thresh = int(min(255, self.canny_threshold_upper_factor * median))

# 改为固定阈值
lower_thresh = 50   # 固定下阈值
upper_thresh = 150  # 固定上阈值
```

---

## 📐 理解Canny阈值

### Canny边缘检测的双阈值

- **上阈值(upper)**: 高于此值的梯度被认为是"强边缘"
- **下阈值(lower)**: 
  - 高于下阈值且连接到强边缘的被保留
  - 低于下阈值的被丢弃

### 阈值计算方式

```
median = 图像灰度的中位数
lower_threshold = lower_factor × median
upper_threshold = upper_factor × median
```

### 示例计算

假设图像median = 157:

| 参数 | Lower | Upper | 说明 |
|------|-------|-------|------|
| (0.7, 1.3) | 109 | 204 | 保守，边缘少 |
| (0.4, 0.9) | 62 | 141 | 推荐，边缘适中 |
| (0.3, 0.7) | 47 | 110 | 敏感，边缘多 |

---

## 🧪 测试不同参数

在修改参数前，建议先测试效果：

```bash
cd /root/dp/StableSR_Edge_v3

conda run -n sr_infer python << 'EOF'
from PIL import Image
import numpy as np
import cv2
from basicsr.utils.edge_utils import EdgeMapGenerator

# 加载测试图像
img = np.array(Image.open("new_features/EdgeInference/gt_images/0803.png"))
img_np = img.astype(np.float32) / 255.0

# 测试不同参数
params = [
    (0.7, 1.3, "original"),
    (0.4, 0.9, "recommended"),
    (0.3, 0.7, "sensitive"),
]

for lower_f, upper_f, name in params:
    gen = EdgeMapGenerator(
        canny_threshold_lower_factor=lower_f,
        canny_threshold_upper_factor=upper_f
    )
    edge = gen.generate_from_numpy(img_np, input_format='RGB', normalize_input=True)
    pixels = np.count_nonzero((edge * 255).astype(np.uint8))
    print(f"{name:12s} ({lower_f}/{upper_f}): {pixels:5d} pixels ({100*pixels/edge.size:.2f}%)")
    
    # 保存对比图
    Image.fromarray((edge * 255).astype(np.uint8)).save(f"outputs/edge_{name}.png")

print("\n✓ Saved comparison images to outputs/")
EOF
```

---

## 📝 推荐配置

### 一般场景（默认）⭐
```python
EdgeMapGenerator(
    canny_threshold_lower_factor=0.4,
    canny_threshold_upper_factor=0.9
)
```
- 适用于大多数自然图像
- 边缘丰富但不过度
- **已应用到推理脚本**

### 复杂场景（细节丰富的图像）
```python
EdgeMapGenerator(
    canny_threshold_lower_factor=0.3,
    canny_threshold_upper_factor=0.7
)
```
- 检测更多细微边缘
- 适合纹理丰富的图像

### 简单场景（干净的图像）
```python
EdgeMapGenerator(
    canny_threshold_lower_factor=0.5,
    canny_threshold_upper_factor=1.0
)
```
- 只保留主要边缘
- 减少噪声

---

## 🎛️ 其他可调参数

除了Canny阈值，还可以调整：

### 1. 高斯模糊（降噪）
```python
EdgeMapGenerator(
    gaussian_kernel_size=(5, 5),  # 核大小，越大越模糊
    gaussian_sigma=1.4,            # 标准差，越大越模糊
    ...
)
```

### 2. 形态学操作（清理边缘）
```python
EdgeMapGenerator(
    morph_kernel_size=(3, 3),     # 核大小
    morph_kernel_shape=cv2.MORPH_ELLIPSE,  # 核形状
    ...
)
```

---

## ⚙️ 修改步骤

### 方式1: 修改推理脚本（推荐）

编辑 `scripts/sr_val_edge_inference.py`:

```python
# 找到这一行（约第70行）
edge_generator = EdgeMapGenerator(
    canny_threshold_lower_factor=0.4,  # 修改这里
    canny_threshold_upper_factor=0.9   # 修改这里
)
```

### 方式2: 修改EdgeMapGenerator默认值

编辑 `basicsr/utils/edge_utils.py`:

```python
# 找到 __init__ 方法（约第18行）
def __init__(
    self, 
    gaussian_kernel_size=(5, 5),
    gaussian_sigma=1.4,
    canny_threshold_lower_factor=0.4,  # 修改默认值
    canny_threshold_upper_factor=0.9,  # 修改默认值
    ...
):
```

---

## 📊 评估Edge质量

### 视觉检查

1. **查看edge map图像**:
   ```bash
   # Edge map保存在
   outputs/edge_inference_test/quick/edge_maps/
   ```

2. **判断标准**:
   - ✅ **好的edge**: 主要物体轮廓清晰，重要边缘都捕获
   - ❌ **太少**: 很多重要边缘缺失
   - ❌ **太多**: 充满噪点，难以区分主要边缘

### 数量统计

运行测试查看边缘像素占比：

```bash
cd /root/dp/StableSR_Edge_v3
python3 -c "
from PIL import Image
import numpy as np
e = np.array(Image.open('outputs/edge_inference_test/quick/edge_maps/0803_edge.png'))
nz = np.count_nonzero(e)
print(f'Edge pixels: {100*nz/e.size:.2f}%')
"
```

**参考范围**:
- < 1%: 太少 ❌
- 2-4%: 合适 ✅
- > 6%: 可能太多 ⚠️

---

## 🔍 故障排查

### 问题1: 修改参数后没有效果

**检查**:
1. 确认修改了正确的文件
2. 重新运行推理（不是使用缓存）
3. 检查日志确认使用了新参数

### 问题2: Edge map全黑

**可能原因**:
- 阈值过高
- 图像本身没有明显边缘

**解决**:
- 大幅降低阈值因子到0.2/0.5
- 或使用固定低阈值(30, 100)

### 问题3: Edge map噪点太多

**可能原因**:
- 阈值过低
- 图像质量差，噪声大

**解决**:
- 提高阈值因子
- 增大高斯模糊核(7,7)或sigma(2.0)

---

## 📚 相关资源

- **Canny边缘检测**: [Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector)
- **OpenCV Canny文档**: [cv2.Canny](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de)
- **Edge参数可视化工具**: [在线Canny调试](https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html)

---

## ✅ 快速总结

1. **问题**: Edge太少（< 2%）
2. **原因**: Canny阈值因子过高(0.7/1.3)
3. **解决**: 降低到(0.4/0.9)
4. **效果**: 边缘增加40%+
5. **位置**: `scripts/sr_val_edge_inference.py` 第70行

---

**最后更新**: 2025-10-15  
**状态**: ✅ 已优化  
**推荐参数**: (0.4, 0.9)

