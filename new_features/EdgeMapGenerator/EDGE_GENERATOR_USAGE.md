# EdgeMapGenerator 使用指南

## 概述

`EdgeMapGenerator` 是一个统一的边缘图生成工具类，用于在训练和推理中生成一致的边缘图。

## 核心特性

- **一致性**: 训练和推理使用相同的边缘生成逻辑
- **可配置**: 支持自定义Canny参数、高斯模糊参数等
- **灵活性**: 支持numpy数组和PyTorch tensor输入
- **批处理**: 自动处理单张和批量图像

## 快速开始

### 1. 基本使用

```python
from basicsr.utils.edge_utils import EdgeMapGenerator

# 创建生成器实例
edge_generator = EdgeMapGenerator()

# 从tensor生成edge map（推理常用）
# 输入: [B, 3, H, W] 范围[-1, 1]
# 输出: [B, 3, H, W] 范围[-1, 1]
edge_map = edge_generator.generate_from_tensor(
    image_tensor,
    input_format='RGB',
    normalize_range='[-1,1]'
)

# 从numpy数组生成edge map（训练常用）
# 输入: [H, W, 3] 范围[0, 1]
# 输出: [H, W, 3] 范围[0, 1]
edge_map_np = edge_generator.generate_from_numpy(
    img_np,
    input_format='BGR',
    normalize_input=True
)
```

### 2. 便捷函数

```python
from basicsr.utils.edge_utils import generate_edge_map

# 自动检测输入类型
edge_map = generate_edge_map(image)  # 可以是numpy或tensor
```

### 3. 自定义参数

```python
edge_generator = EdgeMapGenerator(
    gaussian_kernel_size=(5, 5),      # 高斯模糊核大小
    gaussian_sigma=1.4,                # 高斯模糊标准差
    canny_threshold_lower_factor=0.7,  # Canny下阈值因子
    canny_threshold_upper_factor=1.3,  # Canny上阈值因子
    morph_kernel_size=(3, 3),          # 形态学核大小
    device='cuda'
)
```

## 在训练中使用

### Dataset集成

```python
# basicsr/data/realesrgan_dataset.py

from basicsr.utils.edge_utils import EdgeMapGenerator

class RealESRGANDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        # 初始化edge生成器
        self.edge_generator = EdgeMapGenerator(
            gaussian_kernel_size=opt.get('edge_gaussian_kernel_size', (5, 5)),
            gaussian_sigma=opt.get('edge_gaussian_sigma', 1.4),
        )
    
    def __getitem__(self, index):
        # 加载GT图像 (BGR, [0,1])
        img_gt = load_image(...)
        
        # 生成edge map
        img_edge = self.edge_generator.generate_from_numpy(
            img_gt, 
            input_format='BGR', 
            normalize_input=True
        )
        
        return {'gt': img_gt, 'img_edge': img_edge, ...}
```

## 在推理中使用

### 推理脚本

```python
# predict.py 或 inference script

from basicsr.utils.edge_utils import EdgeMapGenerator

# 创建全局实例
edge_generator = EdgeMapGenerator()

def inference(lr_image):
    # lr_image: [B, 3, H, W] 范围[-1, 1]
    
    # 生成edge map
    edge_map = edge_generator.generate_from_tensor(
        lr_image,
        input_format='RGB',
        normalize_range='[-1,1]'
    )
    
    # 使用edge_map进行推理
    result = model.sample(
        cond=semantic_c,
        struct_cond=init_latent,
        edge_map=edge_map,
        ...
    )
    
    return result
```

## 配置文件支持

可以在训练配置文件中自定义edge生成参数：

```yaml
# configs/train_config.yaml

datasets:
  train:
    name: RealESRGAN
    type: RealESRGANDataset
    # Edge生成参数（可选）
    edge_gaussian_kernel_size: [5, 5]
    edge_gaussian_sigma: 1.4
    edge_canny_lower_factor: 0.7
    edge_canny_upper_factor: 1.3
    edge_morph_kernel_size: [3, 3]
```

## 技术细节

### 边缘检测流程

1. **预处理**: 转换为灰度图
2. **降噪**: 高斯模糊 (kernel=5x5, σ=1.4)
3. **边缘检测**: 自适应Canny算法
   - 下阈值 = 0.7 × median(blurred_image)
   - 上阈值 = 1.3 × median(blurred_image)
4. **后处理**: 形态学闭运算 (椭圆核3x3)
5. **转换**: 单通道→3通道RGB/BGR

### 输入输出格式

| 方法 | 输入格式 | 输入范围 | 输出格式 | 输出范围 |
|------|---------|---------|---------|---------|
| `generate_from_numpy` | [H, W, 3] | [0, 1] | [H, W, 3] | [0, 1] |
| `generate_from_tensor` | [B, 3, H, W] | [-1, 1] | [B, 3, H, W] | [-1, 1] |

### 参数说明

- **gaussian_kernel_size**: 高斯模糊核大小，必须为奇数
- **gaussian_sigma**: 高斯模糊标准差，控制模糊程度
- **canny_threshold_lower_factor**: Canny下阈值因子，范围[0, 1]
- **canny_threshold_upper_factor**: Canny上阈值因子，范围[1, 2]
- **morph_kernel_size**: 形态学操作核大小
- **morph_kernel_shape**: 形态学核形状 (MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS)

## 常见问题

### Q1: 为什么训练用BGR，推理用RGB？

A: 这是由于数据加载方式不同：
- 训练中使用OpenCV加载图像（默认BGR）
- 推理中通常使用PIL或torchvision（默认RGB）

EdgeMapGenerator会自动处理这两种格式。

### Q2: 如何确保训练和推理的一致性？

A: 只要使用相同的参数（默认参数或自定义参数），EdgeMapGenerator保证生成完全一致的边缘图。

### Q3: 可以在CPU上使用吗？

A: 可以，边缘检测主要在CPU上运行（使用OpenCV），device参数主要影响tensor的设备放置。

## 示例代码

完整示例请参考：
- 训练: `basicsr/data/realesrgan_dataset.py`
- 推理: `predict.py` 或 `scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py`

## 迁移指南

### 从旧代码迁移

**旧代码** (手动实现):
```python
# 50+行的手动实现
img_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(...)
median = np.median(img_blurred)
# ... 更多代码
```

**新代码** (使用EdgeMapGenerator):
```python
# 3行搞定
from basicsr.utils.edge_utils import EdgeMapGenerator
edge_generator = EdgeMapGenerator()
img_edge = edge_generator.generate_from_numpy(img_gt, input_format='BGR')
```

## 总结

EdgeMapGenerator提供了一个统一、可配置、易用的边缘图生成解决方案，确保训练和推理的一致性，同时大大简化了代码。

