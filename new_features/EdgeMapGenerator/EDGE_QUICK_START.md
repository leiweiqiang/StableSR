# EdgeMapGenerator 快速开始

## 🚀 1分钟快速上手

### 导入和使用

```python
from basicsr.utils.edge_utils import EdgeMapGenerator

# 创建生成器
edge_gen = EdgeMapGenerator()

# 方式1: 从tensor生成（推理常用）
edge_map = edge_gen.generate_from_tensor(
    image_tensor,          # [B, 3, H, W], 范围[-1, 1]
    input_format='RGB',
    normalize_range='[-1,1]'
)

# 方式2: 从numpy生成（训练常用）
edge_map = edge_gen.generate_from_numpy(
    img_np,               # [H, W, 3], 范围[0, 1]
    input_format='BGR',
    normalize_input=True
)
```

## 📝 实际使用场景

### 场景1: 训练数据集

```python
# basicsr/data/realesrgan_dataset.py

class RealESRGANDataset(data.Dataset):
    def __init__(self, opt):
        # 初始化edge生成器
        self.edge_generator = EdgeMapGenerator()
    
    def __getitem__(self, index):
        img_gt = load_image(...)  # BGR, [0,1]
        
        # 生成edge map
        img_edge = self.edge_generator.generate_from_numpy(
            img_gt, 
            input_format='BGR', 
            normalize_input=True
        )
        
        return {'gt': img_gt, 'img_edge': img_edge}
```

### 场景2: 推理脚本

```python
# predict.py

from basicsr.utils.edge_utils import EdgeMapGenerator

edge_generator = EdgeMapGenerator()

def predict(lr_image):
    # lr_image: [1, 3, H, W], [-1, 1]
    
    # 生成edge map
    edge_map = edge_generator.generate_from_tensor(
        lr_image,
        input_format='RGB',
        normalize_range='[-1,1]'
    )
    
    # 使用edge map进行推理
    samples = model.sample(
        struct_cond=init_latent,
        edge_map=edge_map,
        ...
    )
    return samples
```

## ⚙️ 自定义参数

```python
edge_gen = EdgeMapGenerator(
    gaussian_kernel_size=(7, 7),      # 更大的模糊核
    gaussian_sigma=2.0,                # 更强的模糊
    canny_threshold_lower_factor=0.5,  # 更低的下阈值
    canny_threshold_upper_factor=1.5   # 更高的上阈值
)
```

## ✅ 测试验证

```bash
# 激活环境
conda activate sr_edge

# 运行测试
python test_edge_generator.py
```

预期输出：
```
✓ 所有测试通过!
```

## 📚 更多信息

- 详细文档: `EDGE_GENERATOR_USAGE.md`
- 重构总结: `EDGE_REFACTOR_SUMMARY.md`
- 核心代码: `basicsr/utils/edge_utils.py`

## 🎯 关键优势

✅ **统一**: 训练和推理使用相同逻辑  
✅ **简洁**: 3行代码替代28行手动实现  
✅ **可靠**: 经过充分测试，差异为0  
✅ **灵活**: 支持多种格式和自定义参数  

---

就是这么简单！现在你可以在项目的任何地方使用 `EdgeMapGenerator` 来生成一致的边缘图。

