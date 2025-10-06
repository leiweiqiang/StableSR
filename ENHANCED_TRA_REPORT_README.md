# Enhanced TraReport - StableSR模型比较评估工具

## 概述

Enhanced TraReport是一个增强版的超分辨率模型性能评估工具，专门用于比较StableSR Edge和StableSR Upscale模型的性能。该工具可以同时使用两种模型对验证图片进行超分辨率处理，计算PSNR指标，并提供详细的比较分析。

## 主要功能

### 1. 双模型支持
- **StableSR Edge**: 支持边缘增强的StableSR模型
- **StableSR Upscale**: 支持标准StableSR Upscale模型
- 同时加载和运行两种模型进行对比

### 2. 全面的PSNR评估
- 计算每个图片的PSNR值
- 提供详细的统计信息（平均值、最小值、最大值、标准差）
- 支持批量处理多个图片

### 3. 模型比较分析
- 自动比较两种模型的性能
- 计算PSNR差异和改进百分比
- 识别性能更好的模型
- 提供详细的比较报告

### 4. 灵活的配置选项
- 可自定义DDPM采样步数
- 支持不同的超分辨率倍数
- 多种颜色修复方法（adain、wavelet、none）
- 可配置随机种子

## 文件结构

```
/root/dp/StableSR_Edge_v2/
├── enhanced_tra_report.py          # 主要的EnhancedTraReport类
├── test_enhanced_tra_report.py     # 测试脚本
├── example_enhanced_tra_report.py  # 使用示例
├── ENHANCED_TRA_REPORT_README.md   # 本文档
└── configs/stableSRNew/            # 配置文件目录
    ├── v2-finetune_text_T_512_edge.yaml      # StableSR Edge配置
    └── v2-finetune_text_T_512.yaml           # StableSR Upscale配置
```

## 安装和依赖

### 系统要求
- Python 3.8+
- CUDA支持的GPU（推荐）
- 足够的GPU内存（建议8GB+）

### 依赖包
```bash
pip install torch torchvision
pip install omegaconf
pip install pytorch-lightning
pip install basicsr
pip install pillow
pip install numpy
pip install tqdm
```

## 使用方法

### 1. 基本使用

```python
from enhanced_tra_report import EnhancedTraReport

# 创建EnhancedTraReport实例
enhanced_tra_report = EnhancedTraReport(
    gt_dir="/path/to/gt/images",                    # 高分辨率图片目录
    val_dir="/path/to/val/images",                  # 低分辨率图片目录
    stablesr_edge_model_path="/path/to/edge_model.ckpt",      # StableSR Edge模型
    stablesr_upscale_model_path="/path/to/upscale_model.ckpt", # StableSR Upscale模型
    stablesr_edge_config_path="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
    stablesr_upscale_config_path="./configs/stableSRNew/v2-finetune_text_T_512.yaml",
    device="cuda",
    ddpm_steps=200,
    upscale=4.0,
    colorfix_type="adain",
    seed=42
)

# 运行评估
results = enhanced_tra_report.run_evaluation("comparison_results.json")
```

### 2. 自定义参数

```python
enhanced_tra_report = EnhancedTraReport(
    gt_dir="/path/to/gt",
    val_dir="/path/to/val",
    stablesr_edge_model_path="/path/to/edge_model.ckpt",
    stablesr_upscale_model_path="/path/to/upscale_model.ckpt",
    device="cuda",
    ddpm_steps=100,        # 减少采样步数以加快处理
    upscale=2.0,           # 2倍超分辨率
    colorfix_type="wavelet", # 使用小波颜色修复
    seed=123
)
```

### 3. 批量评估

```python
datasets = [
    {"name": "DIV2K", "gt_dir": "/path/to/DIV2K/HR", "val_dir": "/path/to/DIV2K/LR"},
    {"name": "Set5", "gt_dir": "/path/to/Set5/HR", "val_dir": "/path/to/Set5/LR"},
    {"name": "Set14", "gt_dir": "/path/to/Set14/HR", "val_dir": "/path/to/Set14/LR"}
]

for dataset in datasets:
    enhanced_tra_report = EnhancedTraReport(
        gt_dir=dataset["gt_dir"],
        val_dir=dataset["val_dir"],
        stablesr_edge_model_path="/path/to/edge_model.ckpt",
        stablesr_upscale_model_path="/path/to/upscale_model.ckpt"
    )
    
    results = enhanced_tra_report.run_evaluation(f"{dataset['name']}_results.json")
```

## 输出格式

### JSON结果结构

```json
{
  "evaluation_info": {
    "stablesr_edge_model_path": "模型路径",
    "stablesr_upscale_model_path": "模型路径",
    "gt_dir": "GT目录",
    "val_dir": "验证目录",
    "total_files": 10,
    "parameters": {
      "ddpm_steps": 200,
      "upscale": 4.0,
      "colorfix_type": "adain",
      "seed": 42
    }
  },
  "results": [
    {
      "val_file": "验证图片路径",
      "gt_file": "GT图片路径",
      "stablesr_edge": {
        "psnr": 28.45,
        "sr_shape": [1024, 1024, 3]
      },
      "stablesr_upscale": {
        "psnr": 29.12,
        "sr_shape": [1024, 1024, 3]
      },
      "gt_shape": [1024, 1024, 3],
      "psnr_difference": 0.67,
      "better_model": "StableSR Upscale"
    }
  ],
  "summary": {
    "stablesr_edge": {
      "average_psnr": 28.45,
      "min_psnr": 26.12,
      "max_psnr": 30.78,
      "std_psnr": 1.23
    },
    "stablesr_upscale": {
      "average_psnr": 29.12,
      "min_psnr": 27.45,
      "max_psnr": 31.23,
      "std_psnr": 1.15
    },
    "comparison": {
      "psnr_difference": 0.67,
      "better_model": "StableSR Upscale",
      "improvement_percentage": 2.35
    }
  }
}
```

## 参数说明

### 必需参数
- `gt_dir`: 真实高分辨率图片目录
- `val_dir`: 待处理的低分辨率图片目录
- `stablesr_edge_model_path`: StableSR Edge模型权重文件路径
- `stablesr_upscale_model_path`: StableSR Upscale模型权重文件路径

### 可选参数
- `stablesr_edge_config_path`: StableSR Edge配置文件路径（默认：`./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml`）
- `stablesr_upscale_config_path`: StableSR Upscale配置文件路径（默认：`./configs/stableSRNew/v2-finetune_text_T_512.yaml`）
- `device`: 计算设备（默认：`"cuda"`）
- `ddpm_steps`: DDPM采样步数（默认：`200`）
- `upscale`: 超分辨率倍数（默认：`4.0`）
- `colorfix_type`: 颜色修复类型（默认：`"adain"`，可选：`"wavelet"`、`"none"`）
- `seed`: 随机种子（默认：`42`）

## 测试和验证

### 运行测试
```bash
python test_enhanced_tra_report.py
```

### 查看示例
```bash
python example_enhanced_tra_report.py
```

## 性能优化建议

### 1. 内存优化
- 使用较小的`ddpm_steps`值（如50-100）来减少内存使用
- 确保GPU有足够的内存（建议8GB+）
- 可以分批处理大量图片

### 2. 速度优化
- 减少DDPM采样步数
- 使用较小的超分辨率倍数
- 在CPU上运行（虽然较慢但内存需求更少）

### 3. 质量优化
- 增加DDPM采样步数（如200-500）
- 使用合适的颜色修复方法
- 确保输入图片质量良好

## 常见问题

### Q: 模型文件在哪里下载？
A: 需要从StableSR官方仓库下载相应的模型权重文件：
- StableSR Edge模型：需要训练或下载专门的Edge版本
- StableSR Upscale模型：可以使用标准的StableSR模型

### Q: 如何处理内存不足的问题？
A: 
1. 减少`ddpm_steps`参数
2. 使用CPU而不是GPU
3. 分批处理图片
4. 减少图片尺寸

### Q: 如何解释PSNR结果？
A: 
- PSNR值越高表示图像质量越好
- 通常PSNR > 30dB被认为是好的质量
- 比较两种模型的PSNR差异来评估性能

### Q: 支持哪些图片格式？
A: 支持常见的图片格式：JPG、JPEG、PNG、BMP、TIFF

## 扩展功能

### 1. 添加新的评估指标
可以在`_calculate_psnr`方法中添加SSIM、LPIPS等其他评估指标。

### 2. 支持更多模型
可以扩展支持其他超分辨率模型，如Real-ESRGAN、ESRGAN等。

### 3. 可视化结果
可以添加结果可视化功能，生成对比图片和统计图表。

## 贡献和反馈

如果您在使用过程中遇到问题或有改进建议，请：
1. 检查本文档的常见问题部分
2. 运行测试脚本验证环境配置
3. 提供详细的错误信息和环境配置

## 许可证

本项目遵循与StableSR相同的许可证条款。
