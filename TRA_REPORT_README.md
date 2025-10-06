# TraReport - 超分辨率模型评估工具

TraReport是一个用于评估超分辨率模型性能的工具类，它可以加载训练好的模型，对验证集中的图片进行超分辨率处理，然后与真实高分辨率图片计算PSNR指标，并输出JSON格式的评估结果。

## 功能特性

- 🔧 **模型加载**: 支持从配置文件和检查点加载StableSR模型
- 🖼️ **图片处理**: 自动处理val目录中的所有图片进行超分辨率
- 📊 **PSNR计算**: 与gt目录中对应文件名的图片计算PSNR指标
- 📄 **JSON输出**: 生成详细的JSON格式评估报告
- ⚙️ **灵活配置**: 支持多种参数配置（DDPM步数、放大倍数、颜色修复等）

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install omegaconf
pip install pytorch-lightning
pip install basicsr
pip install pillow
pip install tqdm
pip install opencv-python
```

## 使用方法

### 1. 基本使用

```python
from tra_report import TraReport

# 创建TraReport实例
tra_report = TraReport(
    gt_dir="/path/to/gt/images",      # 真实高分辨率图片目录
    val_dir="/path/to/val/images",    # 待处理的低分辨率图片目录
    model_path="/path/to/model.ckpt", # 模型权重文件路径
    config_path="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"  # 配置文件路径
)

# 运行评估
results = tra_report.run_evaluation(output_path="evaluation_results.json")
```

### 2. 命令行使用

```bash
python run_tra_report.py \
    --gt_dir /path/to/gt/images \
    --val_dir /path/to/val/images \
    --model_path /path/to/model.ckpt \
    --output results.json
```

### 3. 高级参数配置

```python
tra_report = TraReport(
    gt_dir="/data/DIV2K_valid_HR",
    val_dir="/data/DIV2K_valid_LR", 
    model_path="./weights/stablesr_000117.ckpt",
    config_path="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
    ddpm_steps=200,           # DDPM采样步数
    upscale=4.0,              # 超分辨率倍数
    colorfix_type="adain",    # 颜色修复类型: adain/wavelet/none
    seed=42,                  # 随机种子
    device="cuda"             # 计算设备
)
```

## 参数说明

### 必需参数
- `gt_dir`: 真实高分辨率图片目录路径
- `val_dir`: 待处理的低分辨率图片目录路径  
- `model_path`: 模型权重文件路径

### 可选参数
- `config_path`: 模型配置文件路径（默认使用edge配置）
- `ddpm_steps`: DDPM采样步数（默认：200）
- `upscale`: 超分辨率倍数（默认：4.0）
- `colorfix_type`: 颜色修复类型，可选值：`adain`、`wavelet`、`none`（默认：adain）
- `seed`: 随机种子（默认：42）
- `device`: 计算设备（默认：cuda）

## 输出格式

评估结果以JSON格式输出，包含以下信息：

```json
{
  "model_path": "/path/to/model.ckpt",
  "config_path": "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
  "gt_dir": "/path/to/gt/images",
  "val_dir": "/path/to/val/images",
  "total_files": 100,
  "parameters": {
    "ddpm_steps": 200,
    "upscale": 4.0,
    "colorfix_type": "adain",
    "seed": 42
  },
  "results": [
    {
      "val_file": "/path/to/val/images/001.png",
      "gt_file": "/path/to/gt/images/001.png", 
      "psnr": 28.4567,
      "sr_shape": [1024, 1024, 3],
      "gt_shape": [1024, 1024, 3]
    }
  ],
  "summary": {
    "average_psnr": 28.1234,
    "min_psnr": 25.6789,
    "max_psnr": 30.9876,
    "std_psnr": 1.2345
  }
}
```

## 文件匹配规则

TraReport会自动匹配val目录和gt目录中的文件：
- 支持的文件格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- 匹配规则：基于文件名（不含扩展名）进行匹配
- 例如：`val/001.png` 会匹配 `gt/001.jpg`

## 示例脚本

项目包含以下示例脚本：

1. **`example_tra_report.py`**: 基本使用示例和批量评估示例
2. **`run_tra_report.py`**: 命令行运行脚本

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存来加载模型和处理图片
2. **文件格式**: 确保图片文件格式正确且可读取
3. **路径设置**: 确保所有路径参数正确且文件存在
4. **配置文件**: 确保模型配置文件与检查点匹配

## 错误处理

TraReport包含完善的错误处理机制：
- 自动验证输入参数和文件路径
- 处理单个文件处理失败的情况
- 提供详细的错误信息和调试输出

## 性能优化建议

1. **批量处理**: 对于大量图片，建议分批处理以避免内存溢出
2. **GPU优化**: 确保使用GPU加速计算
3. **参数调优**: 根据具体需求调整DDPM步数和颜色修复参数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch size或使用CPU模式
   - 降低图片分辨率

2. **模型加载失败**
   - 检查模型路径和配置文件路径
   - 确认模型文件完整性

3. **文件匹配失败**
   - 检查文件命名规则
   - 确认文件格式支持

4. **PSNR计算异常**
   - 检查图片尺寸匹配
   - 确认图片数据范围正确

如有其他问题，请检查错误日志或联系开发者。
