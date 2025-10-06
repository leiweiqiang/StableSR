# TraReport快速开始指南

## 环境准备

1. **激活conda环境**:
```bash
conda activate sr_edge
```

2. **验证环境**:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 数据准备

### 选项1: 创建测试数据（推荐）

如果你只有高分辨率数据，可以创建低分辨率测试数据：

```bash
# 创建低分辨率测试数据（使用前10个文件进行快速测试）
python create_test_data.py \
    --hr_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR \
    --lr_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR \
    --scale_factor 4 \
    --subset_size 10
```

### 选项2: 使用现有数据

如果你已经有配对的HR和LR数据，确保目录结构如下：
```
/your/data/path/
├── gt/          # 高分辨率图片
└── val/         # 低分辨率图片（文件名对应）
```

## 模型准备

检查可用的模型文件：
```bash
find . -name "*.ckpt" -o -name "*.pth"
```

当前可用的模型：
- `./logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt`
- `./logs/2025-10-05T01-31-33_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt`

## 运行评估

### 方法1: 使用Python脚本（推荐）

```bash
# 修改run_evaluation_example.py中的路径，然后运行
python run_evaluation_example.py
```

### 方法2: 使用命令行

```bash
python run_tra_report.py \
    --gt_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR \
    --val_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR \
    --model_path ./logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --config_path ./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --output evaluation_results.json
```

### 方法3: 使用Shell脚本

```bash
chmod +x run_evaluation_example.sh
./run_evaluation_example.sh
```

## 完整示例

### 步骤1: 创建测试数据
```bash
conda activate sr_edge
python create_test_data.py --subset_size 5  # 快速测试用5个文件
```

### 步骤2: 运行评估
```bash
python run_tra_report.py \
    --gt_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR \
    --val_dir /stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR \
    --model_path ./logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt \
    --output quick_test_results.json
```

### 步骤3: 查看结果
```bash
cat quick_test_results.json | python -m json.tool
```

## 参数说明

- `--gt_dir`: 真实高分辨率图片目录
- `--val_dir`: 待处理的低分辨率图片目录
- `--model_path`: 模型权重文件路径
- `--config_path`: 模型配置文件路径（可选，有默认值）
- `--output`: 输出JSON文件路径
- `--ddpm_steps`: DDPM采样步数（默认：200）
- `--upscale`: 超分辨率倍数（默认：4.0）
- `--colorfix_type`: 颜色修复类型（默认：adain）
- `--seed`: 随机种子（默认：42）

## 输出结果

评估完成后会生成JSON文件，包含：
- 每个文件的PSNR值
- 统计摘要（平均值、最小值、最大值、标准差）
- 模型和配置信息

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少DDPM步数
   python run_tra_report.py ... --ddpm_steps 50
   ```

2. **文件匹配失败**
   - 检查文件名是否对应（不含扩展名）
   - 确认文件格式支持

3. **模型加载失败**
   - 检查模型路径是否正确
   - 确认配置文件与模型匹配

### 快速测试

如果遇到问题，可以先运行快速测试：
```bash
python create_test_data.py --subset_size 2  # 只用2个文件测试
python run_tra_report.py ...  # 运行评估
```

## 下一步

1. 查看详细文档：`TRA_REPORT_README.md`
2. 运行完整测试：`python test_tra_report.py`
3. 查看示例代码：`example_tra_report.py`
