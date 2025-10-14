# StableSR Edge 推理测试脚本使用指南

本项目提供了三个推理测试脚本，用于不同场景的模型测试。

---

## 📁 脚本概览

| 脚本 | 用途 | 速度 | 复杂度 |
|------|------|------|--------|
| `quick_test.sh` | 快速验证模型 | 最快 | 简单 |
| `test_inference.sh` | 标准推理测试 | 中等 | 中等 |
| `batch_test.sh` | 批量参数测试 | 较慢 | 高级 |

---

## 🚀 快速开始

### 1. 准备测试数据

```bash
# 创建输入目录
mkdir -p inputs/test_images

# 将低分辨率图像放入该目录
cp /path/to/your/lr_images/*.png inputs/test_images/
```

### 2. 快速测试（推荐第一次使用）

```bash
# 使用默认配置快速测试
./quick_test.sh

# 或指定输入目录
./quick_test.sh inputs/my_images

# 或指定输入和checkpoint
./quick_test.sh inputs/my_images logs/xxx/checkpoints/epoch-10.ckpt
```

**特点**：
- ⚡ 最快速度（25步DDIM）
- ✅ 验证模型是否正常工作
- 📊 适合快速迭代调试

---

## 📋 详细使用说明

### 脚本1: `quick_test.sh` - 快速测试

**用法**：
```bash
./quick_test.sh [输入目录] [checkpoint路径]
```

**参数**（都可选）：
- 第1个参数：输入图像目录（默认：`inputs/test_images`）
- 第2个参数：模型checkpoint路径（默认：自动查找最新的）

**示例**：
```bash
# 1. 默认配置
./quick_test.sh

# 2. 指定输入目录
./quick_test.sh inputs/validation_set

# 3. 完整指定
./quick_test.sh inputs/test_set logs/exp1/checkpoints/last.ckpt
```

**输出**：
- 位置：`outputs/quick_test_YYYYMMDD_HHMM/`
- 包含：超分辨率后的图像

**采样配置**：
- DDIM步数：25步（快速）
- 颜色修正：AdaIN
- 图像尺寸：512x512

---

### 脚本2: `test_inference.sh` - 标准推理测试

**用法**：
```bash
./test_inference.sh
```

**特点**：
- 🔧 完整的参数检查和错误提示
- 📊 详细的性能统计
- 🎨 自动生成预览拼图（需要ImageMagick）
- ⏱️ 显示处理速度和总耗时

**配置修改**（编辑脚本顶部）：
```bash
# 在脚本中修改这些变量
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
CHECKPOINT="logs/stablesr_edge_loss_*/checkpoints/last.ckpt"
INPUT_DIR="inputs/test_images"
DDIM_STEPS=50           # 调整采样步数
COLORFIX_TYPE="adain"   # adain/wavelet/nofix
```

**输出**：
- 位置：`outputs/inference_test_YYYYMMDD_HHMMSS/`
- 包含：
  - 超分辨率图像
  - `preview.jpg`（如果安装了ImageMagick）
  - 性能统计信息

**推荐DDIM步数**：
- 25步：快速预览（~3秒/图）
- 50步：标准质量（~6秒/图）✅ 推荐
- 100步：高质量（~12秒/图）
- 200步：最高质量（~24秒/图）

---

### 脚本3: `batch_test.sh` - 批量参数测试

**用法**：
```bash
./batch_test.sh [输入目录] [checkpoint路径]
```

**功能**：
自动测试6种不同的配置组合，帮助找到最佳参数：

| 配置名 | 步数 | eta | 说明 |
|--------|------|-----|------|
| steps25_eta0.0 | 25 | 0.0 | 快速确定性 |
| steps25_eta1.0 | 25 | 1.0 | 快速随机 |
| steps50_eta0.0 | 50 | 0.0 | 中速确定性 |
| steps50_eta1.0 | 50 | 1.0 | 中速随机 |
| steps100_eta0.0 | 100 | 0.0 | 高质量确定性 |
| steps200_eta1.0 | 200 | 1.0 | 最高质量 |

**输出**：
- 位置：`outputs/batch_test_YYYYMMDD_HHMMSS/`
- 结构：
  ```
  batch_test_20250114_120000/
  ├── steps25_eta0.0/      # 各配置的输出图像
  ├── steps25_eta1.0/
  ├── steps50_eta0.0/
  ├── steps50_eta1.0/
  ├── steps100_eta0.0/
  ├── steps200_eta1.0/
  └── results.csv          # 性能对比表
  ```

**results.csv 示例**：
```csv
配置名,步数,eta,耗时(秒),图像数,状态
steps25_eta0.0,25,0.0,45,10,成功
steps50_eta1.0,50,1.0,95,10,成功
...
```

**使用场景**：
- 🔍 寻找最佳参数组合
- 📊 对比不同配置的质量和速度
- 📈 性能基准测试

---

## 🎯 使用场景推荐

### 场景1: 第一次测试模型
```bash
./quick_test.sh
```
快速验证模型能否正常运行。

### 场景2: 日常验证训练效果
```bash
./test_inference.sh
```
使用标准配置测试checkpoint质量。

### 场景3: 寻找最佳参数
```bash
./batch_test.sh
```
对比不同配置，选择最优的质量-速度平衡点。

### 场景4: 生产环境推理
编辑 `test_inference.sh` 设置最优参数，然后：
```bash
./test_inference.sh
```

---

## ⚙️ 参数说明

### DDIM_STEPS（采样步数）
- **值越大**：质量越好，速度越慢
- **推荐值**：
  - 开发调试：25-50
  - 正式测试：50-100
  - 论文/展示：200+

### DDIM_ETA（随机性）
- **0.0**：完全确定性，相同输入产生相同输出
- **1.0**：完全随机，每次结果不同
- **推荐值**：
  - 需要可复现结果：0.0
  - 追求多样性：1.0
  - 折中：0.5

### COLORFIX_TYPE（颜色修正）
- **adain**：自适应实例归一化（推荐，论文使用）
- **wavelet**：小波变换颜色修正
- **nofix**：不进行颜色修正

### INPUT_SIZE（输入尺寸）
- **512**：标准尺寸（推荐）
- **768**：高分辨率（需要更多显存）

---

## 📊 性能参考

基于 RTX 4090 的测试结果：

| DDIM步数 | 单图耗时 | 显存占用 | 质量评分 |
|----------|----------|----------|----------|
| 25 | ~3秒 | ~8GB | ⭐⭐⭐ |
| 50 | ~6秒 | ~8GB | ⭐⭐⭐⭐ |
| 100 | ~12秒 | ~8GB | ⭐⭐⭐⭐⭐ |
| 200 | ~24秒 | ~8GB | ⭐⭐⭐⭐⭐⭐ |

---

## 🐛 故障排查

### 问题1: "Checkpoint不存在"
**解决**：
```bash
# 查找可用的checkpoint
ls -lh logs/*/checkpoints/

# 修改脚本中的CHECKPOINT变量
```

### 问题2: "输入目录不存在"
**解决**：
```bash
mkdir -p inputs/test_images
# 添加测试图像
```

### 问题3: CUDA内存不足
**解决**：
- 减少 `INPUT_SIZE`（改为256或384）
- 减少 `N_SAMPLES`（改为1）
- 使用更少的DDIM步数

### 问题4: 生成图像质量不佳
**调整**：
- 增加 `DDIM_STEPS`（50 → 100 → 200）
- 尝试不同的 `COLORFIX_TYPE`
- 检查输入图像质量
- 验证checkpoint是否训练充分

---

## 📂 目录结构

```
StableSR_Edge_v3/
├── quick_test.sh           # 快速测试脚本
├── test_inference.sh       # 标准推理脚本
├── batch_test.sh           # 批量测试脚本
├── inputs/                 # 输入目录
│   └── test_images/        # 测试图像
└── outputs/                # 输出目录
    ├── quick_test_*/       # 快速测试输出
    ├── inference_test_*/   # 标准测试输出
    └── batch_test_*/       # 批量测试输出
```

---

## 🔧 高级用法

### 1. 修改推理脚本

如果需要使用不同的Python推理脚本：

编辑 `test_inference.sh`：
```bash
# 修改这一行
INFERENCE_SCRIPT="scripts/sr_val_ddpm_text_T_vqganfin_old.py"
```

可用的推理脚本：
- `sr_val_ddim_text_T_negativeprompt.py` - DDIM采样（快速）
- `sr_val_ddpm_text_T_vqganfin_old.py` - DDPM采样（高质量）
- `sr_val_ddim_text_T_negativeprompt_canvas.py` - Canvas模式
- `sr_val_ddim_text_T_negativeprompt_canvas_tile.py` - Tile模式（大图）

### 2. 自定义批量测试配置

编辑 `batch_test.sh` 中的配置数组：
```bash
declare -a CONFIGS=(
    "my_config:50:0.5:我的配置"
    # 格式：名称:步数:eta:描述
)
```

### 3. 指定GPU

```bash
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1
./test_inference.sh
```

或多GPU：
```bash
export CUDA_VISIBLE_DEVICES=0,1
./batch_test.sh
```

---

## 📝 输出文件说明

### 标准输出文件命名
```
original_filename_output.png    # 标准输出
original_filename_stage1.png    # VQGAN阶段输出（如果保存）
original_filename_lq.png        # 低分辨率输入（如果保存）
```

### 查看输出
```bash
# 查看所有输出
ls outputs/inference_test_*/

# 快速预览（使用eog/feh等图像查看器）
eog outputs/inference_test_*/*.png

# 统计输出
find outputs/ -name "*.png" | wc -l
```

---

## 🎓 最佳实践

1. **训练期间**：每个epoch用 `quick_test.sh` 快速验证
2. **阶段性评估**：每10个epoch用 `test_inference.sh` 详细测试
3. **最终评估**：训练完成后用 `batch_test.sh` 寻找最优参数
4. **保存结果**：重要的输出记得备份，避免被覆盖
5. **记录参数**：在输出目录中保存使用的配置

---

## 🔗 相关文档

- `train_t5.sh` - 训练脚本
- `EDGE_MONITOR_README.md` - 边缘监控说明
- `configs/stableSRNew/` - 模型配置文件
- `scripts/` - 原始Python推理脚本

---

## 📞 需要帮助？

如遇问题，请检查：
1. 配置文件路径是否正确
2. Checkpoint是否存在且完整
3. 输入图像格式是否支持（PNG/JPG）
4. GPU显存是否足够
5. Python环境是否激活

---

**享受超分辨率的乐趣！🚀**

