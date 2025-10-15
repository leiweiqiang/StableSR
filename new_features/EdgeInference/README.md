# Edge-Enhanced Super-Resolution Inference

完整的edge增强超分辨率推理模块，基于`sr_val_ddpm_text_T_vqganfin_old.py`，集成了统一的EdgeMapGenerator支持。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [文件说明](#文件说明)
- [详细用法](#详细用法)
- [参数说明](#参数说明)
- [Edge模式](#edge模式)
- [测试脚本](#测试脚本)
- [常见问题](#常见问题)
- [技术细节](#技术细节)

---

## ✨ 功能特性

### 核心功能
- ✅ **统一Edge生成**: 使用EdgeMapGenerator类，确保训练和推理的edge生成逻辑完全一致
- ✅ **多种Edge模式**: 支持GT-based、LR-based、black edge、dummy edge等多种edge生成方式
- ✅ **批处理支持**: 可配置batch size进行高效批量处理
- ✅ **颜色校正**: 支持AdaIN、Wavelet、NoFix三种颜色校正方式
- ✅ **完整日志**: 自动生成详细的推理日志文件
- ✅ **中间结果保存**: 自动保存edge map、LR输入、GT等中间结果

### 与原始脚本的改进
1. **Edge支持**: 完全集成edge map生成和处理
2. **代码优化**: 更清晰的结构和更好的错误处理
3. **灵活配置**: 更多的命令行参数选项
4. **调试友好**: 详细的日志和中间结果输出

---

## 🚀 快速开始

### 环境准备

```bash
# 激活conda环境
conda activate sr_infer

# 进入项目根目录
cd /root/dp/StableSR_Edge_v3
```

### 基本用法

```bash
# 最简单的edge推理（使用GT图像生成edge）
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --gt-img inputs/gt_images \
    --outdir outputs/edge_inference \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_edge_model.ckpt
```

### 使用测试脚本

```bash
# 进入EdgeInference目录
cd new_features/EdgeInference

# 查看可用的测试配置
./test_edge_inference.sh help

# 运行基础测试
./test_edge_inference.sh basic

# 快速测试（只处理1张图片）
./test_edge_inference.sh quick
```

---

## 📁 文件说明

### 目录结构

```
new_features/EdgeInference/
├── README.md                    # 本文件 - 完整使用文档
├── sr_val_edge_inference.py     # Edge推理主脚本
└── test_edge_inference.sh       # 测试脚本（多种配置）
```

### 文件详情

| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | ~15KB | 完整的使用文档和参数说明 |
| `sr_val_edge_inference.py` | ~35KB | Edge推理核心脚本 |
| `test_edge_inference.sh` | ~8KB | 自动化测试脚本 |

---

## 📖 详细用法

### 1. 标准Edge推理（推荐）

使用GT图像生成edge map，与训练保持一致：

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --gt-img inputs/gt_images \
    --outdir outputs/standard_edge \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_edge_model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --ddpm_steps 200 \
    --n_samples 1 \
    --input_size 512
```

**特点**:
- ✓ 使用GT图像生成edge map（与训练一致）
- ✓ 最佳的edge质量和SR效果
- ✓ 推荐用于正式推理

### 2. 无Edge推理（baseline）

标准超分辨率，不使用edge信息：

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --outdir outputs/no_edge \
    --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
    --ckpt models/your_standard_model.ckpt \
    --ddpm_steps 200
```

**特点**:
- ✗ 不使用edge处理
- 用于对比baseline效果

### 3. LR-based Edge推理（不推荐）

从LR图像生成edge map：

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --outdir outputs/lr_edge \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_edge_model.ckpt \
    --ddpm_steps 200
```

**特点**:
- ⚠ 从LR图像生成edge（与训练不一致）
- ⚠ 可能存在domain mismatch
- ⚠ 不推荐，除非无GT可用

### 4. Black Edge推理（消融实验）

使用空白edge map：

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --outdir outputs/black_edge \
    --use_edge_processing \
    --use_white_edge \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_edge_model.ckpt \
    --ddpm_steps 200
```

**特点**:
- 使用全黑edge map（无edge信息）
- 用于消融实验，验证edge的作用

### 5. 批量处理

处理大量图像：

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/large_dataset \
    --gt-img inputs/large_dataset_gt \
    --outdir outputs/batch_results \
    --use_edge_processing \
    --n_samples 4 \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_edge_model.ckpt
```

**特点**:
- 批处理模式（batch_size=4）
- 提高处理速度
- 自动处理整个目录

### 6. 颜色校正

使用不同的颜色校正方法：

```bash
# AdaIN颜色校正
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --gt-img inputs/gt_images \
    --outdir outputs/adain \
    --use_edge_processing \
    --colorfix_type adain \
    [其他参数...]

# Wavelet颜色校正
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr_images \
    --gt-img inputs/gt_images \
    --outdir outputs/wavelet \
    --use_edge_processing \
    --colorfix_type wavelet \
    [其他参数...]
```

---

## ⚙️ 参数说明

### 必需参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--init-img` | str | `inputs/user_upload` | LR输入图像目录 |
| `--outdir` | str | `outputs/edge_inference` | 输出结果目录 |
| `--config` | str | - | 模型配置文件路径 |
| `--ckpt` | str | - | 模型权重文件路径 |

### Edge相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_edge_processing` | flag | False | 启用edge处理 |
| `--gt-img` | str | None | GT图像目录（推荐用于edge生成） |
| `--use_white_edge` | flag | False | 使用黑色edge map（无edge） |
| `--use_dummy_edge` | flag | False | 使用固定的dummy edge |
| `--dummy_edge_path` | str | - | Dummy edge图像路径 |

### 采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ddpm_steps` | int | 200 | DDPM采样步数 |
| `--n_samples` | int | 1 | 批处理大小 |
| `--input_size` | int | 512 | LR图像resize尺寸 |
| `--seed` | int | 42 | 随机种子 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vqgan_ckpt` | str | - | VQGAN模型路径 |
| `--dec_w` | float | 0.5 | VQGAN和Diffusion融合权重 |
| `--C` | int | 4 | Latent通道数 |
| `--f` | int | 8 | 下采样因子 |

### 后处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--colorfix_type` | str | `nofix` | 颜色校正类型：adain/wavelet/nofix |
| `--precision` | str | `autocast` | 精度模式：autocast/full |

### 调试参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max_images` | int | -1 | 最大处理图像数（-1=全部） |
| `--specific_file` | str | "" | 只处理指定文件 |

---

## 🎨 Edge模式

### 1. GT-based Edge（推荐）⭐

```bash
--use_edge_processing --gt-img path/to/gt
```

**原理**: 从GT图像生成edge map  
**优点**: 
- ✓ 与训练完全一致
- ✓ Edge质量最高
- ✓ SR效果最佳

**缺点**: 
- ✗ 需要GT图像

**适用场景**: 正式推理、性能评估

---

### 2. LR-based Edge

```bash
--use_edge_processing
```

**原理**: 从LR图像生成edge map  
**优点**: 
- ✓ 不需要GT图像

**缺点**: 
- ✗ 与训练不一致（domain mismatch）
- ✗ Edge质量较低
- ✗ 可能影响SR效果

**适用场景**: 无GT可用时的备选方案

---

### 3. Black Edge（无Edge）

```bash
--use_edge_processing --use_white_edge
```

**原理**: 使用全黑edge map（无edge信息）  
**优点**: 
- ✓ 可用于消融实验

**缺点**: 
- ✗ 没有edge信息
- ✗ 效果可能不如有edge

**适用场景**: 消融实验、验证edge作用

---

### 4. Dummy Edge

```bash
--use_edge_processing --use_dummy_edge --dummy_edge_path path/to/edge.png
```

**原理**: 使用预先准备的固定edge map  
**优点**: 
- ✓ 可控的edge输入

**缺点**: 
- ✗ 与实际图像无关
- ✗ 效果可能不佳

**适用场景**: 特殊测试需求

---

### 5. No Edge（标准SR）

```bash
# 不加 --use_edge_processing
```

**原理**: 不使用edge处理，标准SR  
**优点**: 
- ✓ 简单直接

**缺点**: 
- ✗ 无edge增强

**适用场景**: Baseline对比

---

## 🧪 测试脚本

### 可用测试

```bash
# 查看所有测试
./test_edge_inference.sh help
```

#### 1. basic - 基础Edge推理
```bash
./test_edge_inference.sh basic
```
- Edge处理: ✓
- GT-based: ✓  
- Batch size: 1
- 用途: 标准edge推理测试

#### 2. no_edge - 无Edge推理
```bash
./test_edge_inference.sh no_edge
```
- Edge处理: ✗
- 用途: Baseline对比

#### 3. black_edge - 黑色Edge
```bash
./test_edge_inference.sh black_edge
```
- Edge处理: ✓
- Black edge: ✓
- 用途: 消融实验

#### 4. lr_edge - LR-based Edge
```bash
./test_edge_inference.sh lr_edge
```
- Edge处理: ✓
- LR-based: ✓
- 用途: 无GT时的备选方案

#### 5. batch - 批处理
```bash
./test_edge_inference.sh batch
```
- Edge处理: ✓
- Batch size: 4
- 用途: 批量处理测试

#### 6. quick - 快速测试
```bash
./test_edge_inference.sh quick
```
- Edge处理: ✓
- 图像数: 1
- 用途: 快速验证

### 自定义测试脚本

编辑`test_edge_inference.sh`中的"custom"部分：

```bash
# 找到 "custom" case
"custom")
    CUSTOM_LR_DIR="your/lr/path"
    CUSTOM_GT_DIR="your/gt/path"
    CUSTOM_OUTPUT="outputs/custom"
    CUSTOM_BATCH_SIZE=2
    CUSTOM_DDPM_STEPS=200
    ...
```

然后运行：
```bash
./test_edge_inference.sh custom
```

---

## ❓ 常见问题

### Q1: 如何选择Edge模式？

**A**: 推荐顺序：
1. **有GT图像**: 使用GT-based edge（`--gt-img`）⭐ 
2. **无GT图像**: 使用LR-based edge（不推荐，但可用）
3. **消融实验**: 使用black edge（`--use_white_edge`）

### Q2: GT图像和LR图像的分辨率关系？

**A**: 
- LR图像: 下采样后的低分辨率（如512×512）
- GT图像: 原始高分辨率（如2048×2048，通常是LR的4倍）
- Edge map: 从GT生成，保持GT分辨率
- 推理时LR会被resize到`--input_size`（默认512）

### Q3: 为什么推荐使用GT-based edge？

**A**: 因为训练时就是用GT图像生成edge map的：
```python
# 训练代码（basicsr/data/realesrgan_dataset.py）
img_edge = self.edge_generator.generate_from_numpy(
    img_gt,  # 使用GT图像！
    input_format='BGR',
    normalize_input=True
)
```

推理时也用GT生成edge可以保持一致性，避免domain mismatch。

### Q4: batch_size如何设置？

**A**: 
- GPU显存充足: 可设为2-4
- GPU显存不足: 设为1
- 大量图像: 适当增加batch_size提速

### Q5: ddpm_steps如何选择？

**A**:
- 质量优先: 200-1000
- 速度优先: 50-100  
- 平衡: 200（推荐）

### Q6: 输出目录结构？

**A**:
```
outputs/edge_inference/
├── image1_edge.png          # SR结果
├── image2_edge.png
├── edge_maps/               # Edge map
│   ├── image1_edge.png
│   └── image2_edge.png
├── lr_input/ (如有)         # 原始LR输入
│   ├── image1.png
│   └── image2.png
├── gt_hr/ (如提供GT)        # GT图像
│   ├── image1.png
│   └── image2.png
└── edge_inference_*.log     # 推理日志
```

### Q7: 如何调试edge生成？

**A**: 
1. 查看输出的`edge_maps/`目录中的edge图像
2. 检查日志中的分辨率信息
3. 使用`--max_images 1`快速测试单张图像

### Q8: 颜色校正选哪个？

**A**:
- **adain**: 论文中使用，推荐
- **wavelet**: 备选方案
- **nofix**: 不校正，用于对比

---

## 🔧 技术细节

### Edge生成流程

1. **加载图像**: 读取GT或LR图像
2. **预处理**: 
   - 转灰度图
   - 高斯模糊（kernel=5×5, sigma=1.4）
3. **Canny检测**:
   - 自适应阈值（lower=0.7×median, upper=1.3×median）
4. **后处理**:
   - 形态学闭运算（kernel=3×3 ellipse）
   - 转RGB 3通道
   - 归一化到[-1, 1]

### 与训练的一致性

| 项目 | 训练 | 推理（GT-based） | 一致性 |
|------|------|------------------|--------|
| Edge源 | GT图像 | GT图像 | ✓ |
| Edge算法 | EdgeMapGenerator | EdgeMapGenerator | ✓ |
| 参数配置 | 默认参数 | 默认参数 | ✓ |
| 分辨率 | GT原始分辨率 | GT原始分辨率 | ✓ |

### 代码改进点

相比`sr_val_ddpm_text_T_vqganfin_old.py`:

1. ✅ 集成EdgeMapGenerator
2. ✅ 支持多种edge模式
3. ✅ 更完善的参数检查
4. ✅ 更详细的日志输出
5. ✅ 更好的错误处理
6. ✅ 自动化测试脚本

---

## 📚 相关文件

### 核心代码
- `../../basicsr/utils/edge_utils.py` - EdgeMapGenerator实现
- `../../scripts/sr_val_ddpm_text_T_vqganfin_old.py` - 原始推理脚本
- `../../scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py` - Edge推理参考

### 配置文件
- `../../configs/stableSRNew/v2-finetune_text_T_512_edge.yaml` - Edge模型配置
- `../../configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml` - VQGAN配置

### 相关文档
- `../EdgeMapGenerator/README.md` - EdgeMapGenerator文档
- `../../EDGE_MONITOR_README.md` - Edge监控文档
- `../../INFERENCE_README.md` - 推理说明

---

## 🎯 推荐工作流程

### 1. 准备数据
```bash
# 准备LR图像
mkdir -p inputs/test_lr

# 准备GT图像（推荐）
mkdir -p inputs/test_gt
```

### 2. 快速测试
```bash
# 先用quick测试验证配置
./test_edge_inference.sh quick
```

### 3. 检查结果
```bash
# 查看输出
ls outputs/edge_inference_test/quick/

# 检查edge map
ls outputs/edge_inference_test/quick/edge_maps/

# 查看日志
cat outputs/edge_inference_test/quick/edge_inference_*.log
```

### 4. 正式推理
```bash
# 配置正确后，处理全部数据
./test_edge_inference.sh basic
```

### 5. 结果对比
```bash
# 对比不同配置
./test_edge_inference.sh basic       # GT-based edge
./test_edge_inference.sh lr_edge     # LR-based edge  
./test_edge_inference.sh no_edge     # No edge (baseline)
./test_edge_inference.sh black_edge  # Black edge (ablation)
```

---

## 📝 更新日志

### 2025-10-15
- ✅ 初始版本
- ✅ 基于sr_val_ddpm_text_T_vqganfin_old.py完整实现
- ✅ 集成EdgeMapGenerator统一edge生成
- ✅ 支持多种edge模式
- ✅ 添加自动化测试脚本
- ✅ 完整的参数和文档

---

## 📧 支持

如有问题，请：
1. 查看[常见问题](#常见问题)
2. 检查日志文件
3. 参考相关文档
4. 运行测试脚本验证环境

---

**最后更新**: 2025-10-15  
**测试状态**: ✅ 待测试  
**兼容性**: Python 3.8+, PyTorch 1.12+, CUDA 11.3+

