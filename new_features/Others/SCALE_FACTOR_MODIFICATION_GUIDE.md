# 4倍和16倍超分辨率实验修改指南

本文档说明了在StableSR_Edge_v3项目中进行4倍（4x）和16倍（16x）超分辨率实验时需要修改的所有文件。

## 概述

超分辨率倍数（scale factor，简称sf）是控制模型从低分辨率图像重建到高分辨率图像的关键参数。修改sf会影响：
- 数据处理管道
- 模型架构参数
- 训练和推理脚本

## 核心参数说明

### 1. sf (Scale Factor)
- **含义**: 超分辨率放大倍数
- **可选值**: 4 或 16
- **影响范围**: 数据加载、模型输入输出尺寸计算

### 2. image_size (Model Resolution)
- **含义**: 模型训练/推理的目标高分辨率图像尺寸
- **常用值**: 512, 768
- **说明**: 这是输出图像的分辨率，不随sf变化

### 3. structcond_stage_config.image_size
- **含义**: 结构条件编码器的输入尺寸
- **计算公式**: `image_size / 8 * sf / 4` (对于4x) 或保持96 (对于16x)
- **4x时**: 96 (512/8 * 4/4 = 64, 但配置中使用96)
- **16x时**: 96 (同样使用96)

## 需要修改的文件清单

### 零、推理默认配置文件 ⚠️ **重要**

**文件**: `.inference_defaults.conf` (项目根目录)

此文件保存上次推理的参数，会**覆盖** `run_auto_inference.sh` 的命令行参数。

**切换4x/16x实验时必须修改或删除此文件！**

```bash
# 查看当前配置
cat .inference_defaults.conf

# 方法1：修改配置（4x实验）
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"

# 方法2：修改配置（16x实验）
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"

# 方法3：删除文件（推荐，避免混淆）
rm .inference_defaults.conf
```

### 一、配置文件（*.yaml）

#### 1. 主训练配置文件

**文件位置**: `configs/stableSRNew/`

##### 对于4倍实验:
- `v2-finetune_text_T_512.yaml` (已经是4x配置)
- `v2-finetune_text_T_768v.yaml` (已经是4x配置)

##### 对于16倍实验:
- `v2-finetune_text_T_512_edge_800.yaml` (已经是16x配置)
- `v2-finetune_face_T_512.yaml` (已经是16x配置)

##### 如果需要创建新配置，需要修改以下参数:

```yaml
# 第1行：设置超分辨率倍数
sf: 4  # 4倍实验设为4，16倍实验设为16

model:
  params:
    image_size: 512  # 输出图像分辨率，通常不变
    
    structcond_stage_config:
      params:
        image_size: 96  # 对于512分辨率，通常保持96
        in_channels: 4  # 标准版本使用4通道
        # 如果使用edge版本（将边缘图拼接到输入）:
        # in_channels: 8  # edge版本使用8通道（4 latent + 4 edge channels）
        
    first_stage_config:
      params:
        ddconfig:
          resolution: 512  # 应与image_size一致

degradation:
  gt_size: 512  # Ground Truth尺寸，应与image_size一致
  
data:
  params:
    train:
      target: basicsr.data.realesrgan_dataset.RealESRGANDataset
      params:
        crop_size: 512  # 裁剪尺寸，应与image_size一致
        gt_size: 512    # Ground Truth尺寸
        
    validation:
      params:
        crop_size: 512
        gt_size: 512
```

#### 2. 测试数据配置文件

**文件位置**: `configs/stableSRdata/`

需要修改：
- `test_data.yaml` (当前是4x配置)
- `test_data_face.yaml` (当前是4x配置)

```yaml
# 第1行：设置超分辨率倍数
sf: 4  # 或 16

# 其他参数同主配置文件
```

### 二、训练脚本

#### 1. train_t5.sh / train_t6.sh

**文件位置**: 项目根目录

这些脚本通过`--base`参数指定配置文件，只需确保指向正确的配置文件：

```bash
# 示例：4倍实验
CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"

# 示例：16倍实验（带edge）
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
```

### 三、推理脚本和默认配置

**文件位置**: `scripts/` 和项目根目录

所有推理脚本都会从配置文件中读取`sf`参数，因此主要确保：

1. **加载正确的配置文件**
2. **使用与训练时相同的sf值**
3. **配置正确的LR输入路径**（根据sf调整）

#### 关键推理脚本：

1. `scripts/sr_val_edge_inference.py` - Edge增强推理
2. `scripts/sr_val_ddpm_text_T_vqganfin_old.py` - 标准DDPM推理
3. `scripts/sr_val_ddim_text_T_negativeprompt.py` - DDIM推理
4. `scripts/auto_inference.py` - 自动批量推理

#### ⚠️ **重要：`.inference_defaults.conf` 配置文件**

**文件位置**: 项目根目录 `.inference_defaults.conf`

当使用 `run_auto_inference.sh` 进行推理时，此文件中的参数会**覆盖**命令行的默认值。这个文件保存了上次使用的推理参数，包括：

```bash
# .inference_defaults.conf 文件内容示例
DEFAULT_CKPT="/path/to/checkpoint.ckpt"
DEFAULT_LOGS_DIR="/root/dp/StableSR_Edge_v3/logs"
DEFAULT_OUTPUT_BASE="validation_results"
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"      # ← LR输入路径（重要！）
DEFAULT_GT_IMG="/mnt/nas_dp/test_dataset/512x512_valid_HR"      # ← GT输入路径
DEFAULT_MAX_IMAGES="-1"
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
DEFAULT_VQGAN_CKPT="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
DEFAULT_SELECTED_DIR="2025-10-16T22-37-24_stablesr_edge_loss_20251016_223721"
```

**关键参数说明**：

| 参数 | 4倍实验 | 16倍实验 | 说明 |
|------|---------|----------|------|
| `DEFAULT_INIT_IMG` | `/path/to/128x128_valid_LR` | `/path/to/32x32_valid_LR` | **必须匹配sf！** |
| `DEFAULT_GT_IMG` | `/path/to/512x512_valid_HR` | `/path/to/512x512_valid_HR` | 两者相同 |
| `DEFAULT_CONFIG` | `v2-finetune_text_T_512.yaml` | `v2-finetune_text_T_512_edge_800.yaml` | 配置文件路径 |

**切换实验时必须修改**：

```bash
# 切换到4倍实验时，修改 .inference_defaults.conf：
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"  # 128×128 LR for 4x
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"

# 切换到16倍实验时，修改 .inference_defaults.conf：
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"    # 32×32 LR for 16x
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
```

**或者直接删除该文件，让脚本使用命令行参数**：
```bash
rm .inference_defaults.conf
```

**推理脚本使用示例**:

```bash
# 4倍推理（确保 .inference_defaults.conf 中的路径正确或删除该文件）
python scripts/sr_val_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
    --ckpt logs/your_4x_model/checkpoints/epoch=XXX.ckpt \
    --init-img /path/to/128x128_valid_LR \
    --outdir outputs/4x_results

# 16倍推理（确保 .inference_defaults.conf 中的路径正确或删除该文件）
python scripts/sr_val_edge_inference.py \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
    --ckpt logs/your_16x_model/checkpoints/epoch=XXX.ckpt \
    --init-img /path/to/32x32_valid_LR \
    --outdir outputs/16x_results
```

### 四、核心代码文件（通常不需要修改）

以下文件会自动从配置中读取`sf`参数，**通常不需要手动修改**：

1. `ldm/models/diffusion/ddpm.py`
   - 使用: `self.configs.sf`
   - 作用: 在训练时计算低分辨率图像尺寸

2. `ldm/models/autoencoder.py`
   - 使用: `self.configs.sf`
   - 作用: 在编码/解码时处理尺寸

3. `scripts/util_image.py`
   - 使用: `self.sf`
   - 作用: 图像分块处理时的尺度计算

4. `main.py`
   - 作用: 读取配置文件并初始化模型

## 修改步骤总结

### 进行4倍实验:

1. **使用或创建4x配置文件**:
   ```yaml
   sf: 4
   image_size: 512
   gt_size: 512
   crop_size: 512
   structcond_stage_config.image_size: 96
   ```

2. **训练**:
   ```bash
   # 修改train_t5.sh或train_t6.sh中的CONFIG变量
   CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
   bash train_t5.sh
   ```

3. **推理前修改 `.inference_defaults.conf`** (如果使用 `run_auto_inference.sh`):
   ```bash
   # 方法1：编辑文件
   nano .inference_defaults.conf
   # 修改以下行：
   DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"  # 4x需要128×128
   DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
   
   # 方法2：或直接删除该文件
   rm .inference_defaults.conf
   ```

4. **推理**:
   ```bash
   python scripts/auto_inference.py \
       --log_dir logs/your_4x_experiment \
       --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
       --init_img /path/to/128x128_valid_LR
   ```

### 进行16倍实验:

1. **使用或创建16x配置文件**:
   ```yaml
   sf: 16
   image_size: 512
   gt_size: 512
   crop_size: 512
   structcond_stage_config.image_size: 96
   structcond_stage_config.in_channels: 8  # 如果使用edge版本
   ```

2. **训练**:
   ```bash
   # 修改train_t5.sh或train_t6.sh中的CONFIG变量
   CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
   bash train_t5.sh
   ```

3. **推理前修改 `.inference_defaults.conf`** (如果使用 `run_auto_inference.sh`):
   ```bash
   # 方法1：编辑文件
   nano .inference_defaults.conf
   # 修改以下行：
   DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"    # 16x需要32×32
   DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
   
   # 方法2：或直接删除该文件
   rm .inference_defaults.conf
   ```

4. **推理**:
   ```bash
   python scripts/auto_inference.py \
       --log_dir logs/your_16x_experiment \
       --config configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
       --init_img /path/to/32x32_valid_LR
   ```

## 注意事项

### 1. 模型兼容性
- **4x模型**只能用于4x推理
- **16x模型**只能用于16x推理
- 不能混用不同sf的checkpoint和配置

### 2. 内存和计算需求
- **16x实验**需要更多显存和计算资源
- 建议：
  - 4x: batch_size=6 (单卡4090/A6000)
  - 16x: batch_size=2 (单卡4090/A6000)

### 3. 数据集准备
- 确保Ground Truth图像分辨率足够：
  - 4x: GT至少512×512
  - 16x: GT至少512×512 (将生成32×32的LR图像)

### 4. Edge版本特殊说明
- Edge增强版本（in_channels=8）需要：
  - 将边缘图与latent特征拼接
  - 使用支持边缘输入的checkpoint
  - 当前16x实验主要使用edge版本

### 5. `.inference_defaults.conf` 陷阱

**⚠️ 非常重要：** 当使用 `run_auto_inference.sh` 时，`.inference_defaults.conf` 文件会覆盖命令行参数！

**问题场景**：
- 你训练了一个4x模型，但推理时使用了之前16x实验留下的 `.inference_defaults.conf`
- 结果：推理会使用32×32的LR图像去匹配4x模型，导致结果错误

**解决方法**：
1. **删除旧配置文件**：`rm .inference_defaults.conf`
2. **检查配置内容**：`cat .inference_defaults.conf`
3. **手动更新配置**：编辑文件确保 `DEFAULT_INIT_IMG` 和 `DEFAULT_CONFIG` 正确

### 6. 验证配置正确性

训练/推理前检查：

```bash
# 1. 检查配置文件的sf值
head -n 1 configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml
# 应该显示: sf: 16

# 2. 检查structcond_stage_config参数
grep -A 20 "structcond_stage_config:" configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml

# 3. 检查推理默认配置（如果存在）
if [ -f .inference_defaults.conf ]; then
    echo "发现 .inference_defaults.conf，内容如下："
    cat .inference_defaults.conf
    echo ""
    echo "⚠️ 请确认 DEFAULT_INIT_IMG 和 DEFAULT_CONFIG 与实验匹配！"
fi
```

## 快速对比：4x vs 16x

| 参数 | 4x实验 | 16x实验 |
|------|--------|---------|
| sf | 4 | 16 |
| 输入LR尺寸 (512 GT) | 128×128 | 32×32 |
| 输出HR尺寸 | 512×512 | 512×512 |
| structcond image_size | 96 | 96 |
| in_channels (标准) | 4 | 4 |
| in_channels (edge) | 8 | 8 |
| 推荐batch_size | 6 | 2 |
| 难度 | 较容易 | 较困难 |
| 训练时间 | 相对较短 | 相对较长 |

## 常见配置文件映射

| 配置文件 | sf | 版本 | 用途 |
|----------|-----|------|------|
| `v2-finetune_text_T_512.yaml` | 4 | 标准 | 4x基础训练 |
| `v2-finetune_text_T_768v.yaml` | 4 | 标准 | 4x高分辨率(768) |
| `v2-finetune_text_T_512_edge_800.yaml` | 16 | Edge | 16x Edge增强 |
| `v2-finetune_face_T_512.yaml` | 16 | 标准 | 16x人脸优化 |

## 相关文件索引

### 配置文件
- `configs/stableSRNew/*.yaml` - 主配置文件（训练用）
- `configs/stableSRdata/*.yaml` - 测试数据配置
- **`.inference_defaults.conf`** - 推理默认参数配置 ⚠️ **推理时会覆盖命令行参数**

### 训练
- `train_t5.sh` - T5机器训练脚本
- `train_t6.sh` - T6机器训练脚本  
- `main.py` - 主训练入口

### 推理
- `run_auto_inference.sh` - 交互式推理菜单（使用 `.inference_defaults.conf`）
- `scripts/sr_val_edge_inference.py` - Edge推理
- `scripts/auto_inference.py` - 自动批量推理
- `scripts/sr_val_ddpm_text_T_vqganfin_old.py` - DDPM推理

### 核心模型代码
- `ldm/models/diffusion/ddpm.py` - 扩散模型主类
- `ldm/models/autoencoder.py` - VAE编码器/解码器
- `scripts/util_image.py` - 图像处理工具

---

**最后更新**: 2025-10-17
**项目**: StableSR_Edge_v3

