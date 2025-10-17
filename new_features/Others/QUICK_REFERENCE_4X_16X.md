# 4x/16x 实验快速参考手册

## 🚨 最重要的三个文件

### 1. 配置文件中的 `sf` 参数
```yaml
# configs/stableSRNew/v2-finetune_text_T_512.yaml（第1行）
sf: 4   # 4倍实验

# configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml（第1行）
sf: 16  # 16倍实验
```

### 2. 训练脚本中的 `CONFIG` 变量
```bash
# train_t5.sh 或 train_t6.sh
CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"           # 4x
CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"  # 16x
```

### 3. ⚠️ `.inference_defaults.conf` 中的 LR 路径（推理时）
```bash
# .inference_defaults.conf
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"  # 4x → 128×128 LR
DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"    # 16x → 32×32 LR
DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"
```

**⚠️ 警告**：`.inference_defaults.conf` 会**覆盖**命令行参数！切换实验时必须更新或删除。

---

## 📋 切换实验检查清单

### 切换到 4x 实验
```bash
# ✅ 步骤1：检查/修改配置文件
head -n 1 configs/stableSRNew/v2-finetune_text_T_512.yaml
# 应显示：sf: 4

# ✅ 步骤2：修改训练脚本
nano train_t5.sh  # 或 train_t6.sh
# 设置：CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"

# ✅ 步骤3：删除或更新推理配置（推荐删除）
rm .inference_defaults.conf
# 或者编辑：
nano .inference_defaults.conf
# 修改：
#   DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/128x128_valid_LR"
#   DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512.yaml"

# ✅ 步骤4：准备数据（确保有128×128的LR图像）
ls /mnt/nas_dp/test_dataset/128x128_valid_LR/

# ✅ 步骤5：开始训练
bash train_t5.sh
```

### 切换到 16x 实验
```bash
# ✅ 步骤1：检查/修改配置文件
head -n 1 configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml
# 应显示：sf: 16

# ✅ 步骤2：修改训练脚本
nano train_t5.sh  # 或 train_t6.sh
# 设置：CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"

# ✅ 步骤3：删除或更新推理配置（推荐删除）
rm .inference_defaults.conf
# 或者编辑：
nano .inference_defaults.conf
# 修改：
#   DEFAULT_INIT_IMG="/mnt/nas_dp/test_dataset/32x32_valid_LR"
#   DEFAULT_CONFIG="configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml"

# ✅ 步骤4：准备数据（确保有32×32的LR图像）
ls /mnt/nas_dp/test_dataset/32x32_valid_LR/

# ✅ 步骤5：开始训练
bash train_t5.sh
```

---

## 🔍 常见错误诊断

### 问题1：推理结果很差，明明训练很好
**可能原因**：LR输入尺寸与模型不匹配
```bash
# 检查推理配置
cat .inference_defaults.conf | grep DEFAULT_INIT_IMG
# 4x模型应该用128×128，16x模型应该用32×32

# 解决方法
rm .inference_defaults.conf  # 删除配置，强制使用命令行参数
```

### 问题2：训练时出现尺寸不匹配错误
**可能原因**：配置文件中的 `sf` 值错误
```bash
# 检查配置
head -n 1 configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml
# 检查structcond_stage_config
grep -A 5 "structcond_stage_config:" configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml
```

### 问题3：使用 run_auto_inference.sh 时总是用错误的LR路径
**原因**：`.inference_defaults.conf` 文件在覆盖参数
```bash
# 查看配置
cat .inference_defaults.conf

# 解决方法（推荐）
rm .inference_defaults.conf

# 或手动修改
nano .inference_defaults.conf
```

---

## 📊 快速对比表

| 项目 | 4x 实验 | 16x 实验 |
|------|---------|----------|
| **配置文件** | `v2-finetune_text_T_512.yaml` | `v2-finetune_text_T_512_edge_800.yaml` |
| **sf 值** | `sf: 4` | `sf: 16` |
| **LR 输入尺寸** (512 GT) | 128×128 | 32×32 |
| **LR 数据路径** | `128x128_valid_LR/` | `32x32_valid_LR/` |
| **GT 数据路径** | `512x512_valid_HR/` | `512x512_valid_HR/` ✓ 相同 |
| **输出尺寸** | 512×512 | 512×512 ✓ 相同 |
| **in_channels (标准)** | 4 | 4 |
| **in_channels (edge)** | 8 | 8 |
| **推荐 batch_size** | 6 | 2 |
| **难度** | 较容易 | 较困难 |

---

## 🔧 一键验证脚本

将以下内容保存为 `check_config.sh` 并运行：

```bash
#!/bin/bash
echo "=========================================="
echo "StableSR 4x/16x 配置检查"
echo "=========================================="
echo ""

# 检查当前配置文件
echo "1. 检查推理配置文件："
if [ -f .inference_defaults.conf ]; then
    echo "   ✓ 发现 .inference_defaults.conf"
    echo "   当前配置："
    grep "DEFAULT_INIT_IMG" .inference_defaults.conf
    grep "DEFAULT_CONFIG" .inference_defaults.conf
    
    # 判断是4x还是16x
    if grep -q "128x128" .inference_defaults.conf; then
        echo "   → 配置为 4x 实验"
    elif grep -q "32x32" .inference_defaults.conf; then
        echo "   → 配置为 16x 实验"
    else
        echo "   ⚠️ 无法识别实验类型"
    fi
else
    echo "   ✗ 未发现 .inference_defaults.conf（将使用命令行参数）"
fi
echo ""

# 检查训练配置
echo "2. 检查训练脚本配置："
if [ -f train_t5.sh ]; then
    CONFIG_LINE=$(grep '^CONFIG=' train_t5.sh)
    echo "   $CONFIG_LINE"
    
    if echo "$CONFIG_LINE" | grep -q "512.yaml"; then
        echo "   → train_t5.sh 配置为 4x"
    elif echo "$CONFIG_LINE" | grep -q "edge_800.yaml"; then
        echo "   → train_t5.sh 配置为 16x"
    fi
fi
echo ""

# 检查可用的配置文件
echo "3. 可用的配置文件："
echo "   4x配置："
ls -1 configs/stableSRNew/v2-finetune_text_T_512.yaml 2>/dev/null && \
    echo "      ✓ v2-finetune_text_T_512.yaml (sf: $(head -n 1 configs/stableSRNew/v2-finetune_text_T_512.yaml))"
echo "   16x配置："
ls -1 configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml 2>/dev/null && \
    echo "      ✓ v2-finetune_text_T_512_edge_800.yaml (sf: $(head -n 1 configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml))"
echo ""

# 检查数据路径
echo "4. 检查数据路径："
[ -d "/mnt/nas_dp/test_dataset/128x128_valid_LR" ] && echo "   ✓ 4x LR数据: 128x128_valid_LR" || echo "   ✗ 4x LR数据不存在"
[ -d "/mnt/nas_dp/test_dataset/32x32_valid_LR" ] && echo "   ✓ 16x LR数据: 32x32_valid_LR" || echo "   ✗ 16x LR数据不存在"
[ -d "/mnt/nas_dp/test_dataset/512x512_valid_HR" ] && echo "   ✓ GT数据: 512x512_valid_HR" || echo "   ✗ GT数据不存在"
echo ""

echo "=========================================="
echo "检查完成"
echo "=========================================="
```

---

## 💡 最佳实践

1. **每次切换实验时**，先删除 `.inference_defaults.conf`
   ```bash
   rm .inference_defaults.conf
   ```

2. **训练前验证**配置文件 sf 值
   ```bash
   head -n 1 configs/stableSRNew/你的配置文件.yaml
   ```

3. **推理时显式指定**所有参数，不依赖默认值
   ```bash
   python scripts/sr_val_edge_inference.py \
       --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
       --ckpt your_checkpoint.ckpt \
       --init-img /path/to/128x128_valid_LR \
       --gt-img /path/to/512x512_valid_HR
   ```

4. **保持命名一致性**
   - 4x实验使用包含 "4x" 的实验名称
   - 16x实验使用包含 "16x" 的实验名称

---

**相关文档**：`SCALE_FACTOR_MODIFICATION_GUIDE.md`（详细说明）

**最后更新**: 2025-10-17

