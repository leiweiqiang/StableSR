# EdgeInference 模块索引

Edge增强超分辨率推理完整模块

---

## 📂 文件列表

| 文件/目录 | 类型 | 说明 |
|----------|------|------|
| **sr_val_edge_inference.py** | 脚本 | 核心推理脚本（位于`scripts/`目录） |
| **test_edge_inference.sh** | 脚本 | 自动化测试脚本（6种测试配置） |
| **lr_images/** | 目录 | LR测试图像目录 ✅ (1张图像) |
| **gt_images/** | 目录 | GT测试图像目录 ✅ (1张图像) |
| **README.md** | 文档 | 完整使用文档（15KB，详细参数说明） |
| **QUICK_START.md** | 文档 | 5分钟快速开始指南 |
| **INDEX.md** | 文档 | 本文件 - 模块索引 |

---

## 📖 文档导航

### 🚀 新用户开始
1. [QUICK_START.md](QUICK_START.md) - 5分钟快速上手 ⭐推荐优先阅读

### 📚 深入学习
2. [README.md](README.md) - 完整使用文档

### 🧪 实践测试
3. [test_edge_inference.sh](test_edge_inference.sh) - 运行测试

---

## 🎯 核心功能

### ✅ 已实现功能

1. **完整Edge推理支持**
   - GT-based edge生成（推荐）
   - LR-based edge生成
   - Black edge模式
   - Dummy edge模式
   - 无edge模式（baseline）

2. **EdgeMapGenerator集成**
   - 统一的edge生成逻辑
   - 训练/推理完全一致
   - 自动批处理支持

3. **灵活配置**
   - 多种edge模式
   - 颜色校正选项（AdaIN/Wavelet/NoFix）
   - 批处理支持
   - 完整参数控制

4. **调试友好**
   - 详细日志输出
   - Edge map可视化保存
   - 中间结果保存
   - 自动化测试脚本

---

## 🔧 快速使用

### 最简单的用法

```bash
# 1. 激活环境
conda activate sr_infer

# 2. 运行测试
cd new_features/EdgeInference
./test_edge_inference.sh quick
```

### 基础推理命令

```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr \
    --gt-img inputs/gt \
    --outdir outputs/results \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_model.ckpt
```

---

## 📊 与原始脚本对比

| 特性 | sr_val_ddpm_text_T_vqganfin_old.py | sr_val_edge_inference.py |
|------|-----------------------------------|--------------------------|
| Edge支持 | ✗ | ✓ 多种模式 |
| EdgeMapGenerator | ✗ | ✓ 集成 |
| GT-based edge | ✗ | ✓ |
| 自动化测试 | ✗ | ✓ 6种配置 |
| 详细日志 | 基础 | ✓ 完整 |
| Edge可视化 | ✗ | ✓ 自动保存 |
| 中间结果保存 | 基础 | ✓ 完整 |
| 参数验证 | 基础 | ✓ 完善 |
| 错误处理 | 基础 | ✓ 详细 |

---

## 🧪 测试配置

### 可用测试（test_edge_inference.sh）

```bash
./test_edge_inference.sh help      # 查看所有测试
./test_edge_inference.sh basic     # 基础edge推理 ⭐推荐
./test_edge_inference.sh quick     # 快速测试（1图）
./test_edge_inference.sh batch     # 批处理测试
./test_edge_inference.sh no_edge   # 无edge对比
./test_edge_inference.sh black_edge # 消融实验
./test_edge_inference.sh lr_edge   # LR-based edge
```

---

## 💡 使用建议

### 推荐配置

**最佳质量** (有GT):
```bash
--use_edge_processing \
--gt-img path/to/gt \
--ddpm_steps 200 \
--colorfix_type adain
```

**快速推理**:
```bash
--use_edge_processing \
--gt-img path/to/gt \
--ddpm_steps 100 \
--n_samples 4
```

**无GT备选**:
```bash
--use_edge_processing \
--ddpm_steps 200
```

### Edge模式选择

| 场景 | 推荐模式 | 命令参数 |
|------|----------|----------|
| 正式推理 | GT-based ⭐ | `--use_edge_processing --gt-img xxx` |
| 无GT可用 | LR-based | `--use_edge_processing` |
| Baseline对比 | No edge | 不加`--use_edge_processing` |
| 消融实验 | Black edge | `--use_edge_processing --use_white_edge` |

---

## 📁 输出结构

```
outputs/edge_inference/
├── image1_edge.png              # ⭐ SR结果
├── image2_edge.png
├── edge_maps/                   # Edge可视化
│   ├── image1_edge.png
│   └── image2_edge.png
├── lr_input/                    # 原始LR（如保存）
│   ├── image1.png
│   └── image2.png
├── gt_hr/                       # GT图像（如提供）
│   ├── image1.png
│   └── image2.png
└── edge_inference_*.log         # 详细日志
```

---

## 🔗 相关资源

### 项目内资源

- **EdgeMapGenerator**: `../EdgeMapGenerator/README.md`
- **原始推理脚本**: `../../scripts/sr_val_ddpm_text_T_vqganfin_old.py`
- **Edge推理参考**: `../../scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py`
- **Edge监控**: `../../EDGE_MONITOR_README.md`
- **推理说明**: `../../INFERENCE_README.md`

### 核心代码

- **EdgeMapGenerator实现**: `../../basicsr/utils/edge_utils.py`
- **Edge配置文件**: `../../configs/stableSRNew/v2-finetune_text_T_512_edge.yaml`
- **VQGAN配置**: `../../configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml`

---

## 📝 开发日志

### 2025-10-15 - 初始版本
- ✅ 创建EdgeInference模块
- ✅ 完全基于sr_val_ddpm_text_T_vqganfin_old.py的逻辑
- ✅ 集成EdgeMapGenerator统一edge生成
- ✅ 实现多种edge模式
- ✅ 添加自动化测试脚本（6种配置）
- ✅ 编写完整文档
- ✅ conda sr_infer环境支持

---

## ✨ 核心优势

### 1. 统一性
- ✓ 使用EdgeMapGenerator，训练/推理edge生成逻辑完全一致
- ✓ 与训练代码保持同步

### 2. 灵活性
- ✓ 多种edge模式适应不同场景
- ✓ 丰富的参数配置选项
- ✓ 支持批处理和单图处理

### 3. 易用性
- ✓ 自动化测试脚本
- ✓ 详细文档和快速开始指南
- ✓ 清晰的错误提示

### 4. 可调试性
- ✓ 完整日志输出
- ✓ Edge map可视化
- ✓ 中间结果自动保存
- ✓ 详细的分辨率验证信息

---

## 🎓 学习路径

### 初学者
1. 阅读 [QUICK_START.md](QUICK_START.md) (5分钟)
2. 运行 `./test_edge_inference.sh quick` (3分钟)
3. 查看输出结果
4. 尝试基础推理命令

### 进阶用户
1. 阅读 [README.md](README.md) 完整文档
2. 了解所有参数和edge模式
3. 尝试不同测试配置
4. 对比不同edge模式的效果

### 开发者
1. 研究 `sr_val_edge_inference.py` 源代码
2. 了解EdgeMapGenerator实现
3. 参考 `test_edge_inference.sh` 自定义配置
4. 查看相关模块代码

---

## ⚠️ 注意事项

1. **环境要求**: 需要激活 `sr_infer` conda环境
2. **GT图像**: 强烈推荐提供GT图像用于edge生成（`--gt-img`）
3. **分辨率**: GT分辨率应为LR的4倍（如LR=512×512, GT=2048×2048）
4. **配置文件**: 确保使用支持edge的配置文件（`*_edge.yaml`）
5. **显存**: 根据GPU显存调整batch_size

---

## 🐛 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 找不到GT图像 | 路径错误或文件名不匹配 | 检查`--gt-img`路径，确保文件名一致 |
| GPU显存不足 | batch_size过大 | 减小`--n_samples` |
| Edge map全黑 | 使用了`--use_white_edge` | 去掉该参数或检查GT图像 |
| 推理很慢 | ddpm_steps过大 | 减小`--ddpm_steps` |
| 模型不支持edge | 配置文件错误 | 使用`*_edge.yaml`配置 |

---

## 📞 获取帮助

1. 查看 [README.md](README.md) 常见问题部分
2. 检查日志文件 `edge_inference_*.log`
3. 运行 `./test_edge_inference.sh help`
4. 参考EdgeMapGenerator文档

---

**模块状态**: ✅ 完整实现  
**测试状态**: ⏳ 待测试  
**文档状态**: ✅ 完整  
**最后更新**: 2025-10-15

---

**开始使用**: [QUICK_START.md](QUICK_START.md) ⭐

