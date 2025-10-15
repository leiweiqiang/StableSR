# Edge Inference 快速开始指南

⏱️ 5分钟快速上手Edge推理

---

## 🎯 最快开始方式

### 步骤1: 激活环境 (30秒)

```bash
conda activate sr_infer
cd /root/dp/StableSR_Edge_v3
```

### 步骤2: 准备数据 (1分钟)

```bash
# 创建测试目录
mkdir -p inputs/test_lr inputs/test_gt

# 复制你的测试图像
# LR图像 -> inputs/test_lr/
# GT图像 -> inputs/test_gt/
```

### 步骤3: 运行测试 (3分钟)

```bash
cd new_features/EdgeInference
./test_edge_inference.sh quick
```

### 步骤4: 查看结果 (30秒)

```bash
ls outputs/edge_inference_test/quick/
# ├── result_edge.png      # SR结果
# ├── edge_maps/           # Edge map
# └── *.log                # 日志
```

✅ 完成！

---

## 📝 快速命令参考

### 1. 基础Edge推理（推荐）
```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/test_lr \
    --gt-img inputs/test_gt \
    --outdir outputs/results \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_model.ckpt
```

### 2. 无Edge推理（对比）
```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/test_lr \
    --outdir outputs/no_edge \
    --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
    --ckpt models/your_model.ckpt
```

### 3. 只处理一张图片
```bash
python scripts/sr_val_edge_inference.py \
    --init-img inputs/test_lr \
    --gt-img inputs/test_gt \
    --outdir outputs/single \
    --use_edge_processing \
    --max_images 1 \
    [其他参数...]
```

---

## 🔧 常用测试命令

```bash
cd new_features/EdgeInference

# 查看所有测试
./test_edge_inference.sh help

# 基础测试
./test_edge_inference.sh basic

# 快速测试（1张图）
./test_edge_inference.sh quick

# 批处理测试
./test_edge_inference.sh batch

# 无Edge对比
./test_edge_inference.sh no_edge
```

---

## ⚙️ 必知参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--init-img` | - | LR图像目录 ⭐必需 |
| `--gt-img` | - | GT图像目录（推荐用于edge） |
| `--outdir` | - | 输出目录 ⭐必需 |
| `--use_edge_processing` | False | 启用edge处理 |
| `--config` | - | 模型配置 ⭐必需 |
| `--ckpt` | - | 模型权重 ⭐必需 |
| `--ddpm_steps` | 200 | 采样步数 |
| `--n_samples` | 1 | Batch size |

---

## 💡 三个关键点

### 1. Edge模式选择

```bash
# ✓ 推荐：使用GT生成edge（与训练一致）
--use_edge_processing --gt-img path/to/gt

# ⚠ 备选：使用LR生成edge（可能有domain mismatch）
--use_edge_processing

# ✗ Baseline：不使用edge
# （不加--use_edge_processing）
```

### 2. 分辨率关系

- **LR输入**: 512×512（会被resize到--input_size）
- **GT图像**: 2048×2048（4倍于LR）
- **Edge map**: 2048×2048（从GT生成，保持GT分辨率）

### 3. 输出结构

```
outputs/
├── image_edge.png           # ⭐ SR结果
├── edge_maps/               # Edge可视化
│   └── image_edge.png
└── edge_inference_*.log     # 详细日志
```

---

## ❓ 快速问答

**Q: 没有GT图像怎么办？**  
A: 可以从LR生成edge，但效果可能略差（去掉`--gt-img`参数）

**Q: 如何加速推理？**  
A: 减少`--ddpm_steps`（如100），增加`--n_samples`（batch size）

**Q: 如何验证edge是否生效？**  
A: 查看`edge_maps/`目录的edge可视化，对比有无edge的SR结果

**Q: 报错找不到模型？**  
A: 检查`--config`和`--ckpt`路径是否正确

**Q: GPU显存不足？**  
A: 设置`--n_samples 1`，减少`--input_size`

---

## 📚 下一步

- 📖 详细文档: [README.md](README.md)
- 🧪 测试脚本: [test_edge_inference.sh](test_edge_inference.sh)
- 🔍 EdgeMapGenerator: [../EdgeMapGenerator/](../EdgeMapGenerator/)

---

**快速上手完成！开始你的edge推理之旅吧！** 🚀

