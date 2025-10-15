# EdgeInference 文件位置说明

## 📁 文件组织

### 核心推理脚本

**位置**: `scripts/sr_val_edge_inference.py` (31KB) ⭐

这是edge推理的核心Python脚本，与项目中其他推理脚本放在一起。

**路径**: `/root/dp/StableSR_Edge_v3/scripts/sr_val_edge_inference.py`

---

### 测试脚本和文档

**位置**: `new_features/EdgeInference/`

| 文件 | 大小 | 说明 |
|------|------|------|
| `test_edge_inference.sh` | 8.3KB | 自动化测试脚本（6种配置） |
| `example_usage.sh` | 10KB | 10个使用示例 |
| `README.md` | 15KB | 完整使用文档 |
| `QUICK_START.md` | 3.9KB | 5分钟快速指南 |
| `INDEX.md` | 7.3KB | 模块索引 |
| `SUMMARY.md` | 12KB | 项目总结 |
| `FILE_LOCATION.md` | 本文件 | 文件位置说明 |

---

## 🚀 使用方法

### 方式1: 直接运行推理脚本

```bash
cd /root/dp/StableSR_Edge_v3

python scripts/sr_val_edge_inference.py \
    --init-img inputs/lr \
    --gt-img inputs/gt \
    --outdir outputs/results \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/your_model.ckpt
```

### 方式2: 使用测试脚本

```bash
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference

# 查看可用测试
./test_edge_inference.sh help

# 运行快速测试
./test_edge_inference.sh quick

# 运行基础测试
./test_edge_inference.sh basic
```

---

## 📂 完整目录结构

```
StableSR_Edge_v3/
│
├── scripts/
│   ├── sr_val_ddpm_text_T_vqganfin_old.py          (原始脚本)
│   ├── sr_val_ddpm_text_T_vqganfin_old_edge.py     (edge参考)
│   └── sr_val_edge_inference.py                    (新edge脚本) ⭐
│
└── new_features/
    ├── EdgeInference/
    │   ├── test_edge_inference.sh                  (测试脚本)
    │   ├── example_usage.sh                        (示例脚本)
    │   ├── README.md                               (完整文档)
    │   ├── QUICK_START.md                          (快速指南)
    │   ├── INDEX.md                                (模块索引)
    │   ├── SUMMARY.md                              (项目总结)
    │   └── FILE_LOCATION.md                        (本文件)
    │
    ├── EdgeMapGenerator/                           (Edge生成器)
    └── EdgeMonitorCallback/                        (训练监控)
```

---

## 💡 设计理念

### 为什么分开存放？

1. **核心脚本** (`scripts/sr_val_edge_inference.py`)
   - 与其他推理脚本统一管理
   - 方便直接调用
   - 符合项目结构规范

2. **测试和文档** (`new_features/EdgeInference/`)
   - 集中管理测试配置
   - 完整的文档体系
   - 易于维护和扩展

### 优势

✅ **结构清晰**: 核心代码与测试文档分离  
✅ **易于查找**: 推理脚本在统一的scripts目录  
✅ **便于维护**: 文档和测试集中在new_features  
✅ **符合规范**: 遵循项目现有的目录组织方式

---

## 🔗 相关链接

- **快速开始**: [QUICK_START.md](QUICK_START.md)
- **完整文档**: [README.md](README.md)
- **模块索引**: [INDEX.md](INDEX.md)
- **项目总结**: [SUMMARY.md](SUMMARY.md)

---

**更新日期**: 2025-10-15  
**核心脚本位置**: `scripts/sr_val_edge_inference.py`  
**文档位置**: `new_features/EdgeInference/`

