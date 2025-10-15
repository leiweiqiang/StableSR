# EdgeInference 目录结构说明

## 📂 完整目录树

```
StableSR_Edge_v3/
│
├── scripts/
│   └── sr_val_edge_inference.py          (31KB) 核心推理脚本 ⭐
│
└── new_features/EdgeInference/
    │
    ├── 📜 测试脚本
    │   ├── test_edge_inference.sh        (8.3KB) 自动化测试（6种配置）
    │   └── example_usage.sh              (10KB)  10个使用示例
    │
    ├── 📖 文档
    │   ├── README.md                     (15KB)  完整使用手册
    │   ├── QUICK_START.md                (3.9KB) 5分钟快速指南
    │   ├── INDEX.md                      (7.3KB) 模块索引
    │   ├── SUMMARY.md                    (12KB)  项目总结
    │   ├── FILE_LOCATION.md              (3KB)   文件位置说明
    │   ├── TEST_DATA_README.md           (7KB)   测试数据说明
    │   └── DIRECTORY_STRUCTURE.md        本文件
    │
    ├── 🖼️ 测试数据目录
    │   ├── lr_images/                    LR测试图像目录
    │   │   ├── README.md                 LR图像说明
    │   │   └── [你的LR测试图像]          用户放置
    │   │
    │   └── gt_images/                    GT测试图像目录
    │       ├── README.md                 GT图像说明
    │       └── [你的GT测试图像]          用户放置
    │
    └── ⚙️ 配置
        └── .gitignore                    Git忽略规则
```

---

## 📋 文件清单

### 核心脚本 (1个)

| 位置 | 文件 | 大小 | 说明 |
|------|------|------|------|
| `../../scripts/` | `sr_val_edge_inference.py` | 31KB | Edge推理核心脚本 ⭐ |

### 测试脚本 (2个)

| 文件 | 大小 | 说明 |
|------|------|------|
| `test_edge_inference.sh` | 8.3KB | 自动化测试脚本（6种配置） |
| `example_usage.sh` | 10KB | 10个实用示例 |

### 文档文件 (7个)

| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | 15KB | 完整使用手册 |
| `QUICK_START.md` | 3.9KB | 5分钟快速指南 |
| `INDEX.md` | 7.3KB | 模块索引导航 |
| `SUMMARY.md` | 12KB | 项目完成总结 |
| `FILE_LOCATION.md` | 3KB | 文件位置说明 |
| `TEST_DATA_README.md` | 7KB | 测试数据准备说明 |
| `DIRECTORY_STRUCTURE.md` | 本文件 | 目录结构说明 |

### 测试数据目录 (2个)

| 目录 | 用途 | 说明 |
|------|------|------|
| `lr_images/` | LR测试图像 | 用户放置低分辨率测试图像 |
| `gt_images/` | GT测试图像 | 用户放置高分辨率参考图像 |

### 配置文件 (1个)

| 文件 | 说明 |
|------|------|
| `.gitignore` | Git忽略规则（忽略测试图像，保留README） |

---

## 🎯 文件用途说明

### 1. 核心推理脚本

**`scripts/sr_val_edge_inference.py`**
- 完全基于 `sr_val_ddpm_text_T_vqganfin_old.py` 的逻辑
- 集成EdgeMapGenerator支持
- 5种edge模式（GT-based/LR-based/Black/Dummy/No edge）
- 批处理、颜色校正、完整日志等功能

### 2. 测试脚本

**`test_edge_inference.sh`**
- 6种预配置测试：basic, quick, batch, no_edge, black_edge, lr_edge
- 自动激活conda环境（sr_infer）
- 彩色输出和友好提示
- 默认使用本地测试目录

**`example_usage.sh`**
- 10个实用示例
- 涵盖各种使用场景
- 详细注释说明

### 3. 文档系统

**`README.md`** - 主文档
- 所有参数详细说明
- 6种详细使用场景
- Edge模式对比
- 常见问题解答

**`QUICK_START.md`** - 快速指南
- 5分钟上手
- 最简命令
- 关键点总结

**`INDEX.md`** - 索引导航
- 模块总览
- 学习路径
- 故障排查

**`SUMMARY.md`** - 项目总结
- 完成的功能
- 技术实现
- 对比说明

**`FILE_LOCATION.md`** - 位置说明
- 核心脚本位置
- 目录组织理念
- 设计优势

**`TEST_DATA_README.md`** - 数据说明
- 测试数据准备
- 图像要求
- 常见问题

### 4. 测试数据目录

**`lr_images/`**
- 存放低分辨率测试图像
- 推荐512×512或更小
- 作为超分辨率的输入

**`gt_images/`**
- 存放高分辨率参考图像
- 推荐2048×2048（LR的4倍）
- 用于生成edge map和结果评估

---

## 🚀 使用流程

### 第一次使用

```bash
# 1. 进入EdgeInference目录
cd /root/dp/StableSR_Edge_v3/new_features/EdgeInference

# 2. 放置测试图像
cp your/lr/*.png lr_images/
cp your/gt/*.png gt_images/

# 3. 查看快速指南
cat QUICK_START.md

# 4. 运行快速测试
./test_edge_inference.sh quick

# 5. 查看结果
ls ../../outputs/edge_inference_test/quick/
```

### 日常使用

```bash
# 方式1: 使用测试脚本（推荐）
cd new_features/EdgeInference
./test_edge_inference.sh basic

# 方式2: 直接运行推理脚本
cd /root/dp/StableSR_Edge_v3
python scripts/sr_val_edge_inference.py \
    --init-img new_features/EdgeInference/lr_images \
    --gt-img new_features/EdgeInference/gt_images \
    --outdir outputs/my_results \
    --use_edge_processing \
    [其他参数...]
```

---

## 📊 目录统计

| 类型 | 数量 | 总大小 |
|------|------|--------|
| Python脚本 | 1 | 31KB |
| Shell脚本 | 2 | ~18KB |
| Markdown文档 | 9 | ~60KB |
| 测试数据目录 | 2 | 用户填充 |
| 配置文件 | 1 | <1KB |
| **总计** | **15** | **~110KB** |

---

## 🔗 重要链接

### 快速访问
- [快速开始](QUICK_START.md) - 5分钟上手
- [测试数据准备](TEST_DATA_README.md) - 准备测试图像
- [LR图像说明](lr_images/README.md) - LR目录说明
- [GT图像说明](gt_images/README.md) - GT目录说明

### 详细文档
- [完整使用手册](README.md) - 详细参数和用法
- [模块索引](INDEX.md) - 模块总览
- [项目总结](SUMMARY.md) - 完成情况

### 脚本位置
- [核心脚本](../../scripts/sr_val_edge_inference.py) - 推理主程序
- [测试脚本](test_edge_inference.sh) - 自动化测试
- [示例脚本](example_usage.sh) - 使用示例

---

## 💡 设计理念

### 1. 模块化组织
- **核心代码**: `scripts/` - 与项目其他脚本统一
- **测试工具**: `new_features/EdgeInference/` - 集中管理
- **清晰分离**: 代码和测试分开，易于维护

### 2. 文档完整
- 多层次文档：快速指南 → 完整手册 → 详细说明
- 每个子目录都有README
- 涵盖所有使用场景

### 3. 开箱即用
- 本地测试目录
- 预配置测试脚本
- 详细的使用说明

### 4. Git友好
- `.gitignore` 忽略测试图像
- 保留README和文档
- 目录结构清晰

---

## 🎉 开始使用

### 首次设置

1. **阅读快速指南**: `QUICK_START.md`
2. **准备测试数据**: 按照 `TEST_DATA_README.md` 准备图像
3. **运行快速测试**: `./test_edge_inference.sh quick`
4. **查看结果**: 检查输出目录

### 深入学习

1. **完整文档**: `README.md` - 了解所有参数
2. **示例脚本**: `example_usage.sh` - 学习不同用法
3. **对比实验**: 运行不同edge模式测试
4. **自定义配置**: 根据需求调整参数

---

**创建日期**: 2025-10-15  
**最后更新**: 2025-10-15  
**状态**: ✅ 完整可用  
**维护者**: StableSR_Edge_v3 Team

