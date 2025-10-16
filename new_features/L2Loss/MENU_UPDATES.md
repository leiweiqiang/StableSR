# 菜单更新说明

## 📅 更新：2025-10-16

---

## 🎯 菜单改动

### 之前的菜单（4个选项）

```
1. 推理指定目录下全部 checkpoint (edge & no-edge)
2. 推理指定 checkpoint 文件 (edge)
3. 推理指定 checkpoint 文件 (no-edge)
4. 生成推理结果报告 (CSV格式)
0. 退出
```

### 现在的菜单（3个选项）

```
1. 推理指定目录下全部 checkpoint (edge & no-edge & dummy-edge)
2. 推理指定 checkpoint 文件 (edge & no-edge & dummy-edge)
3. 生成推理结果报告 (CSV格式)
0. 退出
```

---

## 🔄 主要改动

### 改动1：合并选项2和选项3

**之前**：
- 选项2：单个 checkpoint（edge 模式）
- 选项3：单个 checkpoint（no-edge 模式）

**现在**：
- 选项2：单个 checkpoint（**三种模式全部运行**）
  - Edge 模式
  - No-Edge 模式
  - Dummy-Edge 模式

**优势**：
- ✅ 一次性运行三种模式，节省时间
- ✅ 自动计算所有5个指标
- ✅ 结果更完整，便于对比

### 改动2：选项4移至选项3

**之前**：选项4 - 生成报告  
**现在**：选项3 - 生成报告

---

## 📋 选项2的完整功能

### 新的选项2：推理指定 checkpoint 文件

#### 功能流程

```
1. 输入参数
   ├─ Checkpoint 文件路径
   ├─ 输出目录
   ├─ LR 图片目录
   ├─ GT 图片目录
   ├─ Config 文件
   ├─ VQGAN checkpoint
   └─ 其他选项
   ↓
2. 显示配置并确认
   ├─ Checkpoint 路径
   ├─ Epoch 编号
   ├─ 输出目录
   └─ 推理数量
   ↓
3. 确认开始推理
   确认开始推理三种模式? (y/n) [y]:
   ↓
4. 执行三种模式推理
   ├─ [1/3] EDGE 模式
   ├─ [2/3] NO-EDGE 模式
   └─ [3/3] DUMMY-EDGE 模式
   ↓
5. 显示结果统计
   ├─ 各模式成功/失败状态
   ├─ 输出目录
   └─ 指标文件位置
   ↓
6. 生成综合报告（可选）
   是否生成综合报告? (y/n) [y]:
```

#### 使用示例

```bash
./run_auto_inference.sh
选择：2

Checkpoint 路径 [/path/to/checkpoint.ckpt]: /logs/exp/checkpoints/epoch=000083.ckpt
输出目录 [validation_results]: ← 回车
LR 图片目录 [/mnt/nas_dp/test_dataset/128x128_valid_LR]: ← 回车
GT 图片目录 [/mnt/nas_dp/test_dataset/512x512_valid_HR]: ← 回车
Config 文件路径 [configs/.../edge.yaml]: ← 回车
VQGAN 路径 [/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt]: ← 回车

==================================================
  推理配置
==================================================
Checkpoint: /logs/exp/checkpoints/epoch=000083.ckpt
Epoch: 000083
输出目录: validation_results/exp
LR图片: /mnt/nas_dp/test_dataset/128x128_valid_LR
GT图片: /mnt/nas_dp/test_dataset/512x512_valid_HR
推理数量: -1 张
==================================================

确认开始推理三种模式? (y/n) [y]: ← 回车

==================================================
  [1/3] EDGE 模式推理
==================================================
（推理进行中...）
✓ EDGE 模式完成

==================================================
  [2/3] NO-EDGE 模式推理
==================================================
（推理进行中...）
✓ NO-EDGE 模式完成

==================================================
  [3/3] DUMMY-EDGE 模式推理
==================================================
（推理进行中...）
✓ DUMMY-EDGE 模式完成

==================================================
  全部推理完成！
==================================================

结果统计：
  ✓ EDGE 模式: 成功
     输出: validation_results/exp/edge/epochs_83
     指标: metrics.json, metrics.csv
  ✓ NO-EDGE 模式: 成功
     输出: validation_results/exp/no_edge/epochs_83
     指标: metrics.json, metrics.csv
  ✓ DUMMY-EDGE 模式: 成功
     输出: validation_results/exp/dummy_edge/epochs_83
     指标: metrics.json, metrics.csv

所有指标（PSNR, SSIM, LPIPS, Edge PSNR, Edge Overlap）已自动计算

是否生成综合报告? (y/n) [y]: ← 回车

正在生成综合报告...
✓ 报告生成成功: validation_results/exp/exp_inference_report.csv
```

---

## 📊 输出结构

```
validation_results/exp/
├── edge/
│   └── epochs_83/
│       ├── 0801.png
│       ├── ...
│       ├── metrics.json  ← 所有5个指标
│       └── metrics.csv
├── no_edge/
│   └── epochs_83/
│       ├── 0801.png
│       ├── ...
│       ├── metrics.json
│       └── metrics.csv
├── dummy_edge/
│   └── epochs_83/
│       ├── 0801.png
│       ├── ...
│       ├── metrics.json
│       └── metrics.csv
└── exp_inference_report.csv  ← 综合报告
```

---

## ✅ 优势

### 1. 一次运行三种模式

**之前**：需要运行两次（选项2和选项3）  
**现在**：一次运行，自动完成三种模式

### 2. 统一使用 auto_inference.py

**之前**：使用不同的脚本  
**现在**：统一使用 `auto_inference.py`，确保：
- 指标计算一致
- 自动计算所有5个指标
- 代码维护简单

### 3. 自动计算指标

**之前**：需要询问是否计算指标  
**现在**：自动计算，无需询问

**自动计算的指标**：
- PSNR
- SSIM
- LPIPS
- Edge PSNR
- Edge Overlap

### 4. 菜单更简洁

**之前**：4个主要选项  
**现在**：3个主要选项，功能更强

---

## 🎯 使用建议

### 何时使用选项2？

✅ **推荐场景**：
- 测试单个 checkpoint
- 快速验证某个 epoch
- 对比三种 edge 模式效果
- 调试特定 checkpoint

### 何时使用选项1？

✅ **推荐场景**：
- 批量处理多个 checkpoint
- 完整的实验评估
- 定期运行所有新 checkpoint

---

## 📋 完整功能对比

| 功能 | 选项1（批量） | 选项2（单个） | 选项3（报告） |
|-----|-------------|-------------|-------------|
| 处理数量 | 多个 checkpoint | 1个 checkpoint | 不推理 |
| Edge 模式 | ✓ | ✓ | - |
| No-Edge 模式 | ✓ | ✓ | - |
| Dummy-Edge 模式 | ✓ | ✓ | - |
| 计算指标 | 自动 | 自动 | - |
| 批次限制 | 支持 | - | - |
| 重新计算选项 | 支持 | - | - |
| 生成报告 | 自动 | 可选 | 自动 |

---

## ⚠️ 注意事项

### 选项2的限制

选项2**不支持**批次限制（`MAX_CKPTS_PER_RUN`），因为：
- 只处理1个 checkpoint
- 批次限制没有意义

如需批量处理，请使用选项1。

### 指标自动计算

选项2会自动计算所有指标，**不会询问**是否计算。

这是因为：
- 单个 checkpoint 的计算很快
- 通常用户需要完整的指标
- 简化交互流程

---

## 🎊 总结

### 菜单更新

- 选项1：批量推理（增强）
- 选项2：单个 checkpoint 三种模式（合并+增强）
- 选项3：生成报告（移动）
- 删除：旧的选项4

### 主要改进

1. ✅ 合并 edge 和 no-edge 模式
2. ✅ 添加 dummy-edge 模式
3. ✅ 自动计算所有5个指标
4. ✅ 菜单更简洁（3个选项）
5. ✅ 功能更强大

---

**✅ 菜单更新完成，功能更强大，使用更简单！** 🎉

