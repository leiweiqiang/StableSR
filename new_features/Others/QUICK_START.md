# StableSR Edge 快速开始指南

## 🚀 30秒快速开始

```bash
# 1. 激活环境
conda activate sr_infer

# 2. 运行脚本
./run_auto_inference.sh

# 3. 4次回车完成
选择: 1
目录: ← 回车
输出: ← 回车
重新计算指标? [n]: ← 回车
选择checkpoint [all]: ← 回车（推理所有）

# ✅ 完成！自动计算5个指标并生成报告
```

---

## 📊 支持的5个指标

| 指标 | 说明 | 值域 | 方向 |
|-----|------|------|------|
| PSNR | 图像质量 | [0,∞) dB | ↑ 越大越好 |
| SSIM | 结构相似度 | [0,1] | ↑ 越大越好 |
| LPIPS | 感知质量 | [0,∞) | ↓ 越小越好 |
| **Edge PSNR** | 边缘精确度 | [0,∞) dB | ↑ 越大越好 |
| **Edge Overlap** | 边缘覆盖率 | [0,1] | ↑ 越大越好 |

---

## 🎯 4个选择点

### 1. 重新计算指标？[默认: n]

- **n**：快速模式（推荐日常使用）
- **y**：完整模式（推荐数据整理）

### 2. 选择 checkpoint？[默认: all]

- **回车/all**：推理所有（推荐）
- **单个序号**：推理单个（如：3）
- **多个序号**：逗号分隔（如：1,3,5）
- **q**：取消

### 3. （如有新checkpoint）确认开始？

- 自动显示选中的 checkpoint
- 确认后开始推理

---

## 💡 常用场景

### 场景A：日常推理（最快）

```bash
./run_auto_inference.sh
1 → 回车×4
# 推理所有新 checkpoint
```

### 场景B：测试单个

```bash
./run_auto_inference.sh
1 → 回车×3 → 3 → 回车
# 只推理第3个 checkpoint
```

### 场景C：选择多个

```bash
./run_auto_inference.sh
1 → 回车×3 → 1,3,5 → 回车
# 推理第 1、3、5 个
```

### 场景D：完整检查

```bash
./run_auto_inference.sh
1 → 回车×2 → y → 回车×2
# 确保所有数据完整
```

---

## 📁 输出文件

### 每个 epoch 目录
- `metrics.json` - 所有指标
- `metrics.csv` - CSV格式

### 实验根目录
- `..._inference_report.csv` - 综合报告（按epoch排序）

---

## 📚 详细文档

- 📖 **METRICS_SUMMARY.md** - 指标总览
- 📖 **EDGE_METRICS_COMPLETE.md** - 完整实现
- 📖 **RUN_AUTO_INFERENCE_FEATURES.md** - 功能详解
- 📖 **new_features/L2Loss/** - 23个详细文档

---

## ⚠️ 重要提醒

1. **环境**：必须在 `sr_infer` conda 环境中
2. **方向**：Edge PSNR 和 Edge Overlap 都是越大越好
3. **默认**：所有默认值都是合理的，直接回车即可

---

**🎉 就这么简单！5次回车，全自动完成！** ✨

