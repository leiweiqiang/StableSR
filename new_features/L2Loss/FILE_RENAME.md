# 文件重命名说明

## 📅 更新：2025-10-16

---

## 🔄 文件重命名

### 改动

```bash
# 之前
scripts/recalculate_edge_l2_loss.py

# 现在
scripts/recalculate_metrics.py
```

---

## 🎯 重命名原因

### 1. 功能已扩展

**之前**：只计算 Edge L2 Loss  
**现在**：计算多个指标
- ✅ Edge PSNR
- ✅ Edge Overlap
- ✅ 可扩展到其他指标

### 2. 名称更准确

- `recalculate_edge_l2_loss` → 暗示只计算 L2 Loss（已过时）
- `recalculate_metrics` → 通用的指标重新计算（准确）

### 3. 符合功能定位

脚本现在的功能：
- 检查所有必需的指标
- 重新计算缺失的指标
- 更新 metrics.json 和 metrics.csv

名称 `recalculate_metrics.py` 更好地反映了这些功能。

---

## 📝 脚本功能

### 当前功能

```python
# 检查指标完整性
check_metrics_complete(metrics_file)
# 检查：PSNR, SSIM, Edge PSNR, Edge Overlap
# 返回：是否完整 + 缺失的指标列表

# 重新计算指标
recalculate_edge_metrics(output_dir, gt_img_dir)
# 计算：Edge PSNR, Edge Overlap
# 更新：metrics.json, metrics.csv
```

### 使用方法

```bash
# 基本用法
python scripts/recalculate_metrics.py <output_dir> <gt_img_dir>

# 强制重新计算
python scripts/recalculate_metrics.py <output_dir> <gt_img_dir> --force

# 示例
python scripts/recalculate_metrics.py \
    validation_results/exp/edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

---

## 🔧 相关改动

### 更新的引用位置

**`run_auto_inference.sh`** 中的所有调用都已更新：

1. **跳过时检查**（4处）：
   ```bash
   python scripts/recalculate_metrics.py "$OUTPUT_CHECK" "$DEFAULT_GT_IMG"
   ```

2. **批量扫描**（1处）：
   ```bash
   python scripts/recalculate_metrics.py "$METRICS_DIR" "$DEFAULT_GT_IMG"
   ```

---

## 📊 输出示例

### 发现缺失指标

```bash
python scripts/recalculate_metrics.py validation_results/.../edge/epochs_27 /path/to/gt

输出：
→ 发现缺失的指标: edge_overlap
→ 需要重新计算指标: validation_results/.../edge/epochs_27
开始计算 Edge PSNR 和 Edge Overlap...
  ✓ 0801.png: PSNR=29.0891 dB, Overlap=0.8523
  ✓ 0802.png: PSNR=27.3456 dB, Overlap=0.7654
  ...
✓ 已更新 metrics.json
  平均 Edge PSNR: 26.1234 dB
  平均 Edge Overlap: 0.7891
✓ 已更新 metrics.csv

✓ Edge 相关指标计算完成
```

### 所有指标已存在

```bash
python scripts/recalculate_metrics.py validation_results/.../edge/epochs_55 /path/to/gt

输出：
✓ 所有指标已存在: validation_results/.../edge/epochs_55
  如需重新计算，请使用 --force 参数
```

### 强制重新计算

```bash
python scripts/recalculate_metrics.py validation_results/.../edge/epochs_83 /path/to/gt --force

输出：
→ 强制重新计算所有指标: validation_results/.../edge/epochs_83
开始计算 Edge PSNR 和 Edge Overlap...
  ✓ 0801.png: PSNR=29.0891 dB, Overlap=0.8523
  ...
✓ Edge 相关指标计算完成
```

---

## ✅ 优势

### 1. 名称更清晰
- 直观反映脚本功能
- 易于理解和记忆

### 2. 功能更通用
- 不限于 Edge L2 Loss
- 可扩展到更多指标

### 3. 易于维护
- 未来添加新指标时不需要改名
- 代码和功能保持一致

---

## 📋 完整的文件结构

```
scripts/
├── auto_inference.py          # 自动计算所有指标
├── generate_metrics_report.py # 生成 CSV 报告
└── recalculate_metrics.py     # 重新计算缺失的指标 ← 新名称

basicsr/metrics/
├── psnr_ssim.py               # PSNR, SSIM
├── edge_l2_loss.py            # Edge PSNR (文件名保留兼容性)
└── edge_overlap.py            # Edge Overlap
```

---

## 📖 相关文档

- 📘 使用说明：`new_features/L2Loss/EDGE_OVERLAP_README.md`
- 📗 用户指南：`new_features/L2Loss/USER_GUIDE.md`
- 📙 完整更新：`new_features/L2Loss/COMPLETE_CHANGELOG.md`

---

## 🚀 立即使用

```bash
# 在 run_auto_inference.sh 中自动调用
./run_auto_inference.sh
# 选择：1
# 重新计算指标? [n]: y

# 或手动调用
python scripts/recalculate_metrics.py \
    validation_results/exp/edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

---

**✅ 文件重命名完成，所有引用已更新！** 🎉

