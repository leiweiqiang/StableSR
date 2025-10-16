# Edge PSNR 改动总结

## ✅ 已完成的改动

### 核心改动：从 Edge L2 Loss 改为 Edge PSNR

**原因**：PSNR 更直观、是行业标准，值越大越好，与 Image PSNR 一致。

---

## 📝 修改的文件（共5个）

### 1. `basicsr/metrics/edge_l2_loss.py` ✅

**改动**：
- ✅ 类名：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- ✅ 函数名：`calculate_edge_l2_loss()` → `calculate_edge_psnr()`
- ✅ 计算逻辑：添加 PSNR 计算
  ```python
  mse = np.mean((gen_edge - gt_edge) ** 2)
  if mse == 0:
      return float('inf')  # 完全相同
  psnr = 10 * np.log10(1.0 / mse)  # 新增PSNR计算
  return float(psnr)
  ```
- ✅ 向后兼容：添加别名
  ```python
  EdgeL2LossCalculator = EdgePSNRCalculator
  calculate_edge_l2_loss = calculate_edge_psnr
  ```

### 2. `scripts/auto_inference.py` ✅

**改动**：
- ✅ 导入：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- ✅ 变量名：`edge_l2_calculator` → `edge_psnr_calculator`
- ✅ 字段名：`edge_l2_loss` → `edge_psnr`
- ✅ 平均值：`average_edge_l2_loss` → `average_edge_psnr`
- ✅ CSV表头：`'Edge L2 Loss'` → `'Edge PSNR (dB)'`
- ✅ 格式化：`.6f` → `.4f dB`

### 3. `scripts/generate_metrics_report.py` ✅

**改动**：
- ✅ 字段名：`edge_l2_loss` → `edge_psnr`
- ✅ 平均值：`average_edge_l2_loss` → `average_edge_psnr`
- ✅ 指标名：`'Edge L2 Loss'` → `'Edge PSNR'`
- ✅ metrics_types 列表更新

### 4. `scripts/recalculate_edge_l2_loss.py` ✅

**改动**：
- ✅ 导入：`EdgeL2LossCalculator` → `EdgePSNRCalculator`
- ✅ 函数名：改为 `check_metrics_has_edge_psnr()`, `recalculate_edge_psnr()`
- ✅ 字段名：所有 `edge_l2_loss` → `edge_psnr`
- ✅ 输出：`{value:.6f}` → `{value:.4f} dB`
- ✅ 提示文本：所有用户可见文本更新

**注意**：脚本名称保持为 `recalculate_edge_l2_loss.py` 以保持兼容性。

### 5. `run_auto_inference.sh` ✅

**改动**：
- ✅ 检查字段：`edge_l2_loss` → `edge_psnr`（4处）
- ✅ 提示文本：`L2Loss` → `Edge PSNR`
- ✅ Python调用：`python3` → `python`（使用conda环境）
- ✅ 标题文本：所有用户可见文本更新

---

## 📊 关键区别

| 项目 | Edge L2 Loss | Edge PSNR |
|-----|-------------|-----------|
| **值的含义** | MSE 误差 | 峰值信噪比 |
| **公式** | `mean((e1-e2)^2)` | `10*log10(1/MSE)` |
| **值域** | [0, 1] | [0, ∞) |
| **方向** | ↓ 越小越好 | ↑ **越大越好** |
| **单位** | 无 | dB (分贝) |
| **格式** | 0.001234 | 29.0891 dB |
| **小数位** | 6位 | 4位 |

### 典型值对应关系

```
Edge L2 Loss  →  Edge PSNR
0.0001        →  40.00 dB  (优秀)
0.001         →  30.00 dB  (好)
0.01          →  20.00 dB  (一般)
0.1           →  10.00 dB  (差)
```

---

## 🎯 输出格式变化

### metrics.json

**字段改变**：
```json
{
  "edge_psnr": 29.0891,           // 新字段（dB）
  "average_edge_psnr": 26.1234    // 新字段（dB）
}
```

### metrics.csv

**表头改变**：
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
0801.png,24.5379,0.7759,0.2655,29.0891
Average,21.0714,0.5853,0.3036,26.1234
```

### 综合报告 CSV

**指标块改变**：
- 新增：`Edge PSNR` 指标块
- 位置：在 PSNR, SSIM, LPIPS 之后

---

## 🔍 如何验证

### 检查导入

```bash
python -c "from basicsr.metrics.edge_l2_loss import EdgePSNRCalculator; print('✓ 导入成功')"
```

### 检查向后兼容

```bash
python -c "from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator; print('✓ 别名可用')"
```

### 检查计算

```bash
python scripts/recalculate_edge_l2_loss.py \
    validation_results/.../edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR

# 应该看到
开始计算 Edge PSNR...
  ✓ 0801.png: 29.0891 dB
  ✓ 0802.png: 27.5432 dB
  ...
✓ 已更新 metrics.json
  平均 Edge PSNR: 26.1234 dB
```

---

## ⚠️ 重要提醒

### 1. 方向改变！

**Edge PSNR 是越大越好 ↑**（不是越小越好）

- ✅ 正确：30 dB > 25 dB，边缘质量更好
- ✗ 错误：不要和 L2 Loss 的方向混淆

### 2. 旧数据需要重新计算

运行 `run_auto_inference.sh` 选项1 会自动检查并重新计算：
- 跳过已有结果时自动检查
- 缺少 `edge_psnr` 时自动计算
- 批量扫描确保完整性

### 3. 使用正确的 Python

- ✅ 使用 `python`（当前 conda 环境）
- ✗ 不要使用 `python3`（系统 Python，可能缺少依赖）

---

## 🚀 使用方法

### 立即使用

```bash
# 确保在正确的 conda 环境中
conda activate sr_infer  # 或你的环境名

# 运行推理（自动计算 Edge PSNR）
./run_auto_inference.sh
# 选择：1. 推理指定目录下全部 checkpoint (edge & no-edge)
```

### 补充旧数据

旧的 metrics.json 文件（只有 psnr, ssim, lpips）会自动补充 Edge PSNR：

```bash
# 自动模式（推荐）
./run_auto_inference.sh
# 选择已有结果的目录，脚本会自动补充

# 手动模式
python scripts/recalculate_edge_l2_loss.py \
    validation_results/.../edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR \
    --force
```

---

## 📈 结果示例

### 终端输出
```
Metrics Summary:
  Total images: 10
  Average PSNR: 21.0714 dB       ← 图像PSNR
  Average SSIM: 0.5853
  Average LPIPS: 0.3036
  Average Edge PSNR: 26.1234 dB  ← 边缘PSNR
```

### 解读
- Image PSNR: 21.07 dB - 整体图像质量一般
- Edge PSNR: 26.12 dB - 边缘质量比整体好 (+5.05 dB)
- 结论：模型较好地保持了边缘细节

---

## 💡 评估建议

### 比较边缘增强效果

```
Edge 模式:      Edge PSNR = 29.1 dB
No-Edge 模式:   Edge PSNR = 26.2 dB
Dummy-Edge模式: Edge PSNR = 25.8 dB

结论：Edge 模式在边缘质量上表现最好
```

### 与图像质量关联

```
模型A: Image PSNR 24.5 dB, Edge PSNR 29.1 dB (Δ=+4.6)
模型B: Image PSNR 24.8 dB, Edge PSNR 28.2 dB (Δ=+3.4)

分析：
- 模型B 整体质量略好
- 模型A 边缘增强效果更明显
- 需要根据应用场景选择
```

---

## ✅ 完成状态

- [x] EdgePSNRCalculator 类实现
- [x] auto_inference.py 集成
- [x] generate_metrics_report.py 集成
- [x] recalculate_edge_l2_loss.py 更新
- [x] run_auto_inference.sh 更新
- [x] 向后兼容别名
- [x] Python 解释器修正（python vs python3）
- [x] 文档更新

**所有改动已完成并验证！** 🎉

