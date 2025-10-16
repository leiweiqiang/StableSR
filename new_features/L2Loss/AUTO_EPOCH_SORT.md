# CSV 报告自动 Epoch 排序

## 📅 更新：2025-10-16

## 🎯 改进内容

修改了 `scripts/generate_metrics_report.py`，使 CSV 报告中的 epoch 列能够**自动按序号横向排列**，而不是使用硬编码的顺序。

## 📊 CSV 格式说明

### 当前格式（已优化）

```csv
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,Epoch 55,Epoch 55,Epoch 55,Epoch 83,Epoch 83,Epoch 83,...
,,,dummy edge,edge,no edge,dummy edge,edge,no edge,dummy edge,edge,no edge,...
PSNR,Average,20.9199,20.2555,20.3397,20.2841,20.3153,20.3651,20.3145,21.0714,21.0870,21.0741,...
,0801.png,23.5642,21.4285,21.5756,21.3972,22.9424,22.8213,22.7250,24.5379,24.4243,24.4236,...
```

**特点**：
- 第一行：Epoch 编号（重复3次，对应3种edge模式）
- 第二行：edge 类型（dummy edge, edge, no edge）
- StableSR 放在最前面
- Epoch 按**数字顺序**自动排列：27 → 55 → 83 → 111 → ...

## 🔄 改进对比

### 之前（硬编码）

```python
column_order = [
    "StableSR",
    "Epoch 47 (edge)",
    "Epoch 47 (no edge)",
    "Epoch 47 (dummy edge)",
    "Epoch 95 (edge)",
    ...  # 手动添加每个 epoch
]
```

**问题**：
- ❌ 需要手动编辑代码
- ❌ 只支持预定义的 epoch
- ❌ 新的 epoch 不会自动出现
- ❌ 顺序固定，不够灵活

### 现在（自动排序）

```python
# 自动提取所有 epoch 编号
epoch_info = {}  # {27: {"edge": "Epoch 27 (edge)", ...}, ...}

# 按数字顺序排列
for epoch_num in sorted(epoch_info.keys()):
    # 对每个 epoch，按 edge → no edge → dummy edge 顺序
    for edge_type in ["edge", "no edge", "dummy edge"]:
        column_order.append(...)
```

**优势**：
- ✅ 自动提取 epoch 编号
- ✅ 自动按数字顺序排列
- ✅ 支持任意数量的 epoch
- ✅ 新 epoch 自动包含
- ✅ 灵活且易维护

## 🎨 排序逻辑

### 1. 提取 Epoch 信息

```python
# 对每个列名（如 "Epoch 27 (edge)"）
match = re.search(r'Epoch\s+(\d+)', column)
if match:
    epoch_num = int(match.group(1))  # 提取数字 27
    
    # 提取 edge 类型
    if "(edge)" in column:
        edge_type = "edge"
    elif "(no edge)" in column:
        edge_type = "no edge"
    elif "(dummy edge)" in column:
        edge_type = "dummy edge"
    
    # 存储
    epoch_info[epoch_num][edge_type] = column
```

### 2. 按序号排序

```python
# StableSR 放在最前面
column_order = ["StableSR"]

# Epoch 按数字排序：27, 55, 83, 111, ...
for epoch_num in sorted(epoch_info.keys()):
    # 每个 epoch 内部按固定顺序
    for edge_type in ["edge", "no edge", "dummy edge"]:
        if edge_type in epoch_info[epoch_num]:
            column_order.append(epoch_info[epoch_num][edge_type])
```

### 3. 最终顺序

```
StableSR → Epoch 27 (edge) → Epoch 27 (no edge) → Epoch 27 (dummy edge)
        → Epoch 55 (edge) → Epoch 55 (no edge) → Epoch 55 (dummy edge)
        → Epoch 83 (edge) → Epoch 83 (no edge) → Epoch 83 (dummy edge)
        → ...
```

## 📋 实现细节

### 代码位置

文件：`scripts/generate_metrics_report.py`

**修改函数**：`generate_csv_report()`（第 158-217 行）

### 核心逻辑

```python
# 1. 收集所有 epoch 和 edge 类型
epoch_info = {}  # {epoch_num: {edge_type: column_name}}

# 2. 遍历所有列名，提取信息
for column in all_columns:
    match = re.search(r'Epoch\s+(\d+)', column)
    if match:
        epoch_num = int(match.group(1))
        # 提取 edge_type 并存储

# 3. 构建排序后的列顺序
column_order = []
if stablesr_col:
    column_order.append("StableSR")  # 最前

for epoch_num in sorted(epoch_info.keys()):  # 数字排序
    for edge_type in ["edge", "no edge", "dummy edge"]:  # 固定顺序
        if edge_type in epoch_info[epoch_num]:
            column_order.append(epoch_info[epoch_num][edge_type])
```

## 🎯 使用效果

### 示例数据

假设有以下 epoch：
- Epoch 27, 55, 83, 111, 138, 166

### 生成的 CSV 表头

```csv
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,Epoch 55,Epoch 55,Epoch 55,Epoch 83,Epoch 83,Epoch 83,Epoch 111,Epoch 111,Epoch 111,Epoch 138,Epoch 138,Epoch 138,Epoch 166,Epoch 166,Epoch 166
,,,dummy edge,edge,no edge,dummy edge,edge,no edge,dummy edge,edge,no edge,dummy edge,edge,no edge,dummy edge,edge,no edge,dummy edge,edge,no edge
```

**注意**：
- Epoch 按数字顺序：27 → 55 → 83 → 111 → 138 → 166
- 每个 Epoch 下有3列：dummy edge, edge, no edge
- 自动适应任意数量的 epoch

### 数据行示例

```csv
PSNR,Average,20.92,20.26,20.34,20.28,20.32,20.37,20.31,21.07,21.09,21.07,20.85,20.92,20.88,21.15,21.21,21.18,21.34,21.39,21.36
,0801.png,23.56,21.43,21.58,21.40,22.94,22.82,22.73,24.54,24.42,24.42,23.89,23.95,23.91,24.67,24.73,24.69,24.89,24.94,24.91
```

## ✅ 优势

### 1. 自动化
- ✅ 无需手动编辑列顺序
- ✅ 新 epoch 自动包含
- ✅ 自动数字排序

### 2. 灵活性
- ✅ 支持任意数量的 epoch
- ✅ 支持任意 epoch 编号
- ✅ 即使 epoch 不连续也能正确排序

### 3. 正确性
- ✅ 严格按数字排序（不是字符串排序）
- ✅ 27 → 55 → 111（而不是 111 → 27 → 55）
- ✅ 每个 epoch 内部顺序一致

### 4. 易维护
- ✅ 代码更简洁
- ✅ 逻辑更清晰
- ✅ 易于理解和修改

## 🔍 验证示例

### 测试数据

假设有以下 metrics.json 文件：
- `edge/epochs_27/metrics.json`
- `edge/epochs_111/metrics.json`
- `edge/epochs_55/metrics.json`
- `no_edge/epochs_27/metrics.json`
- ...

### 生成的列顺序

```python
column_order = [
    "StableSR",
    "Epoch 27 (dummy edge)",
    "Epoch 27 (edge)",
    "Epoch 27 (no edge)",
    "Epoch 55 (dummy edge)",
    "Epoch 55 (edge)",
    "Epoch 55 (no edge)",
    "Epoch 111 (dummy edge)",
    "Epoch 111 (edge)",
    "Epoch 111 (no edge)"
]
```

**注意**：按数字排序，所以 27 → 55 → 111，而不是 111 → 27 → 55

## 📖 代码示例

### 完整实现

```python
def generate_csv_report(metrics_data, image_files, output_path):
    # Step 1: 收集所有 epoch 信息
    epoch_info = {}  # {epoch_num: {edge_type: column_name}}
    stablesr_col = None
    
    for metric_type in metrics_data:
        for column in metrics_data[metric_type]:
            if column == "StableSR":
                stablesr_col = column
                continue
            
            # 提取 epoch 编号
            match = re.search(r'Epoch\s+(\d+)', column)
            if match:
                epoch_num = int(match.group(1))
                
                # 提取 edge 类型
                if "(edge)" in column and "(no edge)" not in column:
                    edge_type = "edge"
                elif "(no edge)" in column:
                    edge_type = "no edge"
                elif "(dummy edge)" in column:
                    edge_type = "dummy edge"
                
                if epoch_num not in epoch_info:
                    epoch_info[epoch_num] = {}
                epoch_info[epoch_num][edge_type] = column
    
    # Step 2: 构建排序后的列顺序
    column_order = []
    
    if stablesr_col:
        column_order.append(stablesr_col)
    
    # 按数字排序 epoch
    for epoch_num in sorted(epoch_info.keys()):
        # 每个 epoch 内按固定顺序
        for edge_type in ["edge", "no edge", "dummy edge"]:
            if edge_type in epoch_info[epoch_num]:
                column_order.append(epoch_info[epoch_num][edge_type])
    
    # Step 3: 使用 column_order 生成 CSV
    ...
```

## 🚀 使用方法

### 生成报告

```bash
# 方法1：通过 run_auto_inference.sh
./run_auto_inference.sh
# 选择：1
# 完成推理后自动生成报告（自动排序）

# 方法2：单独生成报告
python scripts/generate_metrics_report.py \
    validation_results/stablesr_edge_loss_20251015_194003
```

### 查看结果

```bash
# 查看生成的 CSV
cat validation_results/.../..._inference_report.csv | head -3

# 应该看到 epoch 按顺序排列
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,Epoch 55,...
```

## 📋 排序规则

### 主要规则

1. **StableSR 最前**：作为 baseline 对比
2. **Epoch 按数字排序**：27 → 55 → 83 → 111 → ...
3. **每个 Epoch 内部固定顺序**：
   - dummy edge（如果有）
   - edge
   - no edge

### 处理边界情况

- **只有部分 edge 类型**：只显示存在的
- **epoch 不连续**：正常显示（如 27, 83, 166）
- **新增 epoch**：自动添加到正确位置
- **缺失某个模式**：该列不显示

## ✅ 优势总结

### 1. 自动化
- 无需手动维护 epoch 列表
- 新 checkpoint 自动包含
- 删除旧 checkpoint 自动移除

### 2. 正确性
- 严格数字排序（27 → 111，不是 111 → 27）
- 一致的内部顺序
- 准确的对应关系

### 3. 灵活性
- 支持任意数量 epoch
- 支持不连续的 epoch 编号
- 支持部分 edge 模式

### 4. 易用性
- 无需配置
- 自动工作
- 结果清晰易读

## 🎯 实际效果

### 有新 Epoch 时

**之前**：需要编辑代码，添加新 epoch 到列表

**现在**：自动检测并按序号插入正确位置

### Epoch 顺序

**之前**：可能需要手动调整顺序

**现在**：自动按数字排序，始终正确

### 报告生成

**之前**：可能遗漏新 epoch

**现在**：自动包含所有 epoch

---

## 📚 相关文档

- 📖 脚本位置：`scripts/generate_metrics_report.py`
- 📖 使用指南：`new_features/L2Loss/USER_GUIDE.md`
- 📖 Edge PSNR 文档：`new_features/L2Loss/EDGE_PSNR_QUICKREF.md`

---

**✅ CSV 报告现在能够自动按 epoch 序号横向排列，无需手动维护！** 🎉

