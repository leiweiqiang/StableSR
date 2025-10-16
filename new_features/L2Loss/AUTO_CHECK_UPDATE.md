# Edge L2 Loss 自动检查和更新功能

## 功能说明

已在 `run_auto_inference.sh` 脚本的**选项1**中添加了自动检查和计算 Edge L2 Loss 指标的功能。

当运行"推理指定目录下全部 checkpoint (edge & no-edge)"时，脚本会：

1. ✅ 正常执行所有推理任务
2. ✅ 自动扫描所有生成的 `metrics.json` 文件
3. ✅ 检查是否包含 `edge_l2_loss` 指标
4. ✅ 如果缺失，自动重新计算并更新文件
5. ✅ 生成综合报告（包含更新后的 Edge L2 Loss）

## 工作流程

```
推理所有 checkpoints
  ├─ EDGE 模式
  ├─ NO-EDGE 模式
  ├─ DUMMY-EDGE 模式
  └─ STABLESR baseline
         ↓
检查和计算 Edge L2 Loss
  ├─ 扫描所有 metrics.json
  ├─ 检查是否有 edge_l2_loss
  ├─ 缺失的自动计算
  └─ 更新 metrics.json 和 metrics.csv
         ↓
生成综合报告 (CSV)
  └─ 包含完整的 Edge L2 Loss 指标
```

## 输出示例

### 推理完成后的检查过程

```bash
==================================================
  检查并计算 Edge L2 Loss 指标
==================================================

扫描目录: validation_results/stablesr_edge_loss_20251015_194003

检查: edge/epochs_27
  → 需要计算 Edge L2 Loss
  开始计算 Edge L2 Loss...
  输出目录: validation_results/.../edge/epochs_27
  GT目录: /mnt/nas_dp/test_dataset/512x512_valid_HR
    ✓ 0801.png: 0.001234
    ✓ 0802.png: 0.001567
    ...
  ✓ 已更新 metrics.json
    平均 Edge L2 Loss: 0.002456
  ✓ 已更新 metrics.csv

  ✓ Edge L2 Loss 计算完成

检查: edge/epochs_83
  ✓ Edge L2 Loss 指标已存在
    如需重新计算，请使用 --force 参数

检查: no_edge/epochs_27
  → 需要计算 Edge L2 Loss
  ...

==================================================
Edge L2 Loss 检查完成
==================================================

统计信息：
  找到 12 个 metrics.json 文件
  ✓ 已更新: 8 个
  ✓ 已存在: 4 个
```

## 新增文件

### 1. `scripts/recalculate_edge_l2_loss.py`

**功能**：检查和重新计算 Edge L2 Loss 指标

**用法**：
```bash
# 基本用法
python scripts/recalculate_edge_l2_loss.py <output_dir> <gt_img_dir>

# 强制重新计算（即使已存在）
python scripts/recalculate_edge_l2_loss.py <output_dir> <gt_img_dir> --force

# 示例
python scripts/recalculate_edge_l2_loss.py \
    validation_results/exp_name/edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

**行为**：
- ✅ 检查 `metrics.json` 是否存在
- ✅ 检查是否已包含 `edge_l2_loss` 和 `average_edge_l2_loss`
- ✅ 如果缺失，读取图片并计算 Edge L2 Loss
- ✅ 更新 `metrics.json` 和 `metrics.csv`
- ✅ 如果已存在且未使用 `--force`，跳过计算

**退出码**：
- `0`：成功（指标已存在或已成功计算）
- `1`：失败（文件不存在、读取失败、计算失败等）

## 修改的文件

### 1. `run_auto_inference.sh`

**修改位置**：选项1 - `inference_all_checkpoints()` 函数

**修改内容**（第 499-582 行）：
- 在所有推理完成后，生成报告之前
- 添加了"检查并计算 Edge L2 Loss 指标"的步骤
- 自动扫描所有 `metrics.json` 文件
- 调用 `recalculate_edge_l2_loss.py` 进行检查和计算
- 统计更新、已存在、失败的数量

## 使用方法

### 自动使用（推荐）

运行推理脚本的选项1，无需任何额外操作：

```bash
./run_auto_inference.sh
# 选择：1. 推理指定目录下全部 checkpoint (edge & no-edge)
```

脚本会自动：
1. 执行推理
2. 检查 Edge L2 Loss
3. 缺失的自动计算
4. 生成包含完整指标的报告

### 手动使用

如果需要单独对某个目录重新计算：

```bash
# 对单个目录
python scripts/recalculate_edge_l2_loss.py \
    validation_results/exp_name/edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR

# 对所有目录（批量）
find validation_results/exp_name -name "metrics.json" -type f | while read f; do
    dir=$(dirname "$f")
    echo "处理: $dir"
    python scripts/recalculate_edge_l2_loss.py "$dir" /mnt/nas_dp/test_dataset/512x512_valid_HR
done
```

## 技术细节

### 检查逻辑

脚本检查 `metrics.json` 文件中：
1. `average_edge_l2_loss` 字段是否存在
2. 每个图片的 `edge_l2_loss` 字段是否存在

**只有当两者都存在时才认为指标完整，否则重新计算。**

### 计算过程

1. 读取现有的 `metrics.json`
2. 遍历每张图片
3. 找到生成图片和 GT 图片
4. 使用 `EdgeL2LossCalculator` 计算 Edge L2 Loss
5. 更新图片数据和平均值
6. 保存到 `metrics.json` 和 `metrics.csv`

### 错误处理

- 如果图片不存在：标记为 `-1.0`，不计入平均值
- 如果计算失败：标记为 `-1.0`，记录错误信息
- 不影响其他图片的计算
- 即使部分失败，也会保存成功的结果

## 与现有功能的关系

### 新推理任务
- ✅ `auto_inference.py` 已集成 Edge L2 Loss 计算
- ✅ 新的推理结果会自动包含该指标
- ✅ 不需要额外检查

### 旧的推理结果
- ✅ `run_auto_inference.sh` 选项1 会自动检查和补充
- ✅ 可以手动运行 `recalculate_edge_l2_loss.py` 补充
- ✅ 不会重复计算已存在的指标（除非使用 `--force`）

### 报告生成
- ✅ `generate_metrics_report.py` 已支持 Edge L2 Loss
- ✅ CSV 报告会包含该指标作为独立的指标块
- ✅ 与 PSNR、SSIM、LPIPS 并列显示

## 性能影响

- **计算时间**：每张 512x512 图片约 10-50ms
- **对 10 张图片**：约 0.1-0.5 秒
- **对 100 个 checkpoint**：约 1-5 分钟
- **相对于推理时间**：几乎可以忽略

## 注意事项

1. **GT 图片路径**：确保 GT 图片目录正确
2. **文件命名**：生成图片和 GT 图片的基础名称要匹配
3. **已存在的指标**：默认不会重新计算，除非使用 `--force`
4. **计算失败**：不影响其他指标和报告生成

## 验证方法

### 检查单个 metrics.json

```bash
# 查看是否包含 edge_l2_loss
cat validation_results/.../metrics.json | grep edge_l2_loss
```

应该看到：
```json
{
  "average_edge_l2_loss": 0.002456,
  "images": [
    {
      "image_name": "0801.png",
      "edge_l2_loss": 0.001234
    }
  ]
}
```

### 检查 CSV 文件

```bash
# 查看 CSV 表头
head -1 validation_results/.../metrics.csv
```

应该包含：`Edge L2 Loss` 列

### 检查综合报告

生成的综合 CSV 报告应该包含 "Edge L2 Loss" 指标块。

## 故障排除

### Q: 脚本报错"EdgeL2LossCalculator"未找到

**A**: 确保：
1. 在正确的 conda 环境中
2. `basicsr/metrics/edge_l2_loss.py` 文件存在
3. Python path 包含项目根目录

### Q: 显示"GT图片不存在"

**A**: 检查：
1. GT 图片目录路径是否正确
2. GT 图片文件名是否与生成图片匹配
3. GT 图片格式（.png 或 .jpg）

### Q: 计算很慢

**A**: 
1. Edge L2 Loss 计算本身很快（10-50ms/图）
2. 如果慢，可能是磁盘 I/O 或图片读取问题
3. 确保在 SSD 上运行

## 总结

✅ **自动化**：运行选项1时自动检查和计算  
✅ **向后兼容**：旧的推理结果自动补充指标  
✅ **智能跳过**：已有指标不重复计算  
✅ **健壮性**：失败不影响其他处理  
✅ **透明性**：详细的输出和统计信息  

现在使用 `run_auto_inference.sh` 的选项1时，会自动确保所有推理结果都包含完整的 Edge L2 Loss 指标！🎉

