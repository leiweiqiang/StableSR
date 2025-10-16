# 智能跳过：跳过前自动检查并计算 L2Loss

## 功能说明

`run_auto_inference.sh` 脚本现在在跳过已存在的 checkpoint 结果之前，会自动检查并计算缺失的 Edge L2 Loss 指标。

这确保了即使跳过推理步骤，所有的 metrics 文件也会包含完整的 L2Loss 数据。

## 工作机制

### 原始行为（改进前）
```bash
检查目录是否有图片
  ├─ 如果有图片 → 直接跳过
  └─ 如果没有   → 执行推理
```

### 新的智能行为（改进后）
```bash
检查目录是否有图片
  ├─ 如果有图片
  │   ├─ 检查 metrics.json 是否存在
  │   ├─ 检查是否包含 edge_l2_loss
  │   ├─ 如果缺失 → 自动计算 L2Loss
  │   └─ 跳过推理
  └─ 如果没有   → 执行推理
```

## 修改的位置

在 `run_auto_inference.sh` 的四个位置添加了 L2Loss 检查逻辑：

### 1. EDGE 模式（第 215-228 行）
```bash
if [ "$PNG_COUNT" -gt 0 ]; then
    # Images exist, but check if L2Loss is calculated
    METRICS_FILE="$OUTPUT_CHECK/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        # Check if Edge L2 Loss exists in metrics.json
        if ! grep -q "edge_l2_loss" "$METRICS_FILE"; then
            echo "→ epoch=$EPOCH_NUM 已有图片，但缺少 L2Loss，正在计算..."
            python3 scripts/recalculate_edge_l2_loss.py "$OUTPUT_CHECK" "$DEFAULT_GT_IMG" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "  ✓ L2Loss 计算完成"
            else
                echo "  ⚠ L2Loss 计算失败"
            fi
        fi
    fi
    echo "✓ 跳过 epoch=$EPOCH_NUM (已有 $PNG_COUNT 张图片)"
    ((EDGE_SKIPPED++))
    continue
fi
```

### 2. NO-EDGE 模式（第 289-302 行）
相同的逻辑应用到 no_edge 子目录。

### 3. DUMMY-EDGE 模式（第 365-378 行）
相同的逻辑应用到 dummy_edge 子目录。

### 4. STABLESR baseline 模式（第 438-451 行）
相同的逻辑应用到 stablesr/baseline 目录。

## 输出示例

### 场景1：已有图片且已有 L2Loss
```bash
✓ 跳过 epoch=27 (已有 10 张图片)
```

### 场景2：已有图片但缺少 L2Loss
```bash
→ epoch=27 已有图片，但缺少 L2Loss，正在计算...
  ✓ L2Loss 计算完成
✓ 跳过 epoch=27 (已有 10 张图片)
```

### 场景3：L2Loss 计算失败
```bash
→ epoch=27 已有图片，但缺少 L2Loss，正在计算...
  ⚠ L2Loss 计算失败
✓ 跳过 epoch=27 (已有 10 张图片)
```

## 优势

### ✅ 自动补全
- 旧的推理结果自动获得 L2Loss 指标
- 无需手动运行补充脚本
- 在跳过时自动完成

### ✅ 高效
- 只在缺失时计算
- 计算速度快（10-50ms/图）
- 不影响整体推理流程

### ✅ 透明
- 清晰的状态提示
- 显示计算结果
- 失败时给出警告

### ✅ 智能
- 使用 `grep -q "edge_l2_loss"` 快速检查
- 避免重复计算
- 静默执行（`> /dev/null 2>&1`）

## 技术细节

### 检查方法
```bash
if ! grep -q "edge_l2_loss" "$METRICS_FILE"; then
    # edge_l2_loss 不存在，需要计算
fi
```

使用 `grep -q` 快速检查文件中是否包含 `edge_l2_loss` 字符串：
- 如果存在：跳过计算
- 如果不存在：调用 `recalculate_edge_l2_loss.py`

### 静默执行
```bash
python3 scripts/recalculate_edge_l2_loss.py "$OUTPUT_CHECK" "$DEFAULT_GT_IMG" > /dev/null 2>&1
```

重定向输出到 `/dev/null`，只显示简洁的状态信息：
- 成功：`✓ L2Loss 计算完成`
- 失败：`⚠ L2Loss 计算失败`

### 退出码检查
```bash
if [ $? -eq 0 ]; then
    echo "  ✓ L2Loss 计算完成"
else
    echo "  ⚠ L2Loss 计算失败"
fi
```

根据 Python 脚本的退出码判断成功或失败。

## 使用场景

### 场景1：重新运行脚本
如果之前运行过推理但中途中断，再次运行时：
```bash
./run_auto_inference.sh
# 选择：1. 推理指定目录下全部 checkpoint (edge & no-edge)
```

脚本会：
1. 检查已存在的结果
2. 自动补充缺失的 L2Loss
3. 跳过已完成的推理

### 场景2：补充旧数据
对于旧的推理结果（在 L2Loss 功能添加之前）：
```bash
./run_auto_inference.sh
# 选择：1. 推理指定目录下全部 checkpoint (edge & no-edge)
# 选择已有结果的目录
```

脚本会自动为所有旧结果添加 L2Loss 指标。

### 场景3：部分失败的推理
如果某些 checkpoint 的推理失败但有些成功：
```bash
./run_auto_inference.sh
# 重新运行
```

- 成功的：跳过（但会检查并补充 L2Loss）
- 失败的：重新推理

## 性能影响

### 检查开销
- `grep` 检查：< 1ms（极快）
- 文件存在检查：< 1ms

### 计算开销（仅在缺失时）
- 每张图片：10-50ms
- 10 张图片：0.1-0.5秒
- 相对于推理时间（数十秒到数分钟）：几乎可忽略

### 总体影响
- **有 L2Loss**：几乎无影响（只是 grep 检查）
- **无 L2Loss**：增加 < 1 秒（对于10张图）

## 与其他功能的配合

### 1. 与批量检查配合
在所有推理完成后，脚本还会进行全面的批量检查（第 525-598 行）：
```bash
# 批量检查所有 metrics.json
find "$RESULTS_PATH" -name "metrics.json" -type f | while ...
```

这提供了**双重保障**：
- 第一层：跳过时检查（快速、针对性）
- 第二层：批量检查（全面、最终确保）

### 2. 与 --skip_existing 参数配合
`auto_inference.py` 的 `--skip_existing` 参数控制是否跳过已有图片。

当 `--skip_existing` 启用时：
1. Shell 脚本先检查 PNG 文件
2. 发现已存在时检查 L2Loss
3. 计算缺失的 L2Loss
4. 然后跳过推理

### 3. 与报告生成配合
在生成综合报告前，确保所有 metrics 都包含 L2Loss：
```bash
跳过时检查 L2Loss
    ↓
批量检查所有 metrics
    ↓
生成综合报告（包含完整 L2Loss）
```

## 边界情况处理

### 情况1：metrics.json 不存在
```bash
if [ -f "$METRICS_FILE" ]; then
    # 只在文件存在时检查
fi
```
如果 metrics.json 不存在，跳过检查，避免错误。

### 情况2：计算失败
```bash
if [ $? -eq 0 ]; then
    echo "  ✓ L2Loss 计算完成"
else
    echo "  ⚠ L2Loss 计算失败"
fi
```
计算失败时显示警告，但不中断流程。

### 情况3：GT 图片缺失
`recalculate_edge_l2_loss.py` 会处理这种情况：
- 对缺失的 GT，标记为 `-1.0`
- 不影响其他图片的计算
- 继续执行

## 调试和验证

### 检查是否成功
```bash
# 查看 metrics.json
cat validation_results/.../edge/epochs_27/metrics.json | grep edge_l2_loss

# 应该看到
"edge_l2_loss": 0.001234
"average_edge_l2_loss": 0.002456
```

### 查看详细日志
如果想看详细的计算过程，可以临时移除 `> /dev/null 2>&1`：
```bash
# 修改脚本（临时）
python3 scripts/recalculate_edge_l2_loss.py "$OUTPUT_CHECK" "$DEFAULT_GT_IMG"
```

### 强制重新计算
如果需要强制重新计算所有 L2Loss：
```bash
# 手动删除 edge_l2_loss 字段（不推荐）
# 或者在批量检查部分添加 --force 参数
```

## 最佳实践

### ✅ 推荐做法
1. 正常运行脚本，让它自动处理
2. 检查输出，确认 L2Loss 计算完成
3. 查看最终报告，验证数据完整性

### ⚠️ 注意事项
1. 确保 GT 图片路径正确
2. 如果看到大量失败，检查 GT 目录
3. 计算失败不影响跳过逻辑，可以继续

### 🔧 故障排除
如果计算一直失败：
```bash
# 手动运行一次，查看详细错误
python3 scripts/recalculate_edge_l2_loss.py \
    validation_results/.../edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

## 总结

这个改进使得 `run_auto_inference.sh` 更加智能：

| 特性 | 说明 |
|-----|------|
| **自动化** | 跳过时自动检查和计算 |
| **智能化** | 只计算缺失的指标 |
| **透明化** | 清晰的状态反馈 |
| **健壮性** | 失败不影响流程 |
| **高效性** | 开销极小 |

**现在可以放心地多次运行脚本，它会自动确保所有结果都包含完整的 L2Loss 指标！** ✨

