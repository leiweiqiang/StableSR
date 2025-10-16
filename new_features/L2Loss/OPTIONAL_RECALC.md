# Edge PSNR 可选重新计算功能

## 📅 更新：2025-10-16

## 🎯 功能说明

在 `run_auto_inference.sh` 选项1 中添加了**可选的 Edge PSNR 重新计算**功能。

### 新增交互选项

运行选项1时，会询问用户：

```bash
是否重新计算已有结果的 Edge PSNR 指标？
  注意：新推理的结果会自动计算 Edge PSNR
  此选项仅针对跳过的已有结果

重新计算 Edge PSNR? (y/n) [默认: n]:
```

- **输入 `n` 或直接回车（默认）**：跳过重新计算
- **输入 `y`**：检查并重新计算缺失的 Edge PSNR

## 🔄 工作流程

### 选择 "n" (默认 - 不重新计算)

```
开始推理
  ↓
处理新的 checkpoint
  ├─ 执行推理
  └─ 自动计算 Edge PSNR ✓
  ↓
跳过已有的 checkpoint
  └─ 直接跳过（不检查 Edge PSNR）
  ↓
生成报告
```

**特点**：
- ✅ 速度快，不额外计算
- ✅ 新推理自动包含 Edge PSNR
- ⚠️ 旧结果可能缺少 Edge PSNR

### 选择 "y" (重新计算)

```
开始推理
  ↓
处理新的 checkpoint
  ├─ 执行推理
  └─ 自动计算 Edge PSNR ✓
  ↓
跳过已有的 checkpoint
  ├─ 检查是否有 Edge PSNR
  ├─ 缺失则自动计算 ✓
  └─ 跳过推理
  ↓
批量扫描所有 metrics.json
  ├─ 再次检查
  └─ 补充任何遗漏 ✓
  ↓
生成报告
```

**特点**：
- ✅ 确保所有结果都有 Edge PSNR
- ✅ 双重保障（跳过时 + 批量扫描）
- ⚠️ 稍慢（但增加时间很少）

## 📊 使用场景对比

| 场景 | 选择 | 原因 |
|-----|------|------|
| **首次推理** | n (默认) | 所有都是新推理，会自动计算 |
| **继续未完成的推理** | n (默认) | 只关注新的推理结果 |
| **补充旧数据** | y | 需要为旧结果添加 Edge PSNR |
| **完整性检查** | y | 确保所有结果都有完整指标 |
| **快速验证** | n (默认) | 只看新推理的效果 |

## 🎨 交互示例

### 示例1：选择默认（不重新计算）

```bash
./run_auto_inference.sh
# 选择：1

是否重新计算已有结果的 Edge PSNR 指标？
  注意：新推理的结果会自动计算 Edge PSNR
  此选项仅针对跳过的已有结果

重新计算 Edge PSNR? (y/n) [默认: n]: ← 直接回车

✓ 跳过 Edge PSNR 重新计算（仅计算新推理的结果）

正在运行 EDGE 模式推理...
✓ 跳过 epoch=27 (已有 10 张图片)      ← 直接跳过，不检查
✓ 跳过 epoch=55 (已有 10 张图片)      ← 直接跳过，不检查
→ 处理 epoch=83                        ← 新推理，自动计算
...

✓ 跳过 Edge PSNR 批量检查（用户选择不重新计算）

正在生成推理结果报告...
```

### 示例2：选择重新计算

```bash
./run_auto_inference.sh
# 选择：1

是否重新计算已有结果的 Edge PSNR 指标？
  注意：新推理的结果会自动计算 Edge PSNR
  此选项仅针对跳过的已有结果

重新计算 Edge PSNR? (y/n) [默认: n]: y ← 输入 y

✓ 将检查并重新计算缺失的 Edge PSNR 指标

正在运行 EDGE 模式推理...
→ epoch=27 已有图片，但缺少 Edge PSNR，正在计算...
  ✓ Edge PSNR 计算完成
✓ 跳过 epoch=27 (已有 10 张图片)

✓ 跳过 epoch=55 (已有 10 张图片)      ← 已有 Edge PSNR，直接跳过
→ 处理 epoch=83                        ← 新推理，自动计算
...

==================================================
  批量检查并计算 Edge PSNR 指标
==================================================
扫描目录: validation_results/...
检查: edge/epochs_27
  ✓ Edge PSNR 指标已存在
...
统计信息：
  找到 12 个 metrics.json 文件
  ✓ 已更新: 4 个
  ✓ 已存在: 8 个

正在生成推理结果报告...
```

## 💡 实现细节

### 用户选择变量

```bash
ENABLE_EDGE_PSNR_CHECK=true   # 用户选择 y
ENABLE_EDGE_PSNR_CHECK=false  # 用户选择 n（默认）
```

### 四处检查逻辑

在 EDGE、NO-EDGE、DUMMY-EDGE、STABLESR 四个模式的跳过逻辑中：

```bash
if [ "$PNG_COUNT" -gt 0 ]; then
    # 只有当用户选择重新计算时才检查
    if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
        # 检查并计算 Edge PSNR
        ...
    fi
    # 跳过推理
    echo "✓ 跳过 epoch=$EPOCH_NUM"
    continue
fi
```

### 批量检查逻辑

```bash
if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
    # 批量扫描所有 metrics.json
    ...
else
    echo "✓ 跳过 Edge PSNR 批量检查（用户选择不重新计算）"
fi
```

## 📈 性能影响

### 选择 "n" (默认)

- **新推理**：正常计算（必须）
- **跳过时**：不检查，不计算（节省时间）
- **批量扫描**：跳过（节省时间）
- **总时间**：最快

### 选择 "y"

- **新推理**：正常计算（必须）
- **跳过时**：检查并计算缺失的（轻微增加）
- **批量扫描**：全面检查（轻微增加）
- **总时间**：略慢（但增加很少）

**实际影响**：
- 对于 10 个已有结果，增加约 1-5 秒
- 对于整个推理流程（数小时），几乎可忽略

## 🎯 推荐使用

### 什么时候选 "n" (默认)？

✅ **推荐场景**：
- 首次推理全新的 checkpoint
- 继续未完成的推理
- 只关注新结果
- 快速验证
- 调试过程

### 什么时候选 "y"？

✅ **推荐场景**：
- 补充旧数据（在添加 Edge PSNR 功能之前的结果）
- 确保数据完整性
- 准备最终报告
- 数据归档前

## ⚙️ 技术说明

### 代码位置

**询问用户**（第 166-182 行）：
```bash
read -p "重新计算 Edge PSNR? (y/n) [默认: n]: " RECALC_EDGE_PSNR
RECALC_EDGE_PSNR="${RECALC_EDGE_PSNR:-n}"

if [ "$RECALC_EDGE_PSNR" = "y" ] || [ "$RECALC_EDGE_PSNR" = "Y" ]; then
    ENABLE_EDGE_PSNR_CHECK=true
else
    ENABLE_EDGE_PSNR_CHECK=false
fi
```

**跳过时检查**（4处）：
```bash
if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
    # 检查并计算
fi
```

**批量扫描**（第 582-669 行）：
```bash
if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
    # 批量检查
else
    echo "✓ 跳过 Edge PSNR 批量检查"
fi
```

## 🔍 验证

### 检查脚本语法

```bash
bash -n run_auto_inference.sh
# 应该无输出，表示语法正确
```

### 实际测试

```bash
# 测试默认行为（不重新计算）
./run_auto_inference.sh
# 选择：1
# 重新计算 Edge PSNR? [默认: n]: ← 直接回车
# 应该看到：✓ 跳过 Edge PSNR 重新计算

# 测试重新计算行为
./run_auto_inference.sh
# 选择：1
# 重新计算 Edge PSNR? [默认: n]: y ← 输入 y
# 应该看到：✓ 将检查并重新计算缺失的 Edge PSNR 指标
```

## 💡 使用建议

### 一般情况

```bash
./run_auto_inference.sh
# 选择：1
# 直接回车（使用默认：不重新计算）
```

**优势**：快速、简单、适合大多数场景

### 数据整理时

```bash
./run_auto_inference.sh
# 选择：1
# 输入 y（重新计算）
```

**优势**：确保所有数据完整、适合最终报告

## ⚠️ 重要说明

### 1. 新推理始终计算

**无论选择 y 还是 n**，新执行的推理都会自动计算 Edge PSNR。

此选项**仅影响**：
- 跳过的已有结果是否检查
- 是否执行批量扫描

### 2. 默认值设计

默认为 `n`（不重新计算）的原因：
- 大多数情况下不需要
- 减少不必要的计算
- 加快脚本执行
- 用户可随时选择 `y`

### 3. 手动补充方法

如果选择了 `n` 但后来想补充，可以：

```bash
# 方法1：重新运行脚本，选择 y
./run_auto_inference.sh

# 方法2：手动运行补充脚本
python scripts/recalculate_edge_l2_loss.py \
    validation_results/.../edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

## 📋 修改总结

### 改动位置

1. **第 166-182 行**：添加用户选择
2. **第 234 行**：EDGE 模式条件检查
3. **第 310 行**：NO-EDGE 模式条件检查
4. **第 388 行**：DUMMY-EDGE 模式条件检查
5. **第 463 行**：STABLESR 模式条件检查
6. **第 582-669 行**：批量扫描条件执行

### 核心逻辑

```bash
# 询问用户
read -p "重新计算 Edge PSNR? (y/n) [默认: n]:" RECALC_EDGE_PSNR
if [ "$RECALC_EDGE_PSNR" = "y" ]; then
    ENABLE_EDGE_PSNR_CHECK=true
else
    ENABLE_EDGE_PSNR_CHECK=false  # 默认
fi

# 在跳过时检查（四处）
if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
    # 检查并计算
fi

# 批量扫描
if [ "$ENABLE_EDGE_PSNR_CHECK" = true ]; then
    # 批量检查
else
    echo "✓ 跳过批量检查"
fi
```

## 🎨 用户体验

### 默认行为（快速模式）

```bash
✓ 跳过 Edge PSNR 重新计算（仅计算新推理的结果）

正在运行 EDGE 模式推理...
✓ 跳过 epoch=27 (已有 10 张图片)  ← 直接跳过，快
✓ 跳过 epoch=55 (已有 10 张图片)  ← 直接跳过，快
→ 处理 epoch=83                    ← 新推理，自动计算
...

✓ 跳过 Edge PSNR 批量检查（用户选择不重新计算）
```

**优势**：
- 快速执行
- 适合日常使用
- 减少不必要的计算

### 完整模式（确保完整）

```bash
✓ 将检查并重新计算缺失的 Edge PSNR 指标

正在运行 EDGE 模式推理...
→ epoch=27 已有图片，但缺少 Edge PSNR，正在计算...
  ✓ Edge PSNR 计算完成
✓ 跳过 epoch=27 (已有 10 张图片)

✓ 跳过 epoch=55 (已有 10 张图片)  ← 已有 Edge PSNR
→ 处理 epoch=83                    ← 新推理
...

==================================================
  批量检查并计算 Edge PSNR 指标
==================================================
统计信息：
  ✓ 已更新: 4 个
  ✓ 已存在: 8 个
```

**优势**：
- 确保完整性
- 适合最终整理
- 双重保障

## 📈 性能对比

| 操作 | 默认 (n) | 重新计算 (y) | 增加时间 |
|-----|---------|------------|---------|
| 新推理 | 自动计算 | 自动计算 | 0 |
| 跳过时检查 | ✗ 不检查 | ✓ 检查 | ~0.1s/结果 |
| 批量扫描 | ✗ 跳过 | ✓ 扫描 | ~1-5s 总共 |
| **总增加** | **0** | **1-10s** | **可忽略** |

对于典型的推理任务（耗时数小时），增加的时间几乎感觉不到。

## ✅ 优势总结

### 灵活性
- ✅ 用户可选择
- ✅ 默认快速模式
- ✅ 需要时可启用完整检查

### 用户友好
- ✅ 清晰的提示
- ✅ 默认值合理
- ✅ 支持大小写 y/Y

### 向后兼容
- ✅ 不影响现有功能
- ✅ 新推理始终计算
- ✅ 旧数据可随时补充

## 🎯 总结

通过添加这个可选项，`run_auto_inference.sh` 变得更加灵活：

**默认行为（n）**：
- 快速执行
- 只计算新推理
- 适合日常使用

**完整模式（y）**：
- 确保完整性
- 补充旧数据
- 适合最终整理

**最佳实践**：
- 日常推理：使用默认 (n)
- 最终报告：选择重新计算 (y)
- 数据归档：选择重新计算 (y)

---

**✅ 功能已完成，现在运行脚本会有更好的用户体验！** 🎉

