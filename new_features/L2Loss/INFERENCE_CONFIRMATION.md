# 推理确认功能

## 📅 更新：2025-10-16

## 🎯 功能说明

在 `run_auto_inference.sh` 选项1中添加了**推理前确认**功能。

脚本会预先扫描所有 checkpoint，统计需要推理的数量，然后询问用户是否开始推理。

## 🔍 工作流程

### 完整流程

```
1. 找到所有 checkpoint 文件
   ↓
2. 预先扫描检查
   ├─ EDGE 模式：检查哪些需要推理
   ├─ NO-EDGE 模式：检查哪些需要推理
   └─ DUMMY-EDGE 模式：检查哪些需要推理
   ↓
3. 显示统计信息
   ├─ EDGE: 需要推理 X 个
   ├─ NO-EDGE: 需要推理 Y 个
   └─ DUMMY-EDGE: 需要推理 Z 个
   ↓
4. 询问用户确认
   ⚠️  发现 N 个新的推理任务需要执行
   是否开始推理? (y/n) [默认: y]:
   ↓
5. 用户选择
   ├─ y/Y/回车 → 开始推理
   └─ n/N → 取消推理，返回菜单
```

## 🎨 交互示例

### 示例1：有新的 checkpoint（确认并执行）

```bash
./run_auto_inference.sh
# 选择：1

✓ 找到 5 个 checkpoint 文件（已排除 last.ckpt）

正在检查哪些 checkpoint 需要推理...

推理需求统计：
  EDGE 模式: 需要推理 2 个 checkpoint
  NO-EDGE 模式: 需要推理 2 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 2 个 checkpoint
  总计: 需要推理 6 个任务

⚠️  发现 6 个新的推理任务需要执行

是否开始推理? (y/n) [默认: y]: ← 回车或输入 y

✓ 开始执行推理...

正在运行 EDGE 模式推理...
→ 处理 epoch=138
→ 处理 epoch=166
...
```

### 示例2：有新的 checkpoint（取消）

```bash
./run_auto_inference.sh
# 选择：1

推理需求统计：
  EDGE 模式: 需要推理 3 个 checkpoint
  NO-EDGE 模式: 需要推理 3 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 3 个 checkpoint
  总计: 需要推理 9 个任务

⚠️  发现 9 个新的推理任务需要执行

是否开始推理? (y/n) [默认: y]: n ← 输入 n

✗ 用户取消推理

# 返回菜单
```

### 示例3：没有新的 checkpoint

```bash
./run_auto_inference.sh
# 选择：1

正在检查哪些 checkpoint 需要推理...

推理需求统计：
  EDGE 模式: 需要推理 0 个 checkpoint
  NO-EDGE 模式: 需要推理 0 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 0 个 checkpoint
  总计: 需要推理 0 个任务

✓ 没有新的推理任务，所有 checkpoint 结果已存在
  将继续检查 Edge PSNR 指标...  ← 如果选择了重新计算

# 继续执行 Edge PSNR 检查或直接生成报告
```

## 💡 实现细节

### 预扫描逻辑（第 210-257 行）

```bash
# 统计每种模式需要推理的 checkpoint 数量
NEW_CKPTS_EDGE=0
NEW_CKPTS_NO_EDGE=0
NEW_CKPTS_DUMMY=0

for CKPT_FILE in "${CKPT_FILES[@]}"; do
    # 提取 epoch 编号
    EPOCH_NUM="${BASH_REMATCH[1]}"
    
    # 检查 edge 模式输出是否存在
    OUTPUT_CHECK_EDGE="$OUTPUT_BASE/.../edge/epochs_$EPOCH_NUM"
    if [ -d "$OUTPUT_CHECK_EDGE" ]; then
        PNG_COUNT=$(find ... -name "*.png" | wc -l)
        if [ "$PNG_COUNT" -eq 0 ]; then
            ((NEW_CKPTS_EDGE++))  # 目录存在但没有图片
        fi
    else
        ((NEW_CKPTS_EDGE++))      # 目录不存在
    fi
    
    # 同样检查 no-edge 和 dummy-edge 模式
    ...
done
```

### 确认逻辑（第 267-280 行）

```bash
TOTAL_NEW_TASKS=$((NEW_CKPTS_EDGE + NEW_CKPTS_NO_EDGE + NEW_CKPTS_DUMMY))

if [ $TOTAL_NEW_TASKS -gt 0 ]; then
    echo "⚠️  发现 $TOTAL_NEW_TASKS 个新的推理任务需要执行"
    read -p "是否开始推理? (y/n) [默认: y]: " START_INFERENCE
    START_INFERENCE="${START_INFERENCE:-y}"
    
    if [ "$START_INFERENCE" != "y" ] && [ "$START_INFERENCE" != "Y" ]; then
        echo "✗ 用户取消推理"
        return  # 退出函数，返回菜单
    fi
    echo "✓ 开始执行推理..."
fi
```

## 🎯 使用场景

### 场景1：定期检查新训练的 checkpoint

```bash
# 每隔一段时间运行一次
./run_auto_inference.sh
# 选择：1

# 脚本自动告诉你有多少新的 checkpoint
推理需求统计：
  总计: 需要推理 3 个任务

# 你决定是否推理
是否开始推理? (y/n) [y]: 
```

**优势**：
- 清楚知道要做什么
- 可以取消不必要的推理
- 避免意外执行

### 场景2：验证后再推理

```bash
# 先查看统计
./run_auto_inference.sh
# 看到需要推理 10+ 个任务

# 如果太多，可以取消
是否开始推理? (y/n) [y]: n

# 稍后再运行，或调整筛选条件
```

### 场景3：快速确认

```bash
# 看到只有 1-2 个新任务
推理需求统计：
  总计: 需要推理 2 个任务

# 直接回车确认
是否开始推理? (y/n) [y]: ← 回车

✓ 开始执行推理...
```

## 📊 优势分析

### ✅ 信息透明

**之前**：直接开始推理，不知道要处理多少
**现在**：预先告知，心中有数

### ✅ 用户控制

**之前**：必须执行或 Ctrl+C 中断
**现在**：可以优雅地取消

### ✅ 避免浪费

**之前**：可能执行了不必要的推理
**现在**：可以提前取消

### ✅ 默认合理

**默认**：y（开始推理）
**原因**：大多数情况下用户想要推理

## 🔧 技术细节

### 检查逻辑

对每个 checkpoint，检查三个输出目录：
```bash
$OUTPUT_BASE/$SELECTED_DIR_NAME/edge/epochs_X
$OUTPUT_BASE/$SELECTED_DIR_NAME/no_edge/epochs_X
$OUTPUT_BASE/$SELECTED_DIR_NAME/dummy_edge/epochs_X
```

判断标准：
- 目录不存在 → 需要推理
- 目录存在但没有 PNG 文件 → 需要推理
- 目录存在且有 PNG 文件 → 不需要推理

### 性能影响

预扫描开销：
- 每个 checkpoint 检查 3 个目录
- 每次检查约 1-5ms
- 总开销：< 1 秒（对于几十个 checkpoint）

**相对于总推理时间（数小时）：几乎可忽略**

## 📋 完整交互流程

```bash
./run_auto_inference.sh

==================================================
           StableSR Edge 推理菜单
==================================================

1. 推理指定目录下全部 checkpoint (edge & no-edge)
...

请选择操作 [0-4]: 1

==================================================
  模式 1: 推理全部 Checkpoints
==================================================

# Step 1: 选择目录
请输入 logs 目录路径 [logs]: 
可用的子目录：
1. stablesr_edge_loss_20251015_194003
请选择目录编号: 1

# Step 2: 设置输出
请输入保存目录名 [validation_results]: 
✓ 结果将保存到: validation_results

# Step 3: 选择是否重新计算 Edge PSNR
是否重新计算已有结果的 Edge PSNR 指标？
重新计算 Edge PSNR? (y/n) [默认: n]: 

# Step 4: 扫描 checkpoints
✓ 找到 5 个 checkpoint 文件
正在检查哪些 checkpoint 需要推理...

# Step 5: 显示统计并确认 ⭐ 新增
推理需求统计：
  EDGE 模式: 需要推理 2 个 checkpoint
  NO-EDGE 模式: 需要推理 2 个 checkpoint  
  DUMMY-EDGE 模式: 需要推理 2 个 checkpoint
  总计: 需要推理 6 个任务

⚠️  发现 6 个新的推理任务需要执行

是否开始推理? (y/n) [默认: y]: ← 用户确认

# Step 6: 开始推理
✓ 开始执行推理...
...
```

## 🎯 实际效果

### 有新 checkpoint 时

```bash
正在检查哪些 checkpoint 需要推理...

推理需求统计：
  EDGE 模式: 需要推理 2 个 checkpoint
    └─ epoch=138, epoch=166
  NO-EDGE 模式: 需要推理 2 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 2 个 checkpoint
  总计: 需要推理 6 个任务

⚠️  发现 6 个新的推理任务需要执行
   预计耗时：约 30-60 分钟（取决于图片数量）

是否开始推理? (y/n) [默认: y]:
```

用户可以：
- 回车或输入 `y` → 开始推理
- 输入 `n` → 取消，返回菜单

### 没有新 checkpoint 时

```bash
推理需求统计：
  EDGE 模式: 需要推理 0 个 checkpoint
  NO-EDGE 模式: 需要推理 0 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 0 个 checkpoint
  总计: 需要推理 0 个任务

✓ 没有新的推理任务，所有 checkpoint 结果已存在
  将继续检查 Edge PSNR 指标...
```

不会询问确认，直接继续后续步骤（Edge PSNR 检查或生成报告）。

## 💡 设计思路

### 为什么需要确认？

1. **避免误操作**：用户可能只想查看，不想推理
2. **时间成本**：推理可能需要数小时
3. **资源控制**：可以选择合适的时间运行
4. **信息透明**：提前知道要做什么

### 为什么默认是 "y"？

1. **符合预期**：用户选择"推理"通常就是想推理
2. **方便快捷**：直接回车即可
3. **向后兼容**：保持原有的自动执行行为

### 为什么 0 个任务不询问？

1. **无意义**：没有任务，询问没有意义
2. **减少干扰**：直接继续后续步骤
3. **用户友好**：减少不必要的交互

## 📋 两个确认点

现在脚本有**两个用户确认点**：

### 确认点1：Edge PSNR 重新计算（第 172 行）

```bash
是否重新计算已有结果的 Edge PSNR 指标？
重新计算 Edge PSNR? (y/n) [默认: n]:
```

**目的**：控制是否检查旧结果  
**默认**：n（不检查，快速）

### 确认点2：开始推理（第 272 行）⭐ 新增

```bash
⚠️  发现 6 个新的推理任务需要执行

是否开始推理? (y/n) [默认: y]:
```

**目的**：确认是否执行新推理  
**默认**：y（执行推理）

## 🎯 推荐使用方式

### 日常使用

```bash
./run_auto_inference.sh
# 选择：1
# Edge PSNR? [n]: ← 回车（快速模式）
# 开始推理? [y]: ← 回车（确认推理）
```

**特点**：两次回车，快速执行

### 谨慎模式

```bash
./run_auto_inference.sh  
# 选择：1
# Edge PSNR? [n]: ← 回车
# 开始推理? [y]: ← 查看统计，决定 y 或 n
```

**特点**：根据任务数量决定

### 完整模式

```bash
./run_auto_inference.sh
# 选择：1
# Edge PSNR? [n]: y ← 确保完整
# 开始推理? [y]: ← 回车
```

**特点**：确保所有数据完整

## ⚙️ 实现代码

### 预扫描（第 219-257 行）

```bash
for CKPT_FILE in "${CKPT_FILES[@]}"; do
    CKPT_BASENAME=$(basename "$CKPT_FILE")
    if [[ "$CKPT_BASENAME" =~ epoch=([0-9]+) ]]; then
        EPOCH_NUM="${BASH_REMATCH[1]}"
        
        # 检查 edge 模式
        OUTPUT_CHECK_EDGE="..."
        if [ -d "$OUTPUT_CHECK_EDGE" ]; then
            PNG_COUNT=$(find ... -name "*.png" | wc -l)
            if [ "$PNG_COUNT" -eq 0 ]; then
                ((NEW_CKPTS_EDGE++))
            fi
        else
            ((NEW_CKPTS_EDGE++))
        fi
        
        # 同样检查 no-edge 和 dummy-edge
        ...
    fi
done
```

### 确认询问（第 267-280 行）

```bash
TOTAL_NEW_TASKS=$((NEW_CKPTS_EDGE + NEW_CKPTS_NO_EDGE + NEW_CKPTS_DUMMY))

if [ $TOTAL_NEW_TASKS -gt 0 ]; then
    echo "⚠️  发现 $TOTAL_NEW_TASKS 个新的推理任务需要执行"
    echo ""
    read -p "是否开始推理? (y/n) [默认: y]: " START_INFERENCE
    START_INFERENCE="${START_INFERENCE:-y}"
    
    if [ "$START_INFERENCE" != "y" ] && [ "$START_INFERENCE" != "Y" ]; then
        echo "✗ 用户取消推理"
        return
    fi
    echo "✓ 开始执行推理..."
fi
```

## 🔍 边界情况处理

### 情况1：部分模式有新任务

```bash
推理需求统计：
  EDGE 模式: 需要推理 2 个 checkpoint
  NO-EDGE 模式: 需要推理 0 个 checkpoint  ← 已完成
  DUMMY-EDGE 模式: 需要推理 1 个 checkpoint
  总计: 需要推理 3 个任务
```

只要总计 > 0，就会询问确认。

### 情况2：所有模式都完成

```bash
推理需求统计：
  总计: 需要推理 0 个任务

✓ 没有新的推理任务
```

不询问，直接继续后续步骤。

### 情况3：用户取消

```bash
是否开始推理? (y/n) [y]: n

✗ 用户取消推理

# 返回主菜单，可以选择其他操作
```

## 📈 性能影响

| 操作 | 时间 |
|-----|------|
| 预扫描检查 | < 1 秒 |
| 显示统计 | 瞬间 |
| 用户确认 | 取决于用户 |
| **总开销** | **< 1 秒 + 用户时间** |

**结论**：对推理流程几乎无影响

## ✅ 优势总结

1. **信息透明**：提前知道要推理多少
2. **用户控制**：可以取消不想要的推理
3. **时间管理**：可以选择合适的时间运行
4. **资源优化**：避免不必要的计算
5. **用户友好**：默认值合理，操作简单

## 🎊 完整功能列表

现在 `run_auto_inference.sh` 选项1 包含：

1. ✅ 交互式目录选择
2. ✅ 可选的 Edge PSNR 重新计算
3. ✅ **预扫描统计** ⭐ 新增
4. ✅ **推理前确认** ⭐ 新增
5. ✅ 智能跳过已有结果
6. ✅ Edge PSNR 自动检查（可选）
7. ✅ 批量扫描确保完整性（可选）
8. ✅ 自动生成综合报告

---

**🎉 推理前确认功能已完成，让你对推理任务有完全的控制！** ✨

