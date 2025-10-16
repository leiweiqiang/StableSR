# 批次限制功能

## 📅 添加时间：2025-10-16

---

## 🎯 功能说明

在 `run_auto_inference.sh` 选项1中添加了**每次最多计算 checkpoint 数量**的限制功能。

用户可以设置每次运行最多处理多少个 checkpoint，避免一次性处理过多任务。

---

## 💡 使用场景

### 场景1：分批处理

有很多新 checkpoint（如 10+ 个），一次性推理需要很长时间：

```bash
# 设置每次最多处理 2 个
每次最多计算 checkpoint 数量 [默认: -1]: 2

# 第一次运行：处理 checkpoint 1-2
# 第二次运行：处理 checkpoint 3-4
# ...依此类推
```

### 场景2：测试验证

想快速验证一两个 checkpoint：

```bash
# 只处理 1 个
每次最多计算 checkpoint 数量 [默认: -1]: 1

# 快速看到结果，验证流程正常
```

### 场景3：时间控制

只有有限的时间，不能一次性运行完：

```bash
# 有30分钟，设置处理 3 个
每次最多计算 checkpoint 数量 [默认: -1]: 3

# 30分钟后可以停止，下次继续
```

### 场景4：全部处理（默认）

```bash
# 直接回车，处理所有
每次最多计算 checkpoint 数量 [默认: -1]: ← 回车

# 一次性处理完所有 checkpoint
```

---

## 🎨 交互示例

### 示例1：限制为2个

```bash
./run_auto_inference.sh
选择：1

...

设置每次最多计算多少个 checkpoint：
  -1: 全部计算（默认）
  N: 每次最多计算 N 个

每次最多计算 checkpoint 数量 [默认: -1]: 2

✓ 每次最多处理 2 个 checkpoint

...

推理需求统计：
  EDGE 模式: 需要推理 5 个 checkpoint
    └─ Epochs: 000027 000055 000083 000111 000138
  NO-EDGE 模式: 需要推理 5 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 5 个 checkpoint
  总计: 需要推理 15 个任务

...

正在运行 EDGE 模式推理...
（限制：最多处理 2 个）

→ 处理 epoch=000027
→ 处理 epoch=000055
✓ 已达到处理限制（2 个），停止 EDGE 模式推理

EDGE 模式统计: 已处理 2 个，跳过 0 个

正在运行 NO-EDGE 模式推理...
（限制：最多处理 2 个）

→ 处理 epoch=000027
→ 处理 epoch=000055
✓ 已达到处理限制（2 个），停止 NO-EDGE 模式推理

NO-EDGE 模式统计: 已处理 2 个，跳过 0 个

...

统计信息：
  EDGE 模式: 已处理 2 个，跳过 0 个
  NO-EDGE 模式: 已处理 2 个，跳过 0 个
  DUMMY-EDGE 模式: 已处理 2 个，跳过 0 个
  ⚠ 注意：由于设置了处理限制，还有 9 个任务未完成
     请再次运行脚本继续处理剩余的 checkpoint
```

### 示例2：全部处理（默认）

```bash
每次最多计算 checkpoint 数量 [默认: -1]: ← 直接回车

✓ 将处理所有 checkpoint

...

正在运行 EDGE 模式推理...

→ 处理 epoch=000027
→ 处理 epoch=000055
→ 处理 epoch=000083
...（处理所有）

统计信息：
  EDGE 模式: 已处理 5 个，跳过 0 个
  NO-EDGE 模式: 已处理 5 个，跳过 0 个
  DUMMY-EDGE 模式: 已处理 5 个，跳过 0 个
  总计 checkpoints: 已处理 15 个，跳过 0 个
```

---

## 🔧 实现细节

### 用户输入（第 184-202 行）

```bash
read -p "每次最多计算 checkpoint 数量 [默认: -1]: " MAX_CKPTS_PER_RUN
MAX_CKPTS_PER_RUN="${MAX_CKPTS_PER_RUN:--1}"

# 验证输入
if ! [[ "$MAX_CKPTS_PER_RUN" =~ ^-?[0-9]+$ ]]; then
    echo "⚠ 警告：输入无效，使用默认值 -1"
    MAX_CKPTS_PER_RUN=-1
fi
```

### 循环限制检查（3个模式）

```bash
# EDGE 模式
for CKPT_FILE in "${CKPT_FILES[@]}"; do
    # 检查是否达到限制
    if [ "$MAX_CKPTS_PER_RUN" -ne -1 ] && [ "$EDGE_PROCESSED" -ge "$MAX_CKPTS_PER_RUN" ]; then
        echo "✓ 已达到处理限制，停止 EDGE 模式推理"
        break
    fi
    
    # 处理 checkpoint
    ...
    
    if [ $? -eq 0 ]; then
        ((EDGE_PROCESSED++))  # 只有成功才计数
    fi
done

# NO-EDGE 和 DUMMY-EDGE 模式类似
```

### 统计信息（第 718-727 行）

```bash
if [ "$MAX_CKPTS_PER_RUN" -ne -1 ]; then
    TOTAL_PROCESSED=$((EDGE_PROCESSED + NO_EDGE_PROCESSED + DUMMY_EDGE_PROCESSED))
    TOTAL_NEEDED=$((NEW_CKPTS_EDGE + NEW_CKPTS_NO_EDGE + NEW_CKPTS_DUMMY))
    REMAINING=$((TOTAL_NEEDED - TOTAL_PROCESSED))
    
    if [ $REMAINING -gt 0 ]; then
        echo "  ⚠ 注意：还有 $REMAINING 个任务未完成"
        echo "     请再次运行脚本继续处理"
    fi
fi
```

---

## 📊 使用策略

### 策略1：快速验证

```bash
# 先处理 1 个看看效果
MAX: 1

# 验证没问题后，再处理更多
```

### 策略2：分批处理

```bash
# 每次处理 3 个
MAX: 3

# 好处：
# - 可以随时停止
# - 每批完成后有检查点
# - 出问题影响范围小
```

### 策略3：时间控制

```bash
# 根据可用时间估算
# 假设每个 checkpoint 需要 10 分钟
# 有 1 小时 → 设置为 6

MAX: 6
```

### 策略4：全部处理

```bash
# 一次性处理完
MAX: -1 （默认）

# 适合：
# - 时间充足
# - checkpoint 数量不多
# - 需要完整结果
```

---

## 💡 优势

### 1. 灵活控制

- 可以设置任意数量
- 支持分批处理
- 避免长时间运行

### 2. 进度保存

- 每批完成后结果已保存
- 下次运行自动跳过已完成的
- 不怕中断

### 3. 资源管理

- 控制 GPU 使用时间
- 避免过载
- 合理安排计算资源

### 4. 易于调试

- 小批次测试
- 快速发现问题
- 逐步验证

---

## 📋 完整交互流程

```bash
./run_auto_inference.sh

选择：1

请输入 logs 目录路径 [logs]: ← 回车
选择目录编号: 1
请输入保存目录名 [validation_results]: ← 回车

[确认1] 重新计算指标?
重新计算指标? (y/n) [默认: n]: ← 回车

[新增] 设置批次大小
设置每次最多计算多少个 checkpoint：
  -1: 全部计算（默认）
  N: 每次最多计算 N 个

每次最多计算 checkpoint 数量 [默认: -1]: 2 ← 输入限制

✓ 每次最多处理 2 个 checkpoint

正在检查哪些 checkpoint 需要推理...
推理需求统计：
  总计: 需要推理 15 个任务

[确认2] 开始推理?
⚠️  发现 15 个新的推理任务需要执行
是否开始推理? (y/n) [默认: y]: ← 回车

✓ 开始执行推理...

正在运行 EDGE 模式推理...
（限制：最多处理 2 个）
→ 处理 epoch=000027
→ 处理 epoch=000055
✓ 已达到处理限制（2 个），停止 EDGE 模式推理

...

统计信息：
  EDGE 模式: 已处理 2 个，跳过 3 个
  NO-EDGE 模式: 已处理 2 个，跳过 3 个
  DUMMY-EDGE 模式: 已处理 2 个，跳过 3 个
  ⚠ 注意：由于设置了处理限制，还有 9 个任务未完成
     请再次运行脚本继续处理剩余的 checkpoint
```

---

## 🎯 典型用法

### 用法1：快速模式（5次回车）

```bash
./run_auto_inference.sh
1 → 回车 → 回车 → 回车 → 回车 → 回车
# 全自动，处理所有
```

### 用法2：分批模式

```bash
./run_auto_inference.sh
1 → 回车 → 回车 → 回车 → 2 → 回车
# 每次处理 2 个
```

### 用法3：测试模式

```bash
./run_auto_inference.sh
1 → 回车 → 回车 → 回车 → 1 → 回车
# 只处理 1 个测试
```

---

## ⚙️ 技术说明

### 计数逻辑

```bash
EDGE_PROCESSED=0  # 已处理计数器

for CKPT_FILE in ...; do
    # 检查是否达到限制
    if [ "$MAX_CKPTS_PER_RUN" -ne -1 ] && [ "$EDGE_PROCESSED" -ge "$MAX_CKPTS_PER_RUN" ]; then
        break  # 停止循环
    fi
    
    # 执行推理
    ...
    
    if [ $? -eq 0 ]; then
        ((EDGE_PROCESSED++))  # 成功才计数
    fi
done
```

### 剩余任务计算

```bash
TOTAL_PROCESSED=$((EDGE_PROCESSED + NO_EDGE_PROCESSED + DUMMY_EDGE_PROCESSED))
TOTAL_NEEDED=$((NEW_CKPTS_EDGE + NEW_CKPTS_NO_EDGE + NEW_CKPTS_DUMMY))
REMAINING=$((TOTAL_NEEDED - TOTAL_PROCESSED))
```

### 验证输入

```bash
# 只接受整数（包括负数）
if ! [[ "$MAX_CKPTS_PER_RUN" =~ ^-?[0-9]+$ ]]; then
    # 无效输入，使用默认值
    MAX_CKPTS_PER_RUN=-1
fi
```

---

## 🔄 继续处理

### 如何继续剩余的 checkpoint

如果第一次运行时设置了限制，还有剩余任务：

```bash
# 再次运行脚本
./run_auto_inference.sh
# 选择：1
# 选择相同的目录和输出路径
# 设置限制：2（或其他值）

# 脚本会自动：
# 1. 跳过已完成的 checkpoint
# 2. 继续处理剩余的 checkpoint
# 3. 达到限制后停止
```

**重复运行直到全部完成！**

---

## 📈 优势分析

### 1. 时间管理

| 限制 | 单批耗时（估算） | 适用场景 |
|-----|----------------|---------|
| -1 | 数小时 | 时间充足，一次完成 |
| 5 | 约1小时 | 有1小时空闲时间 |
| 2 | 约20-30分钟 | 快速处理几个 |
| 1 | 约10分钟 | 测试验证 |

### 2. 风险控制

- ✅ 分批处理，降低失败影响
- ✅ 随时可停止
- ✅ 已完成的不会丢失
- ✅ 易于恢复

### 3. 灵活性

- ✅ 可以根据情况调整
- ✅ 不同批次可以用不同限制
- ✅ 完全可控

---

## 📊 实际效果

### 有10个新 checkpoint，限制为3

```bash
第一次运行（限制：3）:
  EDGE: 处理 3 个 (27, 55, 83)
  NO-EDGE: 处理 3 个
  DUMMY-EDGE: 处理 3 个
  剩余: 7 个任务

第二次运行（限制：3）:
  EDGE: 处理 3 个 (111, 138, 166)
  NO-EDGE: 处理 3 个
  DUMMY-EDGE: 处理 3 个
  剩余: 1 个任务

第三次运行（限制：3）:
  EDGE: 处理 1 个 (194)
  NO-EDGE: 处理 1 个
  DUMMY-EDGE: 处理 1 个
  剩余: 0 个任务 ✓ 全部完成
```

---

## ⚠️ 注意事项

### 1. 计数基于成功

只有成功处理的 checkpoint 才计入 `PROCESSED` 计数器。

如果某个失败了，不计入，会继续尝试下一个（直到达到限制）。

### 2. 三个模式独立

每个模式（EDGE, NO-EDGE, DUMMY-EDGE）都有独立的限制。

设置限制为 2，实际会处理：
- EDGE: 最多 2 个
- NO-EDGE: 最多 2 个
- DUMMY-EDGE: 最多 2 个
- **总计：最多 6 个任务**

### 3. 跳过不计数

已有结果被跳过时，不计入 `PROCESSED` 计数。

这样确保每次都处理"新"的推理。

### 4. 默认值

默认 -1 表示没有限制，与之前的行为一致。

---

## 🎯 推荐使用

### 日常使用

```bash
# 默认模式（全部处理）
每次最多: ← 回车（-1）
```

### 有限时间

```bash
# 根据可用时间设置
# 1小时 → 5-6 个
# 30分钟 → 2-3 个
每次最多: 3
```

### 测试验证

```bash
# 只处理 1-2 个验证
每次最多: 1
```

### 分批处理

```bash
# 大量 checkpoint 时
# 分成小批次处理
每次最多: 3-5
```

---

## ✅ 完整功能列表

现在 `run_auto_inference.sh` 选项1 包含：

1. ✅ 交互式目录选择
2. ✅ 输出路径设置
3. ✅ 可选重新计算指标
4. ✅ **批次大小限制** ⭐ 新增
5. ✅ 预扫描统计
6. ✅ 列出 checkpoint 名称
7. ✅ 推理前确认
8. ✅ 智能跳过
9. ✅ 批量检查（可选）
10. ✅ 自动生成报告

**功能完整，用户体验优秀！** 🎉

---

## 📚 相关文档

- 📖 用户指南：`USER_GUIDE.md`
- 📖 推理确认：`INFERENCE_CONFIRMATION.md`
- 📖 可选重新计算：`OPTIONAL_RECALC.md`

---

**✅ 批次限制功能已添加，现在可以更灵活地控制推理流程！** ✨

