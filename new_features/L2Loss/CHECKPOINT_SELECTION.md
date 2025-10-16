# Checkpoint 选择功能

## 📅 添加时间：2025-10-16

---

## 🎯 功能说明

在 `run_auto_inference.sh` 选项1中，**取消"最多推理数量"选项**，改为**交互式选择 checkpoint**。

用户可以从未推理的 checkpoint 列表中选择要处理的 checkpoint，支持多选。

---

## 💡 使用流程

### 完整流程

```bash
./run_auto_inference.sh
选择：1

[步骤1] 选择目录和输出路径
...

[步骤2] 是否重新计算指标？
重新计算指标? (y/n) [默认: n]: ← 回车

[步骤3] 显示未推理的 checkpoint 列表
==================================================
  未推理的 Checkpoint 列表
==================================================

找到 5 个未推理的 checkpoint：

  [ 1] epoch=000027 (需要: edge, no-edge, dummy-edge)
  [ 2] epoch=000055 (需要: edge, no-edge)
  [ 3] epoch=000083 (需要: edge, no-edge, dummy-edge)
  [ 4] epoch=000111 (需要: edge)
  [ 5] epoch=000138 (需要: edge, no-edge, dummy-edge)

==================================================

请选择要推理的 checkpoint：
  - 输入序号，多个序号用逗号分隔，例如：1,3,5
  - 输入 'all' 或直接回车推理所有
  - 输入 'q' 取消

请选择 [all]: 

[步骤4] 开始推理选中的 checkpoint
```

---

## 🎨 使用示例

### 示例1：推理所有（默认）

```bash
请选择 [all]: ← 直接回车

✓ 将推理所有 5 个 checkpoint

✓ 开始执行推理...

正在运行 EDGE 模式推理...
→ 处理 epoch=000027
→ 处理 epoch=000055
→ 处理 epoch=000083
→ 处理 epoch=000111
→ 处理 epoch=000138

...
```

### 示例2：选择单个

```bash
请选择 [all]: 3

✓ 已选择 1 个 checkpoint:
    - epoch=000083

✓ 开始执行推理...

正在运行 EDGE 模式推理...
→ 处理 epoch=000083

正在运行 NO-EDGE 模式推理...
→ 处理 epoch=000083

正在运行 DUMMY-EDGE 模式推理...
→ 处理 epoch=000083

✓ 完成
```

### 示例3：选择多个（逗号分隔）

```bash
请选择 [all]: 1,3,5

✓ 已选择 3 个 checkpoint:
    - epoch=000027
    - epoch=000083
    - epoch=000138

✓ 开始执行推理...

（只处理这3个 checkpoint）
```

### 示例4：选择连续的几个

```bash
请选择 [all]: 2,3,4

✓ 已选择 3 个 checkpoint:
    - epoch=000055
    - epoch=000083
    - epoch=000111

✓ 开始执行推理...
```

### 示例5：取消推理

```bash
请选择 [all]: q

✗ 用户取消推理
```

---

## 🔧 输入格式

### 有效输入

| 输入 | 含义 | 示例 |
|-----|------|------|
| **回车** | 推理所有 | ← 直接回车 |
| **all** | 推理所有 | all |
| **单个序号** | 推理单个 | 3 |
| **逗号分隔** | 推理多个 | 1,3,5 |
| **带空格** | 自动去除空格 | 1, 3, 5 |
| **q** | 取消推理 | q |

### 无效输入

| 输入 | 错误原因 | 提示 |
|-----|---------|------|
| **abc** | 不是数字 | ❌ 错误：'abc' 不是有效的序号 |
| **0** | 超出范围 | ❌ 错误：序号 0 超出范围 (1-5) |
| **99** | 超出范围 | ❌ 错误：序号 99 超出范围 (1-5) |
| **空输入 + 非all** | 未选择 | 使用默认 all |

---

## 📊 显示格式

### Checkpoint 列表

```
  [序号] epoch=epoch编号 (需要: 需要推理的模式)
```

**需要的模式**可能是：
- `edge` - 只需要 edge 模式
- `edge, no-edge` - 需要 edge 和 no-edge
- `edge, no-edge, dummy-edge` - 需要所有三种模式

### 示例

```
  [ 1] epoch=000027 (需要: edge, no-edge, dummy-edge)
  [ 2] epoch=000055 (需要: edge, no-edge)
  [ 3] epoch=000083 (需要: edge, no-edge, dummy-edge)
  [ 4] epoch=000111 (需要: edge)
  [ 5] epoch=000138 (需要: edge, no-edge, dummy-edge)
```

**解读**：
- epoch 000027: 所有三种模式都未推理
- epoch 000055: 只有 dummy-edge 已推理，其他两种未推理
- epoch 000111: 只有 edge 未推理，其他两种已推理

---

## 💡 使用场景

### 场景1：快速测试

```bash
# 只推理最新的 checkpoint
请选择: 5

# 快速验证最新模型
```

### 场景2：分批处理

```bash
# 第一次运行
请选择: 1,2

# 第二次运行
请选择: 3,4

# 第三次运行
请选择: 5
```

### 场景3：选择性推理

```bash
# 只推理感兴趣的 epoch
请选择: 1,5

# 跳过中间的 epoch
```

### 场景4：全部推理

```bash
# 默认推理所有
请选择: ← 直接回车

# 或者显式输入
请选择: all
```

---

## 🔄 与之前的对比

### 之前：批次限制

```bash
每次最多计算 checkpoint 数量 [默认: -1]: 2

# 限制：只能控制数量，不能选择具体哪些
# 结果：处理前2个（按顺序）
```

**问题**：
- ❌ 不能选择具体的 checkpoint
- ❌ 只能按顺序处理
- ❌ 想跳过某些 epoch 不方便

### 现在：交互式选择

```bash
请选择 [all]: 1,3,5

# 优势：精确控制要推理哪些
# 结果：只处理选中的 checkpoint
```

**优势**：
- ✅ 精确选择任意 checkpoint
- ✅ 可以跳过不需要的
- ✅ 更灵活、更直观

---

## 🎯 实际应用

### 应用1：测试最新模型

```bash
找到 10 个未推理的 checkpoint

# 只测试最新的
请选择: 10

# 快速看到最新效果
```

### 应用2：对比关键 epoch

```bash
找到 10 个未推理的 checkpoint

# 选择几个关键的 epoch 对比
请选择: 1,5,10

# 快速对比开始、中间、最新的效果
```

### 应用3：补充缺失的推理

```bash
  [ 1] epoch=000027 (需要: edge)
  [ 2] epoch=000055 (需要: no-edge)
  [ 3] epoch=000083 (需要: dummy-edge)

# 只需要补充某些模式
请选择: 1,2,3

# 只推理缺失的模式，不重复推理已有的
```

---

## ⚙️ 技术细节

### 唯一化 epoch 列表

```bash
# 创建唯一的 epoch 列表
declare -A UNIQUE_EPOCHS
for epoch in "${NEW_EDGE_EPOCHS[@]}" "${NEW_NO_EDGE_EPOCHS[@]}" "${NEW_DUMMY_EPOCHS[@]}"; do
    UNIQUE_EPOCHS[$epoch]=1
done

# 排序
AVAILABLE_EPOCHS=($(for epoch in "${!UNIQUE_EPOCHS[@]}"; do echo "$epoch"; done | sort))
```

### 检测需要的模式

```bash
# 对每个 epoch，检查哪些模式需要推理
modes_needed=()
for check_epoch in "${NEW_EDGE_EPOCHS[@]}"; do
    if [ "$check_epoch" = "$epoch" ]; then
        modes_needed+=("edge")
        break
    fi
done
# ... (no-edge 和 dummy-edge 类似)
```

### 解析用户输入

```bash
# 解析逗号分隔的序号
IFS=',' read -ra SELECTED_IDS <<< "$CKPT_SELECTION"

# 验证每个序号
for id_str in "${SELECTED_IDS[@]}"; do
    # 去除空格
    id_str=$(echo "$id_str" | xargs)
    
    # 验证是数字
    if ! [[ "$id_str" =~ ^[0-9]+$ ]]; then
        echo "❌ 错误：'$id_str' 不是有效的序号"
        INVALID_SELECTION=true
        break
    fi
    
    # 验证范围
    idx=$((id_str - 1))
    if [ $idx -lt 0 ] || [ $idx -ge ${#AVAILABLE_EPOCHS[@]} ]; then
        echo "❌ 错误：序号 $id_str 超出范围"
        INVALID_SELECTION=true
        break
    fi
    
    # 添加到选择列表
    SELECTED_EPOCHS+=("${AVAILABLE_EPOCHS[$idx]}")
done
```

### 过滤循环中的 checkpoint

```bash
# 在三个模式的循环中，只处理选中的 epoch
for CKPT_FILE in "${CKPT_FILES[@]}"; do
    # 提取 epoch
    EPOCH_NUM="${BASH_REMATCH[1]}"
    
    # 检查是否在选择列表中
    EPOCH_SELECTED=false
    if [ ${#SELECTED_EPOCHS[@]} -gt 0 ]; then
        for selected_epoch in "${SELECTED_EPOCHS[@]}"; do
            if [ "$selected_epoch" = "$EPOCH_NUM" ]; then
                EPOCH_SELECTED=true
                break
            fi
        done
    else
        # 如果没有选择（所有已存在），处理所有
        EPOCH_SELECTED=true
    fi
    
    if [ "$EPOCH_SELECTED" = false ]; then
        continue  # 跳过未选中的
    fi
    
    # 继续处理...
done
```

---

## 📈 优势总结

### 1. 灵活性

- ✅ 可以选择任意 checkpoint
- ✅ 可以跳过不需要的
- ✅ 可以单独测试某一个

### 2. 直观性

- ✅ 列表显示所有未推理的
- ✅ 显示每个需要的模式
- ✅ 序号化，易于选择

### 3. 效率

- ✅ 避免不必要的推理
- ✅ 快速测试关键 epoch
- ✅ 精确控制资源使用

### 4. 兼容性

- ✅ 保留 'all' 默认行为
- ✅ 直接回车推理所有
- ✅ 与之前的使用习惯一致

---

## 🎯 推荐用法

### 日常使用

```bash
# 默认推理所有
请选择 [all]: ← 回车
```

### 快速测试

```bash
# 只测试最新的
请选择 [all]: 5
```

### 分批处理

```bash
# 第一批
请选择 [all]: 1,2,3

# 下次运行继续
请选择 [all]: 4,5
```

### 对比实验

```bash
# 选择关键 epoch
请选择 [all]: 1,3,5,7,9
```

---

## ⚠️ 注意事项

### 1. 序号从1开始

列表显示的序号从1开始，不是从0开始。

### 2. 输入验证

脚本会验证输入的序号：
- 必须是数字
- 必须在范围内
- 无效输入会提示重新输入

### 3. 空格自动处理

```bash
# 以下输入等价
1,3,5
1, 3, 5
1 , 3 , 5
```

### 4. 默认行为

直接回车等同于 'all'，推理所有未完成的 checkpoint。

---

## 📚 相关改动

### 删除的功能

- ❌ "每次最多计算 checkpoint 数量"选项
- ❌ 批次限制（MAX_CKPTS_PER_RUN）
- ❌ 循环中的限制检查
- ❌ 统计中的剩余任务提示

### 新增的功能

- ✅ 未推理 checkpoint 列表显示
- ✅ 显示每个需要的模式
- ✅ 交互式选择（单选/多选）
- ✅ 输入验证和错误提示
- ✅ 循环中的选择过滤

---

## ✅ 完整功能列表

现在 `run_auto_inference.sh` 选项1 包含：

1. ✅ 交互式目录选择
2. ✅ 输出路径设置
3. ✅ 可选重新计算指标
4. ✅ **未推理 checkpoint 列表** ⭐ 新增
5. ✅ **交互式选择 checkpoint** ⭐ 新增
6. ✅ 预扫描统计
7. ✅ 智能跳过
8. ✅ 批量检查（可选）
9. ✅ 自动生成报告

**功能更灵活，用户体验更好！** 🎉

---

**✅ Checkpoint 选择功能已添加，现在可以精确控制要推理哪些 checkpoint！** ✨

