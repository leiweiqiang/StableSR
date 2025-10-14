# Edge Monitor Callback 使用说明

## 功能概述

`EdgeMonitorCallback` 是一个用于训练过程中监控 edge map 相关网络层变化的回调函数。它会每隔固定步数记录 `EdgeMapProcessor` 中各层的权重和梯度统计信息。

## 功能特性

### 1. 权重监控
- **weight_mean**: 权重的平均值
- **weight_std**: 权重的标准差
- **weight_max**: 权重的最大值
- **weight_min**: 权重的最小值
- **weight_change**: 相比上次检查的权重变化量（绝对值平均）

### 2. 梯度监控
- **grad_mean**: 梯度的平均值
- **grad_std**: 梯度的标准差
- **grad_max**: 梯度的最大值
- **grad_min**: 梯度的最小值
- **grad_norm**: 梯度的L2范数

## 配置方法

### 方法1：通过配置文件

在训练配置文件（如 `v2-finetune_text_T_512_edge_800.yaml`）中添加：

```yaml
lightning:
  callbacks:
    edge_monitor:
      target: main.EdgeMonitorCallback
      params:
        check_frequency: 10      # 每10步检查一次
        log_gradients: True      # 记录梯度统计
        log_weights: True        # 记录权重统计
```

### 方法2：在代码中直接添加

```python
from main import EdgeMonitorCallback

# 创建callback
edge_monitor = EdgeMonitorCallback(
    check_frequency=10,      # 每10步检查一次
    log_gradients=True,      # 记录梯度
    log_weights=True         # 记录权重
)

# 添加到trainer
trainer = Trainer(
    callbacks=[edge_monitor, ...]
)
```

## 输出示例

### 控制台输出

每隔10步，会在控制台输出关键统计信息：

```
================================================================================
Edge Processor Monitor - Step 100
================================================================================
  edge_processor/backbone.0.weight/weight_change: 1.234567e-04
  edge_processor/backbone.0.weight/grad_norm: 2.345678e-02
  edge_processor/to_four.weight/weight_change: 3.456789e-05
  edge_processor/to_four.weight/grad_norm: 4.567890e-03
```

### TensorBoard/WandB 日志

所有统计信息都会被记录到训练日志中，可以通过 TensorBoard 或 WandB 可视化：

- `edge_processor/*/weight_mean`
- `edge_processor/*/weight_std`
- `edge_processor/*/weight_change`
- `edge_processor/*/grad_norm`
- 等等...

## 监控的网络层

callback 会自动监控 `EdgeMapProcessor` 中的所有可训练参数，包括：

1. **backbone** 中的卷积层和批归一化层
   - `backbone.0.weight` (Conv2d 3->32)
   - `backbone.1.weight`, `backbone.1.bias` (BatchNorm2d)
   - `backbone.3.weight` (Conv2d 32->64)
   - 等等...

2. **to_four** 层
   - `to_four.weight` (Conv2d 128->4)
   - `to_four.bias`

## 使用建议

### 检查频率设置

- **快速验证**: `check_frequency=10` - 每10步检查一次，适合初期验证
- **常规训练**: `check_frequency=50` - 每50步检查一次，平衡性能和监控
- **长期训练**: `check_frequency=100` - 每100步检查一次，减少日志量

### 观察指标

1. **训练初期** (前1000步)
   - 关注 `grad_norm`：确保梯度不为0，说明edge相关层正在学习
   - 关注 `weight_change`：应该能看到明显的权重更新

2. **训练中期**
   - 关注 `weight_change` 趋势：应该逐渐减小但保持稳定
   - 关注 `grad_norm`：不应该出现梯度爆炸或消失

3. **训练后期**
   - `weight_change` 应该趋于稳定
   - 各层的变化应该保持协调

## 故障排查

### 如果梯度为0
- 检查edge map是否正确生成和传递
- 检查loss计算中是否包含edge相关的项
- 确认 `edge_processor` 的参数 `requires_grad=True`

### 如果权重不更新
- 检查优化器配置
- 确认学习率设置
- 查看是否有冻结的层

### 如果梯度爆炸
- 考虑降低学习率
- 检查数据归一化
- 考虑添加梯度裁剪

## 示例训练命令

```bash
python main.py -t \
    --base configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml \
    --gpus 0,1,2,3 \
    --name edge_training_test
```

训练开始后，每10步会自动在控制台和日志系统中记录edge相关层的变化。

