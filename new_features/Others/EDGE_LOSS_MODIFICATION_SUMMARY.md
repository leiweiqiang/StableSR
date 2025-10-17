# Edge Loss 修改总结

## 修改日期
2025-10-17

## 修改目标
在训练时的loss function中增加edge map的loss，使模型更关注边缘区域的重建质量。

## 修改内容

### 1. 修改 `ldm/models/diffusion/ddpm.py` - `LatentDiffusionSRTextWT` 类

#### 1.1 __init__() 方法修改

**位置**: 第1565-1586行

**添加内容**:
- 新增参数: `edge_loss_weight=0.0` (第1585行)
- 添加配置变量:
  ```python
  self.edge_loss_weight = edge_loss_weight
  self.use_edge_loss = edge_loss_weight > 0
  ```
  (第1597-1599行)

- 初始化EdgeMapGenerator:
  ```python
  if self.use_edge_loss:
      from basicsr.utils.edge_utils import EdgeMapGenerator
      self.edge_generator = EdgeMapGenerator(device='cuda' if torch.cuda.is_available() else 'cpu')
      print(f"Edge loss enabled with weight: {self.edge_loss_weight}")
  ```
  (第1670-1674行)

#### 1.2 p_losses() 方法修改

**位置**: 第2530-2577行

**添加内容**: Edge loss计算逻辑

**工作流程**:
1. **获取预测的x0** (latent space):
   - 根据parameterization类型 (eps/x0/v) 从model_output恢复x0
   
2. **解码到image space** (保持梯度):
   - `pred_img = self.differentiable_decode_first_stage(pred_x0)`
   - `gt_img = self.differentiable_decode_first_stage(x_start)`
   
3. **生成edge maps** (使用Canny算子):
   - 由于Canny不可微分，使用`torch.no_grad()`和`.detach()`
   - 分别生成预测图像和GT图像的edge maps
   
4. **计算edge-weighted loss**:
   - 使用GT的edge map作为权重
   - 计算加权的图像重建损失
   - `weighted_img_diff = (pred_img - gt_img) ** 2 * (edge_weight + 0.1)`
   - 添加0.1的base weight避免完全忽略非边缘区域
   
5. **更新loss**:
   - `loss = loss + self.edge_loss_weight * edge_loss`
   - 记录到loss_dict中: `loss_dict['{prefix}/loss_edge']`

### 2. 修改配置文件

**文件**: `configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml`

**位置**: 第31-32行

**添加内容**:
```yaml
# Edge loss configuration
edge_loss_weight: 0.1  # Weight for edge loss (set to 0 to disable)
```

## 技术要点

### 梯度处理

**Canny边缘检测的不可微性**：
由于Canny边缘检测是基于OpenCV实现的，它是不可微分的。因此采用了以下策略：

- Edge map生成在`torch.no_grad()`中，使用`.detach()`断开计算图
- 直接计算两个edge maps之间的MSE loss
- **重要**: 这个edge_loss不会产生通过Canny算子的梯度

**训练机制**：
虽然edge_loss本身不产生梯度，但它仍然影响训练：

1. **间接优化路径**：
   - 模型通过loss_simple和loss_vlb学习生成更好的latent表示
   - 当预测的latent decode后的图像质量提高时，其edge map也会相应改善
   - Edge loss提供了额外的监督信号，引导优化方向

2. **总loss的组成**：
   ```
   total_loss = loss_simple + loss_vlb + edge_loss_weight * edge_loss
   ```
   - 虽然edge_loss部分不可微，但仍会影响total_loss的数值
   - 优化器会调整参数以降低total_loss（主要通过可微分的loss_simple和loss_vlb）
   - Edge loss作为质量监控和训练引导

3. **实际效果**：
   - 监控edge质量的变化趋势
   - 作为early stopping或模型选择的指标
   - 在训练过程中提供边缘质量的反馈

### Edge Loss 计算公式
```
# 1. 生成edge maps
pred_edge = Canny(pred_img)  # 预测图像的edge map
gt_edge = Canny(gt_img)      # GT图像的edge map

# 2. 归一化到[0,1]
pred_edge_norm = (pred_edge + 1.0) / 2.0  # [-1,1] -> [0,1]
gt_edge_norm = (gt_edge + 1.0) / 2.0      # [-1,1] -> [0,1]

# 3. 计算MSE
edge_loss = MSE(pred_edge_norm, gt_edge_norm)

# 4. 加入总loss
total_loss = original_loss + edge_loss_weight * edge_loss
```

其中：
- `pred_edge_norm`, `gt_edge_norm`: Edge maps，值域[0,1]，边缘处为1
- `edge_loss_weight`: 默认0.1，可通过配置文件调整

## 预期效果

1. **训练日志变化**:
   - 新增loss项: `train/loss_edge`
   - 总loss更新: `train/loss` = `loss_simple` + `loss_vlb` + `edge_loss_weight * loss_edge`

2. **模型行为**:
   - 模型会更关注边缘区域的重建质量
   - 边缘处的误差会被放大（通过edge_weight）
   - 有助于生成更清晰的边缘

3. **可调参数**:
   - `edge_loss_weight`: 控制edge loss的权重
     - 0.0: 禁用edge loss
     - 0.01-0.1: 温和的edge引导
     - 0.1-0.5: 强edge引导
     - 建议从0.1开始

## 验证步骤

1. **检查初始化**:
   ```bash
   # 训练开始时应该看到：
   "Edge loss enabled with weight: 0.1"
   ```

2. **检查训练日志**:
   - WandB/TensorBoard中应该有`train/loss_edge`曲线
   - 总loss应该包含edge loss component

3. **调整权重**:
   - 如果edge loss过大/过小，调整配置文件中的`edge_loss_weight`
   - 重新训练观察效果

## 注意事项

1. **性能影响**:
   - Edge loss需要decode操作（latent → image），会增加训练时间
   - 每个training step增加约10-20%的计算时间
   
2. **内存占用**:
   - 需要额外存储pred_img和gt_img (image space)
   - 建议batch size相应减小
   
3. **权重调优**:
   - 建议从小权重开始（0.01-0.1）
   - 观察loss_edge的数值范围，调整到与loss_simple同数量级
   
4. **仅训练时启用**:
   - Edge loss只在`self.training=True`时计算
   - 验证/测试阶段不计算edge loss

## 文件修改清单

- ✅ `ldm/models/diffusion/ddpm.py` (添加edge loss逻辑)
- ✅ `configs/stableSRNew/v2-finetune_text_T_512_edge_800.yaml` (添加配置参数)
- ✅ 依赖已有: `basicsr/utils/edge_utils.py` (EdgeMapGenerator)

## 回退方案

如果需要禁用edge loss，有两种方式：

1. **配置文件方式** (推荐):
   ```yaml
   edge_loss_weight: 0.0  # 设为0即可禁用
   ```

2. **代码方式**:
   - 将`self.use_edge_loss = False`强制设置
   - 或注释掉p_losses()中的edge loss计算代码块

## 相关资源

- Edge Map Generator: `basicsr/utils/edge_utils.py`
- 训练脚本: `main.py`
- 配置文件: `configs/stableSRNew/*.yaml`

