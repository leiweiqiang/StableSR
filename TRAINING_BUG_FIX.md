# 🚨 CRITICAL BUG: Edge Processor训练时被冻结

## 问题发现

用户报告：**训练出来的checkpoint中edge_processor的权重完全没有变化**

通过检查checkpoint发现：
```
model.diffusion_model.edge_processor.backbone.0.weight
   Requires grad: False  ❌
```

**所有edge_processor参数的`requires_grad=False`，导致训练时完全没有更新！**

---

## 根本原因分析

### 问题链路

1. **配置设置**: `unfrozen_diff: False`（冻结UNet）

2. **父类冻结逻辑** (`ldm/models/diffusion/ddpm.py:1630-1637`):
   ```python
   if not self.unfrozen_diff:
       self.model.eval()
       for name, param in self.model.named_parameters():
           if 'spade' not in name:
               param.requires_grad = False  # ❌ 冻结所有非spade参数
           else:
               param.requires_grad = True
   ```

3. **Edge Processor被误伤**:
   - edge_processor参数名: `model.diffusion_model.edge_processor.xxx`
   - 不包含'spade' → **被冻结！**

### 调用时序问题

```
LatentDiffusionSRTextWTWithEdge.__init__()
  ↓
super().__init__()  # LatentDiffusionSRTextWT
  ↓
  创建UNet (包含edge_processor)
  ↓
  UNet.__init__() 调用 _ensure_edge_modules_require_grad()
    → edge_processor.requires_grad = True ✅
  ↓
  父类继续执行冻结逻辑
    → if 'spade' not in name: requires_grad = False
    → edge_processor.requires_grad = False ❌  (被覆盖!)
  ↓
返回到子类 (但已经晚了，参数被冻结)
```

**结果**: Edge Processor完全没有训练！

---

## 修复方案

### 修复1: 父类冻结逻辑豁免edge_processor

**文件**: `ldm/models/diffusion/ddpm.py:1634-1639`

```python
# 修改前
if 'spade' not in name:
    param.requires_grad = False

# 修改后  
if 'spade' not in name and 'edge_processor' not in name:
    param.requires_grad = False
else:
    param.requires_grad = True
    print(f"✅ Trainable parameter: {name}")
```

### 修复2: 子类强制确保edge_processor可训练

**文件**: `ldm/models/diffusion/ddpm_with_edge.py:101-121`

```python
def __init__(self, ...):
    super().__init__(...)
    
    # 🔥 CRITICAL FIX: 父类可能冻结了edge_processor，需要重新解冻
    if self.use_edge_processing:
        self._ensure_edge_processor_trainable()

def _ensure_edge_processor_trainable(self):
    """在父类__init__之后，强制解冻edge_processor"""
    if hasattr(self.model, 'diffusion_model'):
        edge_processor = self.model.diffusion_model.edge_processor
        if edge_processor is not None:
            edge_processor.train()
            for param in edge_processor.parameters():
                param.requires_grad = True
```

---

## 验证修复

### 1. 代码层面验证

启动训练，查看日志：
```bash
python main.py --base configs/.../edge_loss.yaml --train ...
```

应该看到：
```
🔥 Edge Processor - Trainable Parameters:
  ✅ backbone.0.weight: requires_grad=True
  ✅ backbone.0.bias: requires_grad=True
  ✅ backbone.1.weight: requires_grad=True
  ...
```

### 2. Checkpoint验证

训练几个epoch后检查：
```bash
python 2.py logs/.../checkpoints/epoch=000005.ckpt
```

应该看到：
```
model.diffusion_model.edge_processor.backbone.0.weight
   Requires grad: True  ✅  (注意：保存时会是False，但训练时是True)
   Mean: xxx (应该与初始值不同)
```

### 3. 对比初始权重

```python
import torch

# 初始checkpoint
ckpt0 = torch.load('epoch=000000.ckpt')
weight0 = ckpt0['state_dict']['model.diffusion_model.edge_processor.backbone.0.weight']

# 训练后checkpoint  
ckpt5 = torch.load('epoch=000005.ckpt')
weight5 = ckpt5['state_dict']['model.diffusion_model.edge_processor.backbone.0.weight']

# 应该不同！
diff = (weight5 - weight0).abs().mean()
print(f"Weight change: {diff}")  # 应该 > 0
```

---

## 问题总结

### 发现的Bug

| Bug | 描述 | 影响 | 修复状态 |
|-----|------|------|---------|
| **训练Bug1** | `edge_loss_weight=0` | 模型不学edge特征 | ⚠️ 需改配置 |
| **训练Bug2** | edge_processor被冻结 | 参数完全不更新 | ✅ 已修复 |
| **推理Bug** | tile推理丢弃edge_map | tile模式无效 | ✅ 已修复 |

### 综合影响

**三个bug叠加 = Edge功能彻底失效！**

1. 即使想训练，参数也不更新（被冻结）
2. 即使参数更新了，也没有loss监督（loss_weight=0）  
3. 即使训练好了，推理也用不上（tile模式丢弃edge_map）

---

## 完整修复清单

### ✅ 已修复（代码层面）

1. **权重初始化**: 0.01 → 0.1
2. **Tile推理链路**: override sample_canvas系列方法
3. **冻结逻辑豁免**: 父类不冻结edge_processor  
4. **子类强制解冻**: __init__最后确保可训练

### ⚠️ 仍需配置修改

5. **边缘损失**: `edge_loss_weight: 0 → 0.3`

---

## 重新训练步骤

### 步骤1: 使用修复后的代码

```bash
cd /root/dp/StableSR_Edge_v2_loss

# 已修复的文件：
# ✅ ldm/models/diffusion/ddpm.py (父类冻结逻辑)
# ✅ ldm/models/diffusion/ddpm_with_edge.py (子类解冻+tile推理)
```

### 步骤2: 修改配置

```yaml
# configs/stableSRNew/v2-finetune_text_T_512_edge_loss.yaml

# 启用边缘损失
edge_loss_weight: 0.3  # 从0改为0.3

# 可选：完全解冻UNet (更激进)
# unfrozen_diff: True
```

### 步骤3: 启动训练

```bash
source /root/miniconda/bin/activate sr_edge
bash train_edge_loss_t5.sh
```

### 步骤4: 监控训练

**立即检查日志**:
```
🔥 Edge Processor - Trainable Parameters:
  ✅ backbone.0.weight: requires_grad=True  # 必须是True!
```

**监控tensorboard**:
```bash
tensorboard --logdir logs/
```

应该看到：
- `train/edge_loss`: 从初始值逐渐下降
- `train/edge_loss_weighted`: 0.3 × edge_loss

---

## 预期效果

### 修复前
- Edge processor参数: **完全不更新** ❌
- Edge loss: 始终为0 ❌
- Tile推理: edge_map被丢弃 ❌

### 修复后  
- Edge processor参数: **正常更新** ✅
- Edge loss: 逐渐下降 ✅
- Tile推理: edge_map正常传递 ✅
- **性能**: edge vs no_edge 应该差距 > 1dB PSNR ✅

---

## 诊断命令

### 检查参数是否可训练
```python
import torch
model = ...  # 加载模型
for name, param in model.model.diffusion_model.edge_processor.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

### 检查权重是否变化
```python
# 对比两个epoch的权重
ckpt1 = torch.load('epoch=000010.ckpt')
ckpt2 = torch.load('epoch=000020.ckpt')

key = 'model.diffusion_model.edge_processor.backbone.0.weight'
diff = (ckpt2['state_dict'][key] - ckpt1['state_dict'][key]).abs().mean()
print(f"Weight change: {diff}")  # 应该 > 0!
```

### 检查edge_loss
```bash
# 查看训练日志
grep "edge_loss" logs/.../log.txt
```

---

## 技术要点

### 为什么保存的checkpoint中requires_grad=False?

**正常现象**！PyTorch保存checkpoint时不保存requires_grad状态。

- **训练时**: `param.requires_grad = True` (通过代码设置)
- **保存时**: checkpoint只保存权重值，不保存requires_grad
- **加载时**: 默认requires_grad=False，需要代码重新设置

所以看到checkpoint中requires_grad=False不代表训练时也是False。

**关键是**: 训练时日志应该显示requires_grad=True!

### 如何确认真的在训练?

1. **日志输出**: 看到"✅ Trainable parameter: edge_processor.xxx"
2. **权重变化**: 不同epoch的权重值应该不同
3. **Loss下降**: edge_loss应该逐渐减小
4. **梯度检查**: 可以打印梯度是否为None

---

## 总结

### 问题原因
`unfrozen_diff: False` + 冻结逻辑没有豁免edge_processor = **edge_processor被错误冻结**

### 修复方案  
1. ✅ 父类豁免edge_processor
2. ✅ 子类强制解冻edge_processor
3. ⚠️ 配置启用edge_loss_weight

### 下一步
**必须重新训练**，之前的checkpoint因为edge_processor没训练，所以无效。

使用修复后的代码 + edge_loss_weight=0.3 训练50+ epochs，应该能看到明显效果。

---

**修复日期**: 2025-10-13  
**相关文件**: 
- `ldm/models/diffusion/ddpm.py`
- `ldm/models/diffusion/ddpm_with_edge.py`
- `CRITICAL_BUG_FIX.md` (推理bug)

