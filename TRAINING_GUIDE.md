# Edge增强StableSR训练指南（基于Turbo）

## 📋 概述

本指南说明如何基于StableSR Turbo checkpoint训练Edge增强模型。

## 🔧 关键改进

### 1. **Edge特征归一化**（已修复）
- **问题**：之前的模型产生latent范围异常（[-30, 35] vs 正常的 [-4, 4]）
- **解决**：在 `edge_processor.py` 中添加输出归一化：
  ```python
  x = torch.tanh(x) * 0.5  # 限制在[-0.5, 0.5]范围
  ```

### 2. **使用Turbo作为基础**
- **原因**：Turbo模型比v2-1标准模型清晰度高59.4%
- **配置**：`v2-finetune_text_T_512_edge_fixed.yaml` 已指向turbo checkpoint

### 3. **优化的学习率**
- Base learning rate: `3.0e-05`（降低以提高稳定性）

## 📁 关键文件

```
configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml  # 训练配置
ldm/modules/diffusionmodules/edge_processor.py              # Edge处理器（已归一化）
ldm/modules/diffusionmodules/unet_with_edge.py              # Edge UNet
train_edge_turbo.sh                                         # 训练脚本
```

## 🚀 开始训练

### 方式1：使用训练脚本（推荐）

```bash
cd /root/dp/StableSR_Edge_v2
./train_edge_turbo.sh
```

### 方式2：手动命令

```bash
cd /root/dp/StableSR_Edge_v2
conda activate sr_edge

python main.py \
    --base configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml \
    --train \
    --gpus 0,1,2,3 \
    --logdir logs \
    --name stablesr_edge_turbo \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from /stablesr_dataset/checkpoints/stablesr_turbo.ckpt
```

## 📊 监控训练

### 查看日志
```bash
tail -f logs/stablesr_edge_turbo_*/train.log
```

### TensorBoard（如果可用）
```bash
tensorboard --logdir logs/stablesr_edge_turbo_*
```

## 🧪 训练后测试

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml \
    --ckpt logs/stablesr_edge_turbo_*/checkpoints/epoch=000030.ckpt \
    --vqgan_ckpt /stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt \
    --init-img /mnt/nas_dp/test_dataset/128x128_valid_LR \
    --gt-img /mnt/nas_dp/test_dataset/512x512_valid_HR \
    --outdir ./output_edge_turbo \
    --ddpm_steps 200 \
    --dec_w 0.5 \
    --colorfix_type wavelet
```

## 📈 预期效果

基于Turbo + Edge归一化，预期：

| 指标 | 目标 |
|------|------|
| Latent范围 | [-4, 4] ✓（正常） |
| 极端像素比例 | < 30% |
| 清晰度提升 | > Turbo基线 |

## ⚠️ 注意事项

1. **GPU内存**：
   - 每GPU建议至少24GB显存（batch_size=2）
   - 如果OOM，降低batch_size或增加梯度累积

2. **训练时长**：
   - 建议训练至少30-50 epochs
   - 每15 epochs检查一次checkpoint质量

3. **数据集**：
   - 确保训练数据包含edge map（`img_edge`字段）
   - Edge map应为3通道，值域[0, 1]

## 🐛 故障排查

### 问题1：OOM（显存不足）
**解决**：
```yaml
# 在配置文件中修改
data:
  params:
    batch_size: 1  # 降低batch size
```

### 问题2：训练不稳定
**解决**：
- 降低学习率：`base_learning_rate: 1.0e-05`
- 增加warmup steps

### 问题3：生成的图片仍然模糊
**检查**：
1. 确认使用了正确的checkpoint
2. 使用200步采样（不要用4步）
3. 检查edge map质量

## 📝 版本历史

- **v2**: 基于Turbo + Edge归一化 (当前版本)
- **v1**: 基于v2-1 + 无归一化 (已弃用，latent范围问题)

## 🎯 下一步

训练完成后：
1. 在验证集上测试多个checkpoints（epoch 30, 45, 60）
2. 对比Turbo基线 vs Edge增强的效果
3. 如果Edge效果好，部署到生产环境
4. 如果效果不佳，调整edge_weight或归一化参数

---
最后更新：2025-10-07



