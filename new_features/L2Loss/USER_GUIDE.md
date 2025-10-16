# Edge PSNR 完整用户指南

## 🎯 快速开始

### 最简单的使用方法

```bash
# 1. 确保环境正确
conda activate sr_infer

# 2. 运行脚本
./run_auto_inference.sh

# 3. 按提示操作
选择：1
Edge PSNR? [n]: ← 回车（使用默认）
开始推理? [y]: ← 回车（确认推理）

# 4. 等待完成
# Edge PSNR 自动计算并保存
```

**就这么简单！** 🎉

---

## 📋 完整交互流程

### 步骤1：选择功能

```bash
==================================================
           StableSR Edge 推理菜单
==================================================

1. 推理指定目录下全部 checkpoint (edge & no-edge)  ← 选这个
2. 推理指定 checkpoint 文件 (edge)
3. 推理指定 checkpoint 文件 (no-edge)
4. 生成推理结果报告 (CSV格式)
0. 退出

请选择操作 [0-4]: 1
```

### 步骤2：选择目录

```bash
请输入 logs 目录路径 [logs]: ← 回车使用默认

可用的子目录：
1. stablesr_edge_loss_20251015_194003
2. experiment_20251016_120000

请选择目录编号 [1-2]: 1
✓ 将处理目录: stablesr_edge_loss_20251015_194003
```

### 步骤3：设置输出

```bash
请输入保存目录名 [validation_results]: ← 回车使用默认
✓ 结果将保存到: validation_results
```

### 步骤4：选择 Edge PSNR 重新计算

```bash
是否重新计算已有结果的 Edge PSNR 指标？
  注意：新推理的结果会自动计算 Edge PSNR
  此选项仅针对跳过的已有结果

重新计算 Edge PSNR? (y/n) [默认: n]:
```

**选择建议**：
- 首次推理或只关注新结果 → 回车（n）
- 需要补充旧数据或最终整理 → 输入 y

### 步骤5：预扫描和确认 ⭐

```bash
✓ 找到 5 个 checkpoint 文件（已排除 last.ckpt）

正在检查哪些 checkpoint 需要推理...

推理需求统计：
  EDGE 模式: 需要推理 2 个 checkpoint
  NO-EDGE 模式: 需要推理 2 个 checkpoint
  DUMMY-EDGE 模式: 需要推理 2 个 checkpoint
  总计: 需要推理 6 个任务

⚠️  发现 6 个新的推理任务需要执行

是否开始推理? (y/n) [默认: y]:
```

**选择建议**：
- 准备好推理 → 回车（y）
- 任务太多或时间不合适 → 输入 n

### 步骤6：推理执行

```bash
✓ 开始执行推理...

正在运行 EDGE 模式推理...
✓ 跳过 epoch=27 (已有 10 张图片)
✓ 跳过 epoch=55 (已有 10 张图片)
→ 处理 epoch=138                    ← 新推理
→ 处理 epoch=166                    ← 新推理

EDGE 模式统计: 已处理 2 个，跳过 3 个

正在运行 NO-EDGE 模式推理...
...
```

### 步骤7：Edge PSNR 检查（可选）

如果选择了重新计算（y）：

```bash
==================================================
  批量检查并计算 Edge PSNR 指标
==================================================

扫描目录: validation_results/...
检查: edge/epochs_138
  ✓ Edge PSNR 指标已存在
...

统计信息：
  找到 15 个 metrics.json 文件
  ✓ 已更新: 5 个
  ✓ 已存在: 10 个
```

如果选择了默认（n）：

```bash
✓ 跳过 Edge PSNR 批量检查（用户选择不重新计算）
```

### 步骤8：生成报告

```bash
==================================================
  正在生成推理结果报告
==================================================

扫描结果目录: validation_results/...
✓ 报告生成成功
```

---

## 🎛️ 两个关键选择

### 选择1：Edge PSNR 重新计算

| 选择 | 效果 | 适用场景 |
|-----|------|---------|
| **n (默认)** | 只计算新推理 | 日常使用、快速验证 |
| **y** | 检查并补充所有 | 补充旧数据、最终整理 |

### 选择2：开始推理

| 选择 | 效果 | 适用场景 |
|-----|------|---------|
| **y (默认)** | 开始推理 | 准备就绪、时间充足 |
| **n** | 取消返回 | 任务太多、时间不合适 |

---

## 📊 典型使用场景

### 场景A：快速验证新 checkpoint

```bash
./run_auto_inference.sh
# 1 → 回车 → 回车 → 回车 → 回车
```

**4次回车，全自动！**

### 场景B：补充完整数据

```bash
./run_auto_inference.sh
# 1 → 回车 → 回车 → y → 回车
```

**确保所有数据都有 Edge PSNR**

### 场景C：先查看再决定

```bash
./run_auto_inference.sh
# 1 → 回车 → 回车 → 回车

# 看到：发现 20 个新任务
# 太多了，取消

# 开始推理? [y]: n
# ✗ 用户取消推理

# 稍后有时间再运行
```

### 场景D：只生成报告

```bash
./run_auto_inference.sh
# 1 → 回车 → 回车 → 回车

# 看到：0 个新任务
# ✓ 没有新任务，将直接生成报告...

# 不需要确认，自动继续生成报告
```

---

## 📈 输出结果

### metrics.json（每个 epoch 目录）

```json
{
  "images": [
    {
      "image_name": "0801.png",
      "psnr": 24.5379,
      "ssim": 0.7759,
      "lpips": 0.2655,
      "edge_psnr": 29.0891    ← Edge PSNR (dB)
    }
  ],
  "average_psnr": 21.0714,
  "average_ssim": 0.5853,
  "average_lpips": 0.3036,
  "average_edge_psnr": 26.1234  ← 平均 Edge PSNR (dB)
}
```

### metrics.csv（每个 epoch 目录）

```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
0801.png,24.5379,0.7759,0.2655,29.0891
0802.png,25.7358,0.6455,0.2015,27.3456
...
Average,21.0714,0.5853,0.3036,26.1234
```

### 综合报告 CSV

包含4个指标块：
1. PSNR
2. SSIM
3. LPIPS
4. **Edge PSNR** ⭐

每个指标块显示所有 epoch 的对比。

---

## 🎯 Edge PSNR 解读指南

### 单个值解读

| Edge PSNR | 评价 | 建议 |
|-----------|------|------|
| > 35 dB | 优秀 | 保持当前方法 |
| 30-35 dB | 很好 | 可以考虑优化 |
| 25-30 dB | 一般 | 需要改进边缘处理 |
| < 25 dB | 较差 | 重点改进边缘 |

### 与 Image PSNR 对比

```
示例结果：
  Image PSNR: 24.54 dB
  Edge PSNR:  29.09 dB
  Δ = +4.55 dB

解读：
  - 边缘质量比整体好 4.55 dB
  - 模型较好地保持了边缘细节
  - Edge 增强有效果
```

### 不同模式对比

```
Edge 模式:      Edge PSNR = 29.1 dB  ← 最好
No-Edge 模式:   Edge PSNR = 26.2 dB
Dummy-Edge模式: Edge PSNR = 25.8 dB  ← 最差

结论：使用真实 edge 效果最好
```

---

## ⚠️ 常见问题

### Q1: 推理很慢，可以中断吗？

**A**: 可以按 `Ctrl+C` 中断。下次运行时：
- 已完成的会被跳过
- 未完成的会继续处理

### Q2: 忘记选择重新计算 Edge PSNR 怎么办？

**A**: 可以：
1. 再次运行脚本，选择 y
2. 手动运行：
   ```bash
   python scripts/recalculate_edge_l2_loss.py <dir> <gt_dir>
   ```

### Q3: 显示"计算失败"是什么原因？

**A**: 可能的原因：
1. 环境问题（不在 conda 环境中）
2. GT 图片路径不对
3. 图片文件缺失

**解决**：
```bash
# 手动运行查看详细错误
python scripts/recalculate_edge_l2_loss.py \
    validation_results/.../edge/epochs_27 \
    /mnt/nas_dp/test_dataset/512x512_valid_HR
```

### Q4: Edge PSNR 是越大越好还是越小越好？

**A**: **越大越好！** PSNR 是峰值信噪比，值越大表示质量越好。
- 不要和之前的 L2 Loss（越小越好）混淆

---

## 🎊 总结

### 完整功能

`run_auto_inference.sh` 选项1 现在提供：

1. **交互式选择**
   - 目录选择
   - 输出路径设置

2. **可选功能**
   - Edge PSNR 重新计算（默认：否）
   - 推理前确认（默认：是）

3. **智能处理**
   - 自动跳过已有结果
   - 可选检查 Edge PSNR
   - 批量扫描确保完整

4. **自动化**
   - Edge PSNR 自动计算
   - 报告自动生成
   - 完整的错误处理

### 推荐工作流

```bash
# 日常使用（最快）
./run_auto_inference.sh
1 → 回车 → 回车 → 回车 → 回车

# 完整模式（确保完整）
./run_auto_inference.sh
1 → 回车 → 回车 → y → 回车

# 谨慎模式（查看后决定）
./run_auto_inference.sh
1 → 回车 → 回车 → 回车 → 查看统计 → 决定 y/n
```

---

**✅ 现在你可以完全控制推理流程，同时享受自动化的便利！** 🚀

