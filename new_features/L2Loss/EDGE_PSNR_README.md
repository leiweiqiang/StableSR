# Edge PSNR 功能说明

## 🎯 一句话总结

新增了 **Edge PSNR** 指标，用于评估超分辨率图像的边缘质量，单位 dB，**值越大越好**。

---

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate sr_infer

# 2. 运行脚本
./run_auto_inference.sh

# 3. 选择并确认（4次回车）
选择: 1
目录: ← 回车
输出: ← 回车
重新计算Edge PSNR? [n]: ← 回车（默认）
开始推理? [y]: ← 回车（确认）

# Edge PSNR 自动计算并保存！
```

---

## 📊 Edge PSNR 解读

| Edge PSNR (dB) | 质量评价 |
|---------------|---------|
| > 40 dB | 优秀 ⭐⭐⭐⭐⭐ |
| 35-40 dB | 很好 ⭐⭐⭐⭐ |
| 30-35 dB | 好 ⭐⭐⭐ |
| 25-30 dB | 一般 ⭐⭐ |
| < 25 dB | 较差 ⭐ |

**重要**：值越大越好 ↑

---

## 💡 两个关键选择

### 选择1：Edge PSNR 重新计算

```
重新计算 Edge PSNR? (y/n) [默认: n]:
```

- **n (默认)**：快速模式，只计算新推理
- **y**：完整模式，检查并补充所有结果

**推荐**：日常使用选 n，最终整理选 y

### 选择2：开始推理

```
⚠️  发现 6 个新的推理任务需要执行
是否开始推理? (y/n) [默认: y]:
```

- **y (默认)**：开始推理
- **n**：取消，返回菜单

**推荐**：查看任务数量后决定

---

## 📁 输出文件

### metrics.json（每个 epoch）
```json
{
  "average_edge_psnr": 26.1234,
  "images": [
    {"image_name": "0801.png", "edge_psnr": 29.0891}
  ]
}
```

### metrics.csv（每个 epoch）
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge PSNR (dB)
0801.png,24.5379,0.7759,0.2655,29.0891
Average,21.0714,0.5853,0.3036,26.1234
```

### 综合报告 CSV（自动按 Epoch 排序）
```csv
Metric,Filename,StableSR,Epoch 27,Epoch 27,Epoch 27,Epoch 55,...
,,,dummy edge,edge,no edge,dummy edge,...
PSNR,Average,20.92,20.26,20.34,20.28,...
Edge PSNR,Average,26.12,25.34,26.78,25.91,...
```

---

## 📚 详细文档

所有文档位于：`new_features/L2Loss/`

**推荐阅读顺序**：
1. `USER_GUIDE.md` - 用户指南 ⭐
2. `EDGE_PSNR_QUICKREF.md` - 快速参考
3. `COMPLETE_CHANGELOG.md` - 完整更新日志

---

## ⚠️ 重要提醒

1. **方向**：Edge PSNR 越大越好（不是越小）
2. **环境**：使用 `python` 而不是 `python3`
3. **默认**：两个选项都有合理的默认值
4. **单位**：dB（分贝）

---

## 🎯 核心优势

- ✅ 自动计算：新推理自动包含
- ✅ 智能跳过：可选检查已有结果
- ✅ 推理确认：避免意外执行
- ✅ 自动排序：CSV 按 epoch 序号排列
- ✅ 用户友好：两次回车即可完成

---

**完整文档**: `new_features/L2Loss/`  
**核心代码**: `basicsr/metrics/edge_l2_loss.py`  
**版本**: v2.0 (2025-10-16)

**现在就可以使用！** 🎉

