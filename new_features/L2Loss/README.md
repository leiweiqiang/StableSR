# Edge L2 Loss Metric

## 目录说明

本目录包含 **Edge L2 Loss (MSE)** 指标的相关文档和测试文件。

## 文件列表

### 📖 文档

1. **EDGE_L2_LOSS_QUICKSTART.md** - 快速入门指南 ⭐ 推荐先看这个
   - 一句话总结
   - 快速使用方法
   - 结果解读
   - 常见问题

2. **EDGE_L2_LOSS_README.md** - 完整技术文档
   - 详细的需求分析
   - 完整的技术实现说明
   - API 文档
   - 使用示例
   - 输出格式说明
   - 常见问题（FAQ）
   - 技术细节

3. **EDGE_L2_LOSS_SUMMARY.md** - 实现总结
   - 完成工作清单
   - 文件修改说明
   - 输出格式示例
   - 兼容性说明
   - 验证清单

4. **AUTO_CHECK_UPDATE.md** - 自动检查和更新功能 ⭐ 新功能
   - 在 run_auto_inference.sh 选项1中的自动检查
   - 批量扫描和更新所有 metrics.json
   - 使用方法和示例
   - 故障排除

5. **SKIP_WITH_L2LOSS_CHECK.md** - 智能跳过功能 ⭐ 新功能
   - 跳过前自动检查并计算 L2Loss
   - 四个位置的改进（edge, no-edge, dummy-edge, stablesr）
   - 工作机制和优势
   - 使用场景和最佳实践

6. **INSTALL_AND_USAGE.md** - 安装与使用指南
   - 文件位置说明
   - 使用方法
   - 输出格式
   - 高级用法

7. **FILE_INDEX.md** - 文件索引
   - 所有文件的详细说明
   - 推荐阅读顺序
   - 使用场景指南

### 🧪 测试

8. **test_edge_l2_loss.py** - 测试脚本
   - 测试从 numpy 数组计算
   - 测试从文件计算
   - 测试从 PyTorch tensor 计算
   - 测试便捷调用方法
   - 测试不同尺寸图片处理

## 快速开始

### 推荐阅读顺序

1. 📖 `EDGE_L2_LOSS_QUICKSTART.md` - 快速了解和上手
2. 🧪 运行 `test_edge_l2_loss.py` - 验证功能
3. 📖 `EDGE_L2_LOSS_README.md` - 深入了解技术细节
4. 📝 `EDGE_L2_LOSS_SUMMARY.md` - 查看完整实现清单

### 使用方法

```bash
# 方法1：在推理时自动计算
python scripts/auto_inference.py \
    --ckpt checkpoint.ckpt \
    --init_img lr_images/ \
    --gt_img gt_images/ \
    --calculate_metrics

# 方法2：运行测试
cd /root/dp/StableSR_Edge_v3
python new_features/L2Loss/test_edge_l2_loss.py

# 方法3：在代码中使用
from basicsr.metrics.edge_l2_loss import calculate_edge_l2_loss
loss = calculate_edge_l2_loss(gen_img, gt_img)
```

## 核心实现位置

虽然文档在本目录，但核心代码在项目的标准位置：

- **核心实现**: `basicsr/metrics/edge_l2_loss.py`
- **推理集成**: `scripts/auto_inference.py` (已修改)
- **报告集成**: `scripts/generate_metrics_report.py` (已修改)

## 功能概述

**Edge L2 Loss** 是一个用于评估超分辨率图像边缘保真度的指标：

- 📥 **输入**: 生成图片 + Ground Truth 图片
- 🔧 **处理**: 使用 EdgeMapGenerator 生成边缘图
- 📊 **输出**: MSE 值 (范围 [0, 1]，越小越好)
- 💡 **意义**: 量化边缘相似度，值 < 0.001 表示边缘非常相似

## 特性

- ✅ 多种输入格式 (numpy/file/tensor)
- ✅ 自动尺寸匹配
- ✅ 完全向后兼容
- ✅ 集成到推理和报告系统
- ✅ 健壮的错误处理
- ✅ 完整的文档和测试

## 输出示例

### metrics.csv
```csv
Image Name,PSNR (dB),SSIM,LPIPS,Edge L2 Loss
0801.png,24.5379,0.7759,0.2655,0.001234
Average,21.0714,0.5853,0.3036,0.002456
```

## 与其他指标的关系

| 指标 | 度量内容 | 值域 | 越小越好 |
|------|---------|------|---------|
| PSNR | 像素级差异 | [0, ∞) | ✗ (越大越好) |
| SSIM | 结构相似度 | [0, 1] | ✗ (越大越好) |
| LPIPS | 感知相似度 | [0, 1] | ✓ |
| **Edge L2 Loss** | **边缘相似度** | **[0, 1]** | **✓** |

## 版本信息

- **版本**: v1.0
- **日期**: 2025-10-16
- **作者**: StableSR_Edge_v3 Team

## 反馈与贡献

如有问题或建议，欢迎联系项目维护者。

