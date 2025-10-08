# 📤 StableSR Notion集成完整指南

## 🎯 功能概述

本项目提供了完整的Notion集成功能，可以：
- ✅ 自动生成专业的验证报告
- ✅ 直接上传到Notion工作区
- ✅ 包含详细的性能指标和技术细节
- ✅ 支持环境变量和命令行参数
- ✅ 一键设置向导

## 📁 文件说明

### 核心脚本
| 文件 | 说明 |
|------|------|
| `upload_to_notion.py` | 主上传脚本，将验证报告上传到Notion |
| `setup_notion.sh` | 一键设置脚本，引导配置Notion集成 |
| `generate_notion_report.py` | 生成Markdown格式报告（离线使用） |
| `valid_edge_turbo.sh` | 运行模型验证 |

### 文档
| 文件 | 说明 |
|------|------|
| `NOTION_QUICK_START.md` | 快速开始指南 ⭐ **从这里开始** |
| `NOTION_SETUP.md` | 详细设置指南 |
| `README_NOTION.md` | 本文件 - 完整说明 |

### 生成的文件
| 文件 | 说明 |
|------|------|
| `.notion_env` | Notion凭证配置（由setup_notion.sh生成） |
| `notion_validation_report.md` | 离线Markdown报告 |

## 🚀 快速开始

### 三步上传到Notion

```bash
# 步骤1: 一键设置
bash setup_notion.sh

# 步骤2: 运行验证（如果还没有结果）
bash valid_edge_turbo.sh \
  logs/your_model/checkpoints/epoch=000215.ckpt \
  128x128_valid_LR \
  validation_results

# 步骤3: 上传到Notion
source .notion_env
python upload_to_notion.py \
  --result-dir validation_results/your_results \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/your_model/checkpoint.ckpt
```

## 📊 Notion报告内容

上传的报告包含：

### 1️⃣ 概览部分
- 模型名称和标识
- 处理图片总数
- 平均和总输出大小

### 2️⃣ 配置信息
- 模型路径
- 验证图片路径
- 输出目录

### 3️⃣ 验证参数表格
- DDPM步数
- 解码器权重
- 颜色修正类型
- 采样配置

### 4️⃣ 性能指标
- 处理图片数量
- 平均处理时间
- 总处理时间
- 文件大小统计

### 5️⃣ 技术细节
- 模型架构说明
- 边缘处理配置
- 训练信息

### 6️⃣ 总结
- 验证结果概述
- 模型性能评估

## 🛠️ 使用场景

### 场景1：定期验证报告
```bash
# 每次训练完成后
bash valid_edge_turbo.sh model.ckpt val_images results
source .notion_env
python upload_to_notion.py \
  --result-dir results/latest \
  --val-img-dir val_images \
  --model-path model.ckpt
```

### 场景2：多模型对比
```bash
# 上传多个模型的验证结果到同一个Notion页面
for epoch in 100 150 200 215; do
  python upload_to_notion.py \
    --result-dir results/epoch_${epoch} \
    --val-img-dir val_images \
    --model-path checkpoints/epoch=${epoch}.ckpt
done
```

### 场景3：团队协作
```bash
# 设置共享的Notion页面ID
export NOTION_PAGE_ID="team_shared_page_id"

# 团队成员都可以上传到同一个页面
python upload_to_notion.py \
  --result-dir my_results \
  --val-img-dir val_images \
  --model-path my_model.ckpt
```

## 🔐 安全说明

### ⚠️ 重要：保护你的Token

`.notion_env` 文件包含敏感的API Token，请：
- ❌ 不要提交到Git仓库
- ❌ 不要分享给他人
- ✅ 添加到 `.gitignore`
- ✅ 使用环境变量

```bash
# 推荐：添加到 .gitignore
echo ".notion_env" >> .gitignore
```

## 📋 完整工作流程示例

```bash
# ==================================================
# 完整的验证和上传工作流程
# ==================================================

# 1. 首次设置Notion集成（只需一次）
bash setup_notion.sh

# 2. 训练模型（假设已完成）
# ...

# 3. 运行验证
bash valid_edge_turbo.sh \
  logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt \
  128x128_valid_LR \
  validation_results

# 4. 加载Notion配置
source .notion_env

# 5. 上传到Notion
python upload_to_notion.py \
  --result-dir validation_results/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506_epoch=000215 \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt

# 6. 在Notion中查看报告
# 打开命令行输出的URL链接

# 7. （可选）生成离线Markdown报告
python generate_notion_report.py \
  --result-dir validation_results/... \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/.../checkpoint.ckpt \
  --output notion_report.md
```

## 🎨 自定义报告

你可以修改报告的内容和格式：

### 修改 `upload_to_notion.py`

```python
# 在 create_notion_blocks() 函数中
# 添加自定义blocks

# 例如：添加一个新的章节
blocks.append({
    "object": "block",
    "type": "heading_2",
    "heading_2": {
        "rich_text": [{"type": "text", "text": {"content": "🎯 Custom Section"}}]
    }
})

blocks.append({
    "object": "block",
    "type": "paragraph",
    "paragraph": {
        "rich_text": [{"type": "text", "text": {"content": "Your custom content here"}}]
    }
})
```

## 🔍 调试和测试

### 测试Notion连接
```bash
source .notion_env
python -c "
from notion_client import Client
import os
notion = Client(auth=os.environ['NOTION_TOKEN'])
page = notion.pages.retrieve(os.environ['NOTION_PAGE_ID'])
print('✅ Connection successful!')
print(f'Page: {page}')
"
```

### 查看详细错误
```bash
# 启用Python调试模式
python -u upload_to_notion.py \
  --result-dir ... \
  --val-img-dir ... \
  --model-path ... \
  2>&1 | tee upload_log.txt
```

## 📚 参考资源

### Notion相关
- [Notion API文档](https://developers.notion.com/)
- [Notion Python SDK](https://github.com/ramnes/notion-sdk-py)
- [Block对象参考](https://developers.notion.com/reference/block)

### 项目相关
- `NOTION_QUICK_START.md` - 快速开始
- `NOTION_SETUP.md` - 详细设置指南
- `valid_edge_turbo.sh` - 验证脚本
- `generate_notion_report.py` - Markdown报告生成器

## ❓ 常见问题

### Q: Token在哪里找？
A: 访问 https://www.notion.so/my-integrations 创建集成并复制token

### Q: 页面ID是什么？
A: 打开Notion页面，从URL中复制32位的ID

### Q: 为什么上传失败？
A: 检查：1) Token正确 2) 页面已与集成共享 3) 页面ID正确

### Q: 可以批量上传吗？
A: 可以！写一个循环脚本调用upload_to_notion.py

### Q: 报告能自定义吗？
A: 可以！修改create_notion_blocks()函数

### Q: 支持图片上传吗？
A: 当前版本需要手动上传图片到Notion，未来版本会支持自动上传

## 🎉 总结

现在你有了完整的Notion集成工具：
- ✅ 自动化验证报告生成
- ✅ 一键上传到Notion
- ✅ 专业的报告格式
- ✅ 灵活的配置选项
- ✅ 完整的文档支持

开始使用：**`bash setup_notion.sh`** 🚀

---

*Last updated: 2025-10-07*

