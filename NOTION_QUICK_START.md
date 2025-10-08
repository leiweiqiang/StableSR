# 🚀 Notion上传 - 快速开始

## 一键设置

只需运行一个命令：

```bash
bash setup_notion.sh
```

这个脚本会引导你：
1. ✅ 安装必要依赖
2. ✅ 输入Notion集成Token
3. ✅ 输入父页面ID
4. ✅ 测试连接
5. ✅ 保存配置
6. ✅ （可选）立即上传最新结果

## 📋 前置要求

### 在Notion中：

1. **创建集成**：https://www.notion.so/my-integrations
   - 点击 "+ New integration"
   - 命名为 "StableSR Validator"
   - 复制 Integration Token

2. **准备页面**：
   - 创建或选择一个页面
   - 点击右上角 "Share"
   - 邀请你的集成

3. **获取页面ID**：
   - 从页面URL中复制：`https://www.notion.so/Page-Name-XXXXXXXXXX`
   - 或点击 "..." → "Copy link"

## 🎯 使用方法

### 方法1：自动化设置（推荐）

```bash
# 一键设置
bash setup_notion.sh

# 设置完成后，以后只需source配置文件
source .notion_env

# 上传报告
python upload_to_notion.py \
  --result-dir validation_results/your_results \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/your_model/checkpoint.ckpt
```

### 方法2：手动设置

```bash
# 安装依赖
pip install notion-client

# 设置环境变量
export NOTION_TOKEN="secret_your_token_here"
export NOTION_PAGE_ID="your_page_id_here"

# 上传报告
python upload_to_notion.py \
  --result-dir validation_results/your_results \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/your_model/checkpoint.ckpt
```

### 方法3：命令行参数

```bash
python upload_to_notion.py \
  --result-dir validation_results/your_results \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/your_model/checkpoint.ckpt \
  --notion-token "secret_xxx..." \
  --parent-page-id "xxx..."
```

## 💡 实际例子

```bash
# 1. 运行验证
bash valid_edge_turbo.sh \
  logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt \
  128x128_valid_LR \
  validation_results

# 2. 设置Notion（首次）
bash setup_notion.sh

# 3. 上传到Notion
source .notion_env
python upload_to_notion.py \
  --result-dir validation_results/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506_epoch=000215 \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt
```

## ✅ 成功标志

上传成功后，你会看到：

```
============================================================
✅ Successfully uploaded to Notion!
============================================================

📄 Page URL: https://www.notion.so/Validation-Report-2025-10-07-xxx
🆔 Page ID: xxxxxxxxxxxxx
```

然后在Notion中打开链接，你会看到：
- 📊 完整的验证报告
- 📈 性能指标表格
- 🔧 配置详情
- ✅ 处理结果汇总

## 🔧 故障排除

| 问题 | 解决方案 |
|------|----------|
| ❌ Invalid token | 检查token格式，应以 `secret_` 开头 |
| ❌ Page not found | 确保页面已与集成共享 |
| ❌ Forbidden | 重新分享页面给集成 |
| ❌ Module not found | 运行 `pip install notion-client` |

## 📚 更多信息

- 详细设置：查看 `NOTION_SETUP.md`
- API文档：https://developers.notion.com/
- 问题反馈：检查错误消息并参考上面的故障排除

---

**提示**：将 `source .notion_env` 添加到你的 `~/.bashrc` 以便每次自动加载配置！

