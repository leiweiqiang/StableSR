# 📤 Notion上传设置指南

## 🎯 概述

这个指南将帮助你设置Notion集成，并直接将验证报告上传到你的Notion工作区。

## 📋 步骤1: 创建Notion集成

### 1.1 访问Notion集成页面
打开浏览器，访问：https://www.notion.so/my-integrations

### 1.2 创建新集成
1. 点击 **"+ New integration"** 按钮
2. 填写集成信息：
   - **Name**: `StableSR Validator`（或任何你喜欢的名字）
   - **Associated workspace**: 选择你的工作区
   - **Type**: Internal integration
3. 点击 **Submit**

### 1.3 获取集成Token
1. 创建成功后，你会看到 **"Internal Integration Token"**
2. 点击 **"Show"** 并复制这个token
3. **重要**：这个token类似 `secret_xxxxxxxxxxxxxxxxxxxxxxxxx`
4. 保存好这个token，稍后会用到

## 📋 步骤2: 准备Notion页面

### 2.1 创建或选择目标页面
1. 在Notion中创建一个新页面，或选择现有页面
2. 这个页面将作为报告的**父页面**
3. 建议创建一个专门的页面，比如 "Validation Reports"

### 2.2 分享页面给集成
1. 点击页面右上角的 **"Share"** 按钮
2. 在弹出的对话框中，点击 **"Invite"**
3. 找到并选择你刚创建的集成（例如 "StableSR Validator"）
4. 确认分享

### 2.3 获取页面ID
有两种方法获取页面ID：

**方法1：从URL获取**
1. 打开你的Notion页面
2. 查看浏览器地址栏的URL
3. URL格式：`https://www.notion.so/Your-Page-Name-<PAGE_ID>`
4. 复制32位的PAGE_ID（可能包含破折号）

例如：
```
https://www.notion.so/Validation-Reports-1234567890abcdef1234567890abcdef
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        这就是你的 PAGE_ID
```

**方法2：复制链接**
1. 点击页面右上角的 **"..."** 菜单
2. 选择 **"Copy link"**
3. 从链接中提取PAGE_ID

## 📋 步骤3: 设置环境变量（推荐）

为了方便使用，你可以设置环境变量：

```bash
# 设置Notion Token
export NOTION_TOKEN="secret_your_token_here"

# 设置父页面ID
export NOTION_PAGE_ID="your_page_id_here"

# 永久保存（添加到 ~/.bashrc）
echo 'export NOTION_TOKEN="secret_your_token_here"' >> ~/.bashrc
echo 'export NOTION_PAGE_ID="your_page_id_here"' >> ~/.bashrc
source ~/.bashrc
```

## 📋 步骤4: 安装依赖

安装Notion Python客户端：

```bash
pip install notion-client
```

## 🚀 步骤5: 上传报告

### 方法1：使用环境变量

如果你已经设置了环境变量：

```bash
python upload_to_notion.py \
  --result-dir /root/dp/StableSR_Edge_v2/validation_results/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506_epoch=000215 \
  --val-img-dir /root/dp/StableSR_Edge_v2/128x128_valid_LR \
  --model-path /root/dp/StableSR_Edge_v2/logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt
```

### 方法2：使用命令行参数

```bash
python upload_to_notion.py \
  --result-dir /root/dp/StableSR_Edge_v2/validation_results/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506_epoch=000215 \
  --val-img-dir /root/dp/StableSR_Edge_v2/128x128_valid_LR \
  --model-path /root/dp/StableSR_Edge_v2/logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt \
  --notion-token "secret_your_token_here" \
  --parent-page-id "your_page_id_here"
```

### 方法3：创建便捷脚本

创建一个包装脚本 `upload_latest_validation.sh`:

```bash
#!/bin/bash
# 上传最新的验证结果到Notion

# 设置这些值
NOTION_TOKEN="secret_your_token_here"
NOTION_PAGE_ID="your_page_id_here"

# 最新的验证结果
RESULT_DIR="/root/dp/StableSR_Edge_v2/validation_results/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506_epoch=000215"
VAL_IMG_DIR="/root/dp/StableSR_Edge_v2/128x128_valid_LR"
MODEL_PATH="/root/dp/StableSR_Edge_v2/logs/2025-10-07T13-25-09_stablesr_edge_turbo_20251007_132506/checkpoints/epoch=000215.ckpt"

python upload_to_notion.py \
  --result-dir "$RESULT_DIR" \
  --val-img-dir "$VAL_IMG_DIR" \
  --model-path "$MODEL_PATH" \
  --notion-token "$NOTION_TOKEN" \
  --parent-page-id "$NOTION_PAGE_ID"
```

然后：
```bash
chmod +x upload_latest_validation.sh
./upload_latest_validation.sh
```

## ✅ 验证成功

上传成功后，你会看到：

```
============================================================
✅ Successfully uploaded to Notion!
============================================================

📄 Page URL: https://www.notion.so/Validation-Report-2025-10-07-xxx
🆔 Page ID: xxxxxxxxxxxxx

💡 Tip: You can now add images to the page manually
```

## 🖼️ 添加图片到Notion页面

报告上传后，你可以手动添加图片：

1. 打开Notion中新创建的报告页面
2. 在"Sample Comparisons"部分
3. 拖拽或粘贴图片到相应位置
4. 或使用 `/image` 命令插入图片

## 🔧 故障排除

### 错误: Invalid token
- 检查你的token是否正确复制
- 确保token以 `secret_` 开头
- 重新生成集成token

### 错误: Page not found
- 确认页面已经与集成共享
- 检查页面ID是否正确
- 尝试重新分享页面

### 错误: Forbidden
- 集成可能没有权限
- 重新分享页面给集成
- 检查工作区设置

## 📝 完整工作流程示例

```bash
# 1. 运行验证
bash valid_edge_turbo.sh \
  logs/model/checkpoints/epoch=000215.ckpt \
  128x128_valid_LR \
  validation_results

# 2. 上传到Notion
python upload_to_notion.py \
  --result-dir validation_results/... \
  --val-img-dir 128x128_valid_LR \
  --model-path logs/model/checkpoints/epoch=000215.ckpt

# 3. 在Notion中查看和编辑报告
```

## 🎨 自定义报告

你可以修改 `upload_to_notion.py` 中的 `create_notion_blocks()` 函数来自定义报告格式、添加更多内容或修改样式。

## 📚 更多资源

- [Notion API文档](https://developers.notion.com/)
- [Notion Python SDK](https://github.com/ramnes/notion-sdk-py)
- [Notion集成指南](https://developers.notion.com/docs/getting-started)

---

**需要帮助？** 查看上面的故障排除部分或参考Notion官方文档。
