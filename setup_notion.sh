#!/bin/bash
# Quick setup script for Notion integration
# Notion集成快速设置脚本

echo "============================================================"
echo "🎨 StableSR Notion Integration Setup"
echo "============================================================"
echo ""

# 检查是否安装了notion-client
echo "📦 Checking dependencies..."
if ! python -c "import notion_client" 2>/dev/null; then
    echo "⚠️  notion-client not installed. Installing now..."
    pip install notion-client
    echo "✓ notion-client installed"
else
    echo "✓ notion-client already installed"
fi
echo ""

# 获取Notion Token
echo "============================================================"
echo "Step 1: Notion Integration Token"
echo "============================================================"
echo ""
echo "Please follow these steps:"
echo "1. Visit: https://www.notion.so/my-integrations"
echo "2. Click '+ New integration'"
echo "3. Name it (e.g., 'StableSR Validator')"
echo "4. Copy the 'Internal Integration Token'"
echo ""
read -p "Enter your Notion Integration Token (secret_xxx...): " NOTION_TOKEN
echo ""

if [[ ! $NOTION_TOKEN =~ ^secret_ ]]; then
    echo "⚠️  Warning: Token should start with 'secret_'"
    read -p "Are you sure this is correct? (y/n): " confirm
    if [[ $confirm != "y" ]]; then
        echo "❌ Setup cancelled"
        exit 1
    fi
fi

# 获取Page ID
echo "============================================================"
echo "Step 2: Parent Page ID"
echo "============================================================"
echo ""
echo "Please follow these steps:"
echo "1. Create or open a page in Notion"
echo "2. Click 'Share' in the top-right"
echo "3. Invite your integration (e.g., 'StableSR Validator')"
echo "4. Copy the page URL or ID"
echo ""
echo "Page URL format:"
echo "https://www.notion.so/Page-Name-1234567890abcdef1234567890abcdef"
echo "                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "                                  This is your Page ID"
echo ""
read -p "Enter your Parent Page ID: " NOTION_PAGE_ID

# 清理Page ID（移除破折号和URL前缀）
NOTION_PAGE_ID=$(echo "$NOTION_PAGE_ID" | sed 's/.*notion\.so\///' | sed 's/.*-//' | tr -d '-')

echo ""
echo "Cleaned Page ID: $NOTION_PAGE_ID"
echo ""

# 保存到环境变量文件
ENV_FILE=".notion_env"
cat > "$ENV_FILE" <<EOF
# Notion Integration Configuration
# Generated on $(date)

export NOTION_TOKEN="$NOTION_TOKEN"
export NOTION_PAGE_ID="$NOTION_PAGE_ID"
EOF

echo "============================================================"
echo "✅ Configuration saved to: $ENV_FILE"
echo "============================================================"
echo ""

# 测试连接
echo "🔍 Testing connection to Notion..."
echo ""

# 创建测试脚本
cat > test_notion_connection.py <<'PYTHON_EOF'
import os
import sys

try:
    from notion_client import Client
except ImportError:
    print("❌ notion-client not installed")
    sys.exit(1)

notion_token = os.environ.get('NOTION_TOKEN')
parent_page_id = os.environ.get('NOTION_PAGE_ID')

if not notion_token or not parent_page_id:
    print("❌ Environment variables not set")
    sys.exit(1)

try:
    notion = Client(auth=notion_token)
    page = notion.pages.retrieve(parent_page_id)
    print("✅ Successfully connected to Notion!")
    print(f"   Page title: {page.get('properties', {}).get('title', {})}")
    print(f"   Page ID: {parent_page_id}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("")
    print("Common issues:")
    print("1. Invalid token - check your integration token")
    print("2. Page not shared - share the page with your integration")
    print("3. Invalid page ID - verify the page ID is correct")
    sys.exit(1)
PYTHON_EOF

# 运行测试
source "$ENV_FILE"
python test_notion_connection.py
TEST_RESULT=$?

# 清理测试脚本
rm -f test_notion_connection.py

echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo "============================================================"
    echo "🎉 Setup Complete!"
    echo "============================================================"
    echo ""
    echo "To use Notion upload in future sessions, run:"
    echo ""
    echo "    source .notion_env"
    echo ""
    echo "Or add to your ~/.bashrc:"
    echo ""
    echo "    echo 'source $(pwd)/.notion_env' >> ~/.bashrc"
    echo ""
    echo "============================================================"
    echo "📤 Upload Example"
    echo "============================================================"
    echo ""
    echo "python upload_to_notion.py \\"
    echo "  --result-dir validation_results/your_results \\"
    echo "  --val-img-dir 128x128_valid_LR \\"
    echo "  --model-path logs/your_model/checkpoint.ckpt"
    echo ""
    
    # 询问是否立即上传
    read -p "Would you like to upload the latest validation report now? (y/n): " upload_now
    if [[ $upload_now == "y" ]]; then
        echo ""
        echo "Finding latest validation results..."
        
        # 查找最新的验证结果
        LATEST_RESULT=$(ls -td validation_results/*/ 2>/dev/null | head -1)
        
        if [[ -n $LATEST_RESULT ]]; then
            echo "Found: $LATEST_RESULT"
            echo ""
            
            # 尝试推断相关路径
            MODEL_PATH=$(find logs -name "*.ckpt" -type f 2>/dev/null | head -1)
            VAL_IMG_DIR="128x128_valid_LR"
            
            if [[ -n $MODEL_PATH ]] && [[ -d $VAL_IMG_DIR ]]; then
                echo "Uploading to Notion..."
                python upload_to_notion.py \
                    --result-dir "$LATEST_RESULT" \
                    --val-img-dir "$VAL_IMG_DIR" \
                    --model-path "$MODEL_PATH"
            else
                echo "⚠️  Could not find all required paths"
                echo "Please run upload_to_notion.py manually with correct paths"
            fi
        else
            echo "⚠️  No validation results found in validation_results/"
            echo "Please run validation first using valid_edge_turbo.sh"
        fi
    fi
else
    echo "============================================================"
    echo "❌ Setup failed - please check the errors above"
    echo "============================================================"
    echo ""
    echo "You can try again by running:"
    echo "    bash setup_notion.sh"
    echo ""
fi
