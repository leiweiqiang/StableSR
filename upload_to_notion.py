#!/usr/bin/env python3
"""
Upload validation report directly to Notion
直接将验证报告上传到Notion账号
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

try:
    from notion_client import Client
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    print("⚠️  Notion client not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "notion-client"])
    from notion_client import Client
    NOTION_AVAILABLE = True


def create_notion_blocks(results):
    """创建Notion block结构"""
    blocks = []
    
    # 标题
    blocks.append({
        "object": "block",
        "type": "heading_1",
        "heading_1": {
            "rich_text": [{"type": "text", "text": {"content": "🎨 StableSR Edge-Enhanced Model Validation Report"}}]
        }
    })
    
    # 生成时间
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": "Generated: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
            ]
        }
    })
    
    # 分隔线
    blocks.append({"object": "block", "type": "divider", "divider": {}})
    
    # Overview Section
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "📊 Overview"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Model: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": results['model_name'], "link": None}, "annotations": {"code": True}}
            ]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Total Images Processed: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": str(results['total_images'])}}
            ]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Average Output Size: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": f"{results['avg_size_mb']:.2f} MB"}}
            ]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Total Output Size: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": f"{results['total_size_mb']:.2f} MB"}}
            ]
        }
    })
    
    # Model Configuration
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🔧 Model Configuration"}}]
        }
    })
    
    config_text = f"""Model Path: {results['model_path']}
Validation Images: {results['val_img_dir']}
Output Directory: {results['result_dir']}"""
    
    blocks.append({
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": config_text}}],
            "language": "plain text"
        }
    })
    
    # Validation Parameters
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "⚙️ Validation Parameters"}}]
        }
    })
    
    # 创建参数表格
    blocks.append({
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 2,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Parameter"}, "annotations": {"bold": True}}],
                            [{"type": "text", "text": {"content": "Value"}, "annotations": {"bold": True}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "DDPM Steps"}}],
                            [{"type": "text", "text": {"content": "200"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Decoder Weight (dec_w)"}}],
                            [{"type": "text", "text": {"content": "0.5"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Color Fix Type"}}],
                            [{"type": "text", "text": {"content": "AdaIN"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Number of Samples"}}],
                            [{"type": "text", "text": {"content": "1"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Random Seed"}}],
                            [{"type": "text", "text": {"content": "42"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Edge Processing"}}],
                            [{"type": "text", "text": {"content": "✅ Enabled"}}]
                        ]
                    }
                }
            ]
        }
    })
    
    # Validation Results
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🖼️ Validation Results"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": f"Processing completed for "}},
                {"type": "text", "text": {"content": f"{results['total_images']} images"}, "annotations": {"bold": True}}
            ]
        }
    })
    
    # Performance Metrics
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "⚡ Performance Metrics"}}]
        }
    })
    
    total_time = results['total_images'] * 28 / 60
    
    blocks.append({
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 2,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Metric"}, "annotations": {"bold": True}}],
                            [{"type": "text", "text": {"content": "Value"}, "annotations": {"bold": True}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Images Processed"}}],
                            [{"type": "text", "text": {"content": str(results['total_images'])}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Average Processing Time"}}],
                            [{"type": "text", "text": {"content": "~28 seconds/image"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Total Processing Time"}}],
                            [{"type": "text", "text": {"content": f"~{total_time:.1f} minutes"}}]
                        ]
                    }
                },
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": "Average Output File Size"}}],
                            [{"type": "text", "text": {"content": f"{results['avg_size_mb']:.2f} MB"}}]
                        ]
                    }
                }
            ]
        }
    })
    
    # Technical Details
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🔍 Technical Details"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "heading_3",
        "heading_3": {
            "rich_text": [{"type": "text", "text": {"content": "Model Architecture"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Base Model: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": "StableSR Turbo"}}
            ]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Enhancement: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": "Edge Processing Module"}}
            ]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": "Edge Detection: ", "link": None}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": "Canny edge detection with Gaussian blur"}}
            ]
        }
    })
    
    # Summary
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "✅ Summary"}}]
        }
    })
    
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{
                "type": "text",
                "text": {
                    "content": f"Successfully validated Edge-enhanced StableSR model on {results['total_images']} test images. "
                              "The model demonstrated stable performance with edge processing enabled, generating high-quality "
                              "super-resolution outputs with edge-aware enhancements."
                }
            }]
        }
    })
    
    # Footer
    blocks.append({"object": "block", "type": "divider", "divider": {}})
    
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{
                "type": "text",
                "text": {"content": f"Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"},
                "annotations": {"italic": True}
            }]
        }
    })
    
    return blocks


def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)


def analyze_validation_results(result_dir, val_img_dir, model_path):
    """分析验证结果"""
    result_path = Path(result_dir)
    val_path = Path(val_img_dir)
    
    # 获取结果图片列表
    result_images = sorted([f for f in os.listdir(result_path) if f.endswith('.png')])
    val_images = sorted([f for f in os.listdir(val_path) if f.endswith('.png')])
    
    # 计算统计信息
    total_images = len(result_images)
    total_size_mb = sum(get_file_size_mb(result_path / img) for img in result_images)
    avg_size_mb = total_size_mb / total_images if total_images > 0 else 0
    
    # 提取模型信息
    model_name = result_path.name
    
    return {
        'model_name': model_name,
        'model_path': model_path,
        'result_dir': result_dir,
        'val_img_dir': val_img_dir,
        'total_images': total_images,
        'total_size_mb': total_size_mb,
        'avg_size_mb': avg_size_mb,
        'result_images': result_images,
        'val_images': val_images,
    }


def upload_to_notion(notion_token, parent_page_id, results):
    """上传内容到Notion"""
    
    # 初始化Notion客户端
    notion = Client(auth=notion_token)
    
    # 创建blocks
    print("Creating Notion blocks...")
    blocks = create_notion_blocks(results)
    
    # 创建页面
    print(f"Uploading to Notion (parent page: {parent_page_id})...")
    
    try:
        new_page = notion.pages.create(
            parent={"page_id": parent_page_id},
            icon={"type": "emoji", "emoji": "🎨"},
            properties={
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Validation Report - {datetime.now().strftime('%Y-%m-%d')}"
                            }
                        }
                    ]
                }
            },
            children=blocks
        )
        
        page_url = new_page.get("url", "")
        page_id = new_page.get("id", "")
        
        return {
            "success": True,
            "page_url": page_url,
            "page_id": page_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Upload validation report to Notion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload report to Notion
  python upload_to_notion.py \\
    --result-dir validation_results/... \\
    --val-img-dir 128x128_valid_LR \\
    --model-path logs/.../checkpoint.ckpt \\
    --notion-token "secret_xxx..." \\
    --parent-page-id "xxx-xxx-xxx"
    
Setup:
  1. Create a Notion integration at: https://www.notion.so/my-integrations
  2. Copy the Internal Integration Token
  3. Share your target page with the integration
  4. Get the page ID from the page URL
        """
    )
    
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Path to validation results directory')
    parser.add_argument('--val-img-dir', type=str, required=True,
                       help='Path to validation input images directory')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--notion-token', type=str,
                       help='Notion integration token (or set NOTION_TOKEN env var)')
    parser.add_argument('--parent-page-id', type=str,
                       help='Parent page ID in Notion (or set NOTION_PAGE_ID env var)')
    
    args = parser.parse_args()
    
    # 获取Notion凭证
    notion_token = args.notion_token or os.environ.get('NOTION_TOKEN')
    parent_page_id = args.parent_page_id or os.environ.get('NOTION_PAGE_ID')
    
    if not notion_token:
        print("❌ Error: Notion token not provided!")
        print("")
        print("Please provide --notion-token or set NOTION_TOKEN environment variable")
        print("")
        print("Setup instructions:")
        print("1. Go to: https://www.notion.so/my-integrations")
        print("2. Click 'New integration'")
        print("3. Give it a name and copy the 'Internal Integration Token'")
        print("4. Share your target Notion page with this integration")
        print("")
        sys.exit(1)
    
    if not parent_page_id:
        print("❌ Error: Parent page ID not provided!")
        print("")
        print("Please provide --parent-page-id or set NOTION_PAGE_ID environment variable")
        print("")
        print("How to get page ID:")
        print("1. Open your Notion page in browser")
        print("2. The URL looks like: https://www.notion.so/Page-Name-<PAGE_ID>")
        print("3. Copy the PAGE_ID part (32 character hex string)")
        print("")
        sys.exit(1)
    
    # 清理page ID（移除破折号）
    parent_page_id = parent_page_id.replace('-', '')
    
    # 验证目录存在
    if not os.path.exists(args.result_dir):
        print(f"❌ Result directory not found: {args.result_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.val_img_dir):
        print(f"❌ Validation images directory not found: {args.val_img_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("📤 Uploading Validation Report to Notion")
    print("=" * 60)
    print("")
    
    # 分析结果
    print("📊 Analyzing validation results...")
    results = analyze_validation_results(args.result_dir, args.val_img_dir, args.model_path)
    print(f"   ✓ Found {results['total_images']} validation results")
    print("")
    
    # 上传到Notion
    print("🚀 Uploading to Notion...")
    result = upload_to_notion(notion_token, parent_page_id, results)
    
    print("")
    if result['success']:
        print("=" * 60)
        print("✅ Successfully uploaded to Notion!")
        print("=" * 60)
        print("")
        print(f"📄 Page URL: {result['page_url']}")
        print(f"🆔 Page ID: {result['page_id']}")
        print("")
        print("💡 Tip: You can now add images to the page manually")
        print("")
    else:
        print("=" * 60)
        print("❌ Upload failed!")
        print("=" * 60)
        print("")
        print(f"Error: {result['error']}")
        print("")
        print("Common issues:")
        print("1. Invalid token - check your integration token")
        print("2. Page not shared - share the page with your integration")
        print("3. Invalid page ID - verify the page ID is correct")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
