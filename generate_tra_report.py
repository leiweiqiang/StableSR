#!/usr/bin/env python3
"""
使用TraReport生成评估报告
仅使用StableSR Edge模型，避免配置不匹配问题
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import random
from datetime import datetime

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def create_test_subset(gt_dir, val_dir, num_samples=10):
    """创建测试子集"""
    print(f"创建包含 {num_samples} 个样本的测试子集...")
    
    # 获取所有文件
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # 确保文件匹配
    matching_files = []
    for gt_file in gt_files:
        if gt_file in val_files:
            matching_files.append(gt_file)
    
    print(f"找到 {len(matching_files)} 对匹配的文件")
    
    # 随机选择样本
    selected_files = random.sample(matching_files, min(num_samples, len(matching_files)))
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    temp_gt_dir = os.path.join(temp_dir, 'gt')
    temp_val_dir = os.path.join(temp_dir, 'val')
    os.makedirs(temp_gt_dir, exist_ok=True)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    # 复制选中的文件
    for file in selected_files:
        shutil.copy2(os.path.join(gt_dir, file), os.path.join(temp_gt_dir, file))
        shutil.copy2(os.path.join(val_dir, file), os.path.join(temp_val_dir, file))
    
    print(f"测试子集创建完成: {temp_dir}")
    print(f"GT文件: {len(selected_files)} 个")
    print(f"Val文件: {len(selected_files)} 个")
    
    return temp_dir, temp_gt_dir, temp_val_dir

def generate_tra_report():
    """生成TraReport评估报告"""
    print("=" * 80)
    print("TraReport 评估报告生成器")
    print("=" * 80)
    
    try:
        from tra_report import TraReport
        print("✅ TraReport导入成功")
        
        # 创建测试数据
        temp_dir, gt_dir, val_dir = create_test_subset(
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR',
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR',
            num_samples=10
        )
        
        # 初始化TraReport
        model_path = './logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt'
        config_path = './configs/stableSRNew/v2-finetune_text_T_512_edge.yaml'
        
        print(f"使用模型: {model_path}")
        print(f"使用配置: {config_path}")
        
        tra_report = TraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            model_path=model_path,
            config_path=config_path,
            device='cuda',
            ddpm_steps=50,  # 适中的步数
            upscale=4.0,
            colorfix_type='adain',
            seed=42
        )
        
        print("✅ TraReport初始化成功")
        
        # 运行评估
        print("\n开始评估...")
        results = tra_report.evaluate()
        
        # 生成报告
        report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "config_path": config_path,
                "dataset": "DIV2K Validation Set",
                "num_samples": len(results.get("results", [])),
                "ddpm_steps": 50,
                "upscale_factor": 4.0,
                "colorfix_type": "adain",
                "device": "cuda"
            },
            "summary": {
                "total_images": len(results.get("results", [])),
                "average_psnr": results.get("average_psnr", 0),
                "best_psnr": results.get("best_psnr", 0),
                "worst_psnr": results.get("worst_psnr", 0)
            },
            "detailed_results": results.get("results", []),
            "model_info": {
                "model_type": "StableSR Edge",
                "parameters": "920.95M",
                "architecture": "LatentDiffusionSRTextWTWithEdge",
                "edge_processing": True
            }
        }
        
        # 保存报告
        output_file = "tra_report_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 评估完成！")
        print(f"📊 评估结果:")
        print(f"  - 处理图片数量: {report['summary']['total_images']}")
        print(f"  - 平均PSNR: {report['summary']['average_psnr']:.2f} dB")
        print(f"  - 最佳PSNR: {report['summary']['best_psnr']:.2f} dB")
        print(f"  - 最差PSNR: {report['summary']['worst_psnr']:.2f} dB")
        print(f"  - 报告文件: {output_file}")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        return report
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary_report(report):
    """生成总结报告"""
    if not report:
        return
    
    print("\n" + "=" * 80)
    print("TraReport 评估总结报告")
    print("=" * 80)
    
    print(f"📅 评估时间: {report['evaluation_info']['timestamp']}")
    print(f"🤖 模型类型: {report['model_info']['model_type']}")
    print(f"📊 数据集: {report['evaluation_info']['dataset']}")
    print(f"🖼️  处理图片: {report['summary']['total_images']} 张")
    print(f"⚙️  超分辨率倍数: {report['evaluation_info']['upscale_factor']}x")
    print(f"🎯 DDPM步数: {report['evaluation_info']['ddpm_steps']}")
    print(f"🎨 颜色修复: {report['evaluation_info']['colorfix_type']}")
    
    print(f"\n📈 PSNR统计:")
    print(f"  - 平均PSNR: {report['summary']['average_psnr']:.2f} dB")
    print(f"  - 最佳PSNR: {report['summary']['best_psnr']:.2f} dB")
    print(f"  - 最差PSNR: {report['summary']['worst_psnr']:.2f} dB")
    
    if report['summary']['average_psnr'] > 0:
        print(f"\n🎯 性能评估:")
        if report['summary']['average_psnr'] >= 30:
            print("  ✅ 优秀 - PSNR >= 30dB")
        elif report['summary']['average_psnr'] >= 25:
            print("  ✅ 良好 - PSNR >= 25dB")
        elif report['summary']['average_psnr'] >= 20:
            print("  ⚠️  一般 - PSNR >= 20dB")
        else:
            print("  ❌ 需要改进 - PSNR < 20dB")
    
    print(f"\n📁 详细结果已保存到: tra_report_evaluation.json")

def main():
    """主函数"""
    print("TraReport 评估报告生成器")
    print("使用StableSR Edge模型对DIV2K数据集进行评估")
    print("=" * 80)
    
    # 检查环境
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA不可用，将使用CPU（速度较慢）")
    
    # 检查数据集
    gt_dir = '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR'
    val_dir = '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR'
    
    if not os.path.exists(gt_dir):
        print(f"❌ GT目录不存在: {gt_dir}")
        return False
    
    if not os.path.exists(val_dir):
        print(f"❌ Val目录不存在: {val_dir}")
        return False
    
    # 检查模型文件
    model_path = './logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"✅ 环境检查通过")
    print(f"✅ GT目录: {gt_dir}")
    print(f"✅ Val目录: {val_dir}")
    print(f"✅ 模型文件: {model_path}")
    
    # 生成报告
    report = generate_tra_report()
    
    if report:
        generate_summary_report(report)
        print(f"\n🎉 TraReport评估报告生成完成！")
        return True
    else:
        print(f"\n❌ TraReport评估报告生成失败！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
