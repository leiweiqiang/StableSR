#!/usr/bin/env python3
"""
TraReport评估示例脚本
使用真实数据运行超分辨率模型评估
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tra_report import TraReport

def main():
    """运行TraReport评估示例"""
    
    # 设置路径变量
    gt_dir = "/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR"
    val_dir = "/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR"  # 你需要准备低分辨率版本
    model_path = "./weights/stablesr_000117.ckpt"  # 或者你的模型路径
    config_path = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    
    # 检查路径是否存在
    print("检查路径...")
    if not os.path.exists(gt_dir):
        print(f"错误: GT目录不存在: {gt_dir}")
        print("请检查DIV2K数据集路径")
        return 1
        
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请检查模型路径")
        return 1
        
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return 1
    
    # 创建输出目录
    os.makedirs("./evaluation_results", exist_ok=True)
    
    # 运行评估
    print("开始运行TraReport评估...")
    print(f"GT目录: {gt_dir}")
    print(f"Val目录: {val_dir}")
    print(f"模型路径: {model_path}")
    print(f"配置文件: {config_path}")
    print()
    
    try:
        # 创建TraReport实例
        tra_report = TraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            model_path=model_path,
            config_path=config_path,
            ddpm_steps=200,
            upscale=4.0,
            colorfix_type="adain",
            seed=42
        )
        
        # 运行评估
        results = tra_report.run_evaluation(
            output_path="./evaluation_results/tra_report_results.json"
        )
        
        print("\n=== 评估结果摘要 ===")
        print(f"处理文件总数: {results['total_files']}")
        print(f"平均PSNR: {results['summary']['average_psnr']:.4f}")
        print(f"最小PSNR: {results['summary']['min_psnr']:.4f}")
        print(f"最大PSNR: {results['summary']['max_psnr']:.4f}")
        print(f"PSNR标准差: {results['summary']['std_psnr']:.4f}")
        
        print("\n评估完成！结果保存在 ./evaluation_results/tra_report_results.json")
        return 0
        
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def check_environment():
    """检查环境配置"""
    print("=== 环境检查 ===")
    
    # 检查CUDA
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查其他依赖
    dependencies = [
        "omegaconf",
        "basicsr", 
        "pytorch_lightning",
        "PIL",
        "tqdm",
        "cv2"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")
            return False
    
    return True

if __name__ == "__main__":
    print("TraReport评估示例")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，请安装所需依赖")
        sys.exit(1)
    
    print()
    
    # 运行评估
    sys.exit(main())
