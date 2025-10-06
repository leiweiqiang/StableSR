#!/usr/bin/env python3
"""
TraReport命令行运行脚本
简化的命令行接口，用于快速运行TraReport评估
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tra_report import TraReport

def main():
    parser = argparse.ArgumentParser(
        description="TraReport - Super Resolution Model Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_tra_report.py --gt_dir /path/to/gt --val_dir /path/to/val --model_path /path/to/model.ckpt
  
  python run_tra_report.py --gt_dir /data/DIV2K_valid_HR --val_dir /data/DIV2K_valid_LR --model_path ./weights/stablesr_000117.ckpt --output results.json
        """
    )
    
    # 必需参数
    parser.add_argument("--gt_dir", type=str, required=True,
                       help="Ground truth high-resolution images directory")
    parser.add_argument("--val_dir", type=str, required=True,
                       help="Validation low-resolution images directory")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Model checkpoint file path")
    
    # 可选参数
    parser.add_argument("--config_path", type=str, default=None,
                       help="Model configuration file path (default: use edge config)")
    parser.add_argument("--output", type=str, default="tra_report_results.json",
                       help="Output JSON file path (default: tra_report_results.json)")
    parser.add_argument("--ddpm_steps", type=int, default=200,
                       help="DDPM sampling steps (default: 200)")
    parser.add_argument("--upscale", type=float, default=4.0,
                       help="Upscale factor (default: 4.0)")
    parser.add_argument("--colorfix_type", type=str, default="adain",
                       choices=["adain", "wavelet", "none"],
                       help="Color fix type (default: adain)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.gt_dir):
        print(f"错误: GT目录不存在: {args.gt_dir}")
        return 1
        
    if not os.path.exists(args.val_dir):
        print(f"错误: Val目录不存在: {args.val_dir}")
        return 1
        
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return 1
    
    # 设置默认配置文件
    if args.config_path is None:
        if "edge" in args.model_path.lower():
            args.config_path = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        else:
            args.config_path = "./configs/stableSRNew/v2-finetune_text_T_512.yaml"
    
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        return 1
    
    print("=== TraReport 超分辨率模型评估工具 ===")
    print(f"GT目录: {args.gt_dir}")
    print(f"Val目录: {args.val_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"配置文件: {args.config_path}")
    print(f"输出文件: {args.output}")
    print(f"DDPM步数: {args.ddpm_steps}")
    print(f"放大倍数: {args.upscale}")
    print(f"颜色修复: {args.colorfix_type}")
    print(f"随机种子: {args.seed}")
    print(f"计算设备: {args.device}")
    print()
    
    try:
        # 创建TraReport实例
        tra_report = TraReport(
            gt_dir=args.gt_dir,
            val_dir=args.val_dir,
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            ddpm_steps=args.ddpm_steps,
            upscale=args.upscale,
            colorfix_type=args.colorfix_type,
            seed=args.seed
        )
        
        # 运行评估
        print("开始模型评估...")
        results = tra_report.run_evaluation(output_path=args.output)
        
        print("\n=== 评估完成 ===")
        print(f"结果已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
