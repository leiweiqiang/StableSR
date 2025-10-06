#!/usr/bin/env python3
"""
Enhanced TraReport命令行运行脚本
提供简单的命令行接口来运行StableSR Edge和StableSR Upscale模型的比较评估
"""

import argparse
import os
import sys
from pathlib import Path
from enhanced_tra_report import EnhancedTraReport


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Enhanced TraReport - StableSR模型比较评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用
  python run_enhanced_tra_report.py \\
    --gt_dir /path/to/gt/images \\
    --val_dir /path/to/val/images \\
    --edge_model /path/to/edge_model.ckpt \\
    --upscale_model /path/to/upscale_model.ckpt \\
    --output results.json

  # 自定义参数
  python run_enhanced_tra_report.py \\
    --gt_dir /path/to/gt \\
    --val_dir /path/to/val \\
    --edge_model /path/to/edge_model.ckpt \\
    --upscale_model /path/to/upscale_model.ckpt \\
    --ddpm_steps 100 \\
    --upscale 2.0 \\
    --colorfix wavelet \\
    --output custom_results.json
        """
    )
    
    # 必需参数
    parser.add_argument("--gt_dir", required=True, help="高分辨率图片目录")
    parser.add_argument("--val_dir", required=True, help="低分辨率图片目录")
    parser.add_argument("--edge_model", required=True, help="StableSR Edge模型权重文件路径")
    parser.add_argument("--upscale_model", required=True, help="StableSR Upscale模型权重文件路径")
    
    # 可选参数
    parser.add_argument("--edge_config", help="StableSR Edge配置文件路径")
    parser.add_argument("--upscale_config", help="StableSR Upscale配置文件路径")
    parser.add_argument("--output", default="enhanced_tra_report_results.json", help="输出结果文件路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--ddpm_steps", type=int, default=200, help="DDPM采样步数")
    parser.add_argument("--upscale", type=float, default=4.0, help="超分辨率倍数")
    parser.add_argument("--colorfix", default="adain", choices=["adain", "wavelet", "none"], help="颜色修复类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.gt_dir):
        print(f"错误: GT目录不存在: {args.gt_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.val_dir):
        print(f"错误: 验证目录不存在: {args.val_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.edge_model):
        print(f"错误: StableSR Edge模型文件不存在: {args.edge_model}")
        sys.exit(1)
        
    if not os.path.exists(args.upscale_model):
        print(f"错误: StableSR Upscale模型文件不存在: {args.upscale_model}")
        sys.exit(1)
    
    # 设置默认配置文件路径
    if args.edge_config is None:
        args.edge_config = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    if args.upscale_config is None:
        args.upscale_config = "./configs/stableSRNew/v2-finetune_text_T_512.yaml"
    
    # 验证配置文件
    if not os.path.exists(args.edge_config):
        print(f"警告: StableSR Edge配置文件不存在: {args.edge_config}")
        print("将使用默认配置")
        
    if not os.path.exists(args.upscale_config):
        print(f"警告: StableSR Upscale配置文件不存在: {args.upscale_config}")
        print("将使用默认配置")
    
    # 打印配置信息
    print("="*60)
    print("Enhanced TraReport - 模型比较评估")
    print("="*60)
    print(f"GT目录: {args.gt_dir}")
    print(f"验证目录: {args.val_dir}")
    print(f"StableSR Edge模型: {args.edge_model}")
    print(f"StableSR Upscale模型: {args.upscale_model}")
    print(f"Edge配置文件: {args.edge_config}")
    print(f"Upscale配置文件: {args.upscale_config}")
    print(f"输出文件: {args.output}")
    print(f"计算设备: {args.device}")
    print(f"DDPM步数: {args.ddpm_steps}")
    print(f"超分辨率倍数: {args.upscale}")
    print(f"颜色修复: {args.colorfix}")
    print(f"随机种子: {args.seed}")
    print("="*60)
    
    try:
        # 创建EnhancedTraReport实例
        print("\n创建EnhancedTraReport实例...")
        enhanced_tra_report = EnhancedTraReport(
            gt_dir=args.gt_dir,
            val_dir=args.val_dir,
            stablesr_edge_model_path=args.edge_model,
            stablesr_upscale_model_path=args.upscale_model,
            stablesr_edge_config_path=args.edge_config,
            stablesr_upscale_config_path=args.upscale_config,
            device=args.device,
            ddpm_steps=args.ddpm_steps,
            upscale=args.upscale,
            colorfix_type=args.colorfix,
            seed=args.seed
        )
        
        print("EnhancedTraReport实例创建成功!")
        
        # 运行评估
        print("\n开始评估...")
        results = enhanced_tra_report.run_evaluation(args.output)
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        print(f"处理文件数: {results['evaluation_info']['total_files']}")
        print(f"StableSR Edge平均PSNR: {results['summary']['stablesr_edge']['average_psnr']:.4f}")
        print(f"StableSR Upscale平均PSNR: {results['summary']['stablesr_upscale']['average_psnr']:.4f}")
        print(f"PSNR差异: {results['summary']['comparison']['psnr_difference']:.4f}")
        print(f"更好的模型: {results['summary']['comparison']['better_model']}")
        print(f"改进百分比: {results['summary']['comparison']['improvement_percentage']:.2f}%")
        
        # 详细统计
        if args.verbose:
            print("\n详细统计信息:")
            print("-"*40)
            edge_stats = results['summary']['stablesr_edge']
            upscale_stats = results['summary']['stablesr_upscale']
            
            print("StableSR Edge:")
            print(f"  最小PSNR: {edge_stats['min_psnr']:.4f}")
            print(f"  最大PSNR: {edge_stats['max_psnr']:.4f}")
            print(f"  标准差: {edge_stats['std_psnr']:.4f}")
            
            print("StableSR Upscale:")
            print(f"  最小PSNR: {upscale_stats['min_psnr']:.4f}")
            print(f"  最大PSNR: {upscale_stats['max_psnr']:.4f}")
            print(f"  标准差: {upscale_stats['std_psnr']:.4f}")
            
            print("\n单个图片结果:")
            print("-"*40)
            for i, result in enumerate(results["results"]):
                val_name = Path(result["val_file"]).name
                edge_psnr = result["stablesr_edge"]["psnr"]
                upscale_psnr = result["stablesr_upscale"]["psnr"]
                difference = result["psnr_difference"]
                better = result["better_model"]
                
                print(f"  {i+1}. {val_name}:")
                print(f"     Edge PSNR: {edge_psnr:.4f}")
                print(f"     Upscale PSNR: {upscale_psnr:.4f}")
                print(f"     差异: {difference:.4f}")
                print(f"     更好: {better}")
        
        print(f"\n评估完成! 结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
