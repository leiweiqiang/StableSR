"""
StableSR_ScaleLR 完整使用示例
展示如何使用StableSR_ScaleLR类进行图像超分辨率处理
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_sr_scale_lr import StableSR_ScaleLR


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="StableSR_ScaleLR 完整使用示例")
    
    # 必需参数
    parser.add_argument("--config", type=str, 
                       default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
                       help="配置文件路径")
    parser.add_argument("--ckpt", type=str, 
                       default="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt",
                       help="模型检查点路径")
    parser.add_argument("--vqgan_ckpt", type=str, 
                       default="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt",
                       help="VQGAN检查点路径")
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入图像路径（文件或目录）")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="输出目录")
    
    # 可选参数
    parser.add_argument("--hq_path", type=str, default=None,
                       help="高质量图像路径（可选）")
    parser.add_argument("--ddpm_steps", type=int, default=4,
                       help="DDPM采样步数")
    parser.add_argument("--dec_w", type=float, default=0.5,
                       help="VQGAN和Diffusion结合权重")
    parser.add_argument("--colorfix_type", type=str, default="adain",
                       choices=["adain", "wavelet", "nofix"],
                       help="颜色修正类型")
    parser.add_argument("--input_size", type=int, default=512,
                       help="输入尺寸")
    parser.add_argument("--upscale", type=float, default=4.0,
                       help="上采样倍数")
    parser.add_argument("--tile_overlap", type=int, default=32,
                       help="瓦片重叠大小")
    parser.add_argument("--vqgantile_stride", type=int, default=1000,
                       help="VQGAN瓦片步长")
    parser.add_argument("--vqgantile_size", type=int, default=1280,
                       help="VQGAN瓦片大小")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--precision", type=str, default="autocast",
                       choices=["autocast", "full"],
                       help="精度类型")
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径不存在: {args.input_path}")
        return 1
    
    # 验证配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return 1
    
    # 验证检查点文件
    if not os.path.exists(args.ckpt):
        print(f"错误: 模型检查点不存在: {args.ckpt}")
        return 1
    
    if not os.path.exists(args.vqgan_ckpt):
        print(f"错误: VQGAN检查点不存在: {args.vqgan_ckpt}")
        return 1
    
    # 验证HQ路径（如果提供）
    if args.hq_path and not os.path.exists(args.hq_path):
        print(f"错误: HQ路径不存在: {args.hq_path}")
        return 1
    
    print("StableSR_ScaleLR 图像超分辨率处理")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"模型检查点: {args.ckpt}")
    print(f"VQGAN检查点: {args.vqgan_ckpt}")
    print(f"输入路径: {args.input_path}")
    print(f"输出目录: {args.out_dir}")
    if args.hq_path:
        print(f"HQ路径: {args.hq_path}")
    print(f"DDPM步数: {args.ddpm_steps}")
    print(f"颜色修正: {args.colorfix_type}")
    print(f"上采样倍数: {args.upscale}")
    print("=" * 50)
    
    try:
        # 创建处理器实例
        print("正在初始化StableSR_ScaleLR...")
        processor = StableSR_ScaleLR(
            config_path=args.config,
            ckpt_path=args.ckpt,
            vqgan_ckpt_path=args.vqgan_ckpt,
            ddpm_steps=args.ddpm_steps,
            dec_w=args.dec_w,
            colorfix_type=args.colorfix_type,
            input_size=args.input_size,
            upscale=args.upscale,
            tile_overlap=args.tile_overlap,
            vqgantile_stride=args.vqgantile_stride,
            vqgantile_size=args.vqgantile_size,
            seed=args.seed,
            precision=args.precision
        )
        print("✓ 初始化完成")
        
        # 处理图像
        print("开始处理图像...")
        processor.process_images(
            input_path=args.input_path,
            out_dir=args.out_dir,
            hq_path=args.hq_path
        )
        
        print("✓ 处理完成！")
        print(f"结果保存在: {args.out_dir}")
        print("目录结构:")
        print(f"  - {args.out_dir}/RES/  : 超分辨率结果")
        print(f"  - {args.out_dir}/LR/   : 原始低分辨率图像")
        if args.hq_path:
            print(f"  - {args.out_dir}/HQ/   : 高质量参考图像")
        
        return 0
        
    except Exception as e:
        print(f"✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
