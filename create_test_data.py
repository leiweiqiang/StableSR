#!/usr/bin/env python3
"""
创建测试数据脚本
从DIV2K高分辨率数据创建低分辨率测试数据
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse

def create_lr_dataset(hr_dir, lr_dir, scale_factor=4, subset_size=None, force=False):
    """
    从高分辨率数据集创建低分辨率数据集
    
    Args:
        hr_dir: 高分辨率图片目录
        lr_dir: 低分辨率图片输出目录
        scale_factor: 缩放因子
        subset_size: 子集大小（用于快速测试）
        force: 是否强制重新生成所有图片
    """
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    
    # 创建输出目录
    lr_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    hr_files = []
    
    for ext in image_extensions:
        hr_files.extend(hr_path.glob(f"*{ext}"))
        hr_files.extend(hr_path.glob(f"*{ext.upper()}"))
    
    # 限制子集大小用于快速测试
    if subset_size and len(hr_files) > subset_size:
        hr_files = hr_files[:subset_size]
        print(f"使用前 {subset_size} 个文件进行测试")
    
    print(f"找到 {len(hr_files)} 个高分辨率图片")
    print(f"缩放因子: {scale_factor}")
    
    processed = 0
    for hr_file in hr_files:
        try:
            # 检查输出文件是否已存在
            lr_file = lr_path / hr_file.name
            if lr_file.exists() and not force:
                print(f"跳过已存在的文件: {lr_file.name}")
                continue
            
            # 加载高分辨率图片
            img = Image.open(hr_file)
            
            # 计算低分辨率尺寸
            lr_width = img.width // scale_factor
            lr_height = img.height // scale_factor
            
            # 缩放到低分辨率
            lr_img = img.resize((lr_width, lr_height), Image.LANCZOS)
            
            # 保存低分辨率图片
            lr_img.save(lr_file)
            
            processed += 1
            if processed % 10 == 0:
                print(f"已处理: {processed}/{len(hr_files)}")
                
        except Exception as e:
            print(f"处理文件 {hr_file} 时出错: {str(e)}")
            continue
    
    print(f"完成！创建了 {processed} 个低分辨率图片")
    print(f"高分辨率目录: {hr_dir}")
    print(f"低分辨率目录: {lr_dir}")

def main():
    parser = argparse.ArgumentParser(description="创建低分辨率测试数据集")
    parser.add_argument("--hr_dir", type=str, 
                       default="/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR",
                       help="高分辨率图片目录")
    parser.add_argument("--lr_dir", type=str,
                       default="/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR",
                       help="低分辨率图片输出目录")
    parser.add_argument("--scale_factor", type=int, default=4,
                       help="缩放因子 (默认: 4)")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="子集大小，用于快速测试 (默认: 使用全部)")
    parser.add_argument("--force", action="store_true",
                       help="强制重新生成所有图片")
    
    args = parser.parse_args()
    
    print("创建低分辨率测试数据集")
    print("=" * 50)
    print(f"高分辨率目录: {args.hr_dir}")
    print(f"低分辨率目录: {args.lr_dir}")
    print(f"缩放因子: {args.scale_factor}")
    if args.subset_size:
        print(f"子集大小: {args.subset_size}")
    print()
    
    # 检查输入目录
    if not os.path.exists(args.hr_dir):
        print(f"错误: 高分辨率目录不存在: {args.hr_dir}")
        return 1
    
    # 创建低分辨率数据集
    create_lr_dataset(
        args.hr_dir, 
        args.lr_dir, 
        args.scale_factor, 
        args.subset_size,
        args.force
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
