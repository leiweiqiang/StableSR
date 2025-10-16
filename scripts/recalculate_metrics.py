#!/usr/bin/env python3
"""
重新计算指标

这个脚本检查现有的 metrics.json 文件，如果缺少任何指标
（PSNR, SSIM, Edge PSNR, Edge Overlap 等），则重新计算并更新
metrics.json 和 metrics.csv 文件。

Usage:
    python scripts/recalculate_metrics.py <output_dir> <gt_img_dir>
    python scripts/recalculate_metrics.py <output_dir> <gt_img_dir> --force
"""

import os
import sys
import json
import csv
from pathlib import Path
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
from basicsr.metrics.edge_l2_loss import EdgePSNRCalculator
from basicsr.metrics.edge_overlap import EdgeOverlapCalculator


def check_metrics_complete(metrics_file):
    """检查 metrics.json 是否包含所有必需的指标"""
    if not os.path.exists(metrics_file):
        return False, []
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # 定义所有需要检查的指标
        required_avg_metrics = ['average_psnr', 'average_ssim', 'average_edge_psnr', 'average_edge_overlap']
        required_img_metrics = ['psnr', 'ssim', 'edge_psnr', 'edge_overlap']
        # LPIPS 是可选的（可能不可用）
        
        missing_metrics = []
        
        # 检查平均值指标
        for metric in required_avg_metrics:
            if metric not in data:
                missing_metrics.append(metric)
        
        # 检查每张图片的指标
        for img_data in data.get('images', []):
            for metric in required_img_metrics:
                if metric not in img_data:
                    if metric not in missing_metrics:
                        missing_metrics.append(metric)
                    break  # 只要发现一个缺失就够了
        
        # 如果有缺失的指标，返回 False 和缺失列表
        if missing_metrics:
            return False, missing_metrics
        
        return True, []
    except Exception as e:
        print(f"警告：读取 metrics.json 失败: {e}")
        return False, ['read_error']


def recalculate_edge_metrics(output_dir, gt_img_dir):
    """重新计算 Edge PSNR 和 Edge Overlap 并更新 metrics 文件"""
    
    output_path = Path(output_dir)
    gt_path = Path(gt_img_dir)
    
    if not output_path.exists():
        print(f"错误：输出目录不存在: {output_dir}")
        return False
        
    if not gt_path.exists():
        print(f"错误：GT目录不存在: {gt_img_dir}")
        return False
    
    metrics_file = output_path / 'metrics.json'
    if not metrics_file.exists():
        print(f"错误：metrics.json 不存在: {metrics_file}")
        return False
    
    # 读取现有的 metrics
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"错误：无法读取 metrics.json: {e}")
        return False
    
    # 初始化计算器
    psnr_calculator = EdgePSNRCalculator()
    overlap_calculator = EdgeOverlapCalculator()
    
    print(f"开始计算 Edge PSNR 和 Edge Overlap...")
    print(f"输出目录: {output_dir}")
    print(f"GT目录: {gt_img_dir}")
    
    # 计算每张图片的指标
    total_edge_psnr = 0.0
    total_edge_overlap = 0.0
    count = 0
    updated_images = []
    
    for img_data in metrics.get('images', []):
        img_name = img_data.get('image_name')
        if not img_name:
            continue
        
        # 获取生成图片的基础名称（去除 _edge 后缀）
        img_base_name = Path(img_name).stem.replace('_edge', '')
        
        # 找到生成的图片
        gen_img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_paths = [
                output_path / f"{img_base_name}{ext}",
                output_path / f"{img_base_name}_edge{ext}",
            ]
            for p in potential_paths:
                if p.exists():
                    gen_img_path = p
                    break
            if gen_img_path:
                break
        
        if not gen_img_path:
            print(f"  ⚠ 警告：找不到生成图片: {img_base_name}")
            img_data['edge_psnr'] = -1.0
            img_data['edge_overlap'] = -1.0
            updated_images.append(img_data)
            continue
        
        # 找到对应的 GT 图片
        gt_img_path = gt_path / f"{img_base_name}.png"
        if not gt_img_path.exists():
            gt_img_path = gt_path / f"{img_base_name}.jpg"
            if not gt_img_path.exists():
                print(f"  ⚠ 警告：找不到GT图片: {img_base_name}")
                img_data['edge_psnr'] = -1.0
                img_data['edge_overlap'] = -1.0
                updated_images.append(img_data)
                continue
        
        # 读取图片
        gen_img = cv2.imread(str(gen_img_path))
        gt_img = cv2.imread(str(gt_img_path))
        
        if gen_img is None or gt_img is None:
            print(f"  ⚠ 警告：无法读取图片: {img_base_name}")
            img_data['edge_psnr'] = -1.0
            img_data['edge_overlap'] = -1.0
            updated_images.append(img_data)
            continue
        
        # 计算 Edge PSNR
        edge_psnr = -1.0
        edge_overlap = -1.0
        
        try:
            edge_psnr = psnr_calculator.calculate_from_arrays(
                gen_img, gt_img, input_format='BGR'
            )
            img_data['edge_psnr'] = float(edge_psnr)
        except Exception as e:
            print(f"  ⚠ Edge PSNR计算失败: {e}")
            img_data['edge_psnr'] = -1.0
        
        # 计算 Edge Overlap
        try:
            edge_overlap = overlap_calculator.calculate_from_arrays(
                gen_img, gt_img, input_format='BGR'
            )
            img_data['edge_overlap'] = float(edge_overlap)
        except Exception as e:
            print(f"  ⚠ Edge Overlap计算失败: {e}")
            img_data['edge_overlap'] = -1.0
        
        # 累加计数
        if edge_psnr >= 0 and edge_overlap >= 0:
            total_edge_psnr += edge_psnr
            total_edge_overlap += edge_overlap
            count += 1
            print(f"  ✓ {img_name}: PSNR={edge_psnr:.4f} dB, Overlap={edge_overlap:.4f}")
        else:
            print(f"  ✗ {img_name}: 计算失败")
        
        updated_images.append(img_data)
    
    # 计算平均值
    if count > 0:
        metrics['average_edge_psnr'] = total_edge_psnr / count
        metrics['average_edge_overlap'] = total_edge_overlap / count
    else:
        metrics['average_edge_psnr'] = 0.0
        metrics['average_edge_overlap'] = 0.0
    
    metrics['images'] = updated_images
    
    # 保存更新后的 metrics.json
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ 已更新 metrics.json")
        print(f"  平均 Edge PSNR: {metrics['average_edge_psnr']:.4f} dB")
        print(f"  平均 Edge Overlap: {metrics['average_edge_overlap']:.4f}")
    except Exception as e:
        print(f"错误：无法保存 metrics.json: {e}")
        return False
    
    # 更新 metrics.csv
    csv_path = output_path / 'metrics.csv'
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 检查是否有 LPIPS
            lpips_available = metrics.get('average_lpips') is not None
            
            if lpips_available:
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Edge PSNR (dB)', 'Edge Overlap'])
                for img_data in metrics['images']:
                    writer.writerow([
                        img_data['image_name'],
                        f"{img_data['psnr']:.4f}",
                        f"{img_data['ssim']:.4f}",
                        f"{img_data['lpips']:.4f}" if img_data.get('lpips') is not None else 'N/A',
                        f"{img_data['edge_psnr']:.4f}",
                        f"{img_data['edge_overlap']:.4f}"
                    ])
                avg_lpips = f"{metrics['average_lpips']:.4f}" if metrics['average_lpips'] is not None else 'N/A'
                writer.writerow([
                    'Average',
                    f"{metrics['average_psnr']:.4f}",
                    f"{metrics['average_ssim']:.4f}",
                    avg_lpips,
                    f"{metrics['average_edge_psnr']:.4f}",
                    f"{metrics['average_edge_overlap']:.4f}"
                ])
            else:
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM', 'Edge PSNR (dB)', 'Edge Overlap'])
                for img_data in metrics['images']:
                    writer.writerow([
                        img_data['image_name'],
                        f"{img_data['psnr']:.4f}",
                        f"{img_data['ssim']:.4f}",
                        f"{img_data['edge_psnr']:.4f}",
                        f"{img_data['edge_overlap']:.4f}"
                    ])
                writer.writerow([
                    'Average',
                    f"{metrics['average_psnr']:.4f}",
                    f"{metrics['average_ssim']:.4f}",
                    f"{metrics['average_edge_psnr']:.4f}",
                    f"{metrics['average_edge_overlap']:.4f}"
                ])
        
        print(f"✓ 已更新 metrics.csv")
    except Exception as e:
        print(f"错误：无法保存 metrics.csv: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='检查并重新计算缺失的指标（PSNR, SSIM, Edge PSNR, Edge Overlap 等）'
    )
    parser.add_argument(
        'output_dir',
        help='包含推理结果的输出目录'
    )
    parser.add_argument(
        'gt_img_dir',
        help='Ground Truth 图片目录'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新计算，即使已存在指标'
    )
    
    args = parser.parse_args()
    
    output_dir = os.path.expanduser(args.output_dir)
    gt_img_dir = os.path.expanduser(args.gt_img_dir)
    
    # 检查 metrics.json
    metrics_file = Path(output_dir) / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"错误：metrics.json 不存在: {metrics_file}")
        sys.exit(1)
    
    # 检查是否所有指标都完整
    is_complete, missing_metrics = check_metrics_complete(metrics_file)
    
    if not args.force and is_complete:
        print(f"✓ 所有指标已存在: {output_dir}")
        print("  如需重新计算，请使用 --force 参数")
        sys.exit(0)
    
    # 显示缺失的指标
    if not args.force and missing_metrics:
        print(f"→ 发现缺失的指标: {', '.join(missing_metrics)}")
        print(f"→ 需要重新计算指标: {output_dir}")
    elif args.force:
        print(f"→ 强制重新计算所有指标: {output_dir}")
    
    # 重新计算
    success = recalculate_edge_metrics(output_dir, gt_img_dir)
    
    if success:
        print("\n✓ Edge 相关指标计算完成")
        sys.exit(0)
    else:
        print("\n✗ Edge 相关指标计算失败")
        sys.exit(1)


if __name__ == '__main__':
    main()

