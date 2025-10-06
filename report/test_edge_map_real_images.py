#!/usr/bin/env python3
"""
真实图像Edge Map测试脚本
专门用于测试真实图像的edge map生成和处理

使用方法:
python test_edge_map_real_images.py --input_dir /path/to/images
python test_edge_map_real_images.py --input_image /path/to/single/image.jpg
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealImageEdgeTester:
    """真实图像Edge Map测试器"""
    
    def __init__(self, output_dir="real_edge_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "edge_maps").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        
        self.results = []
        
    def load_image(self, image_path, target_size=(512, 512)):
        """加载和预处理图像"""
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 记录原始尺寸
            original_size = image.shape[:2]
            
            # 调整尺寸
            if target_size:
                image = cv2.resize(image, (target_size[1], target_size[0]))
            
            return image, original_size
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None, None
    
    def generate_edge_map_advanced(self, image, method="canny", **kwargs):
        """高级edge map生成"""
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "canny":
            # 自适应Canny边缘检测
            # 计算图像的中值
            median = np.median(gray)
            
            # 使用中值计算阈值
            lower = int(max(0, 0.7 * median))
            upper = int(min(255, 1.3 * median))
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            
            # Canny边缘检测
            edges = cv2.Canny(blurred, lower, upper)
            
        elif method == "canny_otsu":
            # 使用Otsu方法自动确定阈值
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            
            # Otsu阈值
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 使用Otsu阈值作为Canny的高阈值，低阈值为高阈值的一半
            high_thresh = thresh
            low_thresh = high_thresh // 2
            
            edges = cv2.Canny(blurred, low_thresh, high_thresh)
            
        elif method == "sobel":
            # Sobel边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.uint8(edges / edges.max() * 255)
            
        elif method == "laplacian":
            # Laplacian边缘检测
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            
        elif method == "scharr":
            # Scharr边缘检测
            scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(scharr_x**2 + scharr_y**2)
            edges = np.uint8(edges / edges.max() * 255)
            
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")
        
        # 转换为3通道
        if len(edges.shape) == 2:
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edges_3ch = edges
        
        return edges_3ch
    
    def analyze_edge_map(self, edge_map, image_name):
        """分析edge map特征"""
        # 转换为灰度进行分析
        if len(edge_map.shape) == 3:
            edge_gray = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        else:
            edge_gray = edge_map
        
        # 基本统计
        total_pixels = edge_gray.shape[0] * edge_gray.shape[1]
        edge_pixels = np.sum(edge_gray > 0)
        edge_ratio = edge_pixels / total_pixels * 100
        
        # 边缘密度分析
        edge_density = edge_pixels / total_pixels
        
        # 边缘强度分析
        edge_intensity_mean = np.mean(edge_gray[edge_gray > 0]) if edge_pixels > 0 else 0
        edge_intensity_std = np.std(edge_gray[edge_gray > 0]) if edge_pixels > 0 else 0
        
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_gray, connectivity=8)
        num_components = num_labels - 1  # 减去背景
        
        # 边缘长度分析
        contours, _ = cv2.findContours(edge_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_edge_length = sum(cv2.arcLength(contour, True) for contour in contours)
        
        analysis = {
            "image_name": image_name,
            "total_pixels": int(total_pixels),
            "edge_pixels": int(edge_pixels),
            "edge_ratio": float(edge_ratio),
            "edge_density": float(edge_density),
            "edge_intensity_mean": float(edge_intensity_mean),
            "edge_intensity_std": float(edge_intensity_std),
            "num_components": int(num_components),
            "total_edge_length": float(total_edge_length),
            "num_contours": len(contours)
        }
        
        return analysis
    
    def test_single_image(self, image_path, methods=None):
        """测试单张图像"""
        if methods is None:
            methods = ["canny", "canny_otsu", "sobel", "laplacian", "scharr"]
        
        image_name = Path(image_path).stem
        print(f"\n测试图像: {image_name}")
        print("-" * 40)
        
        # 加载图像
        image, original_size = self.load_image(image_path)
        if image is None:
            return None
        
        print(f"原始尺寸: {original_size}")
        print(f"处理后尺寸: {image.shape[:2]}")
        
        # 保存原始图像
        original_path = self.output_dir / "edge_maps" / f"{image_name}_original.png"
        cv2.imwrite(str(original_path), image)
        
        results = {
            "image_name": image_name,
            "original_size": original_size,
            "processed_size": image.shape[:2],
            "methods": {}
        }
        
        # 测试不同方法
        edge_maps = {}
        for method in methods:
            print(f"\n测试 {method.upper()} 方法:")
            
            try:
                # 生成edge map
                edge_map = self.generate_edge_map_advanced(image, method=method)
                edge_maps[method] = edge_map
                
                # 保存edge map
                edge_path = self.output_dir / "edge_maps" / f"{image_name}_edge_{method}.png"
                cv2.imwrite(str(edge_path), edge_map)
                
                # 分析edge map
                analysis = self.analyze_edge_map(edge_map, f"{image_name}_{method}")
                results["methods"][method] = analysis
                
                print(f"  ✓ Edge map已保存: {edge_path}")
                print(f"  ✓ 边缘像素比例: {analysis['edge_ratio']:.2f}%")
                print(f"  ✓ 连通组件数: {analysis['num_components']}")
                print(f"  ✓ 边缘总长度: {analysis['total_edge_length']:.2f}")
                
            except Exception as e:
                print(f"  ❌ {method} 方法失败: {e}")
                results["methods"][method] = {"error": str(e)}
        
        # 创建对比图
        if edge_maps:
            comparison_path = self.output_dir / "comparisons" / f"{image_name}_comparison.png"
            self.create_detailed_comparison(image, edge_maps, comparison_path)
            print(f"  ✓ 对比图已保存: {comparison_path}")
        
        return results
    
    def create_detailed_comparison(self, original, edge_maps, output_path):
        """创建详细的对比图"""
        n_methods = len(edge_maps)
        if n_methods == 0:
            return
        
        # 创建子图布局
        fig, axes = plt.subplots(3, n_methods + 1, figsize=(4 * (n_methods + 1), 12))
        
        # 原始图像
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('原始图像', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 灰度图
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title('灰度图', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 直方图
        axes[2, 0].hist(gray.ravel(), bins=256, range=[0, 256], alpha=0.7, color='blue')
        axes[2, 0].set_title('灰度直方图', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('像素值')
        axes[2, 0].set_ylabel('频次')
        
        # 各种边缘检测方法
        for i, (method, edge_map) in enumerate(edge_maps.items()):
            # 彩色边缘图
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB)
            axes[0, i + 1].imshow(edge_rgb)
            axes[0, i + 1].set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
            axes[0, i + 1].axis('off')
            
            # 灰度边缘图
            if len(edge_map.shape) == 3:
                edge_gray = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
            else:
                edge_gray = edge_map
            axes[1, i + 1].imshow(edge_gray, cmap='gray')
            axes[1, i + 1].set_title(f'{method.upper()} (灰度)', fontsize=10)
            axes[1, i + 1].axis('off')
            
            # 边缘强度直方图
            axes[2, i + 1].hist(edge_gray.ravel(), bins=256, range=[0, 256], alpha=0.7, color='red')
            axes[2, i + 1].set_title(f'{method.upper()} 直方图', fontsize=10)
            axes[2, i + 1].set_xlabel('像素值')
            axes[2, i + 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_image_directory(self, input_dir, max_images=None):
        """测试图像目录"""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"错误: 目录不存在: {input_dir}")
            return False
        
        # 查找图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"在目录中未找到图像文件: {input_dir}")
            return False
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 限制测试数量
        if max_images:
            image_files = image_files[:max_images]
            print(f"将测试前 {len(image_files)} 个图像")
        
        # 测试每个图像
        all_results = []
        for i, image_file in enumerate(image_files):
            print(f"\n进度: {i+1}/{len(image_files)}")
            result = self.test_single_image(image_file)
            if result:
                all_results.append(result)
        
        # 保存汇总结果
        self.save_summary_results(all_results)
        
        print(f"\n✓ 完成 {len(all_results)} 个图像的测试")
        return True
    
    def save_summary_results(self, results):
        """保存汇总结果"""
        if not results:
            return
        
        # 保存JSON结果
        json_path = self.output_dir / "statistics" / "edge_analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建汇总统计
        summary = self.create_summary_statistics(results)
        
        # 保存汇总报告
        report_path = self.output_dir / "statistics" / "summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Edge Map分析汇总报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试图像数量: {len(results)}\n\n")
            
            for method, stats in summary.items():
                f.write(f"{method.upper()} 方法统计:\n")
                f.write(f"  平均边缘比例: {stats['avg_edge_ratio']:.2f}%\n")
                f.write(f"  平均连通组件数: {stats['avg_components']:.1f}\n")
                f.write(f"  平均边缘长度: {stats['avg_edge_length']:.2f}\n")
                f.write(f"  平均边缘强度: {stats['avg_intensity']:.2f}\n")
                f.write("\n")
        
        print(f"汇总结果已保存:")
        print(f"  JSON: {json_path}")
        print(f"  报告: {report_path}")
    
    def create_summary_statistics(self, results):
        """创建汇总统计"""
        summary = {}
        
        # 收集所有方法的数据
        all_methods = set()
        for result in results:
            all_methods.update(result["methods"].keys())
        
        for method in all_methods:
            method_data = []
            for result in results:
                if method in result["methods"] and "error" not in result["methods"][method]:
                    method_data.append(result["methods"][method])
            
            if method_data:
                summary[method] = {
                    "avg_edge_ratio": np.mean([d["edge_ratio"] for d in method_data]),
                    "avg_components": np.mean([d["num_components"] for d in method_data]),
                    "avg_edge_length": np.mean([d["total_edge_length"] for d in method_data]),
                    "avg_intensity": np.mean([d["edge_intensity_mean"] for d in method_data]),
                    "count": len(method_data)
                }
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="真实图像Edge Map测试脚本")
    
    parser.add_argument("--input_dir", type=str, default=None,
                       help="输入图像目录")
    parser.add_argument("--input_image", type=str, default=None,
                       help="单张输入图像")
    parser.add_argument("--output_dir", type=str, default="real_edge_test_results",
                       help="输出目录")
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大测试图像数量")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["canny", "canny_otsu", "sobel", "laplacian", "scharr"],
                       help="边缘检测方法")
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.input_dir and not args.input_image:
        print("请指定 --input_dir 或 --input_image")
        return 1
    
    if args.input_dir and args.input_image:
        print("请只指定 --input_dir 或 --input_image 中的一个")
        return 1
    
    print("真实图像Edge Map测试")
    print("=" * 50)
    
    # 创建测试器
    tester = RealImageEdgeTester(args.output_dir)
    
    try:
        if args.input_image:
            # 测试单张图像
            result = tester.test_single_image(args.input_image, args.methods)
            if result:
                print("\n🎉 单张图像测试完成!")
            else:
                print("\n❌ 单张图像测试失败")
                return 1
        else:
            # 测试图像目录
            success = tester.test_image_directory(args.input_dir, args.max_images)
            if success:
                print("\n🎉 图像目录测试完成!")
            else:
                print("\n❌ 图像目录测试失败")
                return 1
        
        print(f"结果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
