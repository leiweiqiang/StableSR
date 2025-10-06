#!/usr/bin/env python3
"""
综合Edge Map测试脚本
用于测试StableSR Edge处理功能的完整测试套件

功能包括:
1. Edge map生成测试
2. Edge处理器测试
3. 特征融合测试
4. 真实图像测试
5. 性能基准测试
6. 可视化功能
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ldm.modules.diffusionmodules.edge_processor import EdgeMapProcessor, EdgeFusionModule
    from ldm.modules.diffusionmodules.unet_with_edge import UNetModelDualcondV2WithEdge
    from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
    EDGE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入edge处理模块: {e}")
    print("将使用简化版本进行测试")
    EDGE_MODULES_AVAILABLE = False


class EdgeMapTester:
    """Edge Map测试器"""
    
    def __init__(self, device=None, output_dir="edge_test_results"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        
    def create_synthetic_edge_map(self, size=(512, 512), pattern="geometric"):
        """
        创建合成edge map用于测试
        
        Args:
            size: 图像尺寸 (height, width)
            pattern: 图案类型 ("geometric", "texture", "mixed")
            
        Returns:
            edge_map: numpy数组 [H, W, 3]
        """
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if pattern == "geometric":
            # 几何图案
            cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
            cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
            cv2.line(img, (100, 300), (400, 100), (200, 200, 200), 3)
            cv2.ellipse(img, (250, 150), (60, 40), 45, 0, 360, (180, 180, 180), -1)
            
        elif pattern == "texture":
            # 纹理图案
            for i in range(0, h, 50):
                for j in range(0, w, 50):
                    if (i + j) % 100 == 0:
                        cv2.rectangle(img, (j, i), (j+30, i+30), (255, 255, 255), -1)
            
        elif pattern == "mixed":
            # 混合图案
            cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
            cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
            for i in range(0, h, 80):
                cv2.line(img, (0, i), (w, i), (100, 100, 100), 1)
            for j in range(0, w, 80):
                cv2.line(img, (j, 0), (j, h), (100, 100, 100), 1)
        
        # 转换为灰度并应用Canny边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        
        # 转换为3通道
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_3ch
    
    def generate_edge_map_from_image(self, image_path, output_size=None):
        """
        从真实图像生成edge map
        
        Args:
            image_path: 输入图像路径
            output_size: 输出尺寸 (height, width)
            
        Returns:
            edge_map: numpy数组 [H, W, 3]
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 调整尺寸
        if output_size:
            img = cv2.resize(img, (output_size[1], output_size[0]))
        
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # 应用Canny边缘检测
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        
        # 转换为3通道
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_3ch
    
    def test_edge_map_generation(self):
        """测试edge map生成功能"""
        print("\n" + "="*50)
        print("测试Edge Map生成功能")
        print("="*50)
        
        patterns = ["geometric", "texture", "mixed"]
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for pattern in patterns:
            for size in sizes:
                print(f"生成 {pattern} 图案, 尺寸: {size}")
                
                # 生成edge map
                edge_map = self.create_synthetic_edge_map(size, pattern)
                
                # 保存结果
                output_path = self.output_dir / f"edge_map_{pattern}_{size[0]}x{size[1]}.png"
                cv2.imwrite(str(output_path), edge_map)
                
                print(f"  ✓ 保存到: {output_path}")
                print(f"  ✓ 形状: {edge_map.shape}")
                print(f"  ✓ 数据类型: {edge_map.dtype}")
                print(f"  ✓ 值范围: [{edge_map.min()}, {edge_map.max()}]")
        
        print("✓ Edge Map生成测试完成")
    
    def test_edge_processor(self):
        """测试Edge处理器"""
        print("\n" + "="*50)
        print("测试Edge处理器")
        print("="*50)
        
        if not EDGE_MODULES_AVAILABLE:
            print("⚠️  Edge处理模块不可用，跳过此测试")
            return
        
        # 创建处理器
        processor = EdgeMapProcessor(
            input_channels=3, 
            output_channels=4, 
            target_size=64,
            use_checkpoint=False
        ).to(self.device)
        
        # 测试不同输入尺寸
        test_sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for h, w in test_sizes:
            print(f"测试输入尺寸: {h}x{w}")
            
            # 创建测试edge map
            edge_map_np = self.create_synthetic_edge_map((h, w))
            
            # 转换为tensor
            edge_map = torch.from_numpy(edge_map_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            edge_map = edge_map.to(self.device)
            
            # 处理edge map
            start_time = time.time()
            with torch.no_grad():
                output = processor(edge_map)
            end_time = time.time()
            
            print(f"  输入形状: {edge_map.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  处理时间: {end_time - start_time:.4f}s")
            print(f"  内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f}MB" if torch.cuda.is_available() else "N/A")
            
            # 验证输出形状
            assert output.shape == (1, 4, 64, 64), f"期望输出形状 (1, 4, 64, 64), 实际: {output.shape}"
            
            # 清理内存
            del edge_map, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✓ Edge处理器测试完成")
    
    def test_edge_fusion(self):
        """测试特征融合模块"""
        print("\n" + "="*50)
        print("测试特征融合模块")
        print("="*50)
        
        if not EDGE_MODULES_AVAILABLE:
            print("⚠️  Edge处理模块不可用，跳过此测试")
            return
        
        # 创建融合模块
        fusion_module = EdgeFusionModule().to(self.device)
        
        # 测试不同batch size
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            print(f"测试batch size: {bs}")
            
            # 创建测试输入
            unet_input = torch.randn(bs, 4, 64, 64).to(self.device)
            edge_features = torch.randn(bs, 4, 64, 64).to(self.device)
            
            # 执行融合
            start_time = time.time()
            with torch.no_grad():
                fused = fusion_module(unet_input, edge_features)
            end_time = time.time()
            
            print(f"  U-Net输入: {unet_input.shape}")
            print(f"  Edge特征: {edge_features.shape}")
            print(f"  融合输出: {fused.shape}")
            print(f"  融合时间: {end_time - start_time:.4f}s")
            
            # 验证输出形状
            assert fused.shape == (bs, 8, 64, 64), f"期望输出形状 ({bs}, 8, 64, 64), 实际: {fused.shape}"
            
            # 清理内存
            del unet_input, edge_features, fused
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✓ 特征融合测试完成")
    
    def test_with_real_images(self, image_dir=None):
        """使用真实图像测试"""
        print("\n" + "="*50)
        print("使用真实图像测试")
        print("="*50)
        
        if image_dir is None:
            print("⚠️  未提供图像目录，跳过真实图像测试")
            return
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"⚠️  图像目录不存在: {image_dir}")
            return
        
        # 查找图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠️  在目录中未找到图像文件: {image_dir}")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 测试前几个图像
        test_count = min(3, len(image_files))
        
        for i, image_file in enumerate(image_files[:test_count]):
            print(f"\n测试图像 {i+1}/{test_count}: {image_file.name}")
            
            try:
                # 生成edge map
                edge_map = self.generate_edge_map_from_image(image_file, (512, 512))
                
                # 保存edge map
                edge_output_path = self.output_dir / f"real_edge_{image_file.stem}.png"
                cv2.imwrite(str(edge_output_path), edge_map)
                
                print(f"  ✓ Edge map已保存: {edge_output_path}")
                print(f"  ✓ Edge map形状: {edge_map.shape}")
                
                # 如果edge处理模块可用，进行进一步测试
                if EDGE_MODULES_AVAILABLE:
                    # 转换为tensor并处理
                    edge_tensor = torch.from_numpy(edge_map).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    edge_tensor = edge_tensor.to(self.device)
                    
                    processor = EdgeMapProcessor().to(self.device)
                    with torch.no_grad():
                        features = processor(edge_tensor)
                    
                    print(f"  ✓ 处理后的特征形状: {features.shape}")
                    
                    # 清理内存
                    del edge_tensor, features
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  ❌ 处理图像失败: {e}")
        
        print("✓ 真实图像测试完成")
    
    def performance_benchmark(self):
        """性能基准测试"""
        print("\n" + "="*50)
        print("性能基准测试")
        print("="*50)
        
        if not EDGE_MODULES_AVAILABLE:
            print("⚠️  Edge处理模块不可用，跳过性能测试")
            return
        
        # 创建处理器
        processor = EdgeMapProcessor().to(self.device)
        fusion_module = EdgeFusionModule().to(self.device)
        
        # 测试配置
        test_configs = [
            {"size": (256, 256), "batch_size": 1, "iterations": 10},
            {"size": (512, 512), "batch_size": 1, "iterations": 10},
            {"size": (1024, 1024), "batch_size": 1, "iterations": 5},
            {"size": (512, 512), "batch_size": 2, "iterations": 5},
            {"size": (512, 512), "batch_size": 4, "iterations": 3},
        ]
        
        results = []
        
        for config in test_configs:
            size = config["size"]
            batch_size = config["batch_size"]
            iterations = config["iterations"]
            
            print(f"\n测试配置: 尺寸={size}, batch_size={batch_size}, 迭代={iterations}")
            
            # 准备测试数据
            edge_maps = []
            for _ in range(batch_size):
                edge_map_np = self.create_synthetic_edge_map(size)
                edge_tensor = torch.from_numpy(edge_map_np).float().permute(2, 0, 1) / 255.0
                edge_maps.append(edge_tensor)
            
            edge_batch = torch.stack(edge_maps).to(self.device)
            
            # 预热
            with torch.no_grad():
                _ = processor(edge_batch)
            
            # 测试Edge处理器性能
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(iterations):
                with torch.no_grad():
                    edge_features = processor(edge_batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            processor_time = (end_time - start_time) / iterations
            
            # 测试融合性能
            unet_input = torch.randn(batch_size, 4, 64, 64).to(self.device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(iterations):
                with torch.no_grad():
                    fused = fusion_module(unet_input, edge_features)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            fusion_time = (end_time - start_time) / iterations
            
            # 记录结果
            result = {
                "size": size,
                "batch_size": batch_size,
                "processor_time": processor_time,
                "fusion_time": fusion_time,
                "total_time": processor_time + fusion_time,
                "memory_usage": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            }
            results.append(result)
            
            print(f"  Edge处理器时间: {processor_time:.4f}s")
            print(f"  特征融合时间: {fusion_time:.4f}s")
            print(f"  总时间: {result['total_time']:.4f}s")
            print(f"  内存使用: {result['memory_usage']:.2f}MB")
            
            # 清理内存
            del edge_batch, edge_features, unet_input, fused
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 保存性能结果
        self.save_performance_results(results)
        print("✓ 性能基准测试完成")
    
    def save_performance_results(self, results):
        """保存性能测试结果"""
        results_file = self.output_dir / "performance_results.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Edge Map处理性能测试结果\n")
            f.write("="*50 + "\n\n")
            
            for result in results:
                f.write(f"尺寸: {result['size']}\n")
                f.write(f"Batch Size: {result['batch_size']}\n")
                f.write(f"Edge处理器时间: {result['processor_time']:.4f}s\n")
                f.write(f"特征融合时间: {result['fusion_time']:.4f}s\n")
                f.write(f"总时间: {result['total_time']:.4f}s\n")
                f.write(f"内存使用: {result['memory_usage']:.2f}MB\n")
                f.write("-" * 30 + "\n")
        
        print(f"性能结果已保存到: {results_file}")
    
    def visualize_edge_maps(self):
        """可视化edge maps"""
        print("\n" + "="*50)
        print("可视化Edge Maps")
        print("="*50)
        
        # 创建不同图案的edge maps
        patterns = ["geometric", "texture", "mixed"]
        fig, axes = plt.subplots(1, len(patterns), figsize=(15, 5))
        
        if len(patterns) == 1:
            axes = [axes]
        
        for i, pattern in enumerate(patterns):
            edge_map = self.create_synthetic_edge_map((512, 512), pattern)
            
            # 转换为RGB用于显示
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(edge_rgb)
            axes[i].set_title(f'{pattern.capitalize()} Pattern')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "edge_maps_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Edge maps可视化已保存")
    
    def run_all_tests(self, image_dir=None):
        """运行所有测试"""
        print("开始Edge Map综合测试")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # 基础功能测试
            self.test_edge_map_generation()
            self.test_edge_processor()
            self.test_edge_fusion()
            
            # 真实图像测试
            self.test_with_real_images(image_dir)
            
            # 性能测试
            self.performance_benchmark()
            
            # 可视化
            self.visualize_edge_maps()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 所有测试完成!")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"结果保存在: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Edge Map综合测试脚本")
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备")
    parser.add_argument("--output_dir", type=str, default="edge_test_results",
                       help="输出目录")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="真实图像测试目录")
    parser.add_argument("--test_type", type=str, default="all",
                       choices=["all", "generation", "processor", "fusion", "real", "performance", "visualize"],
                       help="测试类型")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 创建测试器
    tester = EdgeMapTester(device=device, output_dir=args.output_dir)
    
    # 运行指定测试
    if args.test_type == "all":
        success = tester.run_all_tests(args.image_dir)
    elif args.test_type == "generation":
        tester.test_edge_map_generation()
        success = True
    elif args.test_type == "processor":
        tester.test_edge_processor()
        success = True
    elif args.test_type == "fusion":
        tester.test_edge_fusion()
        success = True
    elif args.test_type == "real":
        tester.test_with_real_images(args.image_dir)
        success = True
    elif args.test_type == "performance":
        tester.performance_benchmark()
        success = True
    elif args.test_type == "visualize":
        tester.visualize_edge_maps()
        success = True
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
