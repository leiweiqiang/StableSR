#!/usr/bin/env python3
"""
Edge Map性能测试脚本
专门用于测试edge map处理的性能指标

使用方法:
python test_edge_map_performance.py
python test_edge_map_performance.py --device cuda --batch_sizes 1,2,4,8
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import cv2
import psutil
import gc
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EdgeMapPerformanceTester:
    """Edge Map性能测试器"""
    
    def __init__(self, device=None, output_dir="performance_test_results"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        
        # 系统信息
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "device": str(self.device),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
            "python_version": sys.version,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            })
        else:
            info["cuda_available"] = False
        
        return info
    
    def create_test_edge_map(self, size=(512, 512)):
        """创建测试用的edge map"""
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 创建复杂的测试图案
        cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
        cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
        cv2.line(img, (100, 300), (400, 100), (200, 200, 200), 3)
        cv2.ellipse(img, (250, 150), (60, 40), 45, 0, 360, (180, 180, 180), -1)
        
        # 添加纹理
        for i in range(0, h, 50):
            for j in range(0, w, 50):
                if (i + j) % 100 == 0:
                    cv2.rectangle(img, (j, i), (j+20, i+20), (100, 100, 100), -1)
        
        # 转换为edge map
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_3ch
    
    def benchmark_edge_generation(self, sizes=None, iterations=10):
        """基准测试edge map生成"""
        if sizes is None:
            sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        print("\n" + "="*50)
        print("Edge Map生成性能测试")
        print("="*50)
        
        results = []
        
        for size in sizes:
            print(f"\n测试尺寸: {size}")
            
            # 创建测试图像
            test_image = self.create_test_edge_map(size)
            
            # 测试不同方法
            methods = ["canny", "sobel", "laplacian"]
            
            for method in methods:
                print(f"  测试方法: {method}")
                
                times = []
                memory_usage = []
                
                for i in range(iterations):
                    # 记录开始时间和内存
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        start_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    
                    # 执行edge检测
                    if method == "canny":
                        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
                        edges = cv2.Canny(blurred, 100, 200)
                    elif method == "sobel":
                        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                        edges = np.sqrt(sobel_x**2 + sobel_y**2)
                        edges = np.uint8(edges / edges.max() * 255)
                    elif method == "laplacian":
                        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Laplacian(gray, cv2.CV_64F)
                        edges = np.uint8(np.absolute(edges))
                    
                    # 记录结束时间和内存
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        end_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                    
                    # 计算时间和内存使用
                    elapsed_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    times.append(elapsed_time)
                    memory_usage.append(memory_delta)
                    
                    # 清理内存
                    del edges
                    gc.collect()
                
                # 计算统计信息
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                avg_memory = np.mean(memory_usage)
                
                result = {
                    "size": size,
                    "method": method,
                    "iterations": iterations,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "avg_memory": avg_memory,
                    "pixels_per_second": (size[0] * size[1]) / avg_time
                }
                
                results.append(result)
                
                print(f"    平均时间: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"    时间范围: [{min_time:.4f}s, {max_time:.4f}s]")
                print(f"    平均内存: {avg_memory:.2f}MB")
                print(f"    处理速度: {result['pixels_per_second']:.0f} pixels/s")
        
        return results
    
    def benchmark_edge_processor(self, batch_sizes=None, sizes=None, iterations=5):
        """基准测试edge处理器"""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        if sizes is None:
            sizes = [(256, 256), (512, 512)]
        
        print("\n" + "="*50)
        print("Edge处理器性能测试")
        print("="*50)
        
        try:
            from ldm.modules.diffusionmodules.edge_processor import EdgeMapProcessor
        except ImportError:
            print("⚠️  Edge处理器模块不可用，跳过此测试")
            return []
        
        results = []
        
        # 创建处理器
        processor = EdgeMapProcessor(
            input_channels=3,
            output_channels=4,
            target_size=64,
            use_checkpoint=False
        ).to(self.device)
        
        for size in sizes:
            for batch_size in batch_sizes:
                print(f"\n测试配置: 尺寸={size}, batch_size={batch_size}")
                
                # 准备测试数据
                edge_maps = []
                for _ in range(batch_size):
                    edge_map_np = self.create_test_edge_map(size)
                    edge_tensor = torch.from_numpy(edge_map_np).float().permute(2, 0, 1) / 255.0
                    edge_maps.append(edge_tensor)
                
                edge_batch = torch.stack(edge_maps).to(self.device)
                
                # 预热
                with torch.no_grad():
                    _ = processor(edge_batch)
                
                # 性能测试
                times = []
                memory_usage = []
                
                for i in range(iterations):
                    # 记录开始时间和内存
                    start_time = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        start_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    
                    # 执行处理
                    with torch.no_grad():
                        output = processor(edge_batch)
                    
                    # 记录结束时间和内存
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        end_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    
                    end_time = time.time()
                    
                    # 计算时间和内存使用
                    elapsed_time = end_time - start_time
                    gpu_memory_delta = (end_gpu_memory - start_gpu_memory) if torch.cuda.is_available() else 0
                    
                    times.append(elapsed_time)
                    memory_usage.append(gpu_memory_delta)
                    
                    # 清理内存
                    del output
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 计算统计信息
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                avg_memory = np.mean(memory_usage)
                
                result = {
                    "size": size,
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "avg_memory": avg_memory,
                    "throughput": batch_size / avg_time,  # samples per second
                    "pixels_per_second": (batch_size * size[0] * size[1]) / avg_time
                }
                
                results.append(result)
                
                print(f"  平均时间: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"  时间范围: [{min_time:.4f}s, {max_time:.4f}s]")
                print(f"  平均GPU内存: {avg_memory:.2f}MB")
                print(f"  吞吐量: {result['throughput']:.2f} samples/s")
                print(f"  处理速度: {result['pixels_per_second']:.0f} pixels/s")
                
                # 清理内存
                del edge_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def benchmark_memory_usage(self, max_size=(2048, 2048)):
        """内存使用基准测试"""
        print("\n" + "="*50)
        print("内存使用基准测试")
        print("="*50)
        
        results = []
        sizes = [(256, 256), (512, 512), (1024, 1024), (1536, 1536), (2048, 2048)]
        
        for size in sizes:
            if size[0] > max_size[0] or size[1] > max_size[1]:
                continue
                
            print(f"\n测试尺寸: {size}")
            
            try:
                # 测试edge map生成内存使用
                start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                
                edge_map = self.create_test_edge_map(size)
                
                end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                memory_usage = end_memory - start_memory
                
                # 测试tensor转换内存使用
                edge_tensor = torch.from_numpy(edge_map).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                edge_tensor = edge_tensor.to(self.device)
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                else:
                    gpu_memory = 0
                
                result = {
                    "size": size,
                    "pixels": size[0] * size[1],
                    "cpu_memory": memory_usage,
                    "gpu_memory": gpu_memory,
                    "memory_per_pixel": memory_usage / (size[0] * size[1]) * 1024  # KB per pixel
                }
                
                results.append(result)
                
                print(f"  CPU内存使用: {memory_usage:.2f}MB")
                print(f"  GPU内存使用: {gpu_memory:.2f}MB")
                print(f"  每像素内存: {result['memory_per_pixel']:.4f}KB")
                
                # 清理内存
                del edge_map, edge_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ 测试失败: {e}")
        
        return results
    
    def save_results(self, edge_gen_results, processor_results, memory_results):
        """保存测试结果"""
        # 保存JSON结果
        all_results = {
            "system_info": self.system_info,
            "timestamp": datetime.now().isoformat(),
            "edge_generation": edge_gen_results,
            "edge_processor": processor_results,
            "memory_usage": memory_results
        }
        
        json_path = self.output_dir / "performance_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        report_path = self.output_dir / "performance_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Edge Map性能测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {self.system_info['device']}\n")
            f.write(f"CPU核心数: {self.system_info['cpu_count']}\n")
            f.write(f"总内存: {self.system_info['memory_total']:.1f}GB\n")
            
            if self.system_info['cuda_available']:
                f.write(f"CUDA设备: {self.system_info['cuda_device_name']}\n")
                f.write(f"CUDA内存: {self.system_info['cuda_memory_total']:.1f}GB\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Edge Map生成性能\n")
            f.write("=" * 50 + "\n")
            
            for result in edge_gen_results:
                f.write(f"尺寸: {result['size']}, 方法: {result['method']}\n")
                f.write(f"  平均时间: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s\n")
                f.write(f"  处理速度: {result['pixels_per_second']:.0f} pixels/s\n")
                f.write(f"  内存使用: {result['avg_memory']:.2f}MB\n\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Edge处理器性能\n")
            f.write("=" * 50 + "\n")
            
            for result in processor_results:
                f.write(f"尺寸: {result['size']}, Batch: {result['batch_size']}\n")
                f.write(f"  平均时间: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s\n")
                f.write(f"  吞吐量: {result['throughput']:.2f} samples/s\n")
                f.write(f"  处理速度: {result['pixels_per_second']:.0f} pixels/s\n")
                f.write(f"  GPU内存: {result['avg_memory']:.2f}MB\n\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("内存使用分析\n")
            f.write("=" * 50 + "\n")
            
            for result in memory_results:
                f.write(f"尺寸: {result['size']}\n")
                f.write(f"  CPU内存: {result['cpu_memory']:.2f}MB\n")
                f.write(f"  GPU内存: {result['gpu_memory']:.2f}MB\n")
                f.write(f"  每像素内存: {result['memory_per_pixel']:.4f}KB\n\n")
        
        print(f"\n结果已保存:")
        print(f"  JSON: {json_path}")
        print(f"  报告: {report_path}")
    
    def run_all_tests(self, batch_sizes=None, max_size=(2048, 2048)):
        """运行所有性能测试"""
        print("开始Edge Map性能测试")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Edge map生成性能测试
            edge_gen_results = self.benchmark_edge_generation()
            
            # Edge处理器性能测试
            processor_results = self.benchmark_edge_processor(batch_sizes)
            
            # 内存使用测试
            memory_results = self.benchmark_memory_usage(max_size)
            
            # 保存结果
            self.save_results(edge_gen_results, processor_results, memory_results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "=" * 60)
            print("🎉 性能测试完成!")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"结果保存在: {self.output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Edge Map性能测试脚本")
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备")
    parser.add_argument("--output_dir", type=str, default="performance_test_results",
                       help="输出目录")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8",
                       help="测试的batch sizes，用逗号分隔")
    parser.add_argument("--max_size", type=str, default="2048,2048",
                       help="最大测试尺寸，格式: width,height")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 解析batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    # 解析最大尺寸
    max_size = tuple(int(x.strip()) for x in args.max_size.split(','))
    
    # 创建测试器
    tester = EdgeMapPerformanceTester(device=device, output_dir=args.output_dir)
    
    # 运行测试
    success = tester.run_all_tests(batch_sizes=batch_sizes, max_size=max_size)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
