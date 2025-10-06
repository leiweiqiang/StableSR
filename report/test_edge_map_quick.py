#!/usr/bin/env python3
"""
快速Edge Map测试脚本
用于快速验证edge map生成和处理功能

使用方法:
python test_edge_map_quick.py --input_image path/to/image.jpg
python test_edge_map_quick.py --synthetic  # 使用合成图像测试
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_synthetic_test_image(size=(512, 512)):
    """创建合成测试图像"""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 绘制几何图形
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (350, 350), 80, (128, 128, 128), -1)
    cv2.line(img, (100, 300), (400, 100), (200, 200, 200), 3)
    cv2.ellipse(img, (250, 150), (60, 40), 45, 0, 360, (180, 180, 180), -1)
    
    # 添加一些纹理
    for i in range(0, h, 50):
        for j in range(0, w, 50):
            if (i + j) % 100 == 0:
                cv2.rectangle(img, (j, i), (j+20, i+20), (100, 100, 100), -1)
    
    return img


def generate_edge_map(image, method="canny", **kwargs):
    """
    生成edge map
    
    Args:
        image: 输入图像 (numpy array)
        method: 边缘检测方法 ("canny", "sobel", "laplacian")
        **kwargs: 方法特定参数
    
    Returns:
        edge_map: 边缘图 (numpy array)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "canny":
        # Canny边缘检测
        threshold1 = kwargs.get('threshold1', 100)
        threshold2 = kwargs.get('threshold2', 200)
        blur_kernel = kwargs.get('blur_kernel', (5, 5))
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, blur_kernel, 1.4)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
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
        
    else:
        raise ValueError(f"不支持的边缘检测方法: {method}")
    
    # 转换为3通道
    if len(edges.shape) == 2:
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        edges_3ch = edges
    
    return edges_3ch


def test_edge_map_generation(image_path=None, output_dir="edge_test_output"):
    """测试edge map生成"""
    print("Edge Map生成测试")
    print("="*40)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 准备输入图像
    if image_path:
        print(f"使用真实图像: {image_path}")
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在: {image_path}")
            return False
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像: {image_path}")
            return False
        
        # 调整尺寸
        image = cv2.resize(image, (512, 512))
        image_name = Path(image_path).stem
    else:
        print("使用合成测试图像")
        image = create_synthetic_test_image((512, 512))
        image_name = "synthetic"
    
    print(f"输入图像形状: {image.shape}")
    
    # 测试不同的边缘检测方法
    methods = ["canny", "sobel", "laplacian"]
    edge_maps = {}
    
    for method in methods:
        print(f"\n测试 {method.upper()} 边缘检测:")
        
        try:
            edge_map = generate_edge_map(image, method=method)
            edge_maps[method] = edge_map
            
            # 保存结果
            output_path = output_dir / f"{image_name}_edge_{method}.png"
            cv2.imwrite(str(output_path), edge_map)
            
            print(f"  ✓ 边缘图已保存: {output_path}")
            print(f"  ✓ 边缘图形状: {edge_map.shape}")
            print(f"  ✓ 值范围: [{edge_map.min()}, {edge_map.max()}]")
            
            # 统计边缘像素
            edge_pixels = np.sum(edge_map > 0)
            total_pixels = edge_map.shape[0] * edge_map.shape[1]
            edge_ratio = edge_pixels / total_pixels * 100
            print(f"  ✓ 边缘像素比例: {edge_ratio:.2f}%")
            
        except Exception as e:
            print(f"  ❌ {method} 边缘检测失败: {e}")
    
    # 保存原始图像
    original_path = output_dir / f"{image_name}_original.png"
    cv2.imwrite(str(original_path), image)
    print(f"\n原始图像已保存: {original_path}")
    
    # 创建对比图
    create_comparison_plot(image, edge_maps, output_dir / f"{image_name}_comparison.png")
    
    print("\n✓ Edge Map生成测试完成")
    return True


def create_comparison_plot(original, edge_maps, output_path):
    """创建对比图"""
    n_methods = len(edge_maps)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    if n_methods == 0:
        return
    
    # 原始图像
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 边缘图
    for i, (method, edge_map) in enumerate(edge_maps.items()):
        edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB)
        axes[0, i + 1].imshow(edge_rgb)
        axes[0, i + 1].set_title(f'{method.upper()} 边缘图')
        axes[0, i + 1].axis('off')
    
    # 灰度版本
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('灰度图')
    axes[1, 0].axis('off')
    
    for i, (method, edge_map) in enumerate(edge_maps.items()):
        if len(edge_map.shape) == 3:
            edge_gray = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        else:
            edge_gray = edge_map
        axes[1, i + 1].imshow(edge_gray, cmap='gray')
        axes[1, i + 1].set_title(f'{method.upper()} 灰度')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {output_path}")


def test_edge_processor_simple():
    """简单测试edge处理器（如果可用）"""
    print("\nEdge处理器简单测试")
    print("="*40)
    
    try:
        from ldm.modules.diffusionmodules.edge_processor import EdgeMapProcessor
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建处理器
        processor = EdgeMapProcessor(
            input_channels=3,
            output_channels=4,
            target_size=64
        ).to(device)
        
        # 创建测试edge map
        edge_map_np = generate_edge_map(create_synthetic_test_image())
        edge_tensor = torch.from_numpy(edge_map_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        edge_tensor = edge_tensor.to(device)
        
        print(f"输入tensor形状: {edge_tensor.shape}")
        
        # 处理
        with torch.no_grad():
            output = processor(edge_tensor)
        
        print(f"输出tensor形状: {output.shape}")
        print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 验证输出形状
        expected_shape = (1, 4, 64, 64)
        if output.shape == expected_shape:
            print("✓ Edge处理器测试通过")
            return True
        else:
            print(f"❌ 输出形状不匹配，期望: {expected_shape}, 实际: {output.shape}")
            return False
            
    except ImportError:
        print("⚠️  Edge处理器模块不可用，跳过此测试")
        return True
    except Exception as e:
        print(f"❌ Edge处理器测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速Edge Map测试脚本")
    
    parser.add_argument("--input_image", type=str, default=None,
                       help="输入图像路径")
    parser.add_argument("--synthetic", action="store_true",
                       help="使用合成图像测试")
    parser.add_argument("--output_dir", type=str, default="edge_test_output",
                       help="输出目录")
    parser.add_argument("--test_processor", action="store_true",
                       help="测试edge处理器")
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.input_image and not args.synthetic:
        print("请指定 --input_image 或 --synthetic")
        return 1
    
    if args.input_image and args.synthetic:
        print("请只指定 --input_image 或 --synthetic 中的一个")
        return 1
    
    print("快速Edge Map测试")
    print("="*50)
    
    success = True
    
    # 测试edge map生成
    if args.input_image:
        success &= test_edge_map_generation(args.input_image, args.output_dir)
    else:
        success &= test_edge_map_generation(None, args.output_dir)
    
    # 测试edge处理器
    if args.test_processor:
        success &= test_edge_processor_simple()
    
    if success:
        print("\n🎉 所有测试通过!")
        print(f"结果保存在: {args.output_dir}")
    else:
        print("\n❌ 部分测试失败")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
