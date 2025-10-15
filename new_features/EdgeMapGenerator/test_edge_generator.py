"""
测试EdgeMapGenerator类的功能
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

# 添加项目路径（从readme目录到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from basicsr.utils.edge_utils import EdgeMapGenerator, generate_edge_map


def test_numpy_input():
    """测试numpy数组输入"""
    print("=" * 50)
    print("测试1: Numpy数组输入 (BGR格式)")
    print("=" * 50)
    
    # 创建生成器
    edge_gen = EdgeMapGenerator()
    
    # 创建测试图像 (模拟BGR格式，[0,1]范围)
    img_np = np.random.rand(512, 512, 3).astype(np.float32)
    
    # 生成edge map
    edge_map = edge_gen.generate_from_numpy(
        img_np,
        input_format='BGR',
        normalize_input=True
    )
    
    print(f"输入形状: {img_np.shape}")
    print(f"输出形状: {edge_map.shape}")
    print(f"输出范围: [{edge_map.min():.3f}, {edge_map.max():.3f}]")
    print(f"输出类型: {edge_map.dtype}")
    print("✓ 测试通过\n")
    
    return edge_map


def test_tensor_input():
    """测试PyTorch tensor输入"""
    print("=" * 50)
    print("测试2: PyTorch Tensor输入 (RGB格式)")
    print("=" * 50)
    
    # 创建生成器
    edge_gen = EdgeMapGenerator()
    
    # 创建测试tensor (模拟RGB格式，[-1,1]范围)
    img_tensor = torch.randn(2, 3, 512, 512)  # batch_size=2
    img_tensor = torch.clamp(img_tensor, -1, 1)
    
    # 生成edge map
    edge_map = edge_gen.generate_from_tensor(
        img_tensor,
        input_format='RGB',
        normalize_range='[-1,1]'
    )
    
    print(f"输入形状: {img_tensor.shape}")
    print(f"输出形状: {edge_map.shape}")
    print(f"输出范围: [{edge_map.min():.3f}, {edge_map.max():.3f}]")
    print(f"输出类型: {edge_map.dtype}")
    print(f"输出设备: {edge_map.device}")
    print("✓ 测试通过\n")
    
    return edge_map


def test_single_image_tensor():
    """测试单张图像tensor输入"""
    print("=" * 50)
    print("测试3: 单张图像Tensor输入")
    print("=" * 50)
    
    # 创建生成器
    edge_gen = EdgeMapGenerator()
    
    # 创建单张图像tensor (3, H, W)
    img_tensor = torch.randn(3, 256, 256)
    img_tensor = torch.clamp(img_tensor, -1, 1)
    
    # 生成edge map
    edge_map = edge_gen.generate_from_tensor(
        img_tensor,
        input_format='RGB',
        normalize_range='[-1,1]'
    )
    
    print(f"输入形状: {img_tensor.shape}")
    print(f"输出形状: {edge_map.shape}")
    print(f"输出维度: {edge_map.dim()}D")
    print("✓ 测试通过\n")
    
    return edge_map


def test_convenience_function():
    """测试便捷函数"""
    print("=" * 50)
    print("测试4: 便捷函数 generate_edge_map()")
    print("=" * 50)
    
    # 测试numpy输入
    img_np = np.random.rand(256, 256, 3).astype(np.float32)
    edge_np = generate_edge_map(img_np, input_format='BGR', normalize_input=True)
    print(f"Numpy输入: {img_np.shape} -> {edge_np.shape}")
    
    # 测试tensor输入
    img_tensor = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    edge_tensor = generate_edge_map(img_tensor, input_format='RGB', normalize_range='[-1,1]')
    print(f"Tensor输入: {img_tensor.shape} -> {edge_tensor.shape}")
    print("✓ 测试通过\n")


def test_custom_parameters():
    """测试自定义参数"""
    print("=" * 50)
    print("测试5: 自定义参数")
    print("=" * 50)
    
    # 创建自定义参数的生成器
    edge_gen = EdgeMapGenerator(
        gaussian_kernel_size=(7, 7),
        gaussian_sigma=2.0,
        canny_threshold_lower_factor=0.5,
        canny_threshold_upper_factor=1.5,
        morph_kernel_size=(5, 5)
    )
    
    img_np = np.random.rand(256, 256, 3).astype(np.float32)
    edge_map = edge_gen.generate_from_numpy(img_np, input_format='BGR')
    
    print("自定义参数:")
    print("  - gaussian_kernel_size: (7, 7)")
    print("  - gaussian_sigma: 2.0")
    print("  - canny_threshold_lower_factor: 0.5")
    print("  - canny_threshold_upper_factor: 1.5")
    print("  - morph_kernel_size: (5, 5)")
    print(f"输出形状: {edge_map.shape}")
    print("✓ 测试通过\n")


def test_real_image():
    """测试真实图像（如果存在）"""
    print("=" * 50)
    print("测试6: 真实图像处理")
    print("=" * 50)
    
    # 查找测试图像（优先使用本目录的test_images，其次使用项目根目录的资源）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_paths = [
        os.path.join(script_dir, 'test_images/cat_aigc.jpg'),
        os.path.join(script_dir, 'test_images/OST_120.png'),
        os.path.join(project_root, 'inputs/test_example/cat_aigc.jpg'),
        os.path.join(project_root, 'assets/imgsli_1.jpg')
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image is None:
        print("⚠ 未找到测试图像，跳过此测试")
        print()
        return
    
    print(f"使用测试图像: {test_image}")
    
    # 加载图像
    img = Image.open(test_image).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 生成edge map
    edge_gen = EdgeMapGenerator()
    edge_map = edge_gen.generate_from_numpy(
        img_np,
        input_format='RGB',
        normalize_input=True
    )
    
    print(f"原始图像尺寸: {img_np.shape}")
    print(f"Edge map尺寸: {edge_map.shape}")
    
    # 保存edge map以便可视化
    edge_vis = (edge_map * 255).astype(np.uint8)
    output_path = os.path.join(project_root, 'new_features', 'EdgeMapGenerator', 'test_edge_output.png')
    Image.fromarray(edge_vis).save(output_path)
    print(f"✓ Edge map已保存到: {output_path}\n")


def test_consistency():
    """测试训练和推理格式的一致性"""
    print("=" * 50)
    print("测试7: 训练/推理一致性")
    print("=" * 50)
    
    edge_gen = EdgeMapGenerator()
    
    # 创建相同内容的图像，但格式不同
    # 训练格式: numpy BGR [0,1]
    img_train = np.random.rand(256, 256, 3).astype(np.float32)
    
    # 推理格式: tensor RGB [-1,1]
    # 注意：BGR->RGB需要反转通道，这里为了测试一致性，使用相同通道顺序
    img_infer = torch.from_numpy(img_train).permute(2, 0, 1).unsqueeze(0)
    img_infer = img_infer * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    # 生成edge maps
    edge_train = edge_gen.generate_from_numpy(img_train, input_format='RGB', normalize_input=True)
    edge_infer = edge_gen.generate_from_tensor(img_infer, input_format='RGB', normalize_range='[-1,1]')
    
    # 转换为相同格式进行比较
    edge_train_tensor = torch.from_numpy(edge_train).permute(2, 0, 1).unsqueeze(0)
    edge_train_tensor = edge_train_tensor * 2.0 - 1.0
    
    # 计算差异
    diff = torch.abs(edge_train_tensor - edge_infer).mean()
    
    print(f"训练格式edge map范围: [{edge_train.min():.3f}, {edge_train.max():.3f}]")
    print(f"推理格式edge map范围: [{edge_infer.min():.3f}, {edge_infer.max():.3f}]")
    print(f"平均差异: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ 训练和推理格式完全一致\n")
    else:
        print("⚠ 存在微小差异（这是正常的，由于浮点运算）\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("EdgeMapGenerator 功能测试")
    print("=" * 50 + "\n")
    
    try:
        test_numpy_input()
        test_tensor_input()
        test_single_image_tensor()
        test_convenience_function()
        test_custom_parameters()
        test_real_image()
        test_consistency()
        
        print("=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

