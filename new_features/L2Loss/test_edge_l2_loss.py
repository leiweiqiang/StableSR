#!/usr/bin/env python3
"""
测试Edge L2 Loss计算器

这个脚本用于测试EdgeL2LossCalculator类的功能
"""

import os
import sys
import numpy as np
import cv2
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from basicsr.metrics.edge_l2_loss import EdgeL2LossCalculator


def test_from_arrays():
    """测试从numpy数组计算Edge L2 Loss"""
    print("=" * 60)
    print("Test 1: Calculate Edge L2 Loss from numpy arrays")
    print("=" * 60)
    
    # 创建两张测试图片
    # 图片1: 简单的白色矩形
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)
    
    # 图片2: 类似但稍有不同的矩形
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img2, (60, 60), (210, 210), (255, 255, 255), -1)
    
    # 初始化计算器
    calculator = EdgeL2LossCalculator()
    
    # 计算loss
    loss = calculator.calculate_from_arrays(img1, img2, input_format='BGR')
    
    print(f"✓ Edge L2 Loss between img1 and img2: {loss:.6f}")
    print()
    
    # 测试相同图片
    loss_same = calculator.calculate_from_arrays(img1, img1, input_format='BGR')
    print(f"✓ Edge L2 Loss for identical images: {loss_same:.6f}")
    print("  (应该接近0)")
    print()
    
    return True


def test_from_files():
    """测试从文件计算Edge L2 Loss"""
    print("=" * 60)
    print("Test 2: Calculate Edge L2 Loss from image files")
    print("=" * 60)
    
    # 创建临时测试图片
    temp_dir = "/tmp/edge_l2_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 创建并保存测试图片
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)
    img1_path = os.path.join(temp_dir, "test_img1.png")
    cv2.imwrite(img1_path, img1)
    
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img2, (60, 60), (210, 210), (255, 255, 255), -1)
    img2_path = os.path.join(temp_dir, "test_img2.png")
    cv2.imwrite(img2_path, img2)
    
    # 初始化计算器
    calculator = EdgeL2LossCalculator()
    
    # 计算loss
    loss = calculator.calculate_from_files(img1_path, img2_path)
    
    print(f"✓ Edge L2 Loss from files: {loss:.6f}")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print()
    
    # 清理临时文件
    os.remove(img1_path)
    os.remove(img2_path)
    os.rmdir(temp_dir)
    
    return True


def test_from_tensors():
    """测试从PyTorch tensor计算Edge L2 Loss"""
    print("=" * 60)
    print("Test 3: Calculate Edge L2 Loss from PyTorch tensors")
    print("=" * 60)
    
    # 创建测试tensor (C, H, W) 格式，范围[-1, 1]
    img1_np = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img1_np, (50, 50), (200, 200), (255, 255, 255), -1)
    img1_tensor = torch.from_numpy(img1_np.transpose(2, 0, 1)).float() / 255.0
    img1_tensor = img1_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    img2_np = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img2_np, (60, 60), (210, 210), (255, 255, 255), -1)
    img2_tensor = torch.from_numpy(img2_np.transpose(2, 0, 1)).float() / 255.0
    img2_tensor = img2_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    # 初始化计算器
    calculator = EdgeL2LossCalculator()
    
    # 计算loss
    loss = calculator.calculate_from_tensors(img1_tensor, img2_tensor, normalize_range='[-1,1]')
    
    print(f"✓ Edge L2 Loss from tensors: {loss:.6f}")
    print(f"  Tensor shape: {img1_tensor.shape}")
    print()
    
    return True


def test_convenience_function():
    """测试便捷调用方法"""
    print("=" * 60)
    print("Test 4: Test convenience __call__ method")
    print("=" * 60)
    
    # 创建测试图片
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)
    
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img2, (60, 60), (210, 210), (255, 255, 255), -1)
    
    # 初始化计算器
    calculator = EdgeL2LossCalculator()
    
    # 使用__call__方法
    loss = calculator(img1, img2, input_format='BGR')
    
    print(f"✓ Edge L2 Loss using __call__: {loss:.6f}")
    print()
    
    return True


def test_different_sizes():
    """测试不同尺寸的图片"""
    print("=" * 60)
    print("Test 5: Test with different image sizes")
    print("=" * 60)
    
    # 创建不同尺寸的图片
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)
    
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img2, (100, 100), (400, 400), (255, 255, 255), -1)
    
    # 初始化计算器
    calculator = EdgeL2LossCalculator()
    
    # 计算loss (应该自动resize)
    loss = calculator.calculate_from_arrays(img1, img2, input_format='BGR')
    
    print(f"✓ Edge L2 Loss with different sizes:")
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    print(f"  Loss: {loss:.6f}")
    print("  (img2 will be resized to match img1)")
    print()
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("EdgeL2LossCalculator Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Test from numpy arrays", test_from_arrays),
        ("Test from files", test_from_files),
        ("Test from tensors", test_from_tensors),
        ("Test convenience function", test_convenience_function),
        ("Test different sizes", test_different_sizes),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

