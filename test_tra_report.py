#!/usr/bin/env python3
"""
TraReport测试脚本
用于测试TraReport类的各个功能模块
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
from PIL import Image

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_images(test_dir, num_images=5, size=(256, 256)):
    """创建测试图片"""
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(num_images):
        # 创建随机图片
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(test_dir, f"{i+1:03d}.png"))
    
    print(f"Created {num_images} test images in {test_dir}")

def create_lr_images(hr_dir, lr_dir, scale_factor=4):
    """创建低分辨率图片用于测试"""
    os.makedirs(lr_dir, exist_ok=True)
    
    for img_file in os.listdir(hr_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            hr_path = os.path.join(hr_dir, img_file)
            lr_path = os.path.join(lr_dir, img_file)
            
            # 加载高分辨率图片
            img = Image.open(hr_path)
            
            # 缩放到低分辨率
            lr_size = (img.size[0] // scale_factor, img.size[1] // scale_factor)
            lr_img = img.resize(lr_size, Image.LANCZOS)
            
            # 保存低分辨率图片
            lr_img.save(lr_path)
    
    print(f"Created LR images in {lr_dir}")

def test_file_matching():
    """测试文件匹配功能"""
    print("=== 测试文件匹配功能 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试目录
        hr_dir = os.path.join(temp_dir, "hr")
        lr_dir = os.path.join(temp_dir, "lr")
        
        # 创建测试图片
        create_test_images(hr_dir, num_images=3)
        create_lr_images(hr_dir, lr_dir, scale_factor=4)
        
        # 测试TraReport的文件匹配功能
        try:
            from tra_report import TraReport
            
            # 创建一个模拟的TraReport实例（不需要实际模型）
            class MockTraReport(TraReport):
                def __init__(self, gt_dir, val_dir, model_path):
                    self.gt_dir = Path(gt_dir)
                    self.val_dir = Path(val_dir)
                    self.model_path = model_path
                
                def _find_matching_files(self):
                    """重写方法以使用父类实现"""
                    return super()._find_matching_files()
            
            mock_report = MockTraReport(hr_dir, lr_dir, "dummy_model.ckpt")
            matching_pairs = mock_report._find_matching_files()
            
            print(f"找到匹配的文件对数量: {len(matching_pairs)}")
            for val_file, gt_file in matching_pairs:
                print(f"  {val_file} -> {gt_file}")
            
            if len(matching_pairs) > 0:
                print("✅ 文件匹配功能测试通过")
                return True
            else:
                print("❌ 文件匹配功能测试失败")
                return False
                
        except Exception as e:
            print(f"❌ 文件匹配功能测试出错: {str(e)}")
            return False

def test_psnr_calculation():
    """测试PSNR计算功能"""
    print("\n=== 测试PSNR计算功能 ===")
    
    try:
        from tra_report import TraReport
        import torch
        
        # 创建测试图片
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = img1.copy()  # 完全相同的图片，PSNR应该为无穷大
        
        # 创建模拟实例
        class MockTraReport(TraReport):
            def __init__(self):
                pass
        
        mock_report = MockTraReport()
        psnr = mock_report._calculate_psnr(img1, img2)
        
        print(f"相同图片的PSNR: {psnr}")
        
        if psnr == float('inf'):
            print("✅ PSNR计算功能测试通过")
            return True
        else:
            print("❌ PSNR计算功能测试失败")
            return False
            
    except Exception as e:
        print(f"❌ PSNR计算功能测试出错: {str(e)}")
        return False

def test_json_output():
    """测试JSON输出功能"""
    print("\n=== 测试JSON输出功能 ===")
    
    try:
        from tra_report import TraReport
        
        # 创建测试结果
        test_results = {
            "model_path": "test_model.ckpt",
            "total_files": 2,
            "results": [
                {
                    "val_file": "test1.png",
                    "gt_file": "test1.png",
                    "psnr": 28.5
                },
                {
                    "val_file": "test2.png", 
                    "gt_file": "test2.png",
                    "psnr": 30.2
                }
            ],
            "summary": {
                "average_psnr": 29.35,
                "min_psnr": 28.5,
                "max_psnr": 30.2
            }
        }
        
        # 创建模拟实例
        class MockTraReport(TraReport):
            def __init__(self):
                pass
        
        mock_report = MockTraReport()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            mock_report.save_results(test_results, temp_file)
            
            # 验证文件内容
            with open(temp_file, 'r') as f:
                loaded_results = json.load(f)
            
            if loaded_results == test_results:
                print("✅ JSON输出功能测试通过")
                return True
            else:
                print("❌ JSON输出功能测试失败")
                return False
                
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ JSON输出功能测试出错: {str(e)}")
        return False

def test_import():
    """测试导入功能"""
    print("=== 测试导入功能 ===")
    
    try:
        from tra_report import TraReport
        print("✅ TraReport类导入成功")
        
        # 测试类初始化（不加载模型）
        try:
            tra_report = TraReport.__new__(TraReport)
            print("✅ TraReport类创建成功")
            return True
        except Exception as e:
            print(f"❌ TraReport类创建失败: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
        print("请确保已安装所需依赖包")
        return False

def main():
    """运行所有测试"""
    print("TraReport功能测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_import),
        ("文件匹配测试", test_file_matching),
        ("PSNR计算测试", test_psnr_calculation),
        ("JSON输出测试", test_json_output),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 执行出错: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！TraReport可以正常使用。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
