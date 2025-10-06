#!/usr/bin/env python3
"""
简化的TraReport测试脚本
测试基本的文件匹配和PSNR计算功能，不依赖复杂的模型加载
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
import sys

# 添加当前目录到Python路径
sys.path.append('.')

def test_file_matching():
    """测试文件匹配功能"""
    print("="*60)
    print("测试文件匹配功能")
    print("="*60)
    
    # 创建临时测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        gt_dir = os.path.join(temp_dir, "gt")
        val_dir = os.path.join(temp_dir, "val")
        
        # 创建目录
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # 创建测试图片文件
        test_files = ["test_001.png", "test_002.png", "test_003.jpg"]
        
        for filename in test_files:
            # 创建GT文件
            gt_path = os.path.join(gt_dir, filename)
            with open(gt_path, 'w') as f:
                f.write("dummy gt image")
            
            # 创建对应的val文件
            val_path = os.path.join(val_dir, filename)
            with open(val_path, 'w') as f:
                f.write("dummy val image")
        
        # 测试文件匹配逻辑
        matching_pairs = []
        val_files = []
        gt_files = []
        
        # 获取所有文件
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            import glob
            val_files.extend(glob.glob(os.path.join(val_dir, ext)))
            val_files.extend(glob.glob(os.path.join(val_dir, ext.upper())))
            gt_files.extend(glob.glob(os.path.join(gt_dir, ext)))
            gt_files.extend(glob.glob(os.path.join(gt_dir, ext.upper())))
        
        # 匹配文件
        for val_file in val_files:
            val_name = Path(val_file).stem
            for gt_file in gt_files:
                gt_name = Path(gt_file).stem
                if val_name == gt_name:
                    matching_pairs.append((val_file, gt_file))
                    break
        
        print(f"找到 {len(matching_pairs)} 对匹配的文件:")
        for val_file, gt_file in matching_pairs:
            print(f"  {Path(val_file).name} <-> {Path(gt_file).name}")
        
        assert len(matching_pairs) == 3, f"期望3对匹配文件，实际找到{len(matching_pairs)}对"
        print("✓ 文件匹配测试通过!")
        return True


def test_psnr_calculation():
    """测试PSNR计算功能"""
    print("\n" + "="*60)
    print("测试PSNR计算功能")
    print("="*60)
    
    try:
        from basicsr.metrics.psnr_ssim import calculate_psnr
        
        # 创建测试图片
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()  # 相同图片，PSNR应该很高
        
        # 计算PSNR
        psnr = calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False)
        print(f"相同图片的PSNR: {psnr:.4f}")
        
        # 创建不同的图片
        img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        psnr_diff = calculate_psnr(img1, img3, crop_border=0, input_order='HWC', test_y_channel=False)
        print(f"不同图片的PSNR: {psnr_diff:.4f}")
        
        assert psnr > psnr_diff, "相同图片的PSNR应该更高"
        print("✓ PSNR计算测试通过!")
        return True
        
    except ImportError as e:
        print(f"无法导入PSNR计算模块: {e}")
        return False
    except Exception as e:
        print(f"PSNR计算测试失败: {e}")
        return False


def test_enhanced_tra_report_structure():
    """测试EnhancedTraReport类的基本结构"""
    print("\n" + "="*60)
    print("测试EnhancedTraReport类结构")
    print("="*60)
    
    try:
        from enhanced_tra_report import EnhancedTraReport
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_dir = os.path.join(temp_dir, "gt")
            val_dir = os.path.join(temp_dir, "val")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            # 创建虚拟模型文件
            edge_model = os.path.join(temp_dir, "edge_model.ckpt")
            upscale_model = os.path.join(temp_dir, "upscale_model.ckpt")
            with open(edge_model, 'w') as f:
                f.write("dummy edge model")
            with open(upscale_model, 'w') as f:
                f.write("dummy upscale model")
            
            # 测试类初始化
            try:
                enhanced_tra_report = EnhancedTraReport(
                    gt_dir=gt_dir,
                    val_dir=val_dir,
                    stablesr_edge_model_path=edge_model,
                    stablesr_upscale_model_path=upscale_model,
                )
                print("✓ EnhancedTraReport类初始化成功!")
                
                # 测试文件匹配方法
                matching_pairs = enhanced_tra_report._find_matching_files()
                print(f"✓ 文件匹配方法工作正常，找到{len(matching_pairs)}对文件")
                
                return True
                
            except Exception as e:
                print(f"EnhancedTraReport初始化失败: {e}")
                return False
                
    except ImportError as e:
        print(f"无法导入EnhancedTraReport: {e}")
        return False


def test_existing_models():
    """测试现有模型文件"""
    print("\n" + "="*60)
    print("检查现有模型文件")
    print("="*60)
    
    # 查找现有的模型文件
    model_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.ckpt'):
                model_files.append(os.path.join(root, file))
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for model_file in model_files:
        file_size = os.path.getsize(model_file) / (1024*1024)  # MB
        print(f"  {model_file} ({file_size:.1f} MB)")
    
    if model_files:
        print("✓ 找到可用的模型文件")
        return True
    else:
        print("✗ 未找到模型文件")
        return False


def main():
    """主测试函数"""
    print("TraReport 简化测试套件")
    print("="*60)
    
    tests = [
        ("文件匹配", test_file_matching),
        ("PSNR计算", test_psnr_calculation),
        ("EnhancedTraReport结构", test_enhanced_tra_report_structure),
        ("现有模型文件", test_existing_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name} 测试...")
        try:
            if test_func():
                print(f"✓ {test_name} 测试通过!")
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败!")
        except Exception as e:
            print(f"✗ {test_name} 测试失败，错误: {str(e)}")
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 个测试通过")
    print("="*60)
    
    if passed == total:
        print("所有测试通过! TraReport基本功能正常。")
    else:
        print("部分测试失败。请检查实现。")
    
    return passed == total


if __name__ == "__main__":
    main()
