#!/usr/bin/env python3
"""
测试Enhanced TraReport功能的脚本
演示如何使用EnhancedTraReport进行StableSR Edge和StableSR Upscale模型的比较评估
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from enhanced_tra_report import EnhancedTraReport


def create_test_data(gt_dir: str, val_dir: str, num_images: int = 3):
    """创建测试数据"""
    print(f"Creating test data with {num_images} images...")
    
    # 创建目录
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for i in range(num_images):
        # 创建高分辨率图片 (256x256)
        gt_size = 256
        gt_image = np.random.randint(0, 255, (gt_size, gt_size, 3), dtype=np.uint8)
        gt_pil = Image.fromarray(gt_image)
        gt_path = os.path.join(gt_dir, f"test_image_{i:03d}.png")
        gt_pil.save(gt_path)
        
        # 创建对应的低分辨率图片 (64x64)
        val_size = 64
        val_image = np.random.randint(0, 255, (val_size, val_size, 3), dtype=np.uint8)
        val_pil = Image.fromarray(val_image)
        val_path = os.path.join(val_dir, f"test_image_{i:03d}.png")
        val_pil.save(val_path)
        
        print(f"Created: {gt_path} ({gt_size}x{gt_size}) and {val_path} ({val_size}x{val_size})")


def test_enhanced_tra_report():
    """测试Enhanced TraReport功能"""
    print("="*60)
    print("Testing Enhanced TraReport")
    print("="*60)
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        gt_dir = os.path.join(temp_dir, "gt")
        val_dir = os.path.join(temp_dir, "val")
        
        # 创建测试数据
        create_test_data(gt_dir, val_dir, num_images=2)
        
        # 检查是否有可用的模型文件
        stablesr_edge_model = "./weights/stablesr_edge_model.ckpt"  # 假设的路径
        stablesr_upscale_model = "./weights/stablesr_upscale_model.ckpt"  # 假设的路径
        
        # 如果模型文件不存在，使用现有的模型文件
        if not os.path.exists(stablesr_edge_model):
            # 尝试使用现有的模型文件
            possible_edge_models = [
                "./weights/stablesr_000117.ckpt",
                "./weights/stablesr_768v_000139.ckpt"
            ]
            for model_path in possible_edge_models:
                if os.path.exists(model_path):
                    stablesr_edge_model = model_path
                    break
        
        if not os.path.exists(stablesr_upscale_model):
            # 尝试使用现有的模型文件
            possible_upscale_models = [
                "./weights/stablesr_000117.ckpt",
                "./weights/stablesr_768v_000139.ckpt"
            ]
            for model_path in possible_upscale_models:
                if os.path.exists(model_path):
                    stablesr_upscale_model = model_path
                    break
        
        # 检查模型文件是否存在
        if not os.path.exists(stablesr_edge_model):
            print(f"Warning: StableSR Edge model not found at {stablesr_edge_model}")
            print("Please provide the correct path to the StableSR Edge model")
            return False
            
        if not os.path.exists(stablesr_upscale_model):
            print(f"Warning: StableSR Upscale model not found at {stablesr_upscale_model}")
            print("Please provide the correct path to the StableSR Upscale model")
            return False
        
        try:
            # 创建EnhancedTraReport实例
            print("\nCreating EnhancedTraReport instance...")
            enhanced_tra_report = EnhancedTraReport(
                gt_dir=gt_dir,
                val_dir=val_dir,
                stablesr_edge_model_path=stablesr_edge_model,
                stablesr_upscale_model_path=stablesr_upscale_model,
                stablesr_edge_config_path="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
                stablesr_upscale_config_path="./configs/stableSRNew/v2-finetune_text_T_512.yaml",
                device="cuda" if torch.cuda.is_available() else "cpu",
                ddpm_steps=50,  # 减少步数以加快测试
                upscale=4.0,
                colorfix_type="adain",
                seed=42
            )
            
            print("EnhancedTraReport instance created successfully!")
            
            # 运行评估
            print("\nRunning evaluation...")
            results = enhanced_tra_report.run_evaluation("test_enhanced_results.json")
            
            # 验证结果
            print("\nValidating results...")
            assert "evaluation_info" in results
            assert "results" in results
            assert "summary" in results
            assert "stablesr_edge" in results["summary"]
            assert "stablesr_upscale" in results["summary"]
            assert "comparison" in results["summary"]
            
            print("Results validation passed!")
            
            # 打印详细结果
            print("\n" + "="*60)
            print("DETAILED RESULTS")
            print("="*60)
            
            print(f"Total files processed: {results['evaluation_info']['total_files']}")
            print(f"StableSR Edge Average PSNR: {results['summary']['stablesr_edge']['average_psnr']:.4f}")
            print(f"StableSR Upscale Average PSNR: {results['summary']['stablesr_upscale']['average_psnr']:.4f}")
            print(f"PSNR Difference: {results['summary']['comparison']['psnr_difference']:.4f}")
            print(f"Better Model: {results['summary']['comparison']['better_model']}")
            print(f"Improvement: {results['summary']['comparison']['improvement_percentage']:.2f}%")
            
            print("\nIndividual Results:")
            for i, result in enumerate(results["results"]):
                print(f"  Image {i+1}:")
                print(f"    StableSR Edge PSNR: {result['stablesr_edge']['psnr']:.4f}")
                print(f"    StableSR Upscale PSNR: {result['stablesr_upscale']['psnr']:.4f}")
                print(f"    Difference: {result['psnr_difference']:.4f}")
                print(f"    Better: {result['better_model']}")
            
            print("\nTest completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def test_file_matching():
    """测试文件匹配功能"""
    print("\n" + "="*60)
    print("Testing File Matching")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        gt_dir = os.path.join(temp_dir, "gt")
        val_dir = os.path.join(temp_dir, "val")
        
        # 创建测试数据
        create_test_data(gt_dir, val_dir, num_images=3)
        
        # 创建EnhancedTraReport实例（不需要模型文件来测试文件匹配）
        enhanced_tra_report = EnhancedTraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            stablesr_edge_model_path="./dummy_edge.ckpt",  # 虚拟路径
            stablesr_upscale_model_path="./dummy_upscale.ckpt",  # 虚拟路径
        )
        
        # 测试文件匹配
        matching_pairs = enhanced_tra_report._find_matching_files()
        print(f"Found {len(matching_pairs)} matching file pairs:")
        for val_file, gt_file in matching_pairs:
            print(f"  {Path(val_file).name} <-> {Path(gt_file).name}")
        
        assert len(matching_pairs) == 3, f"Expected 3 matching pairs, got {len(matching_pairs)}"
        print("File matching test passed!")
        return True


def test_psnr_calculation():
    """测试PSNR计算功能"""
    print("\n" + "="*60)
    print("Testing PSNR Calculation")
    print("="*60)
    
    # 创建测试图片
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()  # 相同图片，PSNR应该很高
    
    # 创建EnhancedTraReport实例
    enhanced_tra_report = EnhancedTraReport(
        gt_dir="./dummy_gt",
        val_dir="./dummy_val",
        stablesr_edge_model_path="./dummy_edge.ckpt",
        stablesr_upscale_model_path="./dummy_upscale.ckpt",
    )
    
    # 测试PSNR计算
    psnr = enhanced_tra_report._calculate_psnr(img1, img2)
    print(f"PSNR for identical images: {psnr:.4f}")
    
    # 创建不同的图片
    img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    psnr_diff = enhanced_tra_report._calculate_psnr(img1, img3)
    print(f"PSNR for different images: {psnr_diff:.4f}")
    
    assert psnr > psnr_diff, "PSNR should be higher for identical images"
    print("PSNR calculation test passed!")
    return True


def main():
    """主测试函数"""
    print("Enhanced TraReport Test Suite")
    print("="*60)
    
    tests = [
        ("File Matching", test_file_matching),
        ("PSNR Calculation", test_psnr_calculation),
        ("Enhanced TraReport", test_enhanced_tra_report),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            if test_func():
                print(f"✓ {test_name} test passed!")
                passed += 1
            else:
                print(f"✗ {test_name} test failed!")
        except Exception as e:
            print(f"✗ {test_name} test failed with error: {str(e)}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("All tests passed! Enhanced TraReport is working correctly.")
    else:
        print("Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
