#!/usr/bin/env python3
"""
使用真实DIV2K数据集测试TraReport功能
只测试StableSR Edge模型，避免配置不匹配问题
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def create_test_subset(gt_dir, val_dir, num_samples=5):
    """创建测试子集"""
    print(f"创建包含 {num_samples} 个样本的测试子集...")
    
    # 获取所有文件
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # 确保文件匹配
    matching_files = []
    for gt_file in gt_files:
        if gt_file in val_files:
            matching_files.append(gt_file)
    
    print(f"找到 {len(matching_files)} 对匹配的文件")
    
    # 随机选择样本
    selected_files = random.sample(matching_files, min(num_samples, len(matching_files)))
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    temp_gt_dir = os.path.join(temp_dir, 'gt')
    temp_val_dir = os.path.join(temp_dir, 'val')
    os.makedirs(temp_gt_dir, exist_ok=True)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    # 复制选中的文件
    for file in selected_files:
        shutil.copy2(os.path.join(gt_dir, file), os.path.join(temp_gt_dir, file))
        shutil.copy2(os.path.join(val_dir, file), os.path.join(temp_val_dir, file))
    
    print(f"测试子集创建完成: {temp_dir}")
    print(f"GT文件: {len(selected_files)} 个")
    print(f"Val文件: {len(selected_files)} 个")
    
    return temp_dir, temp_gt_dir, temp_val_dir

def test_tra_report_basic():
    """测试基本TraReport功能"""
    print("=" * 60)
    print("测试基本TraReport功能")
    print("=" * 60)
    
    try:
        from tra_report import TraReport
        print("✅ TraReport导入成功")
        
        # 创建测试数据
        temp_dir, gt_dir, val_dir = create_test_subset(
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR',
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR',
            num_samples=3
        )
        
        # 初始化TraReport
        model_path = './logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt'
        config_path = './configs/stableSRNew/v2-finetune_text_T_512_edge.yaml'
        
        tra_report = TraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            model_path=model_path,
            config_path=config_path,
            device='cuda',
            ddpm_steps=20,  # 减少步数以加快测试
            upscale=4.0,
            colorfix_type='adain',
            seed=42
        )
        
        print("✅ TraReport初始化成功")
        
        # 测试文件匹配
        matching_files = tra_report._find_matching_files()
        print(f"✅ 找到 {len(matching_files)} 对匹配文件")
        
        # 测试PSNR计算（使用bicubic插值）
        if matching_files:
            val_file, gt_file = matching_files[0]
            val_path = os.path.join(val_dir, val_file)
            gt_path = os.path.join(gt_dir, gt_file)
            
            # 加载图片
            val_img = tra_report._load_img(val_path)
            gt_img = tra_report._load_img(gt_path)
            
            # 计算bicubic PSNR
            from torchvision.transforms import functional as F
            val_tensor = F.to_tensor(val_img).unsqueeze(0)
            gt_tensor = F.to_tensor(gt_img).unsqueeze(0)
            
            # Bicubic上采样
            upscaled = F.resize(val_tensor, size=gt_tensor.shape[-2:], interpolation=F.InterpolationMode.BICUBIC)
            
            # 计算PSNR
            psnr = tra_report._calculate_psnr(upscaled, gt_tensor)
            print(f"✅ Bicubic PSNR: {psnr:.2f} dB")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print("✅ 基本功能测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_enhanced_tra_report_edge_only():
    """测试Enhanced TraReport（仅Edge模型）"""
    print("=" * 60)
    print("测试Enhanced TraReport（仅Edge模型）")
    print("=" * 60)
    
    try:
        from enhanced_tra_report import EnhancedTraReport
        print("✅ EnhancedTraReport导入成功")
        
        # 创建测试数据
        temp_dir, gt_dir, val_dir = create_test_subset(
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR',
            '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR',
            num_samples=2
        )
        
        # 初始化EnhancedTraReport（仅Edge模型）
        edge_model_path = './logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt'
        edge_config_path = './configs/stableSRNew/v2-finetune_text_T_512_edge.yaml'
        
        enhanced_tra_report = EnhancedTraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            stablesr_edge_model_path=edge_model_path,
            stablesr_upscale_model_path=None,  # 不加载Upscale模型
            stablesr_edge_config_path=edge_config_path,
            stablesr_upscale_config_path=None,
            device='cuda',
            ddpm_steps=20,
            upscale=4.0,
            colorfix_type='adain',
            seed=42
        )
        
        print("✅ EnhancedTraReport初始化成功")
        
        # 测试文件匹配
        matching_files = enhanced_tra_report._find_matching_files()
        print(f"✅ 找到 {len(matching_files)} 对匹配文件")
        
        # 测试Edge模型加载
        print("正在加载StableSR Edge模型...")
        enhanced_tra_report.load_edge_model()
        print("✅ StableSR Edge模型加载成功")
        
        # 测试单张图片处理
        if matching_files:
            val_file, gt_file = matching_files[0]
            print(f"测试图片: {val_file} -> {gt_file}")
            
            val_path = os.path.join(val_dir, val_file)
            gt_path = os.path.join(gt_dir, gt_file)
            
            # 加载图片
            val_img = enhanced_tra_report._load_img(val_path)
            gt_img = enhanced_tra_report._load_img(gt_path)
            
            print(f"输入图片尺寸: {val_img.size}")
            print(f"目标图片尺寸: {gt_img.size}")
            
            # 测试Edge模型上采样
            print("使用StableSR Edge进行上采样...")
            upscaled_edge = enhanced_tra_report._upscale_with_edge_model(val_img)
            print(f"Edge上采样结果尺寸: {upscaled_edge.size}")
            
            # 计算PSNR
            edge_psnr = enhanced_tra_report._calculate_psnr(upscaled_edge, gt_img)
            print(f"✅ StableSR Edge PSNR: {edge_psnr:.2f} dB")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print("✅ Enhanced TraReport测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced TraReport测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("DIV2K数据集TraReport功能测试")
    print("=" * 60)
    
    # 检查数据集
    gt_dir = '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_HR'
    val_dir = '/stablesr_dataset/dataset/DIV2K/DIV2K_valid_LR'
    
    if not os.path.exists(gt_dir):
        print(f"❌ GT目录不存在: {gt_dir}")
        return False
    
    if not os.path.exists(val_dir):
        print(f"❌ Val目录不存在: {val_dir}")
        return False
    
    print(f"✅ GT目录: {gt_dir}")
    print(f"✅ Val目录: {val_dir}")
    
    # 检查模型文件
    model_path = './logs/2025-10-05T01-24-44_stablesr_edge_T6_DIV2K_2005_10_04/checkpoints/last.ckpt'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"✅ 模型文件: {model_path}")
    
    # 运行测试
    test_results = []
    
    # 测试1: 基本TraReport功能
    result1 = test_tra_report_basic()
    test_results.append(("基本TraReport功能", result1))
    
    # 测试2: Enhanced TraReport（仅Edge模型）
    result2 = test_enhanced_tra_report_edge_only()
    test_results.append(("Enhanced TraReport（仅Edge模型）", result2))
    
    # 输出测试结果
    print("=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！TraReport功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
