#!/usr/bin/env python3
"""
调试scale图片乱码问题的脚本
用于诊断图像处理流程中的问题
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(size=(64, 64), color=(128, 128, 128), filename="test_image.png"):
    """创建测试图像"""
    # 创建彩色测试图像
    image = np.ones((size[0], size[1], 3), dtype=np.uint8)
    image[:, :, 0] = color[0]  # R
    image[:, :, 1] = color[1]  # G  
    image[:, :, 2] = color[2]  # B
    
    # 添加一些图案
    center_x, center_y = size[0] // 2, size[1] // 2
    cv2.circle(image, (center_x, center_y), 20, (255, 0, 0), -1)  # 蓝色圆
    cv2.rectangle(image, (10, 10), (30, 30), (0, 255, 0), -1)     # 绿色矩形
    
    # 保存图像
    test_path = f"/tmp/{filename}"
    Image.fromarray(image).save(test_path)
    print(f"创建测试图像: {test_path}")
    return test_path


def test_image_loading_and_saving():
    """测试图像加载和保存功能"""
    print("\n=== 测试图像加载和保存功能 ===")
    
    # 创建测试图像
    test_image_path = create_test_image()
    
    try:
        # 测试PIL加载
        print("\n1. 测试PIL加载:")
        pil_image = Image.open(test_image_path)
        print(f"   PIL图像尺寸: {pil_image.size}")
        print(f"   PIL图像模式: {pil_image.mode}")
        print(f"   PIL图像格式: {pil_image.format}")
        
        # 测试OpenCV加载
        print("\n2. 测试OpenCV加载:")
        cv_image = cv2.imread(test_image_path)
        if cv_image is not None:
            print(f"   OpenCV图像形状: {cv_image.shape}")
            print(f"   OpenCV图像数据类型: {cv_image.dtype}")
        else:
            print("   OpenCV加载失败")
        
        # 测试numpy数组转换
        print("\n3. 测试numpy数组转换:")
        pil_array = np.array(pil_image)
        print(f"   PIL转numpy形状: {pil_array.shape}")
        print(f"   PIL转numpy数据类型: {pil_array.dtype}")
        print(f"   PIL转numpy值范围: [{pil_array.min()}, {pil_array.max()}]")
        
        # 测试tensor转换
        print("\n4. 测试tensor转换:")
        # 模拟StableSR中的图像处理流程
        im = pil_array.astype(np.float32) / 255.0  # 归一化到[0,1]
        im = im[None].transpose(0, 3, 1, 2)        # [1, 3, H, W]
        im = (torch.from_numpy(im) - 0.5) / 0.5    # 归一化到[-1,1]
        print(f"   Tensor形状: {im.shape}")
        print(f"   Tensor数据类型: {im.dtype}")
        print(f"   Tensor值范围: [{im.min():.3f}, {im.max():.3f}]")
        
        # 测试反向转换
        print("\n5. 测试反向转换:")
        # 模拟保存流程
        im_sr = torch.clamp((im + 1.0) / 2.0, min=0.0, max=1.0)  # 转换回[0,1]
        im_sr_np = im_sr.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255  # 转换回[0,255]
        print(f"   反向转换形状: {im_sr_np.shape}")
        print(f"   反向转换数据类型: {im_sr_np.dtype}")
        print(f"   反向转换值范围: [{im_sr_np.min():.1f}, {im_sr_np.max():.1f}]")
        
        # 测试保存
        print("\n6. 测试图像保存:")
        output_path = "/tmp/test_output.png"
        result_image = Image.fromarray(im_sr_np.astype(np.uint8))
        result_image.save(output_path)
        
        # 验证保存的图像
        saved_image = Image.open(output_path)
        print(f"   保存的图像尺寸: {saved_image.size}")
        print(f"   保存的图像模式: {saved_image.mode}")
        
        # 检查是否保存成功
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("   ✓ 图像保存成功")
            
            # 比较原图和保存的图
            saved_array = np.array(saved_image)
            diff = np.abs(pil_array.astype(float) - saved_array.astype(float))
            max_diff = diff.max()
            print(f"   与原图最大差异: {max_diff}")
            
            if max_diff < 5:  # 允许小的差异
                print("   ✓ 保存的图像质量良好")
            else:
                print("   ⚠ 保存的图像可能有质量问题")
        else:
            print("   ✗ 图像保存失败")
            
    except Exception as e:
        print(f"   ✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


def test_scale_processing():
    """测试scale处理功能"""
    print("\n=== 测试scale处理功能 ===")
    
    try:
        # 创建测试图像
        test_image_path = create_test_image(size=(128, 128))
        
        # 加载图像
        pil_image = Image.open(test_image_path)
        im = np.array(pil_image).astype(np.float32) / 255.0
        im = im[None].transpose(0, 3, 1, 2)
        im = (torch.from_numpy(im) - 0.5) / 0.5
        
        print(f"原始图像尺寸: {im.shape}")
        
        # 测试不同的scale处理
        scales = [2.0, 4.0, 8.0]
        
        for scale in scales:
            print(f"\n测试scale={scale}:")
            
            # 上采样
            new_h = int(im.size(-2) * scale)
            new_w = int(im.size(-1) * scale)
            
            upscaled = torch.nn.functional.interpolate(
                im,
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False
            )
            
            print(f"  上采样后尺寸: {upscaled.shape}")
            print(f"  值范围: [{upscaled.min():.3f}, {upscaled.max():.3f}]")
            
            # 下采样（如果需要）
            if scale > 4.0:  # 模拟StableSR中的逻辑
                target_h = int(im.size(-2) * 4.0)
                target_w = int(im.size(-1) * 4.0)
                
                downscaled = torch.nn.functional.interpolate(
                    upscaled,
                    size=(target_h, target_w),
                    mode='bicubic',
                    align_corners=False
                )
                
                print(f"  下采样后尺寸: {downscaled.shape}")
                print(f"  值范围: [{downscaled.min():.3f}, {downscaled.max():.3f}]")
                
                # 保存结果
                final_result = torch.clamp((downscaled + 1.0) / 2.0, min=0.0, max=1.0)
                final_np = final_result.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                
                output_path = f"/tmp/test_scale_{scale}.png"
                result_image = Image.fromarray(final_np.astype(np.uint8))
                result_image.save(output_path)
                
                print(f"  保存到: {output_path}")
                
                # 检查保存的文件
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print("  ✓ 保存成功")
                else:
                    print("  ✗ 保存失败")
            
    except Exception as e:
        print(f"✗ scale处理测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_color_correction():
    """测试颜色修正功能"""
    print("\n=== 测试颜色修正功能 ===")
    
    try:
        # 创建测试图像
        test_image_path = create_test_image(size=(64, 64), color=(200, 100, 50))
        
        # 加载图像
        pil_image = Image.open(test_image_path)
        im = np.array(pil_image).astype(np.float32) / 255.0
        im = im[None].transpose(0, 3, 1, 2)
        im = (torch.from_numpy(im) - 0.5) / 0.5
        
        print(f"原始图像形状: {im.shape}")
        print(f"原始图像值范围: [{im.min():.3f}, {im.max():.3f}]")
        
        # 模拟处理后的图像（添加一些噪声）
        processed = im + torch.randn_like(im) * 0.1
        processed = torch.clamp(processed, -1, 1)
        
        print(f"处理后图像值范围: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # 测试不同的颜色修正方法
        methods = ['nofix', 'adain', 'wavelet']
        
        for method in methods:
            print(f"\n测试颜色修正方法: {method}")
            
            if method == 'nofix':
                corrected = processed
            elif method == 'adain':
                # 简单的adain模拟
                mean_orig = im.mean(dim=[2, 3], keepdim=True)
                std_orig = im.std(dim=[2, 3], keepdim=True)
                mean_proc = processed.mean(dim=[2, 3], keepdim=True)
                std_proc = processed.std(dim=[2, 3], keepdim=True)
                corrected = (processed - mean_proc) / (std_proc + 1e-8) * std_orig + mean_orig
            else:
                corrected = processed  # wavelet暂时跳过
            
            corrected = torch.clamp(corrected, -1, 1)
            print(f"  修正后值范围: [{corrected.min():.3f}, {corrected.max():.3f}]")
            
            # 保存结果
            final_result = torch.clamp((corrected + 1.0) / 2.0, min=0.0, max=1.0)
            final_np = final_result.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
            
            output_path = f"/tmp/test_colorfix_{method}.png"
            result_image = Image.fromarray(final_np.astype(np.uint8))
            result_image.save(output_path)
            
            print(f"  保存到: {output_path}")
            
    except Exception as e:
        print(f"✗ 颜色修正测试失败: {e}")
        import traceback
        traceback.print_exc()


def check_existing_output_files():
    """检查现有的输出文件"""
    print("\n=== 检查现有输出文件 ===")
    
    output_dirs = [
        "/root/dp/StableSR_Edge_v2/quick_test_results",
        "/root/dp/StableSR_Edge_v2/comprehensive_test_results",
        "/root/dp/StableSR_Edge_v2/test_output",
        "/root/dp/StableSR_Edge_v2/edge_inference_output"
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"\n检查目录: {output_dir}")
            
            # 查找所有图像文件
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(Path(output_dir).rglob(ext))
            
            if image_files:
                print(f"  找到 {len(image_files)} 个图像文件")
                
                for img_file in image_files[:5]:  # 只检查前5个
                    try:
                        # 检查文件大小
                        file_size = os.path.getsize(img_file)
                        print(f"    {img_file.name}: {file_size} bytes")
                        
                        # 尝试加载图像
                        img = Image.open(img_file)
                        print(f"      尺寸: {img.size}, 模式: {img.mode}")
                        
                        # 检查图像数据
                        img_array = np.array(img)
                        if img_array.size > 0:
                            print(f"      数据形状: {img_array.shape}, 值范围: [{img_array.min()}, {img_array.max()}]")
                            
                            # 检查是否有异常值
                            if img_array.max() > 255 or img_array.min() < 0:
                                print(f"      ⚠ 发现异常值!")
                            
                            # 检查是否全黑或全白
                            if img_array.max() == img_array.min():
                                print(f"      ⚠ 图像可能是单色!")
                                
                        else:
                            print(f"      ✗ 图像数据为空!")
                            
                    except Exception as e:
                        print(f"    ✗ 无法读取 {img_file.name}: {e}")
            else:
                print("  未找到图像文件")
        else:
            print(f"目录不存在: {output_dir}")


def main():
    """主函数"""
    print("Scale图片乱码问题调试脚本")
    print("=" * 50)
    
    # 运行所有测试
    test_image_loading_and_saving()
    test_scale_processing()
    test_color_correction()
    check_existing_output_files()
    
    print("\n" + "=" * 50)
    print("调试完成!")
    print("\n建议检查:")
    print("1. 图像数据范围是否正确 (应该在[0,255]或[0,1]范围内)")
    print("2. 数据类型是否正确 (应该是uint8或float32)")
    print("3. 图像维度是否正确 (应该是HWC或CHW格式)")
    print("4. 保存路径是否可写")
    print("5. PIL/OpenCV版本是否兼容")


if __name__ == "__main__":
    main()
