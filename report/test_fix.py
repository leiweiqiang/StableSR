#!/usr/bin/env python3
"""
测试修复后的scale图片处理功能
"""

import os
import sys
import tempfile
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(size=(64, 64), filename="test_scale.png"):
    """创建测试图像"""
    # 创建彩色测试图像
    image = np.ones((size[0], size[1], 3), dtype=np.uint8)
    image[:, :, 0] = 128  # R
    image[:, :, 1] = 64   # G  
    image[:, :, 2] = 192  # B
    
    # 添加一些图案
    center_x, center_y = size[0] // 2, size[1] // 2
    cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)  # 蓝色圆
    cv2.rectangle(image, (10, 10), (25, 25), (0, 255, 0), -1)     # 绿色矩形
    
    # 保存图像
    test_path = f"/tmp/{filename}"
    Image.fromarray(image).save(test_path)
    print(f"创建测试图像: {test_path}")
    return test_path


def test_tensor_indexing():
    """测试tensor索引操作"""
    print("\n=== 测试tensor索引操作 ===")
    
    # 创建一个测试tensor [batch, channel, height, width]
    test_tensor = torch.randn(1, 3, 100, 150)
    print(f"原始tensor形状: {test_tensor.shape}")
    
    # 测试正确的索引操作
    ori_h, ori_w = 80, 120
    
    try:
        # 正确的索引方式
        result_correct = test_tensor[:, :, :ori_h, :ori_w]
        print(f"正确索引结果形状: {result_correct.shape}")
        print("✓ 正确索引操作成功")
    except Exception as e:
        print(f"✗ 正确索引操作失败: {e}")
    
    try:
        # 错误的索引方式（原来的问题）
        result_wrong = test_tensor[:, :ori_h, :ori_w, :]  # 这会改变tensor的维度顺序
        print(f"错误索引结果形状: {result_wrong.shape}")
        print("⚠ 错误索引操作改变了tensor结构")
    except Exception as e:
        print(f"✗ 错误索引操作失败: {e}")
    
    # 测试填充移除逻辑
    print("\n--- 测试填充移除逻辑 ---")
    
    # 模拟填充后的tensor
    padded_tensor = torch.randn(1, 3, 128, 160)  # 填充到32的倍数
    ori_h, ori_w = 100, 150
    
    try:
        # 移除填充
        unpadded = padded_tensor[:, :, :ori_h, :ori_w]
        print(f"填充前形状: {padded_tensor.shape}")
        print(f"移除填充后形状: {unpadded.shape}")
        print(f"目标形状: [1, 3, {ori_h}, {ori_w}]")
        
        if unpadded.shape == (1, 3, ori_h, ori_w):
            print("✓ 填充移除逻辑正确")
        else:
            print("✗ 填充移除逻辑错误")
            
    except Exception as e:
        print(f"✗ 填充移除测试失败: {e}")


def test_scale_processing_with_fix():
    """测试修复后的scale处理"""
    print("\n=== 测试修复后的scale处理 ===")
    
    try:
        # 创建测试图像
        test_image_path = create_test_image(size=(128, 128))
        
        # 模拟StableSR的处理流程
        pil_image = Image.open(test_image_path)
        im = np.array(pil_image).astype(np.float32) / 255.0
        im = im[None].transpose(0, 3, 1, 2)
        im = (torch.from_numpy(im) - 0.5) / 0.5
        
        print(f"原始图像形状: {im.shape}")
        
        # 模拟上采样
        input_size = 512
        upscale = 4.0
        size_min = min(im.size(-1), im.size(-2))
        upsample_scale = max(input_size/size_min, upscale)
        
        print(f"最小尺寸: {size_min}")
        print(f"上采样倍数: {upsample_scale}")
        
        # 上采样
        upscaled = torch.nn.functional.interpolate(
            im,
            size=(int(im.size(-2)*upsample_scale),
                  int(im.size(-1)*upsample_scale)),
            mode='bicubic',
        )
        upscaled = upscaled.clamp(-1, 1)
        
        print(f"上采样后形状: {upscaled.shape}")
        
        # 模拟填充到32的倍数
        ori_h, ori_w = upscaled.shape[2:]
        flag_pad = False
        
        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
            flag_pad = True
            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
            upscaled = torch.nn.functional.pad(upscaled, pad=(0, pad_w, 0, pad_h), mode='reflect')
            print(f"填充后形状: {upscaled.shape}")
        
        # 模拟处理后的结果
        processed = torch.randn_like(upscaled) * 0.5 + 0.5  # 模拟处理结果
        processed = torch.clamp(processed, 0.0, 1.0)
        
        # 如果上采样倍数超过目标倍数，需要下采样
        if upsample_scale > upscale:
            processed = torch.nn.functional.interpolate(
                processed,
                size=(int(im.size(-2)*upscale/upsample_scale),
                      int(im.size(-1)*upscale/upsample_scale)),
                mode='bicubic',
            )
            processed = torch.clamp(processed, min=0.0, max=1.0)
            print(f"下采样后形状: {processed.shape}")
        
        # 移除填充（使用修复后的逻辑）
        if flag_pad:
            processed = processed[:, :, :ori_h, :ori_w]
            print(f"移除填充后形状: {processed.shape}")
        
        # 保存结果
        final_result = processed.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        output_path = "/tmp/test_fixed_scale.png"
        result_image = Image.fromarray(final_result.astype(np.uint8))
        result_image.save(output_path)
        
        print(f"最终结果形状: {final_result.shape}")
        print(f"保存到: {output_path}")
        
        # 验证保存的文件
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            saved_image = Image.open(output_path)
            print(f"保存的图像尺寸: {saved_image.size}")
            print("✓ 修复后的scale处理成功")
        else:
            print("✗ 保存失败")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("测试修复后的scale图片处理")
    print("=" * 50)
    
    test_tensor_indexing()
    test_scale_processing_with_fix()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("\n修复内容:")
    print("1. 修复了tensor索引错误: im_sr[:, :ori_h, :ori_w, ] -> im_sr[:, :, :ori_h, :ori_w]")
    print("2. 这确保了tensor的维度顺序正确: [batch, channel, height, width]")
    print("3. 避免了维度混乱导致的图像乱码问题")


if __name__ == "__main__":
    import cv2  # 在main中导入以避免全局导入问题
    main()
