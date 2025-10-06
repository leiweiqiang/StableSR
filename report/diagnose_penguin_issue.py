#!/usr/bin/env python3
"""
诊断企鹅图片输入但输出彩色花纹乱码的问题
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


def create_penguin_test_image():
    """创建一个企鹅测试图像"""
    print("=== 创建企鹅测试图像 ===")
    
    # 创建一个简单的企鹅图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 企鹅身体 (黑色)
    cv2.ellipse(img, (128, 180), (60, 80), 0, 0, 360, (0, 0, 0), -1)
    
    # 企鹅头部 (黑色)
    cv2.circle(img, (128, 100), 40, (0, 0, 0), -1)
    
    # 企鹅肚子 (白色)
    cv2.ellipse(img, (128, 180), (40, 60), 0, 0, 360, (255, 255, 255), -1)
    
    # 企鹅眼睛 (白色)
    cv2.circle(img, (115, 90), 8, (255, 255, 255), -1)
    cv2.circle(img, (140, 90), 8, (255, 255, 255), -1)
    
    # 企鹅嘴巴 (橙色)
    cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 180, (0, 165, 255), -1)
    
    # 企鹅脚 (橙色)
    cv2.ellipse(img, (110, 240), (15, 10), 0, 0, 360, (0, 165, 255), -1)
    cv2.ellipse(img, (145, 240), (15, 10), 0, 0, 360, (0, 165, 255), -1)
    
    # 保存图像
    penguin_path = "/tmp/penguin_test.png"
    Image.fromarray(img).save(penguin_path)
    print(f"创建企鹅测试图像: {penguin_path}")
    
    return penguin_path


def analyze_image_processing_pipeline():
    """分析图像处理管道"""
    print("\n=== 分析图像处理管道 ===")
    
    # 创建企鹅测试图像
    penguin_path = create_penguin_test_image()
    
    try:
        # 1. 读取原始图像
        print("\n1. 读取原始图像:")
        original_img = Image.open(penguin_path)
        original_array = np.array(original_img)
        print(f"   原始图像尺寸: {original_img.size}")
        print(f"   原始图像模式: {original_img.mode}")
        print(f"   原始图像值范围: [{original_array.min()}, {original_array.max()}]")
        print(f"   原始图像均值: {original_array.mean():.2f}")
        print(f"   原始图像标准差: {original_array.std():.2f}")
        
        # 2. 模拟StableSR的图像预处理
        print("\n2. 模拟图像预处理:")
        im = original_array.astype(np.float32) / 255.0  # 归一化到[0,1]
        print(f"   归一化后范围: [{im.min():.3f}, {im.max():.3f}]")
        
        im = im[None].transpose(0, 3, 1, 2)  # [1, 3, H, W]
        print(f"   转置后形状: {im.shape}")
        
        im = (torch.from_numpy(im) - 0.5) / 0.5  # 归一化到[-1,1]
        print(f"   tensor范围: [{im.min():.3f}, {im.max():.3f}]")
        
        # 3. 模拟上采样
        print("\n3. 模拟上采样:")
        upscale = 2.0
        upscaled = torch.nn.functional.interpolate(
            im,
            size=(int(im.size(-2) * upscale), int(im.size(-1) * upscale)),
            mode='bicubic'
        )
        print(f"   上采样后形状: {upscaled.shape}")
        print(f"   上采样后范围: [{upscaled.min():.3f}, {upscaled.max():.3f}]")
        
        # 4. 模拟填充
        print("\n4. 模拟填充:")
        ori_h, ori_w = upscaled.shape[2:]
        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
            padded = torch.nn.functional.pad(upscaled, pad=(0, pad_w, 0, pad_h), mode='reflect')
            print(f"   填充后形状: {padded.shape}")
        else:
            padded = upscaled
            print(f"   无需填充")
        
        # 5. 模拟模型处理（这里用随机噪声模拟）
        print("\n5. 模拟模型处理:")
        # 这里用随机噪声来模拟模型输出，看看是否会产生彩色花纹
        processed = torch.randn_like(padded) * 0.5 + 0.5
        processed = torch.clamp(processed, 0.0, 1.0)
        print(f"   处理后范围: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # 6. 移除填充
        print("\n6. 移除填充:")
        if padded.shape != upscaled.shape:
            final_result = processed[:, :, :ori_h, :ori_w]
            print(f"   移除填充后形状: {final_result.shape}")
        else:
            final_result = processed
            print(f"   无需移除填充")
        
        # 7. 转换为输出图像
        print("\n7. 转换为输出图像:")
        final_np = final_result.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        print(f"   最终numpy形状: {final_np.shape}")
        print(f"   最终numpy范围: [{final_np.min():.1f}, {final_np.max():.1f}]")
        
        # 保存结果
        output_path = "/tmp/penguin_processed.png"
        result_image = Image.fromarray(final_np.astype(np.uint8))
        result_image.save(output_path)
        print(f"   保存到: {output_path}")
        
        # 8. 分析结果
        print("\n8. 分析结果:")
        result_array = np.array(result_image)
        print(f"   结果图像尺寸: {result_image.size}")
        print(f"   结果图像模式: {result_image.mode}")
        print(f"   结果值范围: [{result_array.min()}, {result_array.max()}]")
        print(f"   结果均值: {result_array.mean():.2f}")
        print(f"   结果标准差: {result_array.std():.2f}")
        
        # 检查是否像彩色花纹
        if result_array.std() > 50:  # 高标准差可能表示彩色花纹
            print("   ⚠️ 高标准差，可能产生彩色花纹效果")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像处理管道分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_space_conversion():
    """测试颜色空间转换问题"""
    print("\n=== 测试颜色空间转换 ===")
    
    try:
        # 创建测试图像
        penguin_path = create_penguin_test_image()
        img = Image.open(penguin_path)
        
        # 测试不同的颜色空间转换
        print("\n1. RGB -> BGR 转换:")
        rgb_array = np.array(img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # 保存BGR版本
        bgr_path = "/tmp/penguin_bgr.png"
        cv2.imwrite(bgr_path, bgr_array)
        print(f"   保存BGR版本: {bgr_path}")
        
        # 测试HSV转换
        print("\n2. RGB -> HSV 转换:")
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        print(f"   HSV范围: [{hsv_array.min()}, {hsv_array.max()}]")
        
        # 测试LAB转换
        print("\n3. RGB -> LAB 转换:")
        lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        print(f"   LAB范围: [{lab_array.min()}, {lab_array.max()}]")
        
        # 测试YUV转换
        print("\n4. RGB -> YUV 转换:")
        yuv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2YUV)
        print(f"   YUV范围: [{yuv_array.min()}, {yuv_array.max()}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 颜色空间转换测试失败: {e}")
        return False


def test_model_inference_with_penguin():
    """使用真实模型测试企鹅图像"""
    print("\n=== 使用真实模型测试企鹅图像 ===")
    
    try:
        # 检查模型文件
        model_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        
        if not os.path.exists(model_path) or not os.path.exists(vqgan_path):
            print("⚠️ 模型文件不存在，跳过真实模型测试")
            return False
        
        # 创建企鹅测试图像
        penguin_path = create_penguin_test_image()
        
        # 导入StableSR类
        from stable_sr_scale_lr import StableSR_ScaleLR
        
        print("正在初始化模型...")
        processor = StableSR_ScaleLR(
            config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml",
            ckpt_path=model_path,
            vqgan_ckpt_path=vqgan_path,
            ddpm_steps=4,  # 快速测试
            upscale=2.0,
            colorfix_type="adain"
        )
        
        # 处理企鹅图像
        output_dir = "/tmp/penguin_model_test"
        print("正在处理企鹅图像...")
        processor.process_images(penguin_path, output_dir)
        
        # 检查输出
        output_files = list(Path(output_dir).rglob("*.png"))
        if output_files:
            print(f"✅ 成功生成 {len(output_files)} 个输出文件")
            
            for output_file in output_files:
                try:
                    output_img = Image.open(output_file)
                    output_array = np.array(output_img)
                    
                    print(f"\n输出文件: {output_file.name}")
                    print(f"  尺寸: {output_img.size}")
                    print(f"  模式: {output_img.mode}")
                    print(f"  值范围: [{output_array.min()}, {output_array.max()}]")
                    print(f"  均值: {output_array.mean():.2f}")
                    print(f"  标准差: {output_array.std():.2f}")
                    
                    # 检查是否像彩色花纹
                    if output_array.std() > 80:
                        print("  ⚠️ 高标准差，可能产生彩色花纹效果")
                    
                except Exception as e:
                    print(f"  ❌ 无法读取输出文件: {e}")
            
            return True
        else:
            print("❌ 没有生成输出文件")
            return False
            
    except Exception as e:
        print(f"❌ 真实模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_existing_outputs():
    """分析现有的输出文件"""
    print("\n=== 分析现有输出文件 ===")
    
    output_dirs = [
        "/root/dp/StableSR_Edge_v2/test_output",
        "/root/dp/StableSR_Edge_v2/edge_inference_output"
    ]
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue
            
        print(f"\n检查目录: {output_dir}")
        
        # 查找结果图像
        result_files = list(Path(output_dir).rglob("*result*.png"))
        
        for result_file in result_files[:3]:  # 只检查前3个
            try:
                img = Image.open(result_file)
                img_array = np.array(img)
                
                print(f"\n文件: {result_file.name}")
                print(f"  尺寸: {img.size}")
                print(f"  模式: {img.mode}")
                print(f"  值范围: [{img_array.min()}, {img_array.max()}]")
                print(f"  均值: {img_array.mean():.2f}")
                print(f"  标准差: {img_array.std():.2f}")
                
                # 分析图像特征
                if img_array.std() > 80:
                    print("  ⚠️ 高标准差，可能是彩色花纹")
                
                # 检查颜色分布
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                print(f"  唯一颜色数: {unique_colors}")
                
                if unique_colors > 10000:
                    print("  ⚠️ 颜色过多，可能是彩色花纹")
                
            except Exception as e:
                print(f"  ❌ 无法读取文件: {e}")


def main():
    """主函数"""
    print("诊断企鹅图片输入但输出彩色花纹乱码的问题")
    print("=" * 60)
    
    tests = [
        ("图像处理管道分析", analyze_image_processing_pipeline),
        ("颜色空间转换测试", test_color_space_conversion),
        ("现有输出分析", analyze_existing_outputs),
        ("真实模型测试", test_model_inference_with_penguin),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结:")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n可能的原因:")
    print("1. 模型输出本身就是彩色花纹（这是正常的，因为模型在学习过程中可能产生抽象图案）")
    print("2. 颜色空间转换问题")
    print("3. 模型参数设置问题（如DDPM步数太少）")
    print("4. 输入图像预处理问题")
    print("5. 模型训练数据问题")


if __name__ == "__main__":
    main()
