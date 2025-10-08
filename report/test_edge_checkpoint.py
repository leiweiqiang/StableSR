#!/usr/bin/env python3
"""
测试edge版本的checkpoint
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


def create_test_images():
    """创建多个测试图像"""
    print("=== 创建测试图像 ===")
    
    test_images = []
    
    # 1. 企鹅图像
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(img1, (128, 180), (60, 80), 0, 0, 360, (0, 0, 0), -1)
    cv2.circle(img1, (128, 100), 40, (0, 0, 0), -1)
    cv2.ellipse(img1, (128, 180), (40, 60), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img1, (115, 90), 8, (255, 255, 255), -1)
    cv2.circle(img1, (140, 90), 8, (255, 255, 255), -1)
    cv2.ellipse(img1, (128, 110), (15, 8), 0, 0, 180, (0, 165, 255), -1)
    
    penguin_path = "/tmp/test_penguin.png"
    Image.fromarray(img1).save(penguin_path)
    test_images.append(('企鹅图像', penguin_path))
    print(f"  创建企鹅图像: {penguin_path}")
    
    # 2. 复杂图案
    img2 = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    for i in range(10):
        x, y = np.random.randint(50, 206), np.random.randint(50, 206)
        cv2.circle(img2, (x, y), 20, tuple(np.random.randint(0, 255, 3).tolist()), -1)
    
    complex_path = "/tmp/test_complex.png"
    Image.fromarray(img2).save(complex_path)
    test_images.append(('复杂图案', complex_path))
    print(f"  创建复杂图案: {complex_path}")
    
    # 3. 渐变图像
    img3 = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            img3[i, j] = [i, j, 128]
    
    gradient_path = "/tmp/test_gradient.png"
    Image.fromarray(img3).save(gradient_path)
    test_images.append(('渐变图像', gradient_path))
    print(f"  创建渐变图像: {gradient_path}")
    
    return test_images


def test_edge_checkpoint():
    """测试edge checkpoint"""
    print("\n=== 测试Edge Checkpoint ===")
    
    # 检查checkpoint文件
    checkpoint_path = "./logs/2025-10-07T07-33-48_stablesr_edge_turbo_20251007_073345/checkpoints/epoch=000015.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return False
    
    print(f"✅ 找到checkpoint: {checkpoint_path}")
    file_size = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"   文件大小: {file_size:.2f} GB")
    
    try:
        # 导入StableSR类
        from stable_sr_scale_lr import StableSR_ScaleLR
        
        # VQGAN路径
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        
        if not os.path.exists(vqgan_path):
            print(f"❌ VQGAN文件不存在: {vqgan_path}")
            return False
        
        print(f"✅ 找到VQGAN: {vqgan_path}")
        
        # 创建测试图像
        test_images = create_test_images()
        
        # 测试不同的DDPM步数
        ddpm_steps_list = [20, 50]
        
        results = []
        
        for ddpm_steps in ddpm_steps_list:
            print(f"\n--- 测试DDPM步数: {ddpm_steps} ---")
            
            try:
                # 创建处理器
                print(f"正在加载模型（DDPM步数={ddpm_steps}）...")
                processor = StableSR_ScaleLR(
                    config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml",
                    ckpt_path=checkpoint_path,
                    vqgan_ckpt_path=vqgan_path,
                    ddpm_steps=ddpm_steps,
                    upscale=2.0,
                    colorfix_type="adain",
                    dec_w=0.5
                )
                
                # 测试每个图像
                for img_name, img_path in test_images:
                    print(f"\n  测试图像: {img_name}")
                    
                    # 处理图像
                    output_dir = f"/tmp/edge_test_ddpm{ddpm_steps}_{img_name.replace(' ', '_')}"
                    print(f"  正在处理...")
                    processor.process_images(img_path, output_dir)
                    
                    # 分析输出
                    output_files = list(Path(output_dir).rglob("*.png"))
                    if output_files:
                        for output_file in output_files:
                            if "RES" in str(output_file):
                                output_img = Image.open(output_file)
                                output_array = np.array(output_img)
                                
                                unique_colors = len(np.unique(output_array.reshape(-1, 3), axis=0))
                                
                                result = {
                                    'ddpm_steps': ddpm_steps,
                                    'image': img_name,
                                    'std': output_array.std(),
                                    'unique_colors': unique_colors,
                                    'mean': output_array.mean(),
                                    'size': output_img.size
                                }
                                results.append(result)
                                
                                print(f"    输出: 尺寸={result['size']}, 标准差={result['std']:.2f}, 颜色数={result['unique_colors']}")
                                
                                # 判断质量
                                if result['std'] < 60 and result['unique_colors'] < 10000:
                                    print(f"    ✅ 质量良好")
                                elif result['std'] < 80 and result['unique_colors'] < 50000:
                                    print(f"    ⚠️ 质量一般")
                                else:
                                    print(f"    ❌ 质量差，可能是彩色花纹")
                    else:
                        print("    ❌ 没有生成输出文件")
                
            except Exception as e:
                print(f"  ❌ DDPM步数 {ddpm_steps} 测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 总结结果
        print(f"\n{'='*60}")
        print("Edge Checkpoint 测试总结:")
        print(f"{'='*60}")
        
        for result in results:
            quality = "✅ 良好" if result['std'] < 60 else "⚠️ 一般" if result['std'] < 80 else "❌ 差"
            print(f"DDPM={result['ddpm_steps']}, {result['image']}: "
                  f"标准差={result['std']:.2f}, 颜色数={result['unique_colors']}, 质量={quality}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_base_model():
    """与基础模型对比"""
    print("\n=== 与基础模型对比 ===")
    
    try:
        from stable_sr_scale_lr import StableSR_ScaleLR
        
        # 创建一个测试图像
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.ellipse(img, (128, 180), (60, 80), 0, 0, 360, (0, 0, 0), -1)
        cv2.circle(img, (128, 100), 40, (0, 0, 0), -1)
        cv2.ellipse(img, (128, 180), (40, 60), 0, 0, 360, (255, 255, 255), -1)
        
        test_path = "/tmp/compare_test.png"
        Image.fromarray(img).save(test_path)
        
        # 测试基础模型
        print("\n1. 测试基础模型:")
        base_model_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        
        processor_base = StableSR_ScaleLR(
            config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml",
            ckpt_path=base_model_path,
            vqgan_ckpt_path=vqgan_path,
            ddpm_steps=50,
            upscale=2.0,
            colorfix_type="adain"
        )
        
        output_dir_base = "/tmp/compare_base"
        processor_base.process_images(test_path, output_dir_base)
        
        # 分析基础模型输出
        output_files = list(Path(output_dir_base).rglob("*.png"))
        for output_file in output_files:
            if "RES" in str(output_file):
                output_img = Image.open(output_file)
                output_array = np.array(output_img)
                unique_colors = len(np.unique(output_array.reshape(-1, 3), axis=0))
                print(f"   基础模型: 标准差={output_array.std():.2f}, 颜色数={unique_colors}")
        
        # 测试edge模型
        print("\n2. 测试Edge模型:")
        edge_model_path = "./logs/2025-10-07T07-33-48_stablesr_edge_turbo_20251007_073345/checkpoints/epoch=000015.ckpt"
        
        if os.path.exists(edge_model_path):
            processor_edge = StableSR_ScaleLR(
                config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml",
                ckpt_path=edge_model_path,
                vqgan_ckpt_path=vqgan_path,
                ddpm_steps=50,
                upscale=2.0,
                colorfix_type="adain"
            )
            
            output_dir_edge = "/tmp/compare_edge"
            processor_edge.process_images(test_path, output_dir_edge)
            
            # 分析edge模型输出
            output_files = list(Path(output_dir_edge).rglob("*.png"))
            for output_file in output_files:
                if "RES" in str(output_file):
                    output_img = Image.open(output_file)
                    output_array = np.array(output_img)
                    unique_colors = len(np.unique(output_array.reshape(-1, 3), axis=0))
                    print(f"   Edge模型: 标准差={output_array.std():.2f}, 颜色数={unique_colors}")
        else:
            print(f"   ❌ Edge模型不存在: {edge_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("测试Edge Checkpoint")
    print("=" * 60)
    print(f"Checkpoint: ./logs/2025-10-07T07-33-48_stablesr_edge_turbo_20251007_073345/checkpoints/epoch=000015.ckpt")
    print("=" * 60)
    
    # 测试edge checkpoint
    success = test_edge_checkpoint()
    
    if success:
        print("\n✅ Edge checkpoint测试完成")
        
        # 与基础模型对比
        print("\n" + "=" * 60)
        compare_with_base_model()
    else:
        print("\n❌ Edge checkpoint测试失败")


if __name__ == "__main__":
    main()

