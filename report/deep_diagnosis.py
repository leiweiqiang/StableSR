#!/usr/bin/env python3
"""
深度诊断scale图片乱码问题
检查所有可能导致乱码的原因
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


def check_image_corruption():
    """检查图像是否真的损坏"""
    print("\n=== 检查图像损坏情况 ===")
    
    # 检查所有输出目录中的图像
    output_dirs = [
        "/root/dp/StableSR_Edge_v2/quick_test_results",
        "/root/dp/StableSR_Edge_v2/comprehensive_test_results", 
        "/root/dp/StableSR_Edge_v2/test_output",
        "/root/dp/StableSR_Edge_v2/edge_inference_output"
    ]
    
    corrupted_images = []
    normal_images = []
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue
            
        print(f"\n检查目录: {output_dir}")
        
        # 查找所有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(output_dir).rglob(ext))
        
        for img_file in image_files:
            try:
                # 尝试打开图像
                img = Image.open(img_file)
                img_array = np.array(img)
                
                # 检查基本属性
                file_size = os.path.getsize(img_file)
                
                # 检查是否真的是乱码
                is_corrupted = False
                corruption_reasons = []
                
                # 1. 检查文件大小
                if file_size < 1000:  # 小于1KB可能有问题
                    is_corrupted = True
                    corruption_reasons.append(f"文件太小({file_size} bytes)")
                
                # 2. 检查图像数据
                if img_array.size == 0:
                    is_corrupted = True
                    corruption_reasons.append("图像数据为空")
                
                # 3. 检查异常值
                if img_array.max() > 255 or img_array.min() < 0:
                    is_corrupted = True
                    corruption_reasons.append(f"异常值范围[{img_array.min()}, {img_array.max()}]")
                
                # 4. 检查是否全黑或全白
                if img_array.max() == img_array.min():
                    is_corrupted = True
                    corruption_reasons.append("单色图像")
                
                # 5. 检查标准差（全黑或全白的标准差接近0）
                if img_array.std() < 1.0:
                    is_corrupted = True
                    corruption_reasons.append(f"标准差过小({img_array.std():.2f})")
                
                # 6. 检查图像尺寸
                if img.size[0] < 10 or img.size[1] < 10:
                    is_corrupted = True
                    corruption_reasons.append(f"尺寸过小{img.size}")
                
                if is_corrupted:
                    corrupted_images.append((img_file, corruption_reasons))
                    print(f"  ❌ {img_file.name}: {', '.join(corruption_reasons)}")
                else:
                    normal_images.append(img_file)
                    print(f"  ✅ {img_file.name}: 正常 ({img.size}, {file_size} bytes)")
                    
            except Exception as e:
                corrupted_images.append((img_file, [f"无法打开: {e}"]))
                print(f"  ❌ {img_file.name}: 无法打开 - {e}")
    
    print(f"\n总结:")
    print(f"正常图像: {len(normal_images)}")
    print(f"损坏图像: {len(corrupted_images)}")
    
    return corrupted_images, normal_images


def check_tensor_operations():
    """检查tensor操作中的潜在问题"""
    print("\n=== 检查tensor操作 ===")
    
    try:
        # 创建测试数据
        batch_size = 1
        channels = 3
        height = 128
        width = 160
        
        # 测试各种tensor操作
        test_tensor = torch.randn(batch_size, channels, height, width)
        print(f"原始tensor: {test_tensor.shape}")
        
        # 1. 测试上采样
        upscaled = torch.nn.functional.interpolate(
            test_tensor,
            size=(height * 2, width * 2),
            mode='bicubic'
        )
        print(f"上采样后: {upscaled.shape}")
        
        # 2. 测试填充
        ori_h, ori_w = height, width
        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
            padded = torch.nn.functional.pad(test_tensor, pad=(0, pad_w, 0, pad_h), mode='reflect')
            print(f"填充后: {padded.shape}")
            
            # 3. 测试索引操作（修复后的版本）
            unpadded = padded[:, :, :ori_h, :ori_w]
            print(f"移除填充后: {unpadded.shape}")
            
            if unpadded.shape != test_tensor.shape:
                print("❌ 索引操作有问题!")
                return False
            else:
                print("✅ 索引操作正确")
        
        # 4. 测试数据范围
        clamped = torch.clamp(test_tensor, -1, 1)
        print(f"clamp后范围: [{clamped.min():.3f}, {clamped.max():.3f}]")
        
        # 5. 测试归一化
        normalized = (clamped + 1.0) / 2.0
        print(f"归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ tensor操作测试失败: {e}")
        return False


def check_color_correction():
    """检查颜色修正功能"""
    print("\n=== 检查颜色修正 ===")
    
    try:
        # 创建测试图像
        test_image = torch.randn(1, 3, 64, 64)
        processed_image = torch.randn_like(test_image)
        
        print(f"原始图像范围: [{test_image.min():.3f}, {test_image.max():.3f}]")
        print(f"处理后图像范围: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # 测试不同的颜色修正方法
        methods = ['nofix', 'adain', 'wavelet']
        
        for method in methods:
            print(f"\n测试方法: {method}")
            
            if method == 'nofix':
                result = processed_image
            elif method == 'adain':
                # 简单的adain实现
                mean_orig = test_image.mean(dim=[2, 3], keepdim=True)
                std_orig = test_image.std(dim=[2, 3], keepdim=True)
                mean_proc = processed_image.mean(dim=[2, 3], keepdim=True)
                std_proc = processed_image.std(dim=[2, 3], keepdim=True)
                result = (processed_image - mean_proc) / (std_proc + 1e-8) * std_orig + mean_orig
            else:
                result = processed_image  # wavelet暂时跳过
            
            print(f"  结果范围: [{result.min():.3f}, {result.max():.3f}]")
            
            # 检查是否有异常值
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("  ❌ 发现NaN或Inf值!")
                return False
        
        print("✅ 颜色修正测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 颜色修正测试失败: {e}")
        return False


def check_model_loading():
    """检查模型加载相关问题"""
    print("\n=== 检查模型加载 ===")
    
    try:
        # 检查配置文件
        config_path = "/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml"
        if os.path.exists(config_path):
            print(f"✅ 配置文件存在: {config_path}")
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        # 检查edge配置文件
        edge_config_path = "/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        if os.path.exists(edge_config_path):
            print(f"✅ Edge配置文件存在: {edge_config_path}")
        else:
            print(f"⚠️ Edge配置文件不存在: {edge_config_path}")
        
        # 检查VQGAN配置
        vqgan_config_path = "/root/dp/StableSR_Edge_v2/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"
        if os.path.exists(vqgan_config_path):
            print(f"✅ VQGAN配置文件存在: {vqgan_config_path}")
        else:
            print(f"❌ VQGAN配置文件不存在: {vqgan_config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置检查失败: {e}")
        return False


def check_memory_and_device():
    """检查内存和设备问题"""
    print("\n=== 检查内存和设备 ===")
    
    try:
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"  设备数量: {torch.cuda.device_count()}")
            print(f"  当前设备: {torch.cuda.current_device()}")
            print(f"  设备名称: {torch.cuda.get_device_name()}")
            
            # 检查内存
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  已分配内存: {memory_allocated:.2f} GB")
            print(f"  已保留内存: {memory_reserved:.2f} GB")
        else:
            print("⚠️ CUDA不可用，使用CPU")
        
        # 检查系统内存
        import psutil
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total / 1024**3:.2f} GB")
        print(f"可用内存: {memory.available / 1024**3:.2f} GB")
        print(f"内存使用率: {memory.percent}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存检查失败: {e}")
        return False


def create_test_with_real_model():
    """使用真实模型创建测试"""
    print("\n=== 使用真实模型测试 ===")
    
    # 检查是否有可用的模型文件
    model_paths = [
        "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt",
        "/root/dp/StableSR_Edge_v2/checkpoints/stablesr_turbo.ckpt",
        "/root/dp/StableSR_Edge_v2/models/stablesr_turbo.ckpt"
    ]
    
    vqgan_paths = [
        "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt",
        "/root/dp/StableSR_Edge_v2/checkpoints/vqgan_cfw_00011.ckpt",
        "/root/dp/StableSR_Edge_v2/models/vqgan_cfw_00011.ckpt"
    ]
    
    model_path = None
    vqgan_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    for path in vqgan_paths:
        if os.path.exists(path):
            vqgan_path = path
            break
    
    if model_path and vqgan_path:
        print(f"✅ 找到模型文件: {model_path}")
        print(f"✅ 找到VQGAN文件: {vqgan_path}")
        
        # 创建测试图像
        test_image_path = "/tmp/test_real_model.png"
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(test_image_path)
        
        try:
            from stable_sr_scale_lr import StableSR_ScaleLR
            
            # 创建处理器
            processor = StableSR_ScaleLR(
                config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml",
                ckpt_path=model_path,
                vqgan_ckpt_path=vqgan_path,
                ddpm_steps=4,  # 快速测试
                upscale=2.0
            )
            
            # 处理图像
            output_dir = "/tmp/test_real_output"
            processor.process_images(test_image_path, output_dir)
            
            # 检查输出
            output_files = list(Path(output_dir).rglob("*.png"))
            if output_files:
                print(f"✅ 成功生成 {len(output_files)} 个输出文件")
                
                # 检查第一个输出文件
                output_img = Image.open(output_files[0])
                print(f"  输出图像尺寸: {output_img.size}")
                
                return True
            else:
                print("❌ 没有生成输出文件")
                return False
                
        except Exception as e:
            print(f"❌ 真实模型测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("⚠️ 未找到模型文件，跳过真实模型测试")
        print(f"查找的模型路径: {model_paths}")
        print(f"查找的VQGAN路径: {vqgan_paths}")
        return False


def main():
    """主函数"""
    print("深度诊断scale图片乱码问题")
    print("=" * 60)
    
    # 运行所有检查
    checks = [
        ("图像损坏检查", check_image_corruption),
        ("tensor操作检查", check_tensor_operations),
        ("颜色修正检查", check_color_correction),
        ("模型配置检查", check_model_loading),
        ("内存设备检查", check_memory_and_device),
        ("真实模型测试", create_test_with_real_model),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            if check_name == "图像损坏检查":
                corrupted, normal = check_func()
                results[check_name] = (len(corrupted), len(normal))
            else:
                results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} 异常: {e}")
            results[check_name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结:")
    
    for check_name, result in results.items():
        if check_name == "图像损坏检查":
            corrupted_count, normal_count = result
            print(f"  {check_name}: {corrupted_count} 损坏, {normal_count} 正常")
        else:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {check_name}: {status}")
    
    # 建议
    print("\n建议:")
    if results.get("图像损坏检查", (0, 0))[0] > 0:
        print("1. 发现损坏的图像，需要检查图像生成流程")
    if not results.get("tensor操作检查", True):
        print("2. tensor操作有问题，需要修复代码")
    if not results.get("颜色修正检查", True):
        print("3. 颜色修正有问题，可能导致图像异常")
    if not results.get("模型配置检查", True):
        print("4. 模型配置有问题，检查配置文件路径")
    if not results.get("真实模型测试", True):
        print("5. 真实模型测试失败，检查模型文件和依赖")


if __name__ == "__main__":
    main()
