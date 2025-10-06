#!/usr/bin/env python3
"""
高级诊断脚本 - 调查DDPM步数50仍然输出彩色花纹的问题
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


def check_model_configuration():
    """检查模型配置"""
    print("=== 检查模型配置 ===")
    
    try:
        # 检查配置文件
        config_path = "/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml"
        if os.path.exists(config_path):
            print(f"✅ 主配置文件存在: {config_path}")
            
            # 读取配置文件内容
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
                
            # 检查关键配置
            if "ddpm_steps" in config_content:
                print("✅ 配置文件中包含ddpm_steps设置")
            else:
                print("⚠️ 配置文件中没有ddpm_steps设置")
                
            if "colorfix_type" in config_content:
                print("✅ 配置文件中包含colorfix_type设置")
            else:
                print("⚠️ 配置文件中没有colorfix_type设置")
        else:
            print(f"❌ 主配置文件不存在: {config_path}")
            return False
        
        # 检查edge配置文件
        edge_config_path = "/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        if os.path.exists(edge_config_path):
            print(f"✅ Edge配置文件存在: {edge_config_path}")
        else:
            print(f"⚠️ Edge配置文件不存在: {edge_config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型配置检查失败: {e}")
        return False


def analyze_input_image_quality():
    """分析输入图像质量"""
    print("\n=== 分析输入图像质量 ===")
    
    # 创建测试图像
    def create_test_image():
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
        
        return img
    
    try:
        # 创建测试图像
        test_img = create_test_image()
        test_path = "/tmp/advanced_test_penguin.png"
        Image.fromarray(test_img).save(test_path)
        
        # 分析图像特征
        print(f"测试图像路径: {test_path}")
        print(f"图像尺寸: {test_img.shape}")
        print(f"图像值范围: [{test_img.min()}, {test_img.max()}]")
        print(f"图像均值: {test_img.mean():.2f}")
        print(f"图像标准差: {test_img.std():.2f}")
        
        # 分析颜色分布
        unique_colors = len(np.unique(test_img.reshape(-1, 3), axis=0))
        print(f"唯一颜色数: {unique_colors}")
        
        # 分析图像复杂度
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        print(f"边缘密度: {edge_density:.4f}")
        
        # 检查是否适合超分辨率
        if unique_colors < 10:
            print("⚠️ 图像颜色过于简单，可能不适合超分辨率")
        if test_img.std() < 50:
            print("⚠️ 图像对比度较低")
        if edge_density < 0.01:
            print("⚠️ 图像边缘较少，可能影响超分辨率效果")
        
        return test_path
        
    except Exception as e:
        print(f"❌ 输入图像分析失败: {e}")
        return None


def test_different_parameter_combinations():
    """测试不同的参数组合"""
    print("\n=== 测试不同参数组合 ===")
    
    # 检查模型文件
    model_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
    vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
    
    if not os.path.exists(model_path) or not os.path.exists(vqgan_path):
        print("⚠️ 模型文件不存在，跳过参数测试")
        return False
    
    try:
        from stable_sr_scale_lr import StableSR_ScaleLR
        
        # 创建测试图像
        test_path = analyze_input_image_quality()
        if not test_path:
            return False
        
        # 测试不同的参数组合
        test_configs = [
            {
                "name": "标准配置",
                "params": {
                    "ddpm_steps": 50,
                    "colorfix_type": "adain",
                    "upscale": 2.0,
                    "dec_w": 0.5
                }
            },
            {
                "name": "高质量配置",
                "params": {
                    "ddpm_steps": 50,
                    "colorfix_type": "adain",
                    "upscale": 2.0,
                    "dec_w": 0.7,
                    "tile_overlap": 32,
                    "vqgantile_size": 1280
                }
            },
            {
                "name": "无颜色修正",
                "params": {
                    "ddpm_steps": 50,
                    "colorfix_type": "nofix",
                    "upscale": 2.0,
                    "dec_w": 0.5
                }
            },
            {
                "name": "Wavelet颜色修正",
                "params": {
                    "ddpm_steps": 50,
                    "colorfix_type": "wavelet",
                    "upscale": 2.0,
                    "dec_w": 0.5
                }
            },
            {
                "name": "低dec_w配置",
                "params": {
                    "ddpm_steps": 50,
                    "colorfix_type": "adain",
                    "upscale": 2.0,
                    "dec_w": 0.3
                }
            }
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\n--- 测试配置: {config['name']} ---")
            
            try:
                # 创建处理器
                processor = StableSR_ScaleLR(
                    config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml",
                    ckpt_path=model_path,
                    vqgan_ckpt_path=vqgan_path,
                    **config['params']
                )
                
                # 处理图像
                output_dir = f"/tmp/advanced_test_{config['name'].replace(' ', '_')}"
                print(f"正在处理...")
                processor.process_images(test_path, output_dir)
                
                # 分析输出
                output_files = list(Path(output_dir).rglob("*.png"))
                if output_files:
                    for output_file in output_files:
                        if "RES" in str(output_file):
                            output_img = Image.open(output_file)
                            output_array = np.array(output_img)
                            
                            unique_colors = len(np.unique(output_array.reshape(-1, 3), axis=0))
                            
                            result = {
                                "config": config['name'],
                                "std": output_array.std(),
                                "unique_colors": unique_colors,
                                "mean": output_array.mean(),
                                "size": output_img.size
                            }
                            results.append(result)
                            
                            print(f"  输出: 标准差={result['std']:.2f}, 颜色数={result['unique_colors']}")
                            
                            # 判断质量
                            if result['std'] < 60 and result['unique_colors'] < 10000:
                                print(f"  ✅ 质量良好")
                            elif result['std'] < 80 and result['unique_colors'] < 50000:
                                print(f"  ⚠️ 质量一般")
                            else:
                                print(f"  ❌ 质量差，可能是彩色花纹")
                
            except Exception as e:
                print(f"  ❌ 配置 {config['name']} 测试失败: {e}")
        
        # 总结结果
        print(f"\n=== 参数测试总结 ===")
        for result in results:
            status = "✅ 良好" if result['std'] < 60 else "⚠️ 一般" if result['std'] < 80 else "❌ 差"
            print(f"{result['config']}: 标准差={result['std']:.2f}, 颜色数={result['unique_colors']}, 质量={status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数组合测试失败: {e}")
        return False


def check_model_loading_issues():
    """检查模型加载问题"""
    print("\n=== 检查模型加载问题 ===")
    
    try:
        # 检查模型文件
        model_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        
        print(f"StableSR模型: {model_path}")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024**3)
            print(f"  文件大小: {file_size:.2f} GB")
        else:
            print("  ❌ 文件不存在")
            return False
        
        print(f"VQGAN模型: {vqgan_path}")
        if os.path.exists(vqgan_path):
            file_size = os.path.getsize(vqgan_path) / (1024**3)
            print(f"  文件大小: {file_size:.2f} GB")
        else:
            print("  ❌ 文件不存在")
            return False
        
        # 检查CUDA设备
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name()}")
            print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️ CUDA不可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载检查失败: {e}")
        return False


def test_alternative_approaches():
    """测试替代方法"""
    print("\n=== 测试替代方法 ===")
    
    try:
        from stable_sr_scale_lr_fast import StableSR_ScaleLR_Fast
        
        # 检查模型文件
        model_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        
        if not os.path.exists(model_path) or not os.path.exists(vqgan_path):
            print("⚠️ 模型文件不存在，跳过替代方法测试")
            return False
        
        # 创建测试图像
        test_path = analyze_input_image_quality()
        if not test_path:
            return False
        
        print("测试StableSR_ScaleLR_Fast...")
        
        try:
            # 使用Fast版本
            processor = StableSR_ScaleLR_Fast(
                config_path="/root/dp/StableSR_Edge_v2/configs/stableSRNew/v2-finetune_text_T_512.yaml",
                ckpt_path=model_path,
                vqgan_ckpt_path=vqgan_path,
                ddpm_steps=50,
                colorfix_type="adain",
                upscale=2.0,
                dec_w=0.5
            )
            
            # 处理图像
            output_dir = "/tmp/advanced_test_fast"
            print("正在处理...")
            processor.process_images(test_path, output_dir)
            
            # 分析输出
            output_files = list(Path(output_dir).rglob("*.png"))
            if output_files:
                for output_file in output_files:
                    if "RES" in str(output_file):
                        output_img = Image.open(output_file)
                        output_array = np.array(output_img)
                        
                        unique_colors = len(np.unique(output_array.reshape(-1, 3), axis=0))
                        
                        print(f"Fast版本输出: 标准差={output_array.std():.2f}, 颜色数={unique_colors}")
                        
                        if output_array.std() < 60:
                            print("  ✅ Fast版本质量良好")
                        else:
                            print("  ❌ Fast版本仍有问题")
            
        except Exception as e:
            print(f"❌ Fast版本测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 替代方法测试失败: {e}")
        return False


def check_existing_outputs():
    """检查现有输出"""
    print("\n=== 检查现有输出 ===")
    
    output_dirs = [
        "/root/dp/StableSR_Edge_v2/test_output",
        "/root/dp/StableSR_Edge_v2/edge_inference_output"
    ]
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue
            
        print(f"\n检查目录: {output_dir}")
        
        # 查找最近的结果图像
        result_files = list(Path(output_dir).rglob("*result*.png"))
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for result_file in result_files[:3]:  # 只检查最新的3个
            try:
                img = Image.open(result_file)
                img_array = np.array(img)
                
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                
                print(f"\n文件: {result_file.name}")
                print(f"  尺寸: {img.size}")
                print(f"  标准差: {img_array.std():.2f}")
                print(f"  唯一颜色数: {unique_colors}")
                print(f"  修改时间: {Path(result_file).stat().st_mtime}")
                
                # 分析问题
                if img_array.std() > 80 and unique_colors > 100000:
                    print("  ❌ 确认是彩色花纹问题")
                elif img_array.std() < 60 and unique_colors < 10000:
                    print("  ✅ 图像质量正常")
                else:
                    print("  ⚠️ 图像质量中等")
                
            except Exception as e:
                print(f"  ❌ 无法读取文件: {e}")


def main():
    """主函数"""
    print("高级诊断 - DDPM步数50仍然输出彩色花纹的问题")
    print("=" * 60)
    
    tests = [
        ("模型配置检查", check_model_configuration),
        ("输入图像质量分析", analyze_input_image_quality),
        ("不同参数组合测试", test_different_parameter_combinations),
        ("模型加载问题检查", check_model_loading_issues),
        ("替代方法测试", test_alternative_approaches),
        ("现有输出检查", check_existing_outputs),
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
    
    # 总结和建议
    print("\n" + "=" * 60)
    print("高级诊断总结:")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n可能的高级原因:")
    print("1. 模型权重文件损坏或不完整")
    print("2. 配置文件参数被覆盖")
    print("3. 输入图像质量不适合超分辨率")
    print("4. VQGAN和Diffusion模型不匹配")
    print("5. 内存不足导致处理异常")
    print("6. CUDA版本兼容性问题")
    print("7. 模型训练数据问题")
    
    print("\n建议的解决方案:")
    print("1. 重新下载模型文件")
    print("2. 检查配置文件设置")
    print("3. 尝试不同的输入图像")
    print("4. 使用更保守的参数设置")
    print("5. 检查系统资源使用情况")


if __name__ == "__main__":
    main()
