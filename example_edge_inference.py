#!/usr/bin/env python3
"""
StableSR Edge Map 推理示例
最简单的edge模型推理示例代码
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
from tra_report import EdgeDDIMSampler
import torch.nn.functional as F
from omegaconf import OmegaConf


def example_edge_inference():
    """Edge模型推理示例"""
    print("StableSR Edge Map 推理示例")
    print("="*40)
    
    # 1. 配置路径（请根据实际情况修改）
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    ckpt_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请修改config_path为正确的配置文件路径")
        return
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 模型检查点不存在: {ckpt_path}")
        print("请修改ckpt_path为正确的模型路径")
        return
    
    try:
        # 2. 加载模型
        print("1. 加载模型...")
        config = OmegaConf.load(config_path)
        
        # 从配置中提取模型参数
        model_config = config.model.params
        
        # 创建模型实例
        model = LatentDiffusionSRTextWTWithEdge(
            first_stage_config=model_config.first_stage_config,
            cond_stage_config=model_config.cond_stage_config,
            structcond_stage_config=model_config.structcond_stage_config,
            num_timesteps_cond=model_config.get('num_timesteps_cond', 1),
            cond_stage_key=model_config.get('cond_stage_key', 'image'),
            cond_stage_trainable=model_config.get('cond_stage_trainable', False),
            concat_mode=model_config.get('concat_mode', True),
            conditioning_key=model_config.get('conditioning_key', 'crossattn'),
            scale_factor=model_config.get('scale_factor', 0.18215),
            scale_by_std=model_config.get('scale_by_std', False),
            unfrozen_diff=model_config.get('unfrozen_diff', False),
            random_size=model_config.get('random_size', False),
            test_gt=model_config.get('test_gt', False),
            p2_gamma=model_config.get('p2_gamma', None),
            p2_k=model_config.get('p2_k', None),
            time_replace=model_config.get('time_replace', 1000),
            use_usm=model_config.get('use_usm', True),
            mix_ratio=model_config.get('mix_ratio', 0.0),
            use_edge_processing=model_config.get('use_edge_processing', True),
            edge_input_channels=model_config.get('edge_input_channels', 3),
            linear_start=model_config.get('linear_start', 0.00085),
            linear_end=model_config.get('linear_end', 0.0120),
            timesteps=model_config.get('timesteps', 1000),
            first_stage_key=model_config.get('first_stage_key', 'image'),
            image_size=model_config.get('image_size', 512),
            channels=model_config.get('channels', 4),
            unet_config=model_config.get('unet_config', None),
            use_ema=model_config.get('use_ema', False)
        ).cuda()
        
        # 加载检查点
        print("加载检查点...")
        model.init_from_ckpt(ckpt_path)
        
        sampler = EdgeDDIMSampler(model)
        print("✓ 模型加载成功")
        
        # 3. 创建测试图像
        print("2. 创建测试图像...")
        # 创建一个简单的测试图像（包含边缘）
        test_img = np.ones((256, 256, 3), dtype=np.float32)
        
        # 添加一些几何形状来产生边缘
        cv2.rectangle(test_img, (50, 50), (150, 150), (0.8, 0.2, 0.2), -1)  # 红色矩形
        cv2.circle(test_img, (200, 200), 50, (0.2, 0.8, 0.2), -1)           # 绿色圆形
        cv2.line(test_img, (100, 200), (250, 100), (0.2, 0.2, 0.8), 3)      # 蓝色线条
        
        # 转换为tensor
        img_tensor = torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5  # 归一化到[-1, 1]
        img_tensor = img_tensor.cuda()
        print(f"✓ 测试图像创建完成: {img_tensor.shape}")
        
        # 4. 上采样图像
        print("3. 上采样图像...")
        upscaled_img = F.interpolate(img_tensor, size=(512, 512), mode='bicubic')
        print(f"✓ 图像上采样完成: {upscaled_img.shape}")
        
        # 5. 生成edge map
        print("4. 生成edge map...")
        edge_map = generate_edge_map(upscaled_img)
        print(f"✓ Edge map生成完成: {edge_map.shape}")
        
        # 6. 准备推理条件
        print("5. 准备推理条件...")
        
        # 文本条件（空字符串表示无文本）
        cross_attn = model.get_learned_conditioning([""])
        
        # 编码到潜在空间
        encoder_posterior = model.encode_first_stage(upscaled_img)
        z_upscaled = model.get_first_stage_encoding(encoder_posterior).detach()
        
        # 生成结构条件
        struct_cond = model.structcond_stage_model(z_upscaled, torch.zeros(1, device='cuda'))
        
        # 构建条件字典
        conditioning = {
            "c_concat": upscaled_img,
            "c_crossattn": cross_attn,
            "struct_cond": struct_cond,
            "edge_map": edge_map
        }
        print("✓ 推理条件准备完成")
        
        # 7. 执行推理
        print("6. 执行推理...")
        with torch.no_grad():
            samples, _ = sampler.sample(
                S=20,  # DDPM步数
                conditioning=conditioning,
                batch_size=1,
                shape=(4, 64, 64),  # 潜在空间形状
                verbose=True
            )
        print("✓ 推理完成")
        
        # 8. 解码结果
        print("7. 解码结果...")
        result = model.decode_first_stage(samples)
        result = torch.clamp((result + 1.0) / 2.0, min=0.0, max=1.0)
        print("✓ 结果解码完成")
        
        # 9. 保存结果
        print("8. 保存结果...")
        os.makedirs("example_output", exist_ok=True)
        
        # 保存输入图像
        input_np = upscaled_img[0].cpu().permute(1, 2, 0).numpy()
        input_np = (input_np + 1.0) / 2.0
        input_np = (input_np * 255).astype(np.uint8)
        Image.fromarray(input_np).save("example_output/input.png")
        
        # 保存edge map
        edge_np = edge_map[0].cpu().permute(1, 2, 0).numpy()
        edge_np = (edge_np + 1.0) / 2.0
        edge_np = (edge_np * 255).astype(np.uint8)
        Image.fromarray(edge_np).save("example_output/edge_map.png")
        
        # 保存最终结果
        result_np = result[0].cpu().permute(1, 2, 0).numpy()
        result_np = (result_np * 255).astype(np.uint8)
        Image.fromarray(result_np).save("example_output/result.png")
        
        print("✓ 结果保存完成")
        print("\n结果文件:")
        print("  - example_output/input.png: 输入图像")
        print("  - example_output/edge_map.png: Edge map")
        print("  - example_output/result.png: 超分辨率结果")
        
        print("\n🎉 示例运行成功！")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


def generate_edge_map(image_tensor):
    """生成edge map的简化函数"""
    # 转换为numpy数组
    img_np = image_tensor[0].cpu().numpy()
    img_np = (img_np + 1.0) / 2.0  # 从[-1, 1]转换到[0, 1]
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # 从[C, H, W]转换到[H, W, C]
    
    # 转换为灰度图
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 应用高斯模糊
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    
    # 应用Canny边缘检测
    edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
    
    # 转换为3通道
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # 转换回tensor格式
    edges_tensor = torch.from_numpy(edges_3ch).permute(2, 0, 1).unsqueeze(0).float()
    edges_tensor = (edges_tensor / 127.5) - 1.0  # 归一化到[-1, 1]
    
    return edges_tensor.to(image_tensor.device)


if __name__ == "__main__":
    example_edge_inference()
