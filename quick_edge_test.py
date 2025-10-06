#!/usr/bin/env python3
"""
StableSR Edge Map 快速测试脚本
简化的edge模型推理测试
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
from tra_report import EdgeDDIMSampler
from edge_model_loader import load_edge_model, create_test_image, generate_edge_map
import torch.nn.functional as F
from omegaconf import OmegaConf


# 使用统一的工具函数


def quick_test():
    """快速测试edge模型推理"""
    print("StableSR Edge Map 快速测试")
    print("="*40)
    
    # 配置路径
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    ckpt_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"  # 请修改为实际路径
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 模型检查点不存在: {ckpt_path}")
        print("请修改ckpt_path为实际的模型路径")
        return
    
    try:
        # 加载模型
        model, sampler = load_edge_model(config_path, ckpt_path)
        
        # 创建测试图像
        print("创建测试图像...")
        img_tensor = create_test_image()
        img_tensor = img_tensor.cuda()
        
        print(f"测试图像尺寸: {img_tensor.shape}")
        
        # 上采样图像
        upscaled_img = F.interpolate(img_tensor, size=(512, 512), mode='bicubic')
        
        # 生成edge map
        print("生成edge map...")
        edge_map = generate_edge_map(upscaled_img)
        print(f"Edge map尺寸: {edge_map.shape}")
        
        # 保存中间结果
        os.makedirs("quick_test_output", exist_ok=True)
        
        # 保存输入图像
        input_np = upscaled_img[0].cpu().permute(1, 2, 0).numpy()
        input_np = (input_np + 1.0) / 2.0
        input_np = (input_np * 255).astype(np.uint8)
        Image.fromarray(input_np).save("quick_test_output/input.png")
        
        # 保存edge map
        edge_np = edge_map[0].cpu().permute(1, 2, 0).numpy()
        edge_np = (edge_np + 1.0) / 2.0
        edge_np = (edge_np * 255).astype(np.uint8)
        Image.fromarray(edge_np).save("quick_test_output/edge_map.png")
        
        # 准备推理条件
        print("准备推理条件...")
        cross_attn = model.get_learned_conditioning([""])
        
        # 编码到潜在空间
        encoder_posterior = model.encode_first_stage(upscaled_img)
        z_upscaled = model.get_first_stage_encoding(encoder_posterior).detach()
        
        # 生成struct_cond
        struct_cond = model.structcond_stage_model(z_upscaled, torch.zeros(1, device='cuda'))
        
        # 构建条件
        conditioning = {
            "c_concat": upscaled_img,
            "c_crossattn": cross_attn,
            "struct_cond": struct_cond,
            "edge_map": edge_map
        }
        
        # 执行推理
        print("执行推理...")
        with torch.no_grad():
            samples, _ = sampler.sample(
                S=20,  # DDPM步数
                conditioning=conditioning,
                batch_size=1,
                shape=(4, 64, 64),
                verbose=True
            )
        
        # 解码结果
        print("解码结果...")
        result = model.decode_first_stage(samples)
        result = torch.clamp((result + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 保存结果
        result_np = result[0].cpu().permute(1, 2, 0).numpy()
        result_np = (result_np * 255).astype(np.uint8)
        Image.fromarray(result_np).save("quick_test_output/result.png")
        
        print("✓ 推理完成！")
        print("结果保存在 quick_test_output/ 目录:")
        print("  - input.png: 输入图像")
        print("  - edge_map.png: edge map")
        print("  - result.png: 超分辨率结果")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
