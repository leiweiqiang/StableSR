#!/usr/bin/env python3
"""
StableSR Edge模型加载工具
提供统一的模型加载接口
"""

import torch
from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
from tra_report import EdgeDDIMSampler


def load_edge_model(config_path, ckpt_path, device="cuda"):
    """
    加载StableSR Edge模型
    
    Args:
        config_path: 配置文件路径
        ckpt_path: 模型检查点路径
        device: 设备类型
        
    Returns:
        model: 加载的模型
        sampler: DDIM采样器
    """
    print(f"加载Edge模型...")
    print(f"配置文件: {config_path}")
    print(f"检查点: {ckpt_path}")
    print(f"设备: {device}")
    
    # 加载配置
    config = OmegaConf.load(config_path)
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
    ).to(device)
    
    # 加载检查点
    print("加载检查点...")
    model.init_from_ckpt(ckpt_path)
    
    # 创建DDIM采样器
    sampler = EdgeDDIMSampler(model)
    
    print("✓ Edge模型加载成功")
    
    # 检查edge处理支持
    if hasattr(model, 'use_edge_processing') and model.use_edge_processing:
        print("✓ 模型支持edge处理")
    else:
        print("⚠️ 模型不支持edge处理")
    
    return model, sampler


def create_test_image(size=(256, 256)):
    """
    创建测试图像
    
    Args:
        size: 图像尺寸 (height, width)
        
    Returns:
        image_tensor: 测试图像张量 [1, 3, H, W]，值范围 [-1, 1]
    """
    import cv2
    import numpy as np
    
    # 创建白色背景
    img = np.ones((size[0], size[1], 3), dtype=np.float32)
    
    # 添加几何形状
    cv2.rectangle(img, (50, 50), (150, 150), (0.8, 0.2, 0.2), -1)  # 红色矩形
    cv2.circle(img, (200, 200), 50, (0.2, 0.8, 0.2), -1)           # 绿色圆形
    cv2.line(img, (100, 200), (250, 100), (0.2, 0.2, 0.8), 3)      # 蓝色线条
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5  # 归一化到[-1, 1]
    
    return img_tensor


def generate_edge_map(image_tensor):
    """
    生成edge map
    
    Args:
        image_tensor: 输入图像张量 [1, 3, H, W]，值范围 [-1, 1]
        
    Returns:
        edge_map: edge map张量 [1, 3, H, W]，值范围 [-1, 1]
    """
    import cv2
    import numpy as np
    
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
    # 测试模型加载
    import os
    
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    ckpt_path = "/stablesr_dataset/checkpoints/stablesr_turbo.ckpt"
    
    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        try:
            model, sampler = load_edge_model(config_path, ckpt_path)
            print("模型加载测试成功！")
        except Exception as e:
            print(f"模型加载测试失败: {e}")
    else:
        print("配置文件或检查点不存在，跳过测试")
