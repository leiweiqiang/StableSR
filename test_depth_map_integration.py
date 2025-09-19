#!/usr/bin/env python3
"""
测试depth map集成功能
"""

import torch
import torch.nn as nn
import numpy as np
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModelWT

def test_encoder_unet_with_depth_map():
    """测试EncoderUNetModelWT的depth map处理功能"""
    
    print("测试EncoderUNetModelWT with depth map...")
    
    # 创建模型实例
    model = EncoderUNetModelWT(
        image_size=96,
        in_channels=4,
        model_channels=256,
        out_channels=256,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0,
        channel_mult=[1, 1, 2, 2],
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_depth_map=True  # 启用depth map处理
    )
    
    # 创建测试数据
    batch_size = 2
    # LR latent image: [N, 4, 64, 64]
    lr_latent = torch.randn(batch_size, 4, 64, 64)
    # Depth map: [N, 1, 2048, 2048] (2Kx2K)
    depth_map = torch.randn(batch_size, 1, 2048, 2048)
    # Timesteps
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"输入形状:")
    print(f"  LR latent: {lr_latent.shape}")
    print(f"  Depth map: {depth_map.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    
    # 测试forward pass
    try:
        with torch.no_grad():
            results = model(lr_latent, timesteps, depth_map)
        
        print(f"输出结果:")
        for key, value in results.items():
            print(f"  {key}: {value.shape}")
        
        print("✅ EncoderUNetModelWT depth map处理测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_encoder_unet_without_depth_map():
    """测试EncoderUNetModelWT不使用depth map的情况"""
    
    print("\n测试EncoderUNetModelWT without depth map...")
    
    # 创建模型实例
    model = EncoderUNetModelWT(
        image_size=96,
        in_channels=4,
        model_channels=256,
        out_channels=256,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0,
        channel_mult=[1, 1, 2, 2],
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_depth_map=False  # 不启用depth map处理
    )
    
    # 创建测试数据
    batch_size = 2
    lr_latent = torch.randn(batch_size, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"输入形状:")
    print(f"  LR latent: {lr_latent.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    
    # 测试forward pass
    try:
        with torch.no_grad():
            results = model(lr_latent, timesteps, depth_map=None)
        
        print(f"输出结果:")
        for key, value in results.items():
            print(f"  {key}: {value.shape}")
        
        print("✅ EncoderUNetModelWT 无depth map测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_depth_map_processing():
    """测试depth map处理的具体步骤"""
    
    print("\n测试depth map处理步骤...")
    
    # 创建depth map处理网络
    depth_conv_3x3 = nn.Sequential(
        nn.Conv2d(1, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 768, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(768, 1024, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.ReLU(inplace=True),
    )
    
    depth_conv_4x4 = nn.Sequential(
        nn.Conv2d(1024, 512, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )
    
    # 测试数据: 2Kx2K depth map
    depth_map = torch.randn(1, 1, 2048, 2048)
    print(f"输入depth map形状: {depth_map.shape}")
    
    try:
        with torch.no_grad():
            # 3x3 conv处理
            features = depth_conv_3x3(depth_map)
            print(f"3x3 conv后形状: {features.shape}")
            
            # 4x4 conv处理
            depth_latent = depth_conv_4x4(features)
            print(f"4x4 conv后形状: {depth_latent.shape}")
            
            # 验证输出尺寸
            expected_shape = (1, 32, 64, 64)  # 2048 / 2^5 = 64
            if depth_latent.shape == expected_shape:
                print(f"✅ Depth map处理测试通过! 输出形状: {depth_latent.shape}")
                return True
            else:
                print(f"❌ 输出形状不匹配! 期望: {expected_shape}, 实际: {depth_latent.shape}")
                return False
                
    except Exception as e:
        print(f"❌ Depth map处理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试depth map集成功能...\n")
    
    # 运行所有测试
    test1 = test_depth_map_processing()
    test2 = test_encoder_unet_without_depth_map()
    test3 = test_encoder_unet_with_depth_map()
    
    print(f"\n测试结果总结:")
    print(f"  Depth map处理: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"  无depth map模式: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"  有depth map模式: {'✅ 通过' if test3 else '❌ 失败'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 所有测试通过! depth map集成功能正常工作!")
    else:
        print("\n⚠️  部分测试失败，请检查实现!")
