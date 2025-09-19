#!/usr/bin/env python3
"""
简单测试depth map功能
"""

import torch
import torch.nn as nn

def test_depth_conv_layers():
    """测试depth map卷积层"""
    
    print("测试depth map卷积层...")
    
    # 创建3x3卷积层
    conv_3x3 = nn.Sequential(
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
    
    # 创建4x4卷积层
    conv_4x4 = nn.Sequential(
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
            features = conv_3x3(depth_map)
            print(f"3x3 conv后形状: {features.shape}")
            
            # 4x4 conv处理
            depth_latent = conv_4x4(features)
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

def test_concatenation():
    """测试latent合并功能"""
    
    print("\n测试latent合并...")
    
    # 模拟LR latent和depth latent
    lr_latent = torch.randn(2, 4, 64, 64)  # [N, 4, 64, 64]
    depth_latent = torch.randn(2, 32, 64, 64)  # [N, 32, 64, 64]
    
    print(f"LR latent形状: {lr_latent.shape}")
    print(f"Depth latent形状: {depth_latent.shape}")
    
    try:
        # 合并latent
        combined_latent = torch.cat([lr_latent, depth_latent], dim=1)
        print(f"合并后形状: {combined_latent.shape}")
        
        expected_shape = (2, 36, 64, 64)  # 4 + 32 = 36
        if combined_latent.shape == expected_shape:
            print(f"✅ Latent合并测试通过! 输出形状: {combined_latent.shape}")
            return True
        else:
            print(f"❌ 输出形状不匹配! 期望: {expected_shape}, 实际: {combined_latent.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Latent合并测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始简单测试...\n")
    
    test1 = test_depth_conv_layers()
    test2 = test_concatenation()
    
    print(f"\n测试结果:")
    print(f"  Depth map卷积: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"  Latent合并: {'✅ 通过' if test2 else '❌ 失败'}")
    
    if test1 and test2:
        print("\n🎉 基本功能测试通过!")
    else:
        print("\n⚠️  部分测试失败!")
