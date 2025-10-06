#!/usr/bin/env python3
"""
高分辨率Edge模型推理测试脚本
支持更大的输出分辨率测试
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edge_model_loader import load_edge_model, create_test_image, generate_edge_map
from tra_report import EdgeDDIMSampler


class HighResEdgeInferenceTester:
    """高分辨率Edge推理测试器"""
    
    def __init__(self, config_path: str, ckpt_path: str):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"使用设备: {self.device}")
        print(f"配置文件: {config_path}")
        print(f"模型检查点: {ckpt_path}")
        
        self._load_model()
    
    def _load_model(self):
        """加载Edge模型"""
        print("加载Edge模型...")
        self.model, self.sampler = load_edge_model(
            self.config_path, 
            self.ckpt_path, 
            self.device
        )
        print("✓ Edge模型加载成功")
        print("✓ 模型支持edge处理")
    
    def upscale_image(self, image: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        上采样图像到目标尺寸
        
        Args:
            image: 输入图像 [1, 3, H, W]
            target_size: 目标尺寸（最短边）
            
        Returns:
            upscaled_image: 上采样后的图像
        """
        current_size = min(image.shape[-2:])
        
        if current_size < target_size:
            scale_factor = target_size / current_size
            new_h = int(image.shape[-2] * scale_factor)
            new_w = int(image.shape[-1] * scale_factor)
            
            upscaled = F.interpolate(
                image, 
                size=(new_h, new_w), 
                mode='bicubic', 
                align_corners=False
            )
            print(f"上采样: {image.shape} -> {upscaled.shape}")
        else:
            upscaled = image
            print(f"图像尺寸已足够大: {image.shape}")
            
        return upscaled
    
    def upscale_image_to_exact_size(self, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        上采样图像到精确尺寸
        
        Args:
            image: 输入图像 [1, 3, H, W]
            target_h: 目标高度
            target_w: 目标宽度
            
        Returns:
            upscaled_image: 上采样后的图像
        """
        upscaled = F.interpolate(
            image, 
            size=(target_h, target_w), 
            mode='bicubic', 
            align_corners=False
        )
        print(f"精确上采样: {image.shape} -> {upscaled.shape}")
        return upscaled
    
    def inference_with_edge_high_res(self, input_image: torch.Tensor, 
                                   target_size: int = 1024,
                                   target_h: int = None, 
                                   target_w: int = None,
                                   caption: str = "a high resolution image",
                                   ddpm_steps: int = 50,
                                   use_edge_detection: bool = True,
                                   save_intermediate: bool = True,
                                   output_dir: str = "test_output") -> torch.Tensor:
        """
        高分辨率Edge推理
        
        Args:
            input_image: 输入图像张量 [1, 3, H, W]
            target_size: 目标尺寸（最短边），如果指定了target_h和target_w则忽略
            target_h: 目标高度（可选）
            target_w: 目标宽度（可选）
            caption: 文本描述
            ddpm_steps: DDPM采样步数
            use_edge_detection: 是否使用edge检测
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            
        Returns:
            output_image: 超分辨率输出图像
        """
        print(f"\n开始高分辨率Edge推理...")
        print(f"DDPM步数: {ddpm_steps}")
        print(f"使用Edge检测: {use_edge_detection}")
        
        # 创建输出目录
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        # 上采样输入图像
        if target_h is not None and target_w is not None:
            upscaled_image = self.upscale_image_to_exact_size(input_image, target_h, target_w)
            print(f"目标输出尺寸: {target_w} × {target_h}")
        else:
            upscaled_image = self.upscale_image(input_image, target_size)
            print(f"目标尺寸（最短边）: {target_size}")
        
        # 保存上采样后的图像
        if save_intermediate:
            upscaled_np = upscaled_image[0].cpu().permute(1, 2, 0).numpy()
            upscaled_np = (upscaled_np + 1.0) / 2.0
            upscaled_np = (upscaled_np * 255).astype(np.uint8)
            Image.fromarray(upscaled_np).save(os.path.join(output_dir, "upscaled_input.png"))
        
        # 获取文本条件
        cross_attn = self.model.get_learned_conditioning([caption])
        
        # 为Edge模型准备struct_cond和edge_map
        if hasattr(self.model, 'use_edge_processing') and self.model.use_edge_processing:
            # 将upscaled_image编码到潜在空间
            encoder_posterior = self.model.encode_first_stage(upscaled_image)
            z_upscaled = self.model.get_first_stage_encoding(encoder_posterior).detach()
            
            # 生成struct_cond
            struct_cond = self.model.structcond_stage_model(
                z_upscaled, 
                torch.zeros(1, device=self.device)
            )
            
            # 生成edge_map（如果启用边缘检测）
            if use_edge_detection:
                edge_map = self.generate_edge_map(upscaled_image)
                print(f"生成edge_map: {edge_map.shape}")
                
                # 保存edge map
                if save_intermediate:
                    edge_np = edge_map[0].cpu().permute(1, 2, 0).numpy()
                    edge_np = (edge_np + 1.0) / 2.0
                    edge_np = (edge_np * 255).astype(np.uint8)
                    Image.fromarray(edge_np).save(os.path.join(output_dir, "edge_map.png"))
                
                conditioning = {
                    "c_concat": upscaled_image, 
                    "c_crossattn": cross_attn,
                    "struct_cond": struct_cond,
                    "edge_map": edge_map
                }
            else:
                print("Edge检测已禁用，仅使用struct_cond")
                conditioning = {
                    "c_concat": upscaled_image, 
                    "c_crossattn": cross_attn,
                    "struct_cond": struct_cond
                }
        else:
            print("模型不支持edge处理，使用标准推理")
            conditioning = {
                "c_concat": upscaled_image, 
                "c_crossattn": cross_attn
            }
        
        # 执行DDIM采样
        print("执行DDIM采样...")
        start_time = time.time()
        
        with torch.no_grad():
            samples, intermediates = self.sampler.sample(
                S=ddpm_steps,
                conditioning=conditioning,
                batch_size=1,
                shape=(4, 64, 64),  # latent shape
                verbose=True
            )
        
        sampling_time = time.time() - start_time
        print(f"采样完成，耗时: {sampling_time:.2f}秒")
        
        # 解码到图像空间
        print("解码结果...")
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 保存结果
        result_image = x_samples[0].cpu().permute(1, 2, 0).numpy()
        result_image = (result_image * 255).astype(np.uint8)
        
        if save_intermediate:
            Image.fromarray(result_image).save(os.path.join(output_dir, "result_high_res.png"))
            print(f"结果已保存: {os.path.join(output_dir, 'result_high_res.png')}")
        
        return x_samples
    
    def generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """生成edge map"""
        return generate_edge_map(image)
    
    def test_multiple_resolutions(self, input_image: torch.Tensor, 
                                resolutions: list = None,
                                caption: str = "a high resolution image",
                                ddpm_steps: int = 50):
        """
        测试多个分辨率输出
        
        Args:
            input_image: 输入图像
            resolutions: 分辨率列表，每个元素可以是:
                        - int: 最短边尺寸
                        - tuple: (height, width)
            caption: 文本描述
            ddpm_steps: DDPM步数
        """
        if resolutions is None:
            resolutions = [
                512,   # 标准尺寸
                768,   # 中等高分辨率
                1024,  # 高分辨率
                1536,  # 超高分辨率
                (1080, 1920),  # 2K分辨率
                (1440, 2560),  # 2.5K分辨率
            ]
        
        print(f"\n{'='*60}")
        print("多分辨率测试")
        print(f"{'='*60}")
        
        results = {}
        
        for i, res in enumerate(resolutions):
            print(f"\n测试分辨率 {i+1}/{len(resolutions)}: {res}")
            
            if isinstance(res, tuple):
                target_h, target_w = res
                output_dir = f"test_output/res_{target_w}x{target_h}"
                result = self.inference_with_edge_high_res(
                    input_image=input_image,
                    target_h=target_h,
                    target_w=target_w,
                    caption=caption,
                    ddpm_steps=ddpm_steps,
                    output_dir=output_dir
                )
                results[f"{target_w}x{target_h}"] = result
            else:
                output_dir = f"test_output/res_{res}"
                result = self.inference_with_edge_high_res(
                    input_image=input_image,
                    target_size=res,
                    caption=caption,
                    ddpm_steps=ddpm_steps,
                    output_dir=output_dir
                )
                results[f"{res}"] = result
            
            print(f"✓ 分辨率 {res} 测试完成")
        
        print(f"\n{'='*60}")
        print("多分辨率测试完成")
        print(f"{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="高分辨率Edge模型推理测试")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input", type=str, help="输入图像路径")
    parser.add_argument("--target_size", type=int, default=1024, help="目标尺寸（最短边）")
    parser.add_argument("--target_h", type=int, help="目标高度")
    parser.add_argument("--target_w", type=int, help="目标宽度")
    parser.add_argument("--caption", type=str, default="a high resolution image", help="文本描述")
    parser.add_argument("--steps", type=int, default=50, help="DDPM采样步数")
    parser.add_argument("--output", type=str, default="test_output_high_res", help="输出目录")
    parser.add_argument("--synthetic", action="store_true", help="使用合成图像测试")
    parser.add_argument("--multi_res", action="store_true", help="测试多个分辨率")
    
    args = parser.parse_args()
    
    print("高分辨率Edge模型推理测试")
    print("="*50)
    
    # 创建测试器
    tester = HighResEdgeInferenceTester(args.config, args.ckpt)
    
    # 准备输入图像
    if args.synthetic:
        print("\n使用合成图像测试")
        input_image = create_test_image(size=(256, 256)).to(tester.device)
        input_np = input_image[0].cpu().permute(1, 2, 0).numpy()
        input_np = (input_np + 1.0) / 2.0
        input_np = (input_np * 255).astype(np.uint8)
        
        # 保存输入图像
        os.makedirs(args.output, exist_ok=True)
        Image.fromarray(input_np).save(os.path.join(args.output, "input_synthetic.png"))
        
    elif args.input:
        print(f"\n加载输入图像: {args.input}")
        input_image = Image.open(args.input).convert('RGB')
        input_array = np.array(input_image)
        input_tensor = torch.from_numpy(input_array).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        input_image = input_tensor.to(tester.device)
        
        # 保存输入图像
        os.makedirs(args.output, exist_ok=True)
        Image.fromarray(input_array).save(os.path.join(args.output, "input_original.png"))
        
    else:
        print("错误: 请指定 --input 图像路径或使用 --synthetic 进行合成图像测试")
        return
    
    print(f"输入图像尺寸: {input_image.shape}")
    
    # 执行推理
    try:
        if args.multi_res:
            # 多分辨率测试
            results = tester.test_multiple_resolutions(
                input_image=input_image,
                caption=args.caption,
                ddpm_steps=args.steps
            )
            print(f"\n✓ 多分辨率测试完成，结果保存在 test_output/res_* 目录中")
            
        else:
            # 单分辨率测试
            result = tester.inference_with_edge_high_res(
                input_image=input_image,
                target_size=args.target_size if not (args.target_h and args.target_w) else None,
                target_h=args.target_h,
                target_w=args.target_w,
                caption=args.caption,
                ddpm_steps=args.steps,
                output_dir=args.output
            )
            print(f"\n✓ 高分辨率测试完成，结果保存在: {args.output}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
