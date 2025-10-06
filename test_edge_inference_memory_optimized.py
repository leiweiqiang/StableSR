#!/usr/bin/env python3
"""
内存优化的高分辨率Edge模型推理测试脚本
针对高分辨率测试进行内存优化
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gc

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edge_model_loader import load_edge_model, create_test_image, generate_edge_map
from tra_report import EdgeDDIMSampler


class MemoryOptimizedEdgeInferenceTester:
    """内存优化的高分辨率Edge推理测试器"""
    
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
    
    def clear_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return allocated, reserved
        return 0, 0
    
    def upscale_image(self, image: torch.Tensor, target_size: int) -> torch.Tensor:
        """上采样图像到目标尺寸"""
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
        """上采样图像到精确尺寸"""
        upscaled = F.interpolate(
            image, 
            size=(target_h, target_w), 
            mode='bicubic', 
            align_corners=False
        )
        print(f"精确上采样: {image.shape} -> {upscaled.shape}")
        return upscaled
    
    def inference_with_edge_optimized(self, input_image: torch.Tensor, 
                                    target_size: int = 1024,
                                    target_h: int = None, 
                                    target_w: int = None,
                                    caption: str = "a high resolution image",
                                    ddpm_steps: int = 30,
                                    use_edge_detection: bool = True,
                                    save_intermediate: bool = True,
                                    output_dir: str = "test_output",
                                    clear_memory_between_steps: bool = True) -> torch.Tensor:
        """
        内存优化的高分辨率Edge推理
        
        Args:
            input_image: 输入图像张量 [1, 3, H, W]
            target_size: 目标尺寸（最短边）
            target_h: 目标高度（可选）
            target_w: 目标宽度（可选）
            caption: 文本描述
            ddpm_steps: DDPM采样步数
            use_edge_detection: 是否使用edge检测
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            clear_memory_between_steps: 是否在步骤间清理内存
            
        Returns:
            output_image: 超分辨率输出图像
        """
        print(f"\n开始内存优化的高分辨率Edge推理...")
        print(f"DDPM步数: {ddpm_steps}")
        print(f"使用Edge检测: {use_edge_detection}")
        
        # 显示初始内存使用
        allocated, reserved = self.get_gpu_memory_info()
        print(f"初始GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
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
        
        # 显示上采样后内存使用
        allocated, reserved = self.get_gpu_memory_info()
        print(f"上采样后GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        # 保存上采样后的图像
        if save_intermediate:
            upscaled_np = upscaled_image[0].cpu().permute(1, 2, 0).numpy()
            upscaled_np = (upscaled_np + 1.0) / 2.0
            upscaled_np = (upscaled_np * 255).astype(np.uint8)
            Image.fromarray(upscaled_np).save(os.path.join(output_dir, "upscaled_input.png"))
        
        # 获取文本条件
        cross_attn = self.model.get_learned_conditioning([caption])
        
        if clear_memory_between_steps:
            self.clear_memory()
        
        # 为Edge模型准备struct_cond和edge_map
        if hasattr(self.model, 'use_edge_processing') and self.model.use_edge_processing:
            # 将upscaled_image编码到潜在空间
            encoder_posterior = self.model.encode_first_stage(upscaled_image)
            z_upscaled = self.model.get_first_stage_encoding(encoder_posterior).detach()
            
            if clear_memory_between_steps:
                self.clear_memory()
            
            # 生成struct_cond
            struct_cond = self.model.structcond_stage_model(
                z_upscaled, 
                torch.zeros(1, device=self.device)
            )
            
            # 释放不需要的张量
            del z_upscaled
            if clear_memory_between_steps:
                self.clear_memory()
            
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
        
        # 显示采样前内存使用
        allocated, reserved = self.get_gpu_memory_info()
        print(f"采样前GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
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
        
        # 清理中间变量
        del conditioning
        if clear_memory_between_steps:
            self.clear_memory()
        
        # 解码到图像空间
        print("解码结果...")
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 清理采样结果
        del samples
        if clear_memory_between_steps:
            self.clear_memory()
        
        # 保存结果
        result_image = x_samples[0].cpu().permute(1, 2, 0).numpy()
        result_image = (result_image * 255).astype(np.uint8)
        
        if save_intermediate:
            Image.fromarray(result_image).save(os.path.join(output_dir, "result_high_res.png"))
            print(f"结果已保存: {os.path.join(output_dir, 'result_high_res.png')}")
        
        # 显示最终内存使用
        allocated, reserved = self.get_gpu_memory_info()
        print(f"最终GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        return x_samples
    
    def generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """生成edge map"""
        return generate_edge_map(image)
    
    def test_resolution_with_memory_check(self, input_image: torch.Tensor, 
                                        resolution_config: dict,
                                        caption: str = "a high resolution image",
                                        ddpm_steps: int = 30) -> bool:
        """
        测试指定分辨率，带内存检查
        
        Args:
            input_image: 输入图像
            resolution_config: 分辨率配置 {'size': int} 或 {'h': int, 'w': int}
            caption: 文本描述
            ddpm_steps: DDPM步数
            
        Returns:
            bool: 是否成功
        """
        try:
            # 清理内存
            self.clear_memory()
            
            # 检查内存使用
            allocated, reserved = self.get_gpu_memory_info()
            print(f"测试前内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
            
            if 'size' in resolution_config:
                target_size = resolution_config['size']
                output_dir = f"test_output/res_{target_size}_optimized"
                result = self.inference_with_edge_optimized(
                    input_image=input_image,
                    target_size=target_size,
                    caption=caption,
                    ddpm_steps=ddpm_steps,
                    output_dir=output_dir,
                    clear_memory_between_steps=True
                )
            else:
                target_h = resolution_config['h']
                target_w = resolution_config['w']
                output_dir = f"test_output/res_{target_w}x{target_h}_optimized"
                result = self.inference_with_edge_optimized(
                    input_image=input_image,
                    target_h=target_h,
                    target_w=target_w,
                    caption=caption,
                    ddpm_steps=ddpm_steps,
                    output_dir=output_dir,
                    clear_memory_between_steps=True
                )
            
            # 清理结果
            del result
            self.clear_memory()
            
            return True
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ 内存不足: {e}")
                return False
            else:
                raise e
        except Exception as e:
            print(f"❌ 其他错误: {e}")
            return False
    
    def test_safe_resolutions(self, input_image: torch.Tensor, 
                            caption: str = "a high resolution image",
                            ddpm_steps: int = 30):
        """
        测试安全的分辨率（逐步增加，遇到内存不足就停止）
        """
        print(f"\n{'='*60}")
        print("安全分辨率测试（内存优化版）")
        print(f"{'='*60}")
        
        # 从低分辨率开始测试
        resolutions = [
            {'size': 512},   # 标准分辨率
            {'size': 768},   # 中等分辨率
            {'size': 1024},  # 高分辨率
            {'size': 1280},  # 更高分辨率
            {'size': 1536},  # 超高分辨率
            {'h': 1080, 'w': 1920},  # 2K分辨率
            {'h': 1440, 'w': 2560},  # 2.5K分辨率
        ]
        
        successful_resolutions = []
        
        for i, res in enumerate(resolutions):
            print(f"\n测试分辨率 {i+1}/{len(resolutions)}: {res}")
            
            success = self.test_resolution_with_memory_check(
                input_image=input_image,
                resolution_config=res,
                caption=caption,
                ddpm_steps=ddpm_steps
            )
            
            if success:
                successful_resolutions.append(res)
                print(f"✓ 分辨率 {res} 测试成功")
            else:
                print(f"✗ 分辨率 {res} 测试失败（内存不足）")
                print("停止测试更高分辨率...")
                break
        
        print(f"\n{'='*60}")
        print("安全分辨率测试完成")
        print(f"成功测试的分辨率: {successful_resolutions}")
        print(f"{'='*60}")
        
        return successful_resolutions


def main():
    parser = argparse.ArgumentParser(description="内存优化的高分辨率Edge模型推理测试")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input", type=str, help="输入图像路径")
    parser.add_argument("--target_size", type=int, default=1024, help="目标尺寸（最短边）")
    parser.add_argument("--target_h", type=int, help="目标高度")
    parser.add_argument("--target_w", type=int, help="目标宽度")
    parser.add_argument("--caption", type=str, default="a high resolution image", help="文本描述")
    parser.add_argument("--steps", type=int, default=30, help="DDPM采样步数")
    parser.add_argument("--output", type=str, default="test_output_optimized", help="输出目录")
    parser.add_argument("--synthetic", action="store_true", help="使用合成图像测试")
    parser.add_argument("--safe_test", action="store_true", help="安全分辨率测试")
    
    args = parser.parse_args()
    
    print("内存优化的高分辨率Edge模型推理测试")
    print("="*50)
    
    # 创建测试器
    tester = MemoryOptimizedEdgeInferenceTester(args.config, args.ckpt)
    
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
        if args.safe_test:
            # 安全分辨率测试
            successful_resolutions = tester.test_safe_resolutions(
                input_image=input_image,
                caption=args.caption,
                ddpm_steps=args.steps
            )
            print(f"\n✓ 安全分辨率测试完成")
            print(f"成功测试的分辨率: {successful_resolutions}")
            
        else:
            # 单分辨率测试
            result = tester.inference_with_edge_optimized(
                input_image=input_image,
                target_size=args.target_size if not (args.target_h and args.target_w) else None,
                target_h=args.target_h,
                target_w=args.target_w,
                caption=args.caption,
                ddpm_steps=args.steps,
                output_dir=args.output,
                clear_memory_between_steps=True
            )
            print(f"\n✓ 内存优化测试完成，结果保存在: {args.output}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
