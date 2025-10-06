#!/usr/bin/env python3
"""
StableSR Edge Map 推理测试脚本
用于测试edge模型的推理功能
"""

import os
import sys
import argparse
import time
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
from tra_report import EdgeDDIMSampler
from edge_model_loader import load_edge_model, create_test_image, generate_edge_map
import torch.nn.functional as F


class EdgeInferenceTester:
    """Edge模型推理测试器"""
    
    def __init__(self, config_path, ckpt_path, device="cuda"):
        """
        初始化Edge推理测试器
        
        Args:
            config_path: 配置文件路径
            ckpt_path: 模型检查点路径
            device: 设备类型
        """
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        
        print(f"使用设备: {self.device}")
        print(f"配置文件: {config_path}")
        print(f"模型检查点: {ckpt_path}")
        
        # 加载配置
        self.config = OmegaConf.load(config_path)
        
        # 加载模型
        self.model = None
        self.sampler = None
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        try:
            # 使用统一的模型加载器
            self.model, self.sampler = load_edge_model(
                self.config_path, 
                self.ckpt_path, 
                self.device
            )
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """使用统一的edge map生成器"""
        return generate_edge_map(image)
    
    def load_image(self, image_path, target_size=None):
        """
        加载图像
        
        Args:
            image_path: 图像路径
            target_size: 目标尺寸 (height, width)
            
        Returns:
            image_tensor: 图像张量 [1, 3, H, W]，值范围 [-1, 1]
        """
        # 读取图像
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 调整尺寸
        if target_size:
            image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        
        # 转换为numpy数组
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = (image_tensor - 0.5) / 0.5  # 归一化到[-1, 1]
        image_tensor = image_tensor.to(self.device)
        
        print(f"加载图像: {image_path}")
        print(f"原始尺寸: {original_size}")
        print(f"当前尺寸: {image_tensor.shape}")
        
        return image_tensor
    
    def upscale_image(self, image: torch.Tensor, target_size=512) -> torch.Tensor:
        """
        上采样图像到目标尺寸
        
        Args:
            image: 输入图像 [1, 3, H, W]
            target_size: 目标尺寸
            
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
    
    def inference_with_edge(self, input_image, caption="", ddpm_steps=20, 
                          use_edge_detection=True, save_intermediate=True, output_dir="inference_output"):
        """
        使用edge map进行推理
        
        Args:
            input_image: 输入图像张量 [1, 3, H, W]，值范围 [-1, 1]
            caption: 文本描述（可选）
            ddpm_steps: DDPM采样步数
            use_edge_detection: 是否使用edge检测
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            
        Returns:
            output_image: 超分辨率输出图像
        """
        print(f"\n开始Edge推理...")
        print(f"DDPM步数: {ddpm_steps}")
        print(f"使用Edge检测: {use_edge_detection}")
        
        # 创建输出目录
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        # 上采样输入图像
        upscaled_image = self.upscale_image(input_image, target_size=512)
        
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
            
            # 生成struct_cond - structcond_stage_model返回字典格式
            struct_cond = self.model.structcond_stage_model(
                z_upscaled, 
                torch.zeros(1, device=self.device)
            )
            print(f"struct_cond类型: {type(struct_cond)}")
            if isinstance(struct_cond, dict):
                print(f"struct_cond字典键: {list(struct_cond.keys())}")
                for key, value in struct_cond.items():
                    print(f"  键 '{key}': 形状 {value.shape}")
            else:
                print(f"struct_cond形状: {struct_cond.shape}")
            
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
            result_path = os.path.join(output_dir, f"result_edge_{ddpm_steps}steps.png")
            Image.fromarray(result_image).save(result_path)
            print(f"结果已保存: {result_path}")
        
        return x_samples
    
    def compare_with_without_edge(self, input_image, ddpm_steps=20, output_dir="comparison_output"):
        """
        对比使用和不使用edge检测的效果
        
        Args:
            input_image: 输入图像
            ddpm_steps: DDPM采样步数
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("对比Edge检测效果")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 不使用edge检测
        print("\n1. 不使用Edge检测...")
        result_without_edge = self.inference_with_edge(
            input_image, 
            ddpm_steps=ddpm_steps,
            use_edge_detection=False,
            save_intermediate=True,
            output_dir=os.path.join(output_dir, "without_edge")
        )
        
        # 使用edge检测
        print("\n2. 使用Edge检测...")
        result_with_edge = self.inference_with_edge(
            input_image, 
            ddpm_steps=ddpm_steps,
            use_edge_detection=True,
            save_intermediate=True,
            output_dir=os.path.join(output_dir, "with_edge")
        )
        
        # 保存对比结果
        self._save_comparison(input_image, result_without_edge, result_with_edge, output_dir)
        
        print(f"\n对比完成，结果保存在: {output_dir}")
    
    def _save_comparison(self, input_image, result_without_edge, result_with_edge, output_dir):
        """保存对比结果"""
        # 保存输入图像
        input_np = input_image[0].cpu().permute(1, 2, 0).numpy()
        input_np = (input_np + 1.0) / 2.0
        input_np = (input_np * 255).astype(np.uint8)
        Image.fromarray(input_np).save(os.path.join(output_dir, "input.png"))
        
        # 保存不使用edge的结果
        result_without_np = result_without_edge[0].cpu().permute(1, 2, 0).numpy()
        result_without_np = (result_without_np * 255).astype(np.uint8)
        Image.fromarray(result_without_np).save(os.path.join(output_dir, "result_without_edge.png"))
        
        # 保存使用edge的结果
        result_with_np = result_with_edge[0].cpu().permute(1, 2, 0).numpy()
        result_with_np = (result_with_np * 255).astype(np.uint8)
        Image.fromarray(result_with_np).save(os.path.join(output_dir, "result_with_edge.png"))
        
        print("对比结果已保存")
    
    def test_synthetic_image(self, output_dir="synthetic_test"):
        """测试合成图像"""
        print("\n" + "="*60)
        print("测试合成图像")
        print("="*60)
        
        # 创建合成测试图像
        synthetic_image = create_test_image()
        synthetic_image = synthetic_image.to(self.device)
        
        # 保存合成图像
        os.makedirs(output_dir, exist_ok=True)
        synth_np = synthetic_image[0].cpu().permute(1, 2, 0).numpy()
        synth_np = (synth_np + 1.0) / 2.0
        synth_np = (synth_np * 255).astype(np.uint8)
        Image.fromarray(synth_np).save(os.path.join(output_dir, "synthetic_input.png"))
        
        # 执行推理
        result = self.inference_with_edge(
            synthetic_image,
            ddpm_steps=20,
            use_edge_detection=True,
            save_intermediate=True,
            output_dir=output_dir
        )
        
        print(f"合成图像测试完成，结果保存在: {output_dir}")
        return result


def main():
    parser = argparse.ArgumentParser(description="StableSR Edge Map 推理测试")
    
    # 必需参数
    parser.add_argument("--config", type=str, required=True, 
                       help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, 
                       help="模型检查点路径")
    
    # 输入参数
    parser.add_argument("--input", type=str, 
                       help="输入图像路径（可选，不提供则使用合成图像）")
    parser.add_argument("--caption", type=str, default="", 
                       help="文本描述（可选）")
    
    # 推理参数
    parser.add_argument("--steps", type=int, default=20, 
                       help="DDPM采样步数")
    parser.add_argument("--no-edge", action="store_true", 
                       help="禁用edge检测")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子")
    
    # 输出参数
    parser.add_argument("--output", type=str, default="inference_output", 
                       help="输出目录")
    parser.add_argument("--compare", action="store_true", 
                       help="对比使用和不使用edge的效果")
    parser.add_argument("--synthetic", action="store_true", 
                       help="使用合成图像进行测试")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("StableSR Edge Map 推理测试")
    print("="*50)
    
    try:
        # 初始化测试器
        tester = EdgeInferenceTester(args.config, args.ckpt)
        
        if args.synthetic:
            # 使用合成图像测试
            tester.test_synthetic_image(output_dir=os.path.join(args.output, "synthetic"))
        
        elif args.input:
            # 使用真实图像
            if not os.path.exists(args.input):
                print(f"❌ 输入图像不存在: {args.input}")
                return
            
            # 加载输入图像
            input_image = tester.load_image(args.input)
            
            if args.compare:
                # 对比测试
                tester.compare_with_without_edge(
                    input_image, 
                    ddpm_steps=args.steps,
                    output_dir=args.output
                )
            else:
                # 单次推理
                result = tester.inference_with_edge(
                    input_image,
                    caption=args.caption,
                    ddpm_steps=args.steps,
                    use_edge_detection=not args.no_edge,
                    save_intermediate=True,
                    output_dir=args.output
                )
                
                print(f"\n推理完成，结果保存在: {args.output}")
        
        else:
            print("❌ 请提供输入图像路径或使用 --synthetic 参数")
            return
        
        print("\n✓ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
