#!/usr/bin/env python3
"""
Enhanced TraReport类用于评估超分辨率模型的性能
支持StableSR Edge和StableSR Upscale模型的比较评估
输入参数：gt目录、val目录、model路径
功能：使用不同model将val中的图片进行超分辨率处理，然后与gt中对应文件名的图片计算PSNR
返回：JSON格式的评估结果，包含两种模型的比较
"""

import os
import json
import glob
import copy
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
import PIL
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from basicsr.metrics.psnr_ssim import calculate_psnr
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


class EnhancedTraReport:
    """增强版超分辨率模型性能评估类，支持多模型比较"""
    
    def __init__(self, gt_dir: str, val_dir: str, 
                 stablesr_edge_model_path: str, stablesr_upscale_model_path: str,
                 stablesr_edge_config_path: Optional[str] = None, 
                 stablesr_upscale_config_path: Optional[str] = None,
                 device: str = "cuda", ddpm_steps: int = 200, upscale: float = 4.0, 
                 colorfix_type: str = "adain", seed: int = 42):
        """
        初始化EnhancedTraReport类
        
        Args:
            gt_dir: 真实高分辨率图片目录
            val_dir: 待处理的低分辨率图片目录  
            stablesr_edge_model_path: StableSR Edge模型权重文件路径
            stablesr_upscale_model_path: StableSR Upscale模型权重文件路径
            stablesr_edge_config_path: StableSR Edge模型配置文件路径
            stablesr_upscale_config_path: StableSR Upscale模型配置文件路径
            device: 计算设备
            ddpm_steps: DDPM采样步数
            upscale: 超分辨率倍数
            colorfix_type: 颜色修复类型
            seed: 随机种子
        """
        self.gt_dir = Path(gt_dir)
        self.val_dir = Path(val_dir)
        self.stablesr_edge_model_path = stablesr_edge_model_path
        self.stablesr_upscale_model_path = stablesr_upscale_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ddpm_steps = ddpm_steps
        self.upscale = upscale
        self.colorfix_type = colorfix_type
        self.seed = seed
        
        # 验证目录存在
        if not self.gt_dir.exists():
            raise ValueError(f"GT directory does not exist: {gt_dir}")
        if not self.val_dir.exists():
            raise ValueError(f"Val directory does not exist: {val_dir}")
        if not os.path.exists(self.stablesr_edge_model_path):
            raise ValueError(f"StableSR Edge model path does not exist: {stablesr_edge_model_path}")
        if self.stablesr_upscale_model_path is not None and not os.path.exists(self.stablesr_upscale_model_path):
            raise ValueError(f"StableSR Upscale model path does not exist: {stablesr_upscale_model_path}")
            
        # 设置默认配置文件路径
        if stablesr_edge_config_path is None:
            self.stablesr_edge_config_path = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        else:
            self.stablesr_edge_config_path = stablesr_edge_config_path
            
        if stablesr_upscale_config_path is None:
            self.stablesr_upscale_config_path = "./configs/stableSRNew/v2-finetune_text_T_512.yaml"
        else:
            self.stablesr_upscale_config_path = stablesr_upscale_config_path
            
        # 初始化模型
        self.stablesr_edge_model = None
        self.stablesr_upscale_model = None
        self.vq_model = None
        self.stablesr_edge_sampler = None
        self.stablesr_upscale_sampler = None
        
        # 设置随机种子
        seed_everything(self.seed)
        
    def load_model_from_config(self, config, ckpt, verbose=False):
        """从配置文件和检查点加载模型"""
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model
        
    def load_models(self):
        """加载所有模型和配置"""
        print("Loading StableSR Edge model...")
        print(f"Edge model path: {self.stablesr_edge_model_path}")
        print(f"Edge config path: {self.stablesr_edge_config_path}")
        
        # 加载StableSR Edge模型
        edge_config = OmegaConf.load(self.stablesr_edge_config_path)
        self.stablesr_edge_model = self.load_model_from_config(edge_config, self.stablesr_edge_model_path)
        self.stablesr_edge_model = self.stablesr_edge_model.to(self.device)
        self.stablesr_edge_model.configs = edge_config
        
        # 注册调度器
        self.stablesr_edge_model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.stablesr_edge_model.num_timesteps = 1000
        
        # 创建DDIM采样器
        self.stablesr_edge_sampler = DDIMSampler(self.stablesr_edge_model)
        
        print("Loading StableSR Upscale model...")
        print(f"Upscale model path: {self.stablesr_upscale_model_path}")
        print(f"Upscale config path: {self.stablesr_upscale_config_path}")
        
        # 加载StableSR Upscale模型
        upscale_config = OmegaConf.load(self.stablesr_upscale_config_path)
        self.stablesr_upscale_model = self.load_model_from_config(upscale_config, self.stablesr_upscale_model_path)
        self.stablesr_upscale_model = self.stablesr_upscale_model.to(self.device)
        self.stablesr_upscale_model.configs = upscale_config
        
        # 注册调度器
        self.stablesr_upscale_model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.stablesr_upscale_model.num_timesteps = 1000
        
        # 创建DDIM采样器
        self.stablesr_upscale_sampler = DDIMSampler(self.stablesr_upscale_model)
        
        # 加载VQGAN模型
        print("Loading VQGAN model...")
        vqgan_config = OmegaConf.load("./configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
        self.vq_model = self.load_model_from_config(vqgan_config, './weights/vqgan_cfw_00011.ckpt')
        self.vq_model = self.vq_model.to(self.device)
        
        print("All models loaded successfully!")
        
    def _load_img(self, path: str) -> torch.Tensor:
        """加载图片并预处理"""
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"Loaded image of size ({w}, {h}) from {path}")
        
        # 转换为tensor并归一化到[-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
        
    def _upscale_image_with_model(self, image: torch.Tensor, model, sampler, model_name: str) -> torch.Tensor:
        """使用指定模型对图片进行超分辨率处理"""
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    # 双三次插值放大
                    upscaled_image = F.interpolate(
                        image.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(int(image.size(0) * self.upscale),
                              int(image.size(1) * self.upscale)),
                        mode='bicubic',
                    )
                    
                    # 确保最小尺寸
                    min_size = 512
                    if upscaled_image.size(-1) < min_size or upscaled_image.size(-2) < min_size:
                        ori_size = upscaled_image.size()
                        rescale = min_size * 1.0 / min(upscaled_image.size(-2), upscaled_image.size(-1))
                        new_h = max(int(ori_size[-2] * rescale), min_size)
                        new_w = max(int(ori_size[-1] * rescale), min_size)
                        upscaled_image = F.interpolate(
                            upscaled_image,
                            size=(new_h, new_w),
                            mode='bicubic',
                        )
                    
                    upscaled_image = upscaled_image.clamp(-1, 1)
                    upscaled_image = upscaled_image.type(torch.float16).to(self.device)
                    
                    # 生成caption
                    caption = ""
                    
                    # 使用DDIM采样
                    samples, _ = sampler.sample(
                        S=self.ddpm_steps,
                        conditioning={"c_concat": [upscaled_image], "c_crossattn": [model.get_learned_conditioning([caption])]},
                        batch_size=1,
                        shape=[4, upscaled_image.shape[2] // 8, upscaled_image.shape[3] // 8],
                        verbose=False
                    )
                    
                    # 解码
                    x_samples = self.vq_model.decode(samples * 1.0 / model.scale_factor)
                    
                    # 颜色修复
                    if self.colorfix_type == "adain":
                        x_samples = adaptive_instance_normalization(x_samples, upscaled_image)
                    elif self.colorfix_type == "wavelet":
                        x_samples = wavelet_reconstruction(x_samples, upscaled_image)
                    
                    # 转换回numpy数组
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                    x_samples = (x_samples * 255).astype(np.uint8)
                    
                    print(f"Generated {model_name} image with shape: {x_samples.shape}")
                    return x_samples[0]
                    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算PSNR"""
        return calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False)
        
    def _find_matching_files(self) -> List[Tuple[str, str]]:
        """查找匹配的文件对"""
        val_files = []
        gt_files = []
        
        # 获取所有图片文件
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            val_files.extend(glob.glob(str(self.val_dir / ext)))
            val_files.extend(glob.glob(str(self.val_dir / ext.upper())))
            gt_files.extend(glob.glob(str(self.gt_dir / ext)))
            gt_files.extend(glob.glob(str(self.gt_dir / ext.upper())))
        
        # 匹配文件
        matching_pairs = []
        for val_file in val_files:
            val_name = Path(val_file).stem
            for gt_file in gt_files:
                gt_name = Path(gt_file).stem
                if val_name == gt_name:
                    matching_pairs.append((val_file, gt_file))
                    break
                    
        return matching_pairs
        
    def evaluate(self) -> Dict:
        """执行评估并返回JSON结果"""
        if self.stablesr_edge_model is None or self.stablesr_upscale_model is None:
            self.load_models()
            
        # 查找匹配的文件对
        matching_pairs = self._find_matching_files()
        if not matching_pairs:
            raise ValueError("No matching files found between val and gt directories")
            
        print(f"Found {len(matching_pairs)} matching file pairs")
        
        results = {
            "evaluation_info": {
                "stablesr_edge_model_path": self.stablesr_edge_model_path,
                "stablesr_upscale_model_path": self.stablesr_upscale_model_path,
                "stablesr_edge_config_path": self.stablesr_edge_config_path,
                "stablesr_upscale_config_path": self.stablesr_upscale_config_path,
                "gt_dir": str(self.gt_dir),
                "val_dir": str(self.val_dir),
                "total_files": len(matching_pairs),
                "parameters": {
                    "ddpm_steps": self.ddpm_steps,
                    "upscale": self.upscale,
                    "colorfix_type": self.colorfix_type,
                    "seed": self.seed
                }
            },
            "results": [],
            "summary": {
                "stablesr_edge": {
                    "average_psnr": 0.0,
                    "min_psnr": float('inf'),
                    "max_psnr": 0.0,
                    "std_psnr": 0.0
                },
                "stablesr_upscale": {
                    "average_psnr": 0.0,
                    "min_psnr": float('inf'),
                    "max_psnr": 0.0,
                    "std_psnr": 0.0
                },
                "comparison": {
                    "psnr_difference": 0.0,
                    "better_model": "",
                    "improvement_percentage": 0.0
                }
            }
        }
        
        stablesr_edge_psnr_values = []
        stablesr_upscale_psnr_values = []
        
        # 处理每个文件对
        for val_file, gt_file in tqdm(matching_pairs, desc="Processing images"):
            try:
                # 加载图片
                val_image = self._load_img(val_file)
                gt_image = self._load_img(gt_file)
                
                # 使用StableSR Edge进行超分辨率处理
                print(f"Processing with StableSR Edge: {Path(val_file).name}")
                sr_edge_image = self._upscale_image_with_model(
                    val_image, self.stablesr_edge_model, self.stablesr_edge_sampler, "StableSR Edge"
                )
                
                # 使用StableSR Upscale进行超分辨率处理
                print(f"Processing with StableSR Upscale: {Path(val_file).name}")
                sr_upscale_image = self._upscale_image_with_model(
                    val_image, self.stablesr_upscale_model, self.stablesr_upscale_sampler, "StableSR Upscale"
                )
                
                # 加载GT图片为numpy数组
                gt_img = Image.open(gt_file).convert("RGB")
                gt_img = gt_img.resize((sr_edge_image.shape[1], sr_edge_image.shape[0]), resample=PIL.Image.LANCZOS)
                gt_array = np.array(gt_img)
                
                # 计算PSNR
                psnr_edge = self._calculate_psnr(sr_edge_image, gt_array)
                psnr_upscale = self._calculate_psnr(sr_upscale_image, gt_array)
                
                stablesr_edge_psnr_values.append(psnr_edge)
                stablesr_upscale_psnr_values.append(psnr_upscale)
                
                # 记录结果
                result_entry = {
                    "val_file": val_file,
                    "gt_file": gt_file,
                    "stablesr_edge": {
                        "psnr": float(psnr_edge),
                        "sr_shape": sr_edge_image.shape
                    },
                    "stablesr_upscale": {
                        "psnr": float(psnr_upscale),
                        "sr_shape": sr_upscale_image.shape
                    },
                    "gt_shape": gt_array.shape,
                    "psnr_difference": float(psnr_upscale - psnr_edge),
                    "better_model": "StableSR Upscale" if psnr_upscale > psnr_edge else "StableSR Edge"
                }
                results["results"].append(result_entry)
                
                print(f"Processed {Path(val_file).name}:")
                print(f"  StableSR Edge PSNR: {psnr_edge:.4f}")
                print(f"  StableSR Upscale PSNR: {psnr_upscale:.4f}")
                print(f"  Difference: {psnr_upscale - psnr_edge:.4f}")
                print(f"  Better: {result_entry['better_model']}")
                
            except Exception as e:
                print(f"Error processing {val_file}: {str(e)}")
                continue
                
        # 计算统计信息
        if stablesr_edge_psnr_values and stablesr_upscale_psnr_values:
            # StableSR Edge统计
            results["summary"]["stablesr_edge"]["average_psnr"] = float(np.mean(stablesr_edge_psnr_values))
            results["summary"]["stablesr_edge"]["min_psnr"] = float(np.min(stablesr_edge_psnr_values))
            results["summary"]["stablesr_edge"]["max_psnr"] = float(np.max(stablesr_edge_psnr_values))
            results["summary"]["stablesr_edge"]["std_psnr"] = float(np.std(stablesr_edge_psnr_values))
            
            # StableSR Upscale统计
            results["summary"]["stablesr_upscale"]["average_psnr"] = float(np.mean(stablesr_upscale_psnr_values))
            results["summary"]["stablesr_upscale"]["min_psnr"] = float(np.min(stablesr_upscale_psnr_values))
            results["summary"]["stablesr_upscale"]["max_psnr"] = float(np.max(stablesr_upscale_psnr_values))
            results["summary"]["stablesr_upscale"]["std_psnr"] = float(np.std(stablesr_upscale_psnr_values))
            
            # 比较统计
            avg_psnr_edge = results["summary"]["stablesr_edge"]["average_psnr"]
            avg_psnr_upscale = results["summary"]["stablesr_upscale"]["average_psnr"]
            results["summary"]["comparison"]["psnr_difference"] = float(avg_psnr_upscale - avg_psnr_edge)
            results["summary"]["comparison"]["better_model"] = "StableSR Upscale" if avg_psnr_upscale > avg_psnr_edge else "StableSR Edge"
            results["summary"]["comparison"]["improvement_percentage"] = float(
                (avg_psnr_upscale - avg_psnr_edge) / avg_psnr_edge * 100
            )
            
        print(f"\nEvaluation completed!")
        print(f"StableSR Edge - Average PSNR: {results['summary']['stablesr_edge']['average_psnr']:.4f}")
        print(f"StableSR Upscale - Average PSNR: {results['summary']['stablesr_upscale']['average_psnr']:.4f}")
        print(f"PSNR Difference: {results['summary']['comparison']['psnr_difference']:.4f}")
        print(f"Better Model: {results['summary']['comparison']['better_model']}")
        print(f"Improvement: {results['summary']['comparison']['improvement_percentage']:.2f}%")
        
        return results
        
    def save_results(self, results: Dict, output_path: str = "enhanced_tra_report_results.json"):
        """保存结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
        
    def run_evaluation(self, output_path: str = "enhanced_tra_report_results.json") -> Dict:
        """运行完整的评估流程"""
        print("Starting enhanced evaluation...")
        results = self.evaluate()
        self.save_results(results, output_path)
        return results


def main():
    """示例用法"""
    # 创建EnhancedTraReport实例
    enhanced_tra_report = EnhancedTraReport(
        gt_dir="/path/to/gt/images",
        val_dir="/path/to/val/images",
        stablesr_edge_model_path="/path/to/stablesr_edge_model.ckpt",
        stablesr_upscale_model_path="/path/to/stablesr_upscale_model.ckpt",
        stablesr_edge_config_path="./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
        stablesr_upscale_config_path="./configs/stableSRNew/v2-finetune_text_T_512.yaml",
        device="cuda",
        ddpm_steps=200,
        upscale=4.0,
        colorfix_type="adain",
        seed=42
    )
    
    # 运行评估
    results = enhanced_tra_report.run_evaluation("enhanced_comparison_results.json")
    
    # 打印总结
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"StableSR Edge Average PSNR: {results['summary']['stablesr_edge']['average_psnr']:.4f}")
    print(f"StableSR Upscale Average PSNR: {results['summary']['stablesr_upscale']['average_psnr']:.4f}")
    print(f"PSNR Difference: {results['summary']['comparison']['psnr_difference']:.4f}")
    print(f"Better Model: {results['summary']['comparison']['better_model']}")
    print(f"Improvement: {results['summary']['comparison']['improvement_percentage']:.2f}%")


if __name__ == "__main__":
    main()
