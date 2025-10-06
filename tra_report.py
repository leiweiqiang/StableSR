#!/usr/bin/env python3
"""
TraReport类用于评估超分辨率模型的性能
输入参数：gt目录、val目录、model路径
功能：使用model将val中的图片进行超分辨率处理，然后与gt中对应文件名的图片计算PSNR
返回：JSON格式的评估结果
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
import cv2

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from basicsr.metrics.psnr_ssim import calculate_psnr
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


class EdgeDDIMSampler(DDIMSampler):
    """Custom DDIM sampler for Edge models that require struct_cond parameter"""
    
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule=schedule, **kwargs)
        # Import noise_like function from util module
        from ldm.modules.diffusionmodules.util import noise_like
        self.noise_like = noise_like
    
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # For Edge models, we need to pass struct_cond and edge_map
            if hasattr(self.model, 'use_edge_processing'):
                # Extract struct_cond and edge_map from conditioning
                struct_cond = c.get('struct_cond', None) if isinstance(c, dict) else None
                edge_map = c.get('edge_map', None) if isinstance(c, dict) else None
                # Extract context (c_crossattn) from conditioning dictionary
                context = c.get('c_crossattn', c) if isinstance(c, dict) else c
                
                # Ensure struct_cond has all required keys for UNet processing
                if struct_cond is not None and isinstance(struct_cond, dict):
                    # Ensure all common UNet feature sizes are available
                    required_keys = ['64', '32', '16', '8']
                    
                    # Create a new struct_cond with all required keys
                    new_struct_cond = {}
                    for key in required_keys:
                        if key in struct_cond:
                            new_struct_cond[key] = struct_cond[key]
                        else:
                            # Find closest available key
                            available_keys = [int(k) for k in struct_cond.keys() if k.isdigit()]
                            if available_keys:
                                closest_key = str(min(available_keys, key=lambda k: abs(k - int(key))))
                                new_struct_cond[key] = struct_cond[closest_key]
                            else:
                                # Use the first available key as fallback
                                new_struct_cond[key] = list(struct_cond.values())[0]
                    struct_cond = new_struct_cond
                
                e_t = self.model.apply_model(x, t, context, struct_cond, edge_map=edge_map)
            else:
                e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if hasattr(self.model, 'use_edge_processing'):
                # Extract struct_cond and edge_map from conditioning
                struct_cond = c.get('struct_cond', None) if isinstance(c, dict) else None
                edge_map = c.get('edge_map', None) if isinstance(c, dict) else None
                # Extract context (c_crossattn) from conditioning dictionary
                context = c.get('c_crossattn', c) if isinstance(c, dict) else c
                
                # Ensure struct_cond has all required keys for unconditional case
                if struct_cond is not None and isinstance(struct_cond, dict):
                    # For unconditional case, we need to ensure all common UNet feature sizes are available
                    # Common UNet feature sizes: 64, 32, 16, 8
                    required_keys = ['64', '32', '16', '8']
                    
                    # Create a new struct_cond with all required keys
                    new_struct_cond = {}
                    for key in required_keys:
                        if key in struct_cond:
                            new_struct_cond[key] = struct_cond[key]
                        else:
                            # Find closest available key
                            available_keys = [int(k) for k in struct_cond.keys() if k.isdigit()]
                            if available_keys:
                                closest_key = str(min(available_keys, key=lambda k: abs(k - int(key))))
                                new_struct_cond[key] = struct_cond[closest_key]
                            else:
                                # Use the first available key as fallback
                                new_struct_cond[key] = list(struct_cond.values())[0]
                    struct_cond = new_struct_cond
                
                # Concatenate unconditional and conditional context
                c_in = torch.cat([unconditional_conditioning, context])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, struct_cond, edge_map=edge_map).chunk(2)
            else:
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * self.noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


class TraReport:
    """超分辨率模型性能评估类"""
    
    def __init__(self, gt_dir: str, val_dir: str, model_path: str, 
                 config_path: Optional[str] = None, device: str = "cuda",
                 ddpm_steps: int = 200, upscale: float = 4.0, 
                 colorfix_type: str = "adain", seed: int = 42,
                 use_edge_detection: bool = True):
        """
        初始化TraReport类
        
        Args:
            gt_dir: 真实高分辨率图片目录
            val_dir: 待处理的低分辨率图片目录  
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径，如果为None则使用默认配置
            device: 计算设备
            ddpm_steps: DDPM采样步数
            upscale: 超分辨率倍数
            colorfix_type: 颜色修复类型
            seed: 随机种子
            use_edge_detection: 是否使用边缘检测生成edge_map
        """
        self.gt_dir = Path(gt_dir)
        self.val_dir = Path(val_dir)
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ddpm_steps = ddpm_steps
        self.upscale = upscale
        self.colorfix_type = colorfix_type
        self.seed = seed
        self.use_edge_detection = use_edge_detection
        
        # 验证目录存在
        if not self.gt_dir.exists():
            raise ValueError(f"GT directory does not exist: {gt_dir}")
        if not self.val_dir.exists():
            raise ValueError(f"Val directory does not exist: {val_dir}")
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # 设置默认配置文件路径
        if config_path is None:
            self.config_path = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        else:
            self.config_path = config_path
            
        # 初始化模型
        self.model = None
        self.vq_model = None
        self.sampler = None
        
        # 设置随机种子
        seed_everything(self.seed)
        
    def load_model(self):
        """加载模型和配置"""
        print(f"Loading model from {self.model_path}")
        print(f"Using config: {self.config_path}")
        
        # 加载配置文件
        config = OmegaConf.load(self.config_path)
        
        # 加载主模型
        self.model = self._load_model_from_config(config, self.model_path)
        self.model = self.model.to(self.device)
        self.model.configs = config
        
        # 注册调度器
        self.model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3
        )
        self.model.num_timesteps = 1000
        
        # 设置DDPM步数
        self._setup_ddpm_timesteps()
        
        # 加载VQGAN模型
        vqgan_config = OmegaConf.load("./configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
        vqgan_path = "/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt"
        self.vq_model = self._load_model_from_config(vqgan_config, vqgan_path)
        self.vq_model = self.vq_model.to(self.device)
        
        # 创建DDIM采样器
        print(f"Debug: model.num_timesteps = {getattr(self.model, 'num_timesteps', 'NOT_FOUND')}")
        print(f"Debug: model.alphas_cumprod shape = {getattr(self.model, 'alphas_cumprod', 'NOT_FOUND')}")
        if hasattr(self.model, 'alphas_cumprod'):
            print(f"Debug: alphas_cumprod.shape[0] = {self.model.alphas_cumprod.shape[0]}")
        
        # 检查是否是Edge模型
        if hasattr(self.model, 'use_edge_processing'):
            print("Detected Edge model, using custom DDIM sampler")
            self.sampler = EdgeDDIMSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)
        print(f"Debug: sampler.ddpm_num_timesteps = {getattr(self.sampler, 'ddpm_num_timesteps', 'NOT_FOUND')}")
        
        print("Model loaded successfully!")
        
    def _load_model_from_config(self, config, ckpt, verbose=False):
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
        
    def _setup_ddpm_timesteps(self):
        """设置DDPM时间步"""
        sqrt_alphas_cumprod = copy.deepcopy(self.model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(self.model.sqrt_one_minus_alphas_cumprod)
        
        use_timesteps = set(self._space_timesteps(1000, [self.ddpm_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        
        # 为所有1000个时间步生成betas
        full_betas = []
        for i in range(1000):
            if i < len(self.model.alphas_cumprod):
                # 使用原始模型的alphas_cumprod
                alpha_cumprod = self.model.alphas_cumprod[i]
                if i == 0:
                    beta = 1 - alpha_cumprod
                else:
                    prev_alpha_cumprod = self.model.alphas_cumprod[i-1]
                    beta = 1 - alpha_cumprod / prev_alpha_cumprod
                full_betas.append(beta.data.cpu().numpy())
            else:
                # 对于超出范围的时间步，使用最后一个beta值
                full_betas.append(full_betas[-1] if full_betas else 0.0001)
        
        # 记录实际使用的时间步
        for i, alpha_cumprod in enumerate(self.model.alphas_cumprod):
            if i in use_timesteps:
                timestep_map.append(i)
                
        self.model.register_schedule(
            given_betas=np.array(full_betas), 
            timesteps=1000  # 使用完整的1000个时间步
        )
        self.model.num_timesteps = 1000
        self.model.ori_timesteps = list(use_timesteps)
        self.model.ori_timesteps.sort()
        
    def _space_timesteps(self, num_timesteps, section_counts):
        """创建时间步列表"""
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim"):])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
            section_counts = [int(x) for x in section_counts.split(",")]
            
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(f"cannot divide section of {size} steps into {section_count}")
            if section_count <= 1:
                stride = 1
            else:
                stride = size // (section_count - 1)
            steps = [start_idx + j * stride for j in range(section_count)]
            all_steps += steps
            start_idx += size
            
        return set(all_steps)
        
    def _generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        从输入图像生成边缘图
        
        Args:
            image: 输入图像张量 [B, C, H, W]，值范围 [-1, 1]
            
        Returns:
            edge_map: 边缘图张量 [B, 3, H, W]，值范围 [-1, 1]
        """
        # 转换为numpy数组进行处理
        if image.dim() == 4:
            # 批量处理，取第一张图片
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()
            
        # 从 [-1, 1] 转换到 [0, 1]
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        
        # 转换维度从 [C, H, W] 到 [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # 转换为uint8格式 [0, 255]
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # 转换为BGR格式（OpenCV使用BGR）
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图进行边缘检测
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
        
        # 应用Canny边缘检测
        edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
        
        # 转换为3通道BGR格式
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 转换回RGB格式
        edges_rgb = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB)
        
        # 转换为float32并归一化到 [0, 1]
        edges_float = edges_rgb.astype(np.float32) / 255.0
        
        # 转换维度从 [H, W, C] 到 [C, H, W]
        edges_tensor = np.transpose(edges_float, (2, 0, 1))
        
        # 转换到 [-1, 1] 范围
        edges_tensor = 2.0 * edges_tensor - 1.0
        
        # 添加batch维度
        if image.dim() == 4:
            edges_tensor = np.expand_dims(edges_tensor, axis=0)
        
        # 转换为torch张量并移动到相同设备
        edge_map = torch.from_numpy(edges_tensor).to(image.device).to(image.dtype)
        
        return edge_map
        
    def _load_img(self, path: str) -> torch.Tensor:
        """加载图片并预处理"""
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"Loaded image of size ({w}, {h}) from {path}")
        
        # 调整尺寸为32的倍数
        w, h = map(lambda x: x - x % 32, (w, h))
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
        
    def _upscale_image(self, image: torch.Tensor) -> torch.Tensor:
        """对图片进行超分辨率处理"""
        try:
            with torch.no_grad():
                with autocast("cuda"):
                    with self.model.ema_scope():
                        # 双三次插值放大
                        upscaled_image = F.interpolate(
                            image,
                            size=(int(image.size(-2) * self.upscale),
                                  int(image.size(-1) * self.upscale)),
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
                        cross_attn = self.model.get_learned_conditioning([caption])
                        
                        # 为Edge模型准备struct_cond和edge_map
                        if hasattr(self.model, 'use_edge_processing'):
                            # 对于Edge模型，需要struct_cond
                            # 将upscaled_image编码到潜在空间
                            encoder_posterior = self.model.encode_first_stage(upscaled_image)
                            z_upscaled = self.model.get_first_stage_encoding(encoder_posterior).detach()
                            # 生成struct_cond
                            struct_cond = self.model.structcond_stage_model(z_upscaled, torch.zeros(1, device=self.device))
                            
                            # 生成edge_map（如果启用边缘检测）
                            if self.use_edge_detection:
                                edge_map = self._generate_edge_map(upscaled_image)
                                print(f"Generated edge_map with shape: {edge_map.shape}")
                                conditioning = {
                                    "c_concat": upscaled_image, 
                                    "c_crossattn": cross_attn,
                                    "struct_cond": struct_cond,
                                    "edge_map": edge_map
                                }
                            else:
                                print("Edge detection disabled, using struct_cond only")
                                conditioning = {
                                    "c_concat": upscaled_image, 
                                    "c_crossattn": cross_attn,
                                    "struct_cond": struct_cond
                                }
                        else:
                            conditioning = {"c_concat": upscaled_image, "c_crossattn": cross_attn}
                        
                        samples, _ = self.sampler.sample(
                            S=self.ddpm_steps,
                            conditioning=conditioning,
                            batch_size=1,
                            shape=[4, upscaled_image.shape[2] // 8, upscaled_image.shape[3] // 8],
                            verbose=False
                        )
                        
                        # 处理samples的格式问题
                        print(f"Debug: samples type: {type(samples)}")
                        if isinstance(samples, list):
                            print(f"Debug: samples is list, length: {len(samples)}")
                            if len(samples) > 0:
                                samples = samples[0]
                                print(f"Debug: extracted first element, type: {type(samples)}")
                            else:
                                raise ValueError("Empty samples list")
                        
                        # 确保samples是tensor
                        if not torch.is_tensor(samples):
                            print(f"Debug: samples is not tensor, converting...")
                            if isinstance(samples, list) and len(samples) > 0:
                                samples = samples[0]
                            else:
                                raise ValueError(f"Cannot convert samples to tensor, got {type(samples)}")
                        
                        print(f"Debug: final samples shape: {samples.shape if hasattr(samples, 'shape') else 'no shape attr'}")
                        
                        # 解码
                        x_samples = self.vq_model.decode(samples)
                        
                        # 颜色修复
                        if self.colorfix_type == "adain":
                            x_samples = adaptive_instance_normalization(x_samples, upscaled_image)
                        elif self.colorfix_type == "wavelet":
                            x_samples = wavelet_reconstruction(x_samples, upscaled_image)
                        
                        # 后处理
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                        x_samples = (255 * x_samples).astype(np.uint8)
                        
                        # 确保返回numpy数组
                        result = x_samples[0] if len(x_samples) > 0 else x_samples
                        if not isinstance(result, np.ndarray):
                            raise ValueError(f"Expected numpy array, got {type(result)}")
                        
                        return result  # 返回第一张图片
                        
        except Exception as e:
            print(f"Error in _upscale_image: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个默认的numpy数组以避免类型错误
            return np.zeros((512, 512, 3), dtype=np.uint8)
                    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图片之间的PSNR"""
        return calculate_psnr(img1, img2, crop_border=0, test_y_channel=False)
        
    def _find_matching_files(self) -> List[Tuple[str, str]]:
        """查找val和gt目录中匹配的文件对"""
        val_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        gt_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        val_files = []
        for ext in val_extensions:
            val_files.extend(glob.glob(str(self.val_dir / f"*{ext}")))
            val_files.extend(glob.glob(str(self.val_dir / f"*{ext.upper()}")))
            
        matching_pairs = []
        for val_file in val_files:
            val_name = Path(val_file).stem
            
            # 查找对应的gt文件
            for ext in gt_extensions:
                gt_file = self.gt_dir / f"{val_name}{ext}"
                if gt_file.exists():
                    matching_pairs.append((val_file, str(gt_file)))
                    break
                gt_file = self.gt_dir / f"{val_name}{ext.upper()}"
                if gt_file.exists():
                    matching_pairs.append((val_file, str(gt_file)))
                    break
                    
        return matching_pairs
        
    def evaluate(self) -> Dict:
        """执行评估并返回JSON结果"""
        if self.model is None:
            self.load_model()
            
        # 查找匹配的文件对
        matching_pairs = self._find_matching_files()
        if not matching_pairs:
            raise ValueError("No matching files found between val and gt directories")
            
        print(f"Found {len(matching_pairs)} matching file pairs")
        
        results = {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "gt_dir": str(self.gt_dir),
            "val_dir": str(self.val_dir),
            "total_files": len(matching_pairs),
            "parameters": {
                "ddpm_steps": self.ddpm_steps,
                "upscale": self.upscale,
                "colorfix_type": self.colorfix_type,
                "seed": self.seed
            },
            "results": [],
            "summary": {
                "average_psnr": 0.0,
                "min_psnr": float('inf'),
                "max_psnr": 0.0
            }
        }
        
        psnr_values = []
        
        # 处理每个文件对
        for val_file, gt_file in tqdm(matching_pairs, desc="Processing images"):
            try:
                # 加载图片
                val_image = self._load_img(val_file)
                gt_image = self._load_img(gt_file)
                
                # 超分辨率处理
                sr_image = self._upscale_image(val_image)
                
                # 加载GT图片为numpy数组
                gt_img = Image.open(gt_file).convert("RGB")
                gt_img = gt_img.resize((sr_image.shape[1], sr_image.shape[0]), resample=PIL.Image.LANCZOS)
                gt_array = np.array(gt_img)
                
                # 计算PSNR
                psnr = self._calculate_psnr(sr_image, gt_array)
                psnr_values.append(psnr)
                
                # 记录结果
                result_entry = {
                    "val_file": val_file,
                    "gt_file": gt_file,
                    "psnr": float(psnr),
                    "sr_shape": sr_image.shape,
                    "gt_shape": gt_array.shape
                }
                results["results"].append(result_entry)
                
                print(f"Processed {Path(val_file).name}: PSNR = {psnr:.4f}")
                
            except Exception as e:
                print(f"Error processing {val_file}: {str(e)}")
                continue
                
        # 计算统计信息
        if psnr_values:
            results["summary"]["average_psnr"] = float(np.mean(psnr_values))
            results["summary"]["min_psnr"] = float(np.min(psnr_values))
            results["summary"]["max_psnr"] = float(np.max(psnr_values))
            results["summary"]["std_psnr"] = float(np.std(psnr_values))
            
        print(f"\nEvaluation completed!")
        print(f"Average PSNR: {results['summary']['average_psnr']:.4f}")
        print(f"Min PSNR: {results['summary']['min_psnr']:.4f}")
        print(f"Max PSNR: {results['summary']['max_psnr']:.4f}")
        
        return results
        
    def save_results(self, results: Dict, output_path: str):
        """保存结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
        
    def run_evaluation(self, output_path: Optional[str] = None) -> Dict:
        """运行完整的评估流程"""
        results = self.evaluate()
        
        if output_path is None:
            output_path = f"tra_report_results_{self.seed}.json"
            
        self.save_results(results, output_path)
        return results


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TraReport - Super Resolution Model Evaluation")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth images directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation images directory")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config_path", type=str, default=None, help="Model config path")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--ddpm_steps", type=int, default=200, help="DDPM sampling steps")
    parser.add_argument("--upscale", type=float, default=4.0, help="Upscale factor")
    parser.add_argument("--colorfix_type", type=str, default="adain", choices=["adain", "wavelet", "none"], help="Color fix type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_edge_detection", action="store_true", default=True, help="Use edge detection for Edge models")
    parser.add_argument("--no_edge_detection", action="store_true", help="Disable edge detection")
    
    args = parser.parse_args()
    
    # 处理边缘检测参数
    use_edge_detection = args.use_edge_detection and not args.no_edge_detection
    
    # 创建TraReport实例
    tra_report = TraReport(
        gt_dir=args.gt_dir,
        val_dir=args.val_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        ddpm_steps=args.ddpm_steps,
        upscale=args.upscale,
        colorfix_type=args.colorfix_type,
        seed=args.seed,
        use_edge_detection=use_edge_detection
    )
    
    # 运行评估
    results = tra_report.run_evaluation(output_path=args.output_path)
    
    return results


if __name__ == "__main__":
    main()
