"""
StableSR_ScaleLR 快速版本
优化处理速度，减少DDPM步数和优化参数
"""

import os
import sys
import glob
import copy
import time
import argparse
from pathlib import Path
from typing import Optional, List, Union

import PIL
import torch
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import torch.nn.functional as F
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.util_image import ImageSpliterTh
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def load_model_from_config(config, ckpt, verbose=False):
    """加载模型从配置和检查点"""
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


def read_image(im_path):
    """读取图像并转换为tensor"""
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32)/255.0
    im = im[None].transpose(0,3,1,2)
    im = (torch.from_numpy(im) - 0.5) / 0.5
    return im.cuda()


class StableSR_ScaleLR_Fast:
    """
    StableSR超分辨率处理类 - 快速版本
    优化了处理速度和内存使用
    """
    
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        vqgan_ckpt_path: str,
        device: str = "cuda",
        ddpm_steps: int = 20,  # 默认减少到20步
        dec_w: float = 0.5,
        colorfix_type: str = "adain",
        input_size: int = 512,
        upscale: float = 4.0,
        tile_overlap: int = 16,  # 减少重叠
        vqgantile_stride: int = 512,  # 减少步长
        vqgantile_size: int = 1024,  # 减少瓦片大小
        seed: int = 42,
        precision: str = "autocast",
        batch_size: int = 2  # 增加批处理大小
    ):
        """
        初始化StableSR_ScaleLR_Fast
        
        Args:
            config_path: 配置文件路径
            ckpt_path: 模型检查点路径
            vqgan_ckpt_path: VQGAN模型检查点路径
            device: 设备类型
            ddpm_steps: DDPM采样步数（默认20步，快速模式）
            dec_w: VQGAN和Diffusion结合权重
            colorfix_type: 颜色修正类型
            input_size: 输入尺寸
            upscale: 上采样倍数
            tile_overlap: 瓦片重叠大小
            vqgantile_stride: VQGAN瓦片步长
            vqgantile_size: VQGAN瓦片大小
            seed: 随机种子
            precision: 精度类型
            batch_size: 批处理大小
        """
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.ddpm_steps = ddpm_steps
        self.dec_w = dec_w
        self.colorfix_type = colorfix_type
        self.input_size = input_size
        self.upscale = upscale
        self.tile_overlap = tile_overlap
        self.vqgantile_stride = vqgantile_stride
        self.vqgantile_size = vqgantile_size
        self.seed = seed
        self.precision = precision
        self.batch_size = batch_size
        
        # 设置随机种子
        seed_everything(self.seed)
        
        # 加载配置和模型
        self.config = OmegaConf.load(config_path)
        self.model = load_model_from_config(self.config, ckpt_path)
        self.model = self.model.to(self.device)
        self.model.configs = self.config
        
        # 加载VQGAN模型
        vqgan_config = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"))
        self.vq_model = load_model_from_config(vqgan_config, vqgan_ckpt_path)
        self.vq_model = self.vq_model.to(self.device)
        self.vq_model.decoder.fusion_w = self.dec_w
        
        # 设置模型调度
        self._setup_model_schedule()
        
        print(f'>>>>>>>>>>color correction>>>>>>>>>>>')
        if self.colorfix_type == 'adain':
            print('Use adain color correction')
        elif self.colorfix_type == 'wavelet':
            print('Use wavelet color correction')
        else:
            print('No color correction')
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'Fast mode: DDPM steps={self.ddpm_steps}, batch_size={self.batch_size}')
    
    def _setup_model_schedule(self):
        """设置模型调度"""
        self.model.register_schedule(
            given_betas=None, 
            beta_schedule="linear", 
            timesteps=1000,
            linear_start=0.00085, 
            linear_end=0.0120, 
            cosine_s=8e-3
        )
        self.model.num_timesteps = 1000

        sqrt_alphas_cumprod = copy.deepcopy(self.model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(self.model.sqrt_one_minus_alphas_cumprod)

        use_timesteps = set(space_timesteps(1000, [self.ddpm_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
        self.model.num_timesteps = 1000
        self.model.ori_timesteps = list(use_timesteps)
        self.model.ori_timesteps.sort()
        self.model = self.model.to(self.device)
        
        # 保存用于采样的参数
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
    
    def process_batch_images(self, image_paths: List[str]) -> List[torch.Tensor]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            处理后的图像tensor列表
        """
        # 读取所有图像
        images = []
        for path in image_paths:
            cur_image = read_image(path)
            size_min = min(cur_image.size(-1), cur_image.size(-2))
            upsample_scale = max(self.input_size/size_min, self.upscale)
            
            # 上采样到合适尺寸
            cur_image = F.interpolate(
                cur_image,
                size=(int(cur_image.size(-2)*upsample_scale),
                      int(cur_image.size(-1)*upsample_scale)),
                mode='bicubic',
            )
            cur_image = cur_image.clamp(-1, 1)
            images.append(cur_image)
        
        # 合并为批次
        batch_images = torch.cat(images, dim=0)
        ori_h, ori_w = batch_images.shape[2:]
        flag_pad = False
        
        # 确保尺寸是32的倍数
        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
            flag_pad = True
            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
            batch_images = F.pad(batch_images, pad=(0, pad_w, 0, pad_h), mode='reflect')
        
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    seed_everything(self.seed)
                    
                    if batch_images.shape[2] > self.vqgantile_size or batch_images.shape[3] > self.vqgantile_size:
                        # 大图像使用瓦片处理
                        im_spliter = ImageSpliterTh(batch_images, self.vqgantile_size, self.vqgantile_stride, sf=1)
                        for im_lq_pch, index_infos in im_spliter:
                            seed_everything(self.seed)
                            init_latent = self.model.get_first_stage_encoding(
                                self.model.encode_first_stage(im_lq_pch)
                            )
                            text_init = [''] * im_lq_pch.size(0)
                            semantic_c = self.model.cond_stage_model(text_init)
                            noise = torch.randn_like(init_latent)
                            
                            t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_pch.size(0))
                            t = t.to(self.device).long()
                            x_T = self.model.q_sample_respace(
                                x_start=init_latent, 
                                t=t, 
                                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
                                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, 
                                noise=noise
                            )
                            
                            samples, _ = self.model.sample_canvas(
                                cond=semantic_c, 
                                struct_cond=init_latent, 
                                batch_size=im_lq_pch.size(0), 
                                timesteps=self.ddpm_steps, 
                                time_replace=self.ddpm_steps, 
                                x_T=x_T, 
                                return_intermediates=True, 
                                tile_size=int(self.input_size/8), 
                                tile_overlap=self.tile_overlap, 
                                batch_size_sample=im_lq_pch.size(0)
                            )
                            
                            _, enc_fea_lq = self.vq_model.encode(im_lq_pch)
                            x_samples = self.vq_model.decode(samples * 1. / self.model.scale_factor, enc_fea_lq)
                            
                            # 颜色修正
                            if self.colorfix_type == 'adain':
                                x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
                            elif self.colorfix_type == 'wavelet':
                                x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
                            
                            im_spliter.update_gaussian(x_samples, index_infos)
                        
                        im_sr = im_spliter.gather()
                        im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
                    else:
                        # 小图像直接处理
                        init_latent = self.model.get_first_stage_encoding(
                            self.model.encode_first_stage(batch_images)
                        )
                        text_init = [''] * batch_images.size(0)
                        semantic_c = self.model.cond_stage_model(text_init)
                        noise = torch.randn_like(init_latent)
                        
                        t = repeat(torch.tensor([999]), '1 -> b', b=batch_images.size(0))
                        t = t.to(self.device).long()
                        x_T = self.model.q_sample_respace(
                            x_start=init_latent, 
                            t=t, 
                            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
                            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, 
                            noise=noise
                        )
                        
                        samples, _ = self.model.sample_canvas(
                            cond=semantic_c, 
                            struct_cond=init_latent, 
                            batch_size=batch_images.size(0), 
                            timesteps=self.ddpm_steps, 
                            time_replace=self.ddpm_steps, 
                            x_T=x_T, 
                            return_intermediates=True, 
                            tile_size=int(self.input_size/8), 
                            tile_overlap=self.tile_overlap, 
                            batch_size_sample=batch_images.size(0)
                        )
                        
                        _, enc_fea_lq = self.vq_model.encode(batch_images)
                        x_samples = self.vq_model.decode(samples * 1. / self.model.scale_factor, enc_fea_lq)
                        
                        # 颜色修正
                        if self.colorfix_type == 'adain':
                            x_samples = adaptive_instance_normalization(x_samples, batch_images)
                        elif self.colorfix_type == 'wavelet':
                            x_samples = wavelet_reconstruction(x_samples, batch_images)
                        
                        im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
                    
                    # 移除填充
                    if flag_pad:
                        im_sr = im_sr[:, :, :ori_h, :ori_w]
                    
                    # 分割回单个图像
                    results = []
                    for i in range(len(image_paths)):
                        results.append(im_sr[i:i+1])
                    
                    return results
    
    def process_images(
        self, 
        input_path: str, 
        out_dir: str, 
        hq_path: Optional[str] = None
    ) -> None:
        """
        处理图像目录
        
        Args:
            input_path: 输入图像路径（文件或目录）
            out_dir: 输出目录
            hq_path: 高质量图像路径（可选）
        """
        # 创建输出目录结构
        res_dir = os.path.join(out_dir, "RES")
        lr_dir = os.path.join(out_dir, "LR")
        hq_dir = os.path.join(out_dir, "HQ")
        
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)
        if hq_path:
            os.makedirs(hq_dir, exist_ok=True)
        
        # 获取输入图像列表
        if os.path.isfile(input_path):
            images_path = [input_path]
        else:
            images_path = sorted(glob.glob(os.path.join(input_path, "*")))
        
        # 过滤已处理的图像
        images_path_ori = copy.deepcopy(images_path)
        for item in images_path_ori:
            img_name = os.path.basename(item)
            if os.path.exists(os.path.join(res_dir, img_name)):
                images_path.remove(item)
        
        print(f"Found {len(images_path)} inputs to process.")
        
        # 批量处理图像
        start_time = time.time()
        for i in tqdm(range(0, len(images_path), self.batch_size), desc="Processing batches"):
            batch_paths = images_path[i:i+self.batch_size]
            
            try:
                # 批量处理图像
                batch_results = self.process_batch_images(batch_paths)
                
                # 保存结果
                for j, (image_path, im_sr) in enumerate(zip(batch_paths, batch_results)):
                    img_name = os.path.basename(image_path)
                    basename = os.path.splitext(img_name)[0]
                    
                    # 保存超分辨率结果
                    res_path = os.path.join(res_dir, f"{basename}.png")
                    im_sr_np = im_sr.cpu().numpy().transpose(0,2,3,1)[0] * 255
                    Image.fromarray(im_sr_np.astype(np.uint8)).save(res_path)
                    
                    # 复制原始图像到LR目录
                    lr_path = os.path.join(lr_dir, img_name)
                    if not os.path.exists(lr_path):
                        import shutil
                        shutil.copy2(image_path, lr_path)
                    
                    print(f"Processed: {img_name}")
                
                # 显示进度
                elapsed_time = time.time() - start_time
                processed_count = i + len(batch_paths)
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = len(images_path) - processed_count
                estimated_remaining_time = remaining_images * avg_time_per_image
                
                print(f"Progress: {processed_count}/{len(images_path)} images")
                print(f"Average time per image: {avg_time_per_image:.1f}s")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"Error processing batch {batch_paths}: {str(e)}")
                continue
        
        # 处理HQ图像（如果提供）
        if hq_path:
            if os.path.isfile(hq_path):
                hq_images = [hq_path]
            else:
                hq_images = sorted(glob.glob(os.path.join(hq_path, "*")))
            
            for hq_image in hq_images:
                try:
                    img_name = os.path.basename(hq_image)
                    hq_dest = os.path.join(hq_dir, img_name)
                    if not os.path.exists(hq_dest):
                        import shutil
                        shutil.copy2(hq_image, hq_dest)
                except Exception as e:
                    print(f"Error copying HQ image {hq_image}: {str(e)}")
        
        total_time = time.time() - start_time
        print(f"Processing completed in {total_time/60:.1f} minutes.")
        print(f"Results saved to: {out_dir}")


def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="StableSR ScaleLR Fast Processing")
    
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--ckpt", type=str, default="/stablesr_dataset/checkpoints/stablesr_turbo.ckpt", help="path to checkpoint")
    parser.add_argument("--vqgan_ckpt", type=str, default="/stablesr_dataset/checkpoints/vqgan_cfw_00011.ckpt", help="path to VQGAN checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="path to input images")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument("--hq_path", type=str, default=None, help="path to HQ images (optional)")
    
    parser.add_argument("--ddpm_steps", type=int, default=4, help="number of ddpm sampling steps (fast mode)")
    parser.add_argument("--dec_w", type=float, default=0.5, help="weight for combining VQGAN and Diffusion")
    parser.add_argument("--colorfix_type", type=str, default="adain", help="Color fix type")
    parser.add_argument("--input_size", type=int, default=512, help="input size")
    parser.add_argument("--upscale", type=float, default=4.0, help="upsample scale")
    parser.add_argument("--tile_overlap", type=int, default=16, help="tile overlap size (reduced)")
    parser.add_argument("--vqgantile_stride", type=int, default=512, help="VQGAN tile stride (reduced)")
    parser.add_argument("--vqgantile_size", type=int, default=1024, help="VQGAN tile size (reduced)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for processing")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--precision", type=str, default="autocast", help="precision type")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = StableSR_ScaleLR_Fast(
        config_path=args.config,
        ckpt_path=args.ckpt,
        vqgan_ckpt_path=args.vqgan_ckpt,
        ddpm_steps=args.ddpm_steps,
        dec_w=args.dec_w,
        colorfix_type=args.colorfix_type,
        input_size=args.input_size,
        upscale=args.upscale,
        tile_overlap=args.tile_overlap,
        vqgantile_stride=args.vqgantile_stride,
        vqgantile_size=args.vqgantile_size,
        batch_size=args.batch_size,
        seed=args.seed,
        precision=args.precision
    )
    
    # 处理图像
    processor.process_images(
        input_path=args.input_path,
        out_dir=args.out_dir,
        hq_path=args.hq_path
    )


if __name__ == "__main__":
    main()
