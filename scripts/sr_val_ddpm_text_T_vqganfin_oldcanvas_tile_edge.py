"""make variations of input image with edge processing and tile support for large images"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import cv2
from util_image import ImageSpliterTh
from pathlib import Path
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

# Edge-specific imports
from tra_report import EdgeDDIMSampler
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge


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


def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
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
	"""Read image and convert to tensor"""
	im = np.array(Image.open(im_path).convert("RGB"))
	im = im.astype(np.float32)/255.0
	im = im[None].transpose(0,3,1,2)
	im = (torch.from_numpy(im) - 0.5) / 0.5
	return im.cuda()


def generate_edge_map(image_tensor):
	"""
	Generate edge map from input image tensor
	
	Args:
		image_tensor: Input image tensor [B, 3, H, W], values in [-1, 1]
		
	Returns:
		edge_map: Edge map tensor [B, 3, H, W], values in [-1, 1]
	"""
	# Convert to numpy array
	img_np = image_tensor[0].cpu().numpy()
	img_np = (img_np + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
	img_np = np.clip(img_np, 0, 1)
	img_np = np.transpose(img_np, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
	
	# Convert to grayscale
	img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
	
	# Apply Gaussian blur
	img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
	
	# Apply Canny edge detection
	edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
	
	# Convert to 3 channels
	edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
	
	# Convert back to tensor format
	edges_tensor = torch.from_numpy(edges_3ch).permute(2, 0, 1).unsqueeze(0).float()
	edges_tensor = (edges_tensor / 127.5) - 1.0  # Normalize to [-1, 1]
	
	return edges_tensor.to(image_tensor.device)


def load_gt_image(gt_image_path, target_size):
	"""
	Load GT image and resize to target size
	
	Args:
		gt_image_path: Path to ground truth image
		target_size: Target size (H, W)
		
	Returns:
		gt_image: GT image tensor [1, 3, H, W], values in [-1, 1]
	"""
	if not os.path.exists(gt_image_path):
		return None
		
	gt_image = Image.open(gt_image_path).convert("RGB")
	gt_image = gt_image.resize((target_size[1], target_size[0]), resample=PIL.Image.LANCZOS)
	
	img_np = np.array(gt_image).astype(np.float32) / 255.0
	img_np = img_np[None].transpose(0, 3, 1, 2)  # [1, 3, H, W]
	
	img_tensor = torch.from_numpy(img_np).float()
	img_tensor = 2.0 * img_tensor - 1.0  # Normalize to [-1, 1]
	
	return img_tensor


def extract_tile_from_gt(gt_image_tensor, tile_coords, original_lr_size, upsampled_lr_size):
	"""
	从GT图片中提取对应的tile用于生成edge map
	
	这是核心函数，处理GT图片的剪切逻辑
	
	Args:
		gt_image_tensor: GT图片张量 [1, 3, H_gt, W_gt], values in [-1, 1]
		tile_coords: LR tile的坐标 (h_start, h_end, w_start, w_end) 在upsampled空间中
		original_lr_size: 原始LR图片的尺寸 (H_lr_orig, W_lr_orig)
		upsampled_lr_size: 上采样后LR图片的尺寸 (H_lr_up, W_lr_up)
		
	Returns:
		gt_tile: GT tile张量 [1, 3, tile_h, tile_w]
		
	剪切逻辑说明：
	1. LR图片经过上采样从 (H_lr_orig, W_lr_orig) -> (H_lr_up, W_lr_up)
	2. Tile坐标是在upsampled空间中的: (h_start, h_end, w_start, w_end)
	3. GT图片尺寸通常是原始LR的scale倍 (比如4x): (H_gt, W_gt)
	4. 需要计算tile在GT图片中对应的坐标
	
	计算公式：
	- upsampled空间坐标 -> 原始LR空间坐标: coord_lr = coord_up * (H_lr_orig / H_lr_up)
	- 原始LR空间坐标 -> GT空间坐标: coord_gt = coord_lr * (H_gt / H_lr_orig)
	- 简化: coord_gt = coord_up * (H_gt / H_lr_up)
	"""
	h_start, h_end, w_start, w_end = tile_coords
	h_lr_orig, w_lr_orig = original_lr_size
	h_lr_up, w_lr_up = upsampled_lr_size
	h_gt, w_gt = gt_image_tensor.shape[2:]
	
	# 计算缩放因子：upsampled LR空间 -> GT空间
	scale_h = h_gt / h_lr_up
	scale_w = w_gt / w_lr_up
	
	# 计算GT tile的坐标
	h_start_gt = int(h_start * scale_h)
	h_end_gt = int(h_end * scale_h)
	w_start_gt = int(w_start * scale_w)
	w_end_gt = int(w_end * scale_w)
	
	# 确保坐标在有效范围内
	h_start_gt = max(0, min(h_start_gt, h_gt))
	h_end_gt = max(0, min(h_end_gt, h_gt))
	w_start_gt = max(0, min(w_start_gt, w_gt))
	w_end_gt = max(0, min(w_end_gt, w_gt))
	
	# 提取GT tile
	gt_tile = gt_image_tensor[:, :, h_start_gt:h_end_gt, w_start_gt:w_end_gt]
	
	# Resize到tile的尺寸（如果需要）
	tile_h = h_end - h_start
	tile_w = w_end - w_start
	
	if gt_tile.shape[2] != tile_h or gt_tile.shape[3] != tile_w:
		gt_tile = F.interpolate(
			gt_tile,
			size=(tile_h, tile_w),
			mode='bicubic',
			align_corners=False
		)
	
	return gt_tile


def generate_edge_map_from_gt_tile(gt_tile):
	"""
	从GT tile生成edge map
	
	Args:
		gt_tile: GT tile张量 [1, 3, H, W], values in [-1, 1]
		
	Returns:
		edge_map: Edge map张量 [1, 3, H, W], values in [-1, 1]
	"""
	return generate_edge_map(gt_tile)


class ImageSpliterWithEdge(ImageSpliterTh):
	"""
	扩展ImageSpliterTh以支持edge map的tile处理
	"""
	def __init__(self, im, pch_size, stride, sf=1, gt_image=None, original_lr_size=None):
		"""
		Args:
			im: LR图片 (已上采样) [B, C, H, W]
			pch_size: tile尺寸
			stride: tile步长
			sf: 缩放因子
			gt_image: GT图片张量 [1, 3, H_gt, W_gt], 可选
			original_lr_size: 原始LR图片尺寸 (H_orig, W_orig), 用于计算GT tile坐标
		"""
		super().__init__(im, pch_size, stride, sf)
		self.gt_image = gt_image
		self.original_lr_size = original_lr_size
		self.upsampled_lr_size = (im.shape[2], im.shape[3])
		
	def get_edge_map_for_current_tile(self, tile_coords):
		"""
		为当前tile生成edge map
		
		Args:
			tile_coords: (h_start, h_end, w_start, w_end)
			
		Returns:
			edge_map: Edge map张量 [1, 3, tile_h, tile_w]
		"""
		h_start, h_end, w_start, w_end = tile_coords
		
		if self.gt_image is not None and self.original_lr_size is not None:
			# 从GT图片提取对应的tile
			gt_tile = extract_tile_from_gt(
				self.gt_image,
				tile_coords,
				self.original_lr_size,
				self.upsampled_lr_size
			)
			# 从GT tile生成edge map
			edge_map = generate_edge_map_from_gt_tile(gt_tile)
		else:
			# 从LR tile生成edge map
			lr_tile = self.im_ori[:, :, h_start:h_end, w_start:w_end]
			edge_map = generate_edge_map(lr_tile)
		
		return edge_map


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image directory",
		default="inputs/user_upload"
	)
	parser.add_argument(
		"--gt-img",
		type=str,
		nargs="?",
		help="path to the ground truth images directory for edge map generation (optional)",
		default=None,
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=200,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=1,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512_edge.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/model.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size (in latent space)",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type: adain; wavelet; nofix",
	)
	parser.add_argument(
		"--vqgantile_stride",
		type=int,
		default=1000,
		help="the stride for tile operation before VQGAN decoder (in pixel)",
	)
	parser.add_argument(
		"--vqgantile_size",
		type=int,
		default=1280,
		help="the size for tile operation before VQGAN decoder (in pixel)",
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--use_edge_processing",
		action="store_true",
		help="Enable edge processing for enhanced super-resolution",
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('='*60)
	print('TILE-BASED EDGE SUPER-RESOLUTION')
	print('='*60)
	print(f'Tile size (VQGAN): {opt.vqgantile_size}')
	print(f'Tile stride (VQGAN): {opt.vqgantile_stride}')
	print(f'Tile overlap (Diffusion latent): {opt.tile_overlap}')
	print(f'Input size: {opt.input_size}')
	print(f'Upscale factor: {opt.upscale}')
	
	print('\n>>>>>>>>>>Edge Processing>>>>>>>>>>>')
	if opt.use_edge_processing:
		print('✓ Edge processing ENABLED')
		if opt.gt_img:
			print(f'✓ Using GT images from: {opt.gt_img}')
		else:
			print('  Using LR images for edge map generation')
	else:
		print('✗ Edge processing DISABLED')
	
	print('\n>>>>>>>>>>Color Correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('✓ Using AdaIN color correction')
	elif opt.colorfix_type == 'wavelet':
		print('✓ Using wavelet color correction')
	else:
		print('  No color correction')
	print('='*60)

	# Load models
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	config = OmegaConf.load(f"{opt.config}")
	
	# Use edge model if edge processing is enabled
	if opt.use_edge_processing:
		print("\n🔧 Loading edge-enhanced model...")
		model = LatentDiffusionSRTextWTWithEdge(
			first_stage_config=config.model.params.first_stage_config,
			cond_stage_config=config.model.params.cond_stage_config,
			structcond_stage_config=config.model.params.structcond_stage_config,
			num_timesteps_cond=config.model.params.get('num_timesteps_cond', 1),
			cond_stage_key=config.model.params.get('cond_stage_key', 'image'),
			cond_stage_trainable=config.model.params.get('cond_stage_trainable', False),
			concat_mode=config.model.params.get('concat_mode', True),
			conditioning_key=config.model.params.get('conditioning_key', 'crossattn'),
			scale_factor=config.model.params.get('scale_factor', 0.18215),
			scale_by_std=config.model.params.get('scale_by_std', False),
			unfrozen_diff=config.model.params.get('unfrozen_diff', False),
			random_size=config.model.params.get('random_size', False),
			test_gt=config.model.params.get('test_gt', False),
			p2_gamma=config.model.params.get('p2_gamma', None),
			p2_k=config.model.params.get('p2_k', None),
			time_replace=config.model.params.get('time_replace', 1000),
			use_usm=config.model.params.get('use_usm', True),
			mix_ratio=config.model.params.get('mix_ratio', 0.0),
			use_edge_processing=True,
			edge_input_channels=config.model.params.get('edge_input_channels', 3),
			linear_start=config.model.params.get('linear_start', 0.00085),
			linear_end=config.model.params.get('linear_end', 0.0120),
			timesteps=config.model.params.get('timesteps', 1000),
			first_stage_key=config.model.params.get('first_stage_key', 'image'),
			image_size=config.model.params.get('image_size', 512),
			channels=config.model.params.get('channels', 4),
			unet_config=config.model.params.get('unet_config', None),
			use_ema=config.model.params.get('use_ema', False)
		).to(device)
		model.init_from_ckpt(opt.ckpt)
		print("✓ Edge-enhanced model loaded successfully")
	else:
		print("\n🔧 Loading standard model...")
		model = load_model_from_config(config, f"{opt.ckpt}")
		model = model.to(device)
		print("✓ Standard model loaded successfully")

	model.configs = config

	# Load VQGAN
	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	# Get image paths
	images_path_ori = sorted(glob.glob(os.path.join(opt.init_img, "*")))
	images_path = copy.deepcopy(images_path_ori)
	for item in images_path_ori:
		img_name = item.split('/')[-1]
		if os.path.exists(os.path.join(outpath, img_name)):
			images_path.remove(item)
	print(f"\n📁 Found {len(images_path)} input images to process.")

	# Setup diffusion schedule
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	
	print("\n🚀 Starting processing...\n")
	
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				for n in trange(len(images_path), desc="Processing images"):
					if (n + 1) % opt.n_samples == 1 or opt.n_samples == 1:
						cur_image = read_image(images_path[n])
						
						# Store original LR size (before upsampling)
						original_lr_h, original_lr_w = cur_image.shape[2:]
						
						size_min = min(cur_image.size(-1), cur_image.size(-2))
						upsample_scale = max(opt.input_size/size_min, opt.upscale)
						cur_image = F.interpolate(
									cur_image,
									size=(int(cur_image.size(-2)*upsample_scale),
										  int(cur_image.size(-1)*upsample_scale)),
									mode='bicubic',
									)
						cur_image = cur_image.clamp(-1, 1)
						im_lq_bs = [cur_image, ]
						im_path_bs = [images_path[n], ]
						original_lr_sizes = [(original_lr_h, original_lr_w), ]
					else:
						cur_image = read_image(images_path[n])
						
						original_lr_h, original_lr_w = cur_image.shape[2:]
						
						size_min = min(cur_image.size(-1), cur_image.size(-2))
						upsample_scale = max(opt.input_size/size_min, opt.upscale)
						cur_image = F.interpolate(
									cur_image,
									size=(int(cur_image.size(-2)*upsample_scale),
										  int(cur_image.size(-1)*upsample_scale)),
									mode='bicubic',
									)
						cur_image = cur_image.clamp(-1, 1)
						im_lq_bs.append(cur_image)
						im_path_bs.append(images_path[n])
						original_lr_sizes.append((original_lr_h, original_lr_w))

					if (n + 1) % opt.n_samples == 0 or (n+1) == len(images_path):
						im_lq_bs = torch.cat(im_lq_bs, dim=0)
						ori_h, ori_w = im_lq_bs.shape[2:]
						
						# Padding to multiple of 32
						if not (ori_h % 32 == 0 and ori_w % 32 == 0):
							flag_pad = True
							pad_h = ((ori_h // 32) + 1) * 32 - ori_h
							pad_w = ((ori_w // 32) + 1) * 32 - ori_w
							im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
						else:
							flag_pad = False

						# Load GT image if provided (for the first image in batch)
						gt_image_tensor = None
						if opt.use_edge_processing and opt.gt_img:
							img_basename = os.path.splitext(os.path.basename(im_path_bs[0]))[0]
							gt_img_path = os.path.join(opt.gt_img, img_basename + '.png')
							
							# Try different extensions if .png not found
							if not os.path.exists(gt_img_path):
								for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
									alt_path = os.path.join(opt.gt_img, img_basename + ext)
									if os.path.exists(alt_path):
										gt_img_path = alt_path
										break
							
							if os.path.exists(gt_img_path):
								# Load GT image (don't resize yet, will resize tiles as needed)
								gt_image_tensor = read_image(gt_img_path)
								print(f"  📷 Loaded GT image: {gt_img_path} [{gt_image_tensor.shape[2]}x{gt_image_tensor.shape[3]}]")

						# Check if we need tile processing
						if im_lq_bs.shape[2] > opt.vqgantile_size or im_lq_bs.shape[3] > opt.vqgantile_size:
							print(f"  🔲 Large image detected [{im_lq_bs.shape[2]}x{im_lq_bs.shape[3]}], using tile processing...")
							
							# Create image spliter with edge support
							if opt.use_edge_processing:
								im_spliter = ImageSpliterWithEdge(
									im_lq_bs, 
									opt.vqgantile_size, 
									opt.vqgantile_stride, 
									sf=1,
									gt_image=gt_image_tensor,
									original_lr_size=original_lr_sizes[0]
								)
							else:
								im_spliter = ImageSpliterTh(
									im_lq_bs, 
									opt.vqgantile_size, 
									opt.vqgantile_stride, 
									sf=1
								)
							
							tile_count = 0
							for im_lq_pch, index_infos in im_spliter:
								tile_count += 1
								h_start, h_end, w_start, w_end = index_infos
								print(f"    Processing tile {tile_count}/{len(im_spliter)}: [{h_start}:{h_end}, {w_start}:{w_end}]")
								
								seed_everything(opt.seed)
								
								# Encode to latent space
								init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))
								text_init = ['']*opt.n_samples
								semantic_c = model.cond_stage_model(text_init)
								noise = torch.randn_like(init_latent)
								
								t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
								t = t.to(device).long()
								x_T = model.q_sample_respace(
									x_start=init_latent, 
									t=t, 
									sqrt_alphas_cumprod=sqrt_alphas_cumprod, 
									sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, 
									noise=noise
								)
								
								# Generate edge map for this tile
								if opt.use_edge_processing:
									if isinstance(im_spliter, ImageSpliterWithEdge):
										# Use the tile coordinates before scaling (in pixel space)
										h_start_px = h_start // 1  # Already in pixel space from __next__
										h_end_px = h_end // 1
										w_start_px = w_start // 1
										w_end_px = w_end // 1
										
										# Get edge map for current tile
										edge_map = im_spliter.get_edge_map_for_current_tile(
											(h_start_px, h_end_px, w_start_px, w_end_px)
										)
									else:
										# Fallback: generate edge map from LR tile
										edge_map = generate_edge_map(im_lq_pch)
									
									print(f"      Edge map range: [{edge_map.min():.3f}, {edge_map.max():.3f}]")
									
									# Sample with edge map
									samples, _ = model.sample_canvas(
										cond=semantic_c, 
										struct_cond=init_latent, 
										batch_size=im_lq_pch.size(0), 
										timesteps=opt.ddpm_steps, 
										time_replace=opt.ddpm_steps, 
										x_T=x_T, 
										return_intermediates=True, 
										tile_size=int(opt.input_size/8), 
										tile_overlap=opt.tile_overlap, 
										batch_size_sample=opt.n_samples,
										edge_map=edge_map
									)
								else:
									# Standard sampling without edge
									samples, _ = model.sample_canvas(
										cond=semantic_c, 
										struct_cond=init_latent, 
										batch_size=im_lq_pch.size(0), 
										timesteps=opt.ddpm_steps, 
										time_replace=opt.ddpm_steps, 
										x_T=x_T, 
										return_intermediates=True, 
										tile_size=int(opt.input_size/8), 
										tile_overlap=opt.tile_overlap, 
										batch_size_sample=opt.n_samples
									)
								
								# Decode
								_, enc_fea_lq = vq_model.encode(im_lq_pch)
								x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
								
								# Color correction
								if opt.colorfix_type == 'adain':
									x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
								elif opt.colorfix_type == 'wavelet':
									x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
								
								# Update spliter with gaussian weighting
								im_spliter.update_gaussian(x_samples, index_infos)
							
							# Gather tiles
							im_sr = im_spliter.gather()
							im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
							
						else:
							print(f"  ✓ Small image [{im_lq_bs.shape[2]}x{im_lq_bs.shape[3]}], processing without tiles...")
							
							# No tiling needed
							init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))
							text_init = ['']*opt.n_samples
							semantic_c = model.cond_stage_model(text_init)
							noise = torch.randn_like(init_latent)
							
							t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
							t = t.to(device).long()
							x_T = model.q_sample_respace(
								x_start=init_latent, 
								t=t, 
								sqrt_alphas_cumprod=sqrt_alphas_cumprod, 
								sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, 
								noise=noise
							)
							
							# Generate edge map
							if opt.use_edge_processing:
								if gt_image_tensor is not None:
									# Resize GT to match processing size
									gt_resized = F.interpolate(
										gt_image_tensor,
										size=(im_lq_bs.shape[2], im_lq_bs.shape[3]),
										mode='bicubic',
										align_corners=False
									)
									edge_map = generate_edge_map(gt_resized)
								else:
									edge_map = generate_edge_map(im_lq_bs)
								
								print(f"  Edge map range: [{edge_map.min():.3f}, {edge_map.max():.3f}]")
								
								# Sample with edge map
								samples, _ = model.sample_canvas(
									cond=semantic_c, 
									struct_cond=init_latent, 
									batch_size=im_lq_bs.size(0), 
									timesteps=opt.ddpm_steps, 
									time_replace=opt.ddpm_steps, 
									x_T=x_T, 
									return_intermediates=True, 
									tile_size=int(opt.input_size/8), 
									tile_overlap=opt.tile_overlap, 
									batch_size_sample=opt.n_samples,
									edge_map=edge_map
								)
							else:
								# Standard sampling
								samples, _ = model.sample_canvas(
									cond=semantic_c, 
									struct_cond=init_latent, 
									batch_size=im_lq_bs.size(0), 
									timesteps=opt.ddpm_steps, 
									time_replace=opt.ddpm_steps, 
									x_T=x_T, 
									return_intermediates=True, 
									tile_size=int(opt.input_size/8), 
									tile_overlap=opt.tile_overlap, 
									batch_size_sample=opt.n_samples
								)
							
							_, enc_fea_lq = vq_model.encode(im_lq_bs)
							x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
							
							if opt.colorfix_type == 'adain':
								x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
							elif opt.colorfix_type == 'wavelet':
								x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
							
							im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)

						# Final resize if needed
						if upsample_scale > opt.upscale:
							im_sr = F.interpolate(
										im_sr,
										size=(int(im_lq_bs.size(-2)*opt.upscale/upsample_scale),
											  int(im_lq_bs.size(-1)*opt.upscale/upsample_scale)),
										mode='bicubic',
										)
							im_sr = torch.clamp(im_sr, min=0.0, max=1.0)

						# Convert to numpy
						im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255

						# Remove padding
						if flag_pad:
							im_sr = im_sr[:, :ori_h, :ori_w, ]

						# Save results
						for jj in range(im_lq_bs.shape[0]):
							img_name = str(Path(im_path_bs[jj]).name)
							basename = os.path.splitext(os.path.basename(img_name))[0]
							suffix = "_edge" if opt.use_edge_processing else ""
							outpath_file = str(Path(opt.outdir)) + '/' + basename + suffix + '.png'
							Image.fromarray(im_sr[jj, ].astype(np.uint8)).save(outpath_file)
							print(f"  ✓ Saved: {outpath_file}")

				toc = time.time()
				print(f"\n✅ Processing complete! Time: {toc-tic:.2f}s")

	print(f"\n📂 Results saved to: {opt.outdir}")


if __name__ == "__main__":
	main()

