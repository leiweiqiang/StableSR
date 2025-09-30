"""StableSR推理脚本 - 支持edge_map增强超分辨率"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
import cv2
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
import math
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
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
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
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

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def load_edge_map(path, target_size=512):
	"""加载edge_map图像"""
	if os.path.exists(path):
		# 加载edge map
		edge_img = Image.open(path)
		# 转换为灰度图
		if edge_img.mode != 'L':
			edge_img = edge_img.convert('L')
		
		# 调整尺寸到target_size
		edge_img = edge_img.resize((target_size, target_size), resample=PIL.Image.LANCZOS)
		
		# 转换为numpy数组并归一化到[0,1]
		edge_array = np.array(edge_img).astype(np.float32) / 255.0
		
		# 添加batch和channel维度: (1, 1, H, W)
		edge_tensor = torch.from_numpy(edge_array)[None, None]
		
		print(f"loaded edge map of size {edge_tensor.shape} from {path}")
		return edge_tensor
	else:
		# 如果没有edge map，创建零张量
		print(f"Edge map not found at {path}, using zero tensor")
		return torch.zeros(1, 1, target_size, target_size)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image directory",
		default="inputs/user_upload",
	)
	parser.add_argument(
		"--edge-img",
		type=str,
		nargs="?",
		help="path to the edge map image directory",
		default=None,
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload",
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=4,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
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
		default="configs/stableSRNew/v2-finetune_text_T_512_with_edge.yaml",
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
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="adain",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# 检查是否使用edge支持的配置
	print(f"Using config: {opt.config}")
	if "edge" not in opt.config:
		print("WARNING: 建议使用带有edge支持的配置文件，如v2-finetune_text_T_512_with_edge.yaml")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	# collect valid image files from input directory, ignore subdirectories
	all_entries = sorted(os.listdir(opt.init_img))
	valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
	img_list_ori = [
		name for name in all_entries
		if os.path.isfile(os.path.join(opt.init_img, name))
		and os.path.splitext(name)[1].lower() in valid_exts
	]
	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	edge_map_list = []
	
	for item in img_list_ori:
		# skip if output already exists (we save as PNG with basename)
		basename = os.path.splitext(os.path.basename(item))[0]
		out_name = basename + '.png'
		if os.path.exists(os.path.join(outpath, out_name)):
			if item in img_list:
				img_list.remove(item)
			continue
			
		# 加载输入图像
		cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
		cur_image = transform(cur_image)
		cur_image = cur_image.clamp(-1, 1)
		init_image_list.append(cur_image)
		
		# 加载对应的edge map
		if opt.edge_img:
			edge_path = os.path.join(opt.edge_img, basename + '.png')
			if not os.path.exists(edge_path):
				# 尝试其他扩展名
				for ext in ['.jpg', '.jpeg', '.bmp']:
					alt_path = os.path.join(opt.edge_img, basename + ext)
					if os.path.exists(alt_path):
						edge_path = alt_path
						break
		else:
			edge_path = ""
			
		edge_map = load_edge_map(edge_path, opt.input_size).to(device)
		edge_map_list.append(edge_map)

	if len(init_image_list) == 0:
		print("No images to process")
		return

	init_image_list = torch.cat(init_image_list, dim=0)
	edge_map_list = torch.cat(edge_map_list, dim=0)
	
	niters = math.ceil(init_image_list.size(0) / batch_size)
	init_image_list = init_image_list.chunk(niters)
	edge_map_list = edge_map_list.chunk(niters)

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

	param_list = []
	untrain_paramlist = []
	name_list = []
	for k, v in model.named_parameters():
		if 'spade' in k or 'structcond_stage_model' in k:
			param_list.append(v)
		else:
			name_list.append(k)
			untrain_paramlist.append(v)
	trainable_params = sum(p.numel() for p in param_list)
	untrainable_params = sum(p.numel() for p in untrain_paramlist)
	print(name_list)
	print(trainable_params)
	print(untrainable_params)

	param_list = []
	untrain_paramlist = []
	for k, v in vq_model.named_parameters():
		if 'fusion_layer' in k:
			param_list.append(v)
		elif 'loss' not in k:
			untrain_paramlist.append(v)
	trainable_params += sum(p.numel() for p in param_list)
	# untrainable_params += sum(p.numel() for p in untrain_paramlist)
	print(trainable_params)
	print(untrainable_params)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				count = 0
				for n in trange(niters, desc="Sampling"):
					init_image = init_image_list[n]
					edge_map = edge_map_list[n]
					
					# 检查模型是否支持edge处理
					if hasattr(model, 'use_edge_fusion') and model.use_edge_fusion:
						print("Using edge-enhanced model")
						# 创建包含edge_map的batch
						batch = {
							'image': init_image,
							'edge_map': edge_map
						}
						# 使用支持edge的get_input方法
						input_data = model.get_input(batch, return_first_stage_outputs=False, 
												   force_c_encode=False, bs=init_image.size(0))
						init_latent = input_data[0]  # 融合后的latent
					else:
						print("Using standard model (no edge support)")
						# 标准处理流程
						init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
						init_latent = model.get_first_stage_encoding(init_latent_generator)
					
					text_init = ['']*init_image.size(0)
					semantic_c = model.cond_stage_model(text_init)

					noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
					t = t.to(device).long()
					x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
					x_T = None

					samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
					
					# 获取编码特征用于解码
					if hasattr(model, 'use_edge_fusion') and model.use_edge_fusion:
						# 对于edge模型，需要重新获取编码特征
						_, enc_fea_lq = vq_model.encode(init_image)
					else:
						# 标准模型已经获取了编码特征
						pass
					
					x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					# count += 1
					# if count==5:
					# 	tic = time.time()
					# if count >= 15:
					# 	print('>>>>>>>>>>>>>>>>>>>>>>>>')
					# 	print(time.time()-tic)
					# 	print(s)

					for i in range(init_image.size(0)):
						img_name = img_list.pop(0)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
