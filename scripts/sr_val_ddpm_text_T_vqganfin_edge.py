"""make variations of input image with edge support"""

import argparse, os, sys, glob

# Ensure we import from the v3 directory, not v2
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
print(f"[INFO] Project root added to sys.path: {project_root}")
print(f"[INFO] sys.path[0]: {sys.path[0]}")
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
import cv2

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import ldm.models.diffusion.ddpm as ddpm_module
import ldm.modules.diffusionmodules.openaimodel as openai_module
print(f"[INFO] ldm.models.diffusion.ddpm loaded from: {ddpm_module.__file__}")
print(f"[INFO] ldm.modules.diffusionmodules.openaimodel loaded from: {openai_module.__file__}")
import math
import copy
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

def extract_edge(image_tensor):
	"""
	Extract edge from image tensor using Canny edge detection.
	This function EXACTLY matches the training-time edge extraction 
	in basicsr/data/realesrgan_dataset.py (lines 180-205) for consistency.
	
	Training pipeline:
	1. Convert RGB to grayscale
	2. Convert float32 [0,1] to uint8 [0,255]
	3. Apply Gaussian blur: kernel=(5, 5), sigma=1.4
	4. Adaptive Canny: lower_thresh=0.7*median, upper_thresh=1.3*median
	5. Morphological close: MORPH_ELLIPSE kernel (3, 3)
	6. Convert GRAY to RGB (3 channels)
	7. Convert to float32 [0,1], then to [-1, 1]
	
	:param image_tensor: torch tensor of shape (B, C, H, W) in range [-1, 1]
	:return: edge tensor of shape (B, 3, H, W) in range [-1, 1]
	"""
	batch_size = image_tensor.shape[0]
	edge_maps = []
	
	for i in range(batch_size):
		# Convert from tensor to numpy [C, H, W] -> [H, W, C]
		img = image_tensor[i].cpu().numpy()
		img = (img + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
		img = np.clip(img, 0, 1)
		img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
		
		# Convert RGB to grayscale
		img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
		
		# Apply Gaussian blur (EXACTLY as training: kernel=(5,5), sigma=1.4)
		img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
		
		# Use adaptive Canny edge detector (EXACTLY as training)
		# Calculate adaptive thresholds based on image statistics
		median = np.median(img_blurred)
		lower_thresh = int(max(0, 0.7 * median))
		upper_thresh = int(min(255, 1.3 * median))
		
		# Apply Canny edge detector with adaptive thresholds
		edges = cv2.Canny(img_blurred, threshold1=lower_thresh, threshold2=upper_thresh)
		
		# Apply morphological operations (EXACTLY as training: MORPH_ELLIPSE (3, 3), MORPH_CLOSE)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
		
		# Convert GRAY to RGB (3 channels, as training does)
		edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
		
		# Convert to float32 [0, 1] (EXACTLY as training)
		edges_float = edges_3ch.astype(np.float32) / 255.0
		
		# Convert to tensor [H, W, C] -> [C, H, W]
		edges_tensor = torch.from_numpy(edges_float).permute(2, 0, 1).float()
		
		# Normalize to [-1, 1] (as done in training)
		edges_tensor = edges_tensor * 2.0 - 1.0
		
		edge_maps.append(edges_tensor)
	
	# Stack all edge maps to [B, C, H, W]
	return torch.stack(edge_maps).to(image_tensor.device)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload",
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
		default=1000,
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
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
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
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--use_edge",
		action="store_true",
		help="whether to use edge as additional condition",
	)
	parser.add_argument(
		"--save_edge",
		action="store_true",
		help="whether to save edge maps",
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	
	print('>>>>>>>>>>edge support>>>>>>>>>>>')
	if opt.use_edge:
		print('Use edge as additional condition')
		print('Edge detection: Adaptive Canny (matches training)')
	else:
		print('No edge condition')
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
	
	# Create edge output directory if needed
	if opt.save_edge:
		edge_outpath = os.path.join(outpath, "edge_maps")
		os.makedirs(edge_outpath, exist_ok=True)

	batch_size = opt.n_samples

	img_list_ori = os.listdir(opt.init_img)
	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	for item in img_list_ori:
		if os.path.exists(os.path.join(outpath, item)):
			img_list.remove(item)
			continue
		cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
		cur_image = transform(cur_image)
		cur_image = cur_image.clamp(-1, 1)
		init_image_list.append(cur_image)
	init_image_list = torch.cat(init_image_list, dim=0)
	niters = math.ceil(init_image_list.size(0) / batch_size)
	init_image_list = init_image_list.chunk(niters)

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
					
					# Extract edge if use_edge is True
					if opt.use_edge:
						edge_map = extract_edge(init_image)
						# Save edge maps if requested
						if opt.save_edge:
							for i in range(edge_map.size(0)):
								img_name = img_list[i]
								basename = os.path.splitext(os.path.basename(img_name))[0]
								# edge_map is [B, 3, H, W], take first channel for visualization
								edge_img = ((edge_map[i, 0].cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
								Image.fromarray(edge_img).save(
									os.path.join(edge_outpath, basename + '_edge.png'))
					
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

					# Pass edge_map directly to sample method if use_edge is True
					# The model's p_sample_loop accepts edge_map parameter in image space [B, 3, H, W]
					if opt.use_edge:
						print(f"[DEBUG] edge_map shape: {edge_map.shape}, dtype: {edge_map.dtype}, device: {edge_map.device}")
						print(f"[DEBUG] init_image shape: {init_image.shape}")
						samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, 
												batch_size=init_image.size(0), timesteps=opt.ddpm_steps, 
												time_replace=opt.ddpm_steps, x_T=x_T, 
												edge_map=edge_map, return_intermediates=True)
					else:
						samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, 
												batch_size=init_image.size(0), timesteps=opt.ddpm_steps, 
												time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
					
					x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

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

