"""make variations of input image with edge processing support

IMPORTANT: Edge Map Generation Strategy
========================================
This script generates edge maps for super-resolution inference.

Resolution Relationship:
- LR (input): downsampled from GT by 4x (e.g., 512x512)
- GT (ground truth): original high resolution (e.g., 2048x2048)
- Edge map: generated from GT at ORIGINAL resolution (2048x2048)
- Edge map resolution = LR resolution × 4 = GT resolution

For best results and consistency with training:
- Use --gt-img to provide ground truth images for edge map generation
- Training uses GT images for edge maps (see basicsr/data/realesrgan_dataset.py)
- Using LR images for edge maps creates domain mismatch with training

Edge generation pipeline (matches training exactly):
1. Load GT image at ORIGINAL resolution (no crop, no resize)
2. Convert to grayscale
3. Apply Gaussian blur: kernel=(5,5), sigma=1.4
4. Adaptive Canny: lower=0.7*median, upper=1.3*median
5. Morphological close: MORPH_ELLIPSE (3,3)
6. Convert to 3-channel RGB
7. Normalize to [-1, 1]
8. Keep edge map at ORIGINAL GT resolution (no resize)

Key point: Edge map is generated from and kept at ORIGINAL high-res GT resolution.
Since LR is 4x downsampled from GT, the edge map is 4x larger than LR input.
The model processes the high-resolution edge map directly without downsampling.
This preserves all fine edge details from the original GT image.
"""

import argparse, os, sys, glob
import shutil
import inspect

# Ensure we import from the current project directory, not any other StableSR installations
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
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
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

# Edge-specific imports
import cv2


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

def generate_edge_map(image_tensor):
	"""
	Generate edge map from input image tensor
	
	Args:
		image_tensor: Input image tensor [B, 3, H, W], values in [-1, 1]
		
	Returns:
		edge_map: Edge map tensor [B, 3, H, W], values in [-1, 1]
	
	Note:
		This function EXACTLY matches the training-time edge map generation
		in basicsr/data/realesrgan_dataset.py for consistency.
		
		Training pipeline (realesrgan_dataset.py lines 178-205):
		1. Convert BGR to grayscale
		2. Convert float32 [0,1] to uint8 [0,255]
		3. Apply Gaussian blur: (5, 5), sigma=1.4
		4. Adaptive Canny: lower=0.7*median, upper=1.3*median
		5. Morphological close: MORPH_ELLIPSE (3, 3)
		6. Convert GRAY to BGR (3 channels)
		7. Convert back to float32 [0,1]
	"""
	batch_size = image_tensor.size(0)
	edge_maps = []
	
	for i in range(batch_size):
		# Convert to numpy array [C, H, W] -> [H, W, C]
		img_np = image_tensor[i].cpu().numpy()
		img_np = (img_np + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
		img_np = np.clip(img_np, 0, 1)
		img_np = np.transpose(img_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
		
		# Convert RGB to grayscale (training uses BGR but we have RGB, so use RGB2GRAY)
		img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
		
		# Apply Gaussian blur (EXACTLY as training: kernel=(5,5), sigma=1.4)
		img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
		
		# Use adaptive Canny edge detector (EXACTLY as training)
		median = np.median(img_blurred)
		lower_thresh = int(max(0, 0.7 * median))
		upper_thresh = int(min(255, 1.3 * median))
		edges = cv2.Canny(img_blurred, threshold1=lower_thresh, threshold2=upper_thresh)
		
		# Apply morphological operations (EXACTLY as training: MORPH_ELLIPSE (3, 3), MORPH_CLOSE)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
		
		# Convert GRAY to RGB (training converts GRAY to BGR, we convert to RGB)
		edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
		
		# Convert to float32 [0, 1] (EXACTLY as training)
		edges_float = edges_3ch.astype(np.float32) / 255.0
		
		# Convert to tensor [H, W, C] -> [C, H, W]
		edges_tensor = torch.from_numpy(edges_float).permute(2, 0, 1).float()
		
		# Normalize to [-1, 1] (done in training loader: img_edge * 2.0 - 1.0)
		edges_tensor = edges_tensor * 2.0 - 1.0
		
		edge_maps.append(edges_tensor)
	
	# Stack all edge maps to [B, C, H, W]
	return torch.stack(edge_maps).to(image_tensor.device)

def generate_edge_map_from_gt(gt_image_path, device):
	"""
	Generate edge map from ground truth image file (TRAINING-CONSISTENT METHOD)
	
	Args:
		gt_image_path: Path to ground truth image (original resolution)
		device: Device to place the tensor on
		
	Returns:
		edge_map: Edge map tensor [1, 3, H_gt, W_gt], values in [-1, 1]
		         at ORIGINAL GT resolution (NOT resized)
		         where H_gt = H_lr × 4, W_gt = W_lr × 4
	
	Note:
		This is the CORRECT way to generate edge maps for inference!
		Training uses GT images for edge map generation, so inference should too.
		
		Resolution relationship (4x super-resolution):
		- LR input: H_lr × W_lr (e.g., 512×512)
		- GT image: H_gt × W_gt = 4×H_lr × 4×W_lr (e.g., 2048×2048)
		- Edge map: H_gt × W_gt (same as GT, i.e., 4× LR)
		
		Pipeline:
		1. Load GT image at ORIGINAL resolution (no crop, no resize)
		2. Generate edge map from original GT image
		3. Return edge map at ORIGINAL resolution (no resize)
		
		The edge map maintains GT resolution (4x LR resolution) and will be 
		processed by the model at its native resolution. Using LR images for 
		edge maps creates domain mismatch with training.
	"""
	# Load GT image at ORIGINAL resolution
	gt_image = Image.open(gt_image_path).convert("RGB")
	
	# Convert to numpy array [H, W, C] - keep original resolution
	img_np = np.array(gt_image).astype(np.float32) / 255.0
	
	# Convert to tensor format [C, H, W]
	img_np = np.transpose(img_np, (2, 0, 1))
	img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
	
	# Normalize to [-1, 1]
	img_tensor = 2.0 * img_tensor - 1.0
	
	# Generate edge map from ORIGINAL resolution GT image
	# Keep at original resolution (4x LR resolution) - no resize
	edge_map = generate_edge_map(img_tensor)
	
	return edge_map.to(device)


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
		"--use_edge_processing",
		action="store_true",
		help="Enable edge processing for enhanced super-resolution",
	)
	parser.add_argument(
		"--use_white_edge",
		action="store_true",
		help="Use black (all negative ones) edge maps instead of generated edge maps (no edge mode)",
	)
	parser.add_argument(
		"--use_dummy_edge",
		action="store_true",
		help="Use a fixed dummy edge map for all images",
	)
	parser.add_argument(
		"--dummy_edge_path",
		type=str,
		default="/stablesr_dataset/default_edge.png",
		help="Path to the dummy edge image to use when --use_dummy_edge is enabled",
	)
	parser.add_argument(
		"--gt-img",
		type=str,
		nargs="?",
		help="[RECOMMENDED] Path to ground truth images directory for edge map generation. "
		     "Training uses GT images at ORIGINAL resolution for edge maps, so using GT images "
		     "during inference ensures consistency and better results. "
		     "When specified, all input images MUST have corresponding GT images. "
		     "Edge maps will be kept at original GT resolution (no resize). "
		     "Without this flag, LR images will be used which causes domain mismatch.",
		default=None,
	)
	parser.add_argument(
		"--max_images",
		type=int,
		default=-1,
		help="Maximum number of images to process (-1 for all images)",
	)
	parser.add_argument(
		"--specific_file",
		type=str,
		default="",
		help="Process only this specific file (filename only, not full path)",
	)

	opt = parser.parse_args()
	
	# Expand user paths (~ symbols)
	opt.ckpt = os.path.expanduser(opt.ckpt)
	opt.vqgan_ckpt = os.path.expanduser(opt.vqgan_ckpt)
	opt.init_img = os.path.expanduser(opt.init_img)
	opt.outdir = os.path.expanduser(opt.outdir)
	if opt.gt_img:
		opt.gt_img = os.path.expanduser(opt.gt_img)
	if opt.dummy_edge_path:
		opt.dummy_edge_path = os.path.expanduser(opt.dummy_edge_path)
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	print('>>>>>>>>>>edge processing>>>>>>>>>>>')
	if opt.use_edge_processing:
		print('Edge processing enabled - using edge-enhanced model')
		if opt.use_white_edge:
			print('Using BLACK edge maps (no edges)')
		elif opt.use_dummy_edge:
			print(f'Using DUMMY edge map from: {opt.dummy_edge_path}')
		elif opt.gt_img:
			print(f'✓ Using GT images for edge map generation from: {opt.gt_img}')
			print('  (This matches the training setup)')
		else:
			print('⚠ WARNING: No GT images provided!')
			print('  Training uses GT images for edge maps, but inference will use LR images.')
			print('  This may cause domain mismatch. Recommend using --gt-img for best results.')
	else:
		print('Edge processing disabled - using standard model')
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
	
	# Check if model supports edge processing
	model_supports_edge = hasattr(model, 'use_edge_processing') and model.use_edge_processing
	if opt.use_edge_processing:
		if model_supports_edge:
			print("✓ Model supports edge processing - edge inference will be used")
		else:
			print("⚠ Warning: --use_edge_processing flag is set, but model does not support edge processing")
			print("  Edge maps will be generated and saved, but may not affect results")
			print("  To use edge processing, ensure config file has use_edge_processing: true")

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	# Get image list
	if opt.specific_file:
		# Process only specific file
		if not os.path.exists(os.path.join(opt.init_img, opt.specific_file)):
			print(f"错误：指定文件不存在: {opt.specific_file}")
			return
		img_list_ori = [opt.specific_file]
		print(f"只处理指定文件: {opt.specific_file}")
	else:
		# Process all or limited number of files
		img_list_ori = os.listdir(opt.init_img)
		
		# Limit number of images if max_images is specified
		if opt.max_images > 0:
			img_list_ori = img_list_ori[:opt.max_images]
			print(f"限制处理图片数量: {len(img_list_ori)} 张")
		else:
			print(f"处理全部图片: {len(img_list_ori)} 张")
	
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

	# Check if model.sample supports edge_map parameter (only once)
	model_sample_supports_edge = False
	if opt.use_edge_processing:
		sample_sig = inspect.signature(model.sample)
		model_sample_supports_edge = 'edge_map' in sample_sig.parameters
		if model_sample_supports_edge:
			print("✓ model.sample() supports edge_map parameter - will use edge-enhanced sampling")
		else:
			print("⚠ model.sample() does not support edge_map parameter - edge maps will be saved but not used in sampling")
	
	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				count = 0
				for n in trange(niters, desc="Sampling"):
					init_image = init_image_list[n]
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

					if opt.use_edge_processing:
						# Generate edge maps for each image in the batch
						edge_maps = []
						
						if opt.use_dummy_edge:
							# Use fixed dummy edge map from file
							# Load and process dummy edge once, then reuse for all images
							if not os.path.exists(opt.dummy_edge_path):
								print(f"ERROR: Dummy edge file not found: {opt.dummy_edge_path}")
								print("Falling back to black edge maps (no edges)")
								for i in range(init_image.size(0)):
									black_edge_map = -torch.ones_like(init_image[i:i+1])
									edge_maps.append(black_edge_map)
								edge_maps = torch.cat(edge_maps, dim=0)
							else:
								# Load dummy edge image (PIL.Image already imported at top)
								dummy_edge_img = Image.open(opt.dummy_edge_path).convert('RGB')
								
								# Resize to match init_image size
								target_size = (init_image.size(3), init_image.size(2))  # (W, H) for PIL
								dummy_edge_img = dummy_edge_img.resize(target_size, Image.BICUBIC)
								
								# Convert to tensor and normalize to [-1, 1]
								# torchvision.transforms already imported at top
								to_tensor = torchvision.transforms.ToTensor()
								dummy_edge_tensor = to_tensor(dummy_edge_img)  # [3, H, W] in [0, 1]
								dummy_edge_tensor = dummy_edge_tensor * 2.0 - 1.0  # Convert to [-1, 1]
								dummy_edge_tensor = dummy_edge_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
								
								# Replicate for all images in batch
								for i in range(init_image.size(0)):
									edge_maps.append(dummy_edge_tensor)
								edge_maps = torch.cat(edge_maps, dim=0)
								print(f"Using DUMMY edge map from: {opt.dummy_edge_path}")
						elif opt.use_white_edge:
							# Use black (all negative ones) edge maps
							# Create black edge map with same spatial size as init_image
							# Edge maps are in range [-1, 1], so black = -1.0
							for i in range(init_image.size(0)):
								black_edge_map = -torch.ones_like(init_image[i:i+1])
								edge_maps.append(black_edge_map)
							edge_maps = torch.cat(edge_maps, dim=0)
							print("Using BLACK edge maps (all negative ones, no edges)")
						else:
							# Generate edge maps from GT images (matching training setup)
							for i in range(init_image.size(0)):
								# Get the corresponding image name for this batch item
								batch_start_idx = n * batch_size
								img_idx = batch_start_idx + i
								if img_idx < len(img_list_ori):
									img_name = img_list_ori[img_idx]
									img_basename = os.path.splitext(os.path.basename(img_name))[0]
									
									# Try to use GT image for edge map generation (same as training)
									edge_map = None
									if opt.gt_img:
										gt_img_path = os.path.join(opt.gt_img, img_basename + '.png')
										if not os.path.exists(gt_img_path):
											# Try other common extensions
											for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
												alt_path = os.path.join(opt.gt_img, img_basename + ext)
												if os.path.exists(alt_path):
													gt_img_path = alt_path
													break
										
										if os.path.exists(gt_img_path):
											# Generate edge map from ORIGINAL resolution GT image (matches training)
											# Keep at original GT resolution (4x LR resolution) - no resize
											# Example: LR=512×512 -> GT=2048×2048 -> Edge map=2048×2048
											# The model will process the high-resolution edge map directly
											edge_map = generate_edge_map_from_gt(gt_img_path, device)
										else:
											raise FileNotFoundError(
												f"GT image not found for {img_basename}. "
												f"When using --gt-img for edge processing, all input images must have "
												f"corresponding GT images. Expected at: {gt_img_path} or with other extensions."
											)
									else:
										# No GT provided, use LR image (not recommended - domain mismatch)
										if n == 0 and i == 0:  # Only warn once
											print("⚠ WARNING: Using LR images for edge maps (domain mismatch with training)")
											print("  This is NOT recommended. Training uses GT images for edge maps.")
											print("  Use --gt-img for best results.")
										edge_map = generate_edge_map(init_image[i:i+1])
								else:
									# Fallback to LR image if index is out of range
									if not opt.gt_img:
										edge_map = generate_edge_map(init_image[i:i+1])
									else:
										raise IndexError(f"Image index {img_idx} out of range")
								
								edge_maps.append(edge_map)
							edge_maps = torch.cat(edge_maps, dim=0)
						
						# Verify resolution relationship: edge_map should be 4x LR
						if n == 0:  # Only print for first batch
							lr_h, lr_w = init_image.size(2), init_image.size(3)
							edge_h, edge_w = edge_maps.size(2), edge_maps.size(3)
							print(f"\n=== Resolution Verification (Batch 0) ===")
							print(f"LR input shape: {init_image.shape} -> spatial: {lr_h}×{lr_w}")
							print(f"Edge map shape: {edge_maps.shape} -> spatial: {edge_h}×{edge_w}")
							
							if opt.gt_img and not opt.use_white_edge and not opt.use_dummy_edge:
								# Edge map from GT should be 4x LR
								expected_edge_h, expected_edge_w = lr_h * 4, lr_w * 4
								if edge_h == expected_edge_h and edge_w == expected_edge_w:
									print(f"✓ Edge map resolution correct: {edge_h}×{edge_w} = 4 × {lr_h}×{lr_w}")
								else:
									print(f"⚠ WARNING: Edge map resolution mismatch!")
									print(f"  Expected: {expected_edge_h}×{expected_edge_w} (4× LR)")
									print(f"  Got: {edge_h}×{edge_w}")
							else:
								# Edge map from LR image (same size as LR)
								if edge_h == lr_h and edge_w == lr_w:
									print(f"Edge map matches LR size (using LR for edge generation)")
								else:
									print(f"Edge map size: {edge_h}×{edge_w}")
							
							print(f"Edge maps range: [{edge_maps.min():.3f}, {edge_maps.max():.3f}], mean: {edge_maps.mean():.3f}")
							print(f"Init latent shape: {init_latent.shape}")
							print(f"Init latent range: [{init_latent.min():.3f}, {init_latent.max():.3f}], mean: {init_latent.mean():.3f}")
							print("=" * 45)
						
						# Save edge maps to edge_map subdirectory
						edge_map_dir = os.path.join(outpath, "edge_map")
						os.makedirs(edge_map_dir, exist_ok=True)
						
						for i in range(edge_maps.size(0)):
							# Get the corresponding image name for this batch item
							batch_start_idx = n * batch_size
							img_idx = batch_start_idx + i
							if img_idx < len(img_list_ori):
								img_name = img_list_ori[img_idx]
								img_basename = os.path.splitext(os.path.basename(img_name))[0]
								
								# Convert edge map from [-1, 1] to [0, 255]
								edge_map_np = edge_maps[i].cpu().numpy()  # [3, H, W]
								edge_map_np = (edge_map_np + 1.0) / 2.0 * 255.0  # Convert to [0, 255]
								edge_map_np = np.clip(edge_map_np, 0, 255)
								edge_map_np = np.transpose(edge_map_np, (1, 2, 0))  # [H, W, 3]
								
								# Save edge map
								edge_map_save_path = os.path.join(edge_map_dir, f"{img_basename}_edge.png")
								Image.fromarray(edge_map_np.astype(np.uint8)).save(edge_map_save_path)
						
						print(f"Edge maps saved to: {edge_map_dir}")
						
						# Use edge-enhanced sampling with edge maps
						if model_sample_supports_edge:
							samples, _ = model.sample(
								cond=semantic_c, 
								struct_cond=init_latent, 
								batch_size=init_image.size(0), 
								timesteps=opt.ddpm_steps, 
								time_replace=opt.ddpm_steps, 
								x_T=x_T, 
								return_intermediates=True,
								edge_map=edge_maps
							)
						else:
							# Fallback to standard sampling
							samples, _ = model.sample(
								cond=semantic_c, 
								struct_cond=init_latent, 
								batch_size=init_image.size(0), 
								timesteps=opt.ddpm_steps, 
								time_replace=opt.ddpm_steps, 
								x_T=x_T, 
								return_intermediates=True
							)
						
						# Debug: Print samples statistics (only for first batch)
						if n == 0:
							print(f"Samples shape: {samples.shape}")
							print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}], mean: {samples.mean():.3f}")
					else:
						# Use standard sampling
						samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
					
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
						
						# Add edge processing suffix to filename if enabled
						suffix = "_edge" if opt.use_edge_processing else ""
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath, basename+suffix+'.png'))
						
						# Save LR input image
						lr_input_dir = os.path.join(outpath, "lr_input")
						os.makedirs(lr_input_dir, exist_ok=True)
						lr_image = init_image[i].cpu().numpy()  # [3, H, W] in [-1, 1]
						lr_image = (lr_image + 1.0) / 2.0 * 255.0  # Convert to [0, 255]
						lr_image = np.clip(lr_image, 0, 255)
						lr_image = np.transpose(lr_image, (1, 2, 0))  # [H, W, 3]
						Image.fromarray(lr_image.astype(np.uint8)).save(
							os.path.join(lr_input_dir, f"{basename}.png"))
						
						# Save GT HR image if available
						if opt.gt_img:
							gt_hr_dir = os.path.join(outpath, "gt_hr")
							os.makedirs(gt_hr_dir, exist_ok=True)
							
							# Find GT image with the same basename
							gt_img_path = os.path.join(opt.gt_img, basename + '.png')
							if not os.path.exists(gt_img_path):
								# Try other common extensions
								for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
									alt_path = os.path.join(opt.gt_img, basename + ext)
									if os.path.exists(alt_path):
										gt_img_path = alt_path
										break
							
							if os.path.exists(gt_img_path):
								# Copy GT image to output directory
								shutil.copy2(gt_img_path, os.path.join(gt_hr_dir, f"{basename}.png"))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()

