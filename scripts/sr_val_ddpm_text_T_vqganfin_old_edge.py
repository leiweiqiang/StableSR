"""make variations of input image with edge processing support"""

import argparse, os, sys, glob
import shutil

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
# from tra_report import EdgeDDIMSampler  # Not needed - unused import
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
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
		This function now matches the training-time edge map generation
		in basicsr/data/realesrgan_dataset.py for consistency.
	"""
	# Convert to numpy array
	img_np = image_tensor[0].cpu().numpy()
	img_np = (img_np + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
	img_np = np.clip(img_np, 0, 1)
	img_np = np.transpose(img_np, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
	
	# Convert to grayscale
	img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
	
	# Apply stronger Gaussian blur (match training: kernel=(7,7), sigma=2.0)
	img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
	
	# Apply adaptive Canny edge detection (match training)
	median = np.median(img_blurred)
	lower_thresh = int(max(0, 0.7 * median))
	upper_thresh = int(min(255, 1.3 * median))
	edges = cv2.Canny(img_blurred, threshold1=lower_thresh, threshold2=upper_thresh)
	
	# Apply morphological operations to clean up edges (match training)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	
	# Convert to 3 channels
	edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
	
	# Convert to float [0, 1] first (match training pipeline)
	edges_float = edges_3ch.astype(np.float32) / 255.0
	
	# Convert back to tensor format
	edges_tensor = torch.from_numpy(edges_float).permute(2, 0, 1).unsqueeze(0).float()
	
	# Then normalize to [-1, 1]
	edges_tensor = edges_tensor * 2.0 - 1.0
	
	return edges_tensor.to(image_tensor.device)

def generate_edge_map_from_gt(gt_image_path, target_size, device):
	"""
	Generate edge map from ground truth image file
	
	Args:
		gt_image_path: Path to ground truth image
		target_size: Target size for the edge map (H, W)
		device: Device to place the tensor on
		
	Returns:
		edge_map: Edge map tensor [1, 3, H, W], values in [-1, 1]
	"""
	# Load GT image
	gt_image = Image.open(gt_image_path).convert("RGB")
	
	# Resize to target size
	gt_image = gt_image.resize((target_size[1], target_size[0]), resample=PIL.Image.LANCZOS)
	
	# Convert to numpy array
	img_np = np.array(gt_image).astype(np.float32) / 255.0
	img_np = np.transpose(img_np, (2, 0, 1))  # Convert from [H, W, C] to [C, H, W]
	
	# Convert to tensor and normalize to [-1, 1]
	img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
	img_tensor = 2.0 * img_tensor - 1.0
	
	# Generate edge map using the existing function
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
		help="path to the ground truth images directory for edge map generation",
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
		if opt.gt_img:
			print(f'Using GT images from: {opt.gt_img}')
		else:
			print('Using LR images for edge map generation')
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
	
	# Use edge model if edge processing is enabled
	if opt.use_edge_processing:
		print("Loading edge-enhanced model...")
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
		
		# Load checkpoint for edge model
		model.init_from_ckpt(opt.ckpt)
		print("✓ Edge-enhanced model loaded successfully")
	else:
		model = load_model_from_config(config, f"{opt.ckpt}")
		model = model.to(device)

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
							# Generate edge maps from images
							for i in range(init_image.size(0)):
								# Get the corresponding image name for this batch item
								batch_start_idx = n * batch_size
								img_idx = batch_start_idx + i
								if img_idx < len(img_list_ori):
									img_name = img_list_ori[img_idx]
									img_basename = os.path.splitext(os.path.basename(img_name))[0]
									
									if opt.gt_img:
										# Use GT image for edge map generation
										gt_img_path = os.path.join(opt.gt_img, img_basename + '.png')
										if not os.path.exists(gt_img_path):
											# Try other common extensions
											for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
												alt_path = os.path.join(opt.gt_img, img_basename + ext)
												if os.path.exists(alt_path):
													gt_img_path = alt_path
													break
										
										if os.path.exists(gt_img_path):
											# Generate edge map from GT image
											target_size = (init_image.size(2), init_image.size(3))  # (H, W)
											edge_map = generate_edge_map_from_gt(gt_img_path, target_size, device)
										else:
											print(f"Warning: GT image not found for {img_basename}, using LR image for edge map")
											edge_map = generate_edge_map(init_image[i:i+1])
									else:
										# Use LR image for edge map generation (original behavior)
										edge_map = generate_edge_map(init_image[i:i+1])
								else:
									# Fallback to LR image if index is out of range
									edge_map = generate_edge_map(init_image[i:i+1])
								
								edge_maps.append(edge_map)
							edge_maps = torch.cat(edge_maps, dim=0)
						
						# Debug: Print edge map statistics
						print(f"Edge maps shape: {edge_maps.shape}")
						print(f"Edge maps range: [{edge_maps.min():.3f}, {edge_maps.max():.3f}], mean: {edge_maps.mean():.3f}")
						print(f"Init latent shape: {init_latent.shape}")
						print(f"Init latent range: [{init_latent.min():.3f}, {init_latent.max():.3f}], mean: {init_latent.mean():.3f}")
						
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
						
						# Debug: Print samples statistics
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
