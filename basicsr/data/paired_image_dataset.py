from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_meta_info_file_2
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import cv2


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
            
        # Load edge map paths if specified
        self.edge_folder = opt.get('dataroot_edge', None)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file_2([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        
        # Ensure both images have the same dimensions
        if img_gt.shape != img_lq.shape:
            # Resize lq to match gt dimensions
            img_lq = cv2.resize(img_lq, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Load edge map if available
        edge_map = None
        if self.edge_folder is not None:
            # Extract filename from gt_path and construct edge map path
            import os
            gt_filename = os.path.basename(gt_path)
            edge_filename = os.path.splitext(gt_filename)[0] + '.png'  # Assume edge maps are PNG
            edge_path = os.path.join(self.edge_folder, edge_filename)
            
            try:
                edge_bytes = self.file_client.get(edge_path, 'edge')
                edge_map = imfrombytes(edge_bytes, float32=True)
                # Convert to grayscale if needed and normalize to [0, 1]
                if len(edge_map.shape) == 3:
                    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
                edge_map = edge_map.astype(np.float32) / 255.0
                
                # Ensure edge map has the same dimensions as the GT image
                if edge_map.shape != img_gt.shape[:2]:
                    # Resize edge map to match GT image dimensions
                    edge_map = cv2.resize(edge_map, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
            except (IOError, OSError) as e:
                # If edge map loading fails, set to None
                edge_map = None

        h, w = img_gt.shape[0:2]
        # pad or resize to ensure consistent dimensions
        target_size = self.opt['gt_size']
        if h != target_size or w != target_size:
            # Resize to target size to ensure consistency
            img_gt = cv2.resize(img_gt, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            img_lq = cv2.resize(img_lq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            if edge_map is not None:
                edge_map = cv2.resize(edge_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Update dimensions after resize
        h, w = img_gt.shape[0:2]

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Apply same crop to edge map if available
            if edge_map is not None:
                h, w = img_gt.shape[0:2]
                edge_map = edge_map[:h, :w]
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            # Apply same augmentation to edge map if available
            if edge_map is not None:
                edge_map = augment([edge_map], self.opt['use_hflip'], self.opt['use_rot'])[0]

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # Ensure consistent sizing for validation/testing
        if self.opt['phase'] != 'train':
            # Images are already resized to target_size, so no additional cropping needed
            pass

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return_dict = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        
        # Add edge map to return data if available
        if edge_map is not None:
            try:
                # Convert edge map to tensor: HWC to CHW
                import torch
                # Ensure the numpy array is contiguous before converting to tensor
                edge_map = np.ascontiguousarray(edge_map)
                edge_map = torch.from_numpy(edge_map).float().unsqueeze(0)  # Add channel dimension
                return_dict['edge_map'] = edge_map
            except Exception as e:
                # If edge map tensor creation fails, skip it
                print(f"Warning: Failed to create edge map tensor: {e}")
                edge_map = None
        
        return return_dict

    def __len__(self):
        return len(self.paths)
