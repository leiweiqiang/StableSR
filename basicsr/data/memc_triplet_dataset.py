from pathlib import Path

import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MemcTripletDataset(data.Dataset):
    """Load MEMC triplets and fuse two frames into one LQ input.

    Expected layout under dataroot:
      frames/clip_xxxx/frame_0001.png  (frame0)
      frames/clip_xxxx/frame_0005.png  (gt_mid)
      frames/clip_xxxx/frame_0010.png  (frame1)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.alpha = opt.get("alpha", 0.5)
        self.gt_size = opt.get("gt_size")
        self.use_hflip = opt.get("use_hflip", True)
        self.use_rot = opt.get("use_rot", False)
        self.scale = opt.get("scale", 1)

        self.frame0_name = opt.get("frame0_name", "frame_0001.png")
        self.gt_name = opt.get("gt_name", "frame_0005.png")
        self.frame1_name = opt.get("frame1_name", "frame_0010.png")

        dataroot = opt.get("dataroot") or opt.get("root") or opt.get("dataroot_gt")
        if dataroot is None:
            raise ValueError("dataroot is required for MemcTripletDataset.")
        self.dataroot = Path(dataroot)
        self.paths = self._scan_paths()

    def _scan_paths(self):
        frames_root = self.dataroot / "frames"
        if not frames_root.exists():
            raise FileNotFoundError(f"frames folder not found: {frames_root}")
        clip_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
        paths = []
        for clip_dir in clip_dirs:
            frame0_path = clip_dir / self.frame0_name
            gt_path = clip_dir / self.gt_name
            frame1_path = clip_dir / self.frame1_name
            if frame0_path.exists() and gt_path.exists() and frame1_path.exists():
                paths.append(
                    {
                        "frame0_path": str(frame0_path),
                        "gt_path": str(gt_path),
                        "frame1_path": str(frame1_path),
                    }
                )
        if not paths:
            raise FileNotFoundError(f"No valid triplets found under {frames_root}")
        return paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        triplet = self.paths[index]
        frame0_path = triplet["frame0_path"]
        gt_path = triplet["gt_path"]
        frame1_path = triplet["frame1_path"]

        frame0_bytes = self.file_client.get(frame0_path, "frame0")
        img_frame0 = imfrombytes(frame0_bytes, float32=True)
        gt_bytes = self.file_client.get(gt_path, "gt")
        img_gt = imfrombytes(gt_bytes, float32=True)
        frame1_bytes = self.file_client.get(frame1_path, "frame1")
        img_frame1 = imfrombytes(frame1_bytes, float32=True)

        if self.opt.get("phase") == "train" and self.gt_size is not None:
            img_gt, img_lqs = paired_random_crop(
                img_gt, [img_frame0, img_frame1], self.gt_size, self.scale, gt_path
            )
            img_frame0, img_frame1 = img_lqs
            img_gt, img_frame0, img_frame1 = augment(
                [img_gt, img_frame0, img_frame1], self.use_hflip, self.use_rot
            )

        img_lq = img_frame0 * self.alpha + img_frame1 * (1.0 - self.alpha)
        img_lq = np.clip(img_lq, 0.0, 1.0)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "frame0_path": frame0_path,
            "frame1_path": frame1_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)
