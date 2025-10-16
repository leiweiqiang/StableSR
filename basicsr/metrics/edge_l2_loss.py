"""
Edge PSNR Metric
计算两张图片的edge map之间的PSNR (Peak Signal-to-Noise Ratio)
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple

from basicsr.utils.edge_utils import EdgeMapGenerator


class EdgePSNRCalculator:
    """
    计算两张图片的edge map之间的PSNR (Peak Signal-to-Noise Ratio)
    
    该类用于评估super-resolution结果的边缘质量，通过比较生成图片和GT图片的
    edge map来量化边缘保真度。
    
    使用方法:
        calculator = EdgePSNRCalculator()
        psnr = calculator.calculate_from_files(gen_img_path, gt_img_path)
        
    或者:
        calculator = EdgePSNRCalculator()
        psnr = calculator.calculate_from_arrays(gen_img_np, gt_img_np)
    """
    
    def __init__(
        self,
        gaussian_kernel_size=(5, 5),
        gaussian_sigma=1.4,
        canny_threshold_lower_factor=0.7,
        canny_threshold_upper_factor=1.3,
        morph_kernel_size=(3, 3),
        morph_kernel_shape=cv2.MORPH_ELLIPSE,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化Edge PSNR计算器
        
        Args:
            gaussian_kernel_size (tuple): 高斯模糊核大小，默认(5, 5)
            gaussian_sigma (float): 高斯模糊标准差，默认1.4
            canny_threshold_lower_factor (float): Canny下阈值因子，默认0.7
            canny_threshold_upper_factor (float): Canny上阈值因子，默认1.3
            morph_kernel_size (tuple): 形态学操作核大小，默认(3, 3)
            morph_kernel_shape: 形态学核形状，默认cv2.MORPH_ELLIPSE
            device (str): 设备类型，默认'cuda'如果可用，否则'cpu'
        """
        self.edge_generator = EdgeMapGenerator(
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            canny_threshold_lower_factor=canny_threshold_lower_factor,
            canny_threshold_upper_factor=canny_threshold_upper_factor,
            morph_kernel_size=morph_kernel_size,
            morph_kernel_shape=morph_kernel_shape,
            device=device
        )
        self.device = device
    
    def _ensure_same_size(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        确保两张图片尺寸相同，如果不同则resize第二张图片
        
        Args:
            img1 (np.ndarray): 第一张图片 (H, W, C)
            img2 (np.ndarray): 第二张图片 (H, W, C)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 尺寸相同的两张图片
        """
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2
    
    def calculate_from_arrays(
        self, 
        gen_img: np.ndarray, 
        gt_img: np.ndarray,
        input_format: str = 'BGR'
    ) -> float:
        """
        从numpy数组计算edge PSNR
        
        Args:
            gen_img (np.ndarray): 生成的图片，形状(H, W, C)，值域[0, 255]
            gt_img (np.ndarray): 真值图片，形状(H, W, C)，值域[0, 255]
            input_format (str): 输入图片格式，'BGR'或'RGB'，默认'BGR'
            
        Returns:
            float: Edge PSNR (dB)，值越大表示边缘质量越好
        """
        # 确保尺寸相同
        gen_img, gt_img = self._ensure_same_size(gen_img, gt_img)
        
        # 生成edge maps (返回值域[0, 1])
        gen_edge = self.edge_generator.generate_from_numpy(
            gen_img, 
            input_format=input_format, 
            normalize_input=False  # 输入是[0, 255]
        )
        
        gt_edge = self.edge_generator.generate_from_numpy(
            gt_img, 
            input_format=input_format, 
            normalize_input=False  # 输入是[0, 255]
        )
        
        # 计算MSE (Mean Squared Error)
        # edge maps值域为[0, 1]
        mse = np.mean((gen_edge - gt_edge) ** 2)
        
        # 计算PSNR
        # PSNR = 10 * log10(MAX^2 / MSE)
        # 对于[0, 1]范围的图像，MAX = 1
        if mse == 0:
            return float('inf')  # 完全相同，PSNR无穷大
        
        psnr = 10 * np.log10(1.0 / mse)
        
        return float(psnr)
    
    def calculate_from_files(
        self, 
        gen_img_path: str, 
        gt_img_path: str
    ) -> float:
        """
        从图片文件路径计算edge PSNR
        
        Args:
            gen_img_path (str): 生成图片的路径
            gt_img_path (str): 真值图片的路径
            
        Returns:
            float: Edge PSNR (dB)，值越大表示边缘质量越好
            
        Raises:
            FileNotFoundError: 如果任一图片文件不存在
            ValueError: 如果图片无法读取
        """
        # 读取图片 (cv2.imread返回BGR格式，值域[0, 255])
        gen_img = cv2.imread(gen_img_path)
        gt_img = cv2.imread(gt_img_path)
        
        if gen_img is None:
            raise ValueError(f"Cannot read image: {gen_img_path}")
        if gt_img is None:
            raise ValueError(f"Cannot read image: {gt_img_path}")
        
        # 计算PSNR
        return self.calculate_from_arrays(gen_img, gt_img, input_format='BGR')
    
    def calculate_from_tensors(
        self, 
        gen_tensor: torch.Tensor, 
        gt_tensor: torch.Tensor,
        normalize_range: str = '[-1,1]'
    ) -> float:
        """
        从PyTorch tensor计算edge PSNR
        
        Args:
            gen_tensor (torch.Tensor): 生成的图片张量，形状(C, H, W)或(B, C, H, W)
            gt_tensor (torch.Tensor): 真值图片张量，形状(C, H, W)或(B, C, H, W)
            normalize_range (str): 输入归一化范围，'[-1,1]'或'[0,1]'，默认'[-1,1]'
            
        Returns:
            float: Edge PSNR (dB)，值越大表示边缘质量越好
            
        Note:
            如果输入是batch，返回batch的平均PSNR
        """
        # 确保是batch格式
        if gen_tensor.dim() == 3:
            gen_tensor = gen_tensor.unsqueeze(0)
        if gt_tensor.dim() == 3:
            gt_tensor = gt_tensor.unsqueeze(0)
        
        # 生成edge maps (返回值域[-1, 1])
        gen_edge = self.edge_generator.generate_from_tensor(
            gen_tensor,
            input_format='RGB',
            normalize_range=normalize_range
        )
        
        gt_edge = self.edge_generator.generate_from_tensor(
            gt_tensor,
            input_format='RGB',
            normalize_range=normalize_range
        )
        
        # 计算MSE
        # 归一化到[0, 1]
        gen_edge_normalized = (gen_edge + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        gt_edge_normalized = (gt_edge + 1.0) / 2.0    # [-1, 1] -> [0, 1]
        
        mse = torch.mean((gen_edge_normalized - gt_edge_normalized) ** 2)
        
        # 计算PSNR
        if mse.item() == 0:
            return float('inf')  # 完全相同，PSNR无穷大
        
        psnr = 10 * torch.log10(1.0 / mse)
        
        return float(psnr.item())
    
    def __call__(
        self, 
        gen_img: Union[str, np.ndarray, torch.Tensor],
        gt_img: Union[str, np.ndarray, torch.Tensor],
        **kwargs
    ) -> float:
        """
        便捷调用方法，自动检测输入类型
        
        Args:
            gen_img: 生成图片（路径/numpy数组/tensor）
            gt_img: 真值图片（路径/numpy数组/tensor）
            **kwargs: 传递给相应计算函数的参数
            
        Returns:
            float: Edge PSNR (dB)
        """
        # 检查输入类型
        if isinstance(gen_img, str) and isinstance(gt_img, str):
            return self.calculate_from_files(gen_img, gt_img)
        elif isinstance(gen_img, np.ndarray) and isinstance(gt_img, np.ndarray):
            return self.calculate_from_arrays(gen_img, gt_img, **kwargs)
        elif isinstance(gen_img, torch.Tensor) and isinstance(gt_img, torch.Tensor):
            return self.calculate_from_tensors(gen_img, gt_img, **kwargs)
        else:
            raise TypeError(
                f"Unsupported input types: {type(gen_img)} and {type(gt_img)}. "
                f"Both inputs must be of the same type (str/np.ndarray/torch.Tensor)"
            )


# 创建默认的全局实例
default_edge_psnr_calculator = EdgePSNRCalculator()


def calculate_edge_psnr(
    gen_img: Union[str, np.ndarray, torch.Tensor],
    gt_img: Union[str, np.ndarray, torch.Tensor],
    **kwargs
) -> float:
    """
    便捷函数：使用默认参数计算edge PSNR
    
    Args:
        gen_img: 生成图片（路径/numpy数组/tensor）
        gt_img: 真值图片（路径/numpy数组/tensor）
        **kwargs: 传递给EdgePSNRCalculator的参数
        
    Returns:
        float: Edge PSNR (dB)
    """
    return default_edge_psnr_calculator(gen_img, gt_img, **kwargs)


# 向后兼容的别名（必须在函数定义之后）
EdgeL2LossCalculator = EdgePSNRCalculator
default_edge_l2_calculator = default_edge_psnr_calculator
calculate_edge_l2_loss = calculate_edge_psnr

