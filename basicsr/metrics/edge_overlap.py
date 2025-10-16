"""
Edge Overlap Metric
计算两张图片的edge map之间的重叠率 (Overlap Ratio)
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple

from basicsr.utils.edge_utils import EdgeMapGenerator


class EdgeOverlapCalculator:
    """
    计算两张图片的edge map之间的重叠率
    
    该类用于评估super-resolution结果的边缘匹配度，通过比较生成图片和GT图片的
    edge map重叠程度来量化边缘保真度。
    
    重叠率 = 交集(生成边缘 ∩ GT边缘) / GT边缘总数
    
    使用方法:
        calculator = EdgeOverlapCalculator()
        overlap = calculator.calculate_from_files(gen_img_path, gt_img_path)
        
    或者:
        calculator = EdgeOverlapCalculator()
        overlap = calculator.calculate_from_arrays(gen_img_np, gt_img_np)
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
        初始化Edge Overlap计算器
        
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
            img1 (np.ndarray): 第一张图片 (H, W) 或 (H, W, C)
            img2 (np.ndarray): 第二张图片 (H, W) 或 (H, W, C)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 尺寸相同的两张图片
        """
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            if len(img1.shape) == 2:  # Grayscale
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            else:  # Color
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2
    
    def calculate_from_arrays(
        self, 
        gen_img: np.ndarray, 
        gt_img: np.ndarray,
        input_format: str = 'BGR'
    ) -> float:
        """
        从numpy数组计算edge overlap
        
        Args:
            gen_img (np.ndarray): 生成的图片，形状(H, W, C)，值域[0, 255]
            gt_img (np.ndarray): 真值图片，形状(H, W, C)，值域[0, 255]
            input_format (str): 输入图片格式，'BGR'或'RGB'，默认'BGR'
            
        Returns:
            float: Edge Overlap，值域[0, 1]，值越大表示边缘重叠越多
        """
        # 确保尺寸相同
        gen_img, gt_img = self._ensure_same_size(gen_img, gt_img)
        
        # 生成edge maps (返回值域[0, 1]，shape: H, W, C)
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
        
        # 转换为灰度图（取任意通道，因为edge map是灰度的）
        # edge maps 值域 [0, 1]
        if len(gen_edge.shape) == 3:
            gen_edge_gray = gen_edge[:, :, 0]  # 取第一个通道
            gt_edge_gray = gt_edge[:, :, 0]
        else:
            gen_edge_gray = gen_edge
            gt_edge_gray = gt_edge
        
        # 二值化：edge map中 > 0 的为边缘
        # edge map 值域是 [0, 1]，边缘处接近1，背景接近0
        gen_edge_bin = (gen_edge_gray > 0.5).astype(np.uint8)
        gt_edge_bin = (gt_edge_gray > 0.5).astype(np.uint8)
        
        # 计算交集和GT边缘总数
        intersection = np.logical_and(gen_edge_bin, gt_edge_bin)
        gt_edge_count = np.sum(gt_edge_bin)
        
        # 计算重叠率
        if gt_edge_count == 0:
            # GT没有边缘，返回0（或可以返回1，取决于定义）
            return 0.0
        
        overlap = np.sum(intersection) * 1.0 / gt_edge_count
        
        return float(overlap)
    
    def calculate_from_files(
        self, 
        gen_img_path: str, 
        gt_img_path: str
    ) -> float:
        """
        从图片文件路径计算edge overlap
        
        Args:
            gen_img_path (str): 生成图片的路径
            gt_img_path (str): 真值图片的路径
            
        Returns:
            float: Edge Overlap，值域[0, 1]，值越大表示边缘重叠越多
            
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
        
        # 计算overlap
        return self.calculate_from_arrays(gen_img, gt_img, input_format='BGR')
    
    def calculate_from_tensors(
        self, 
        gen_tensor: torch.Tensor, 
        gt_tensor: torch.Tensor,
        normalize_range: str = '[-1,1]'
    ) -> float:
        """
        从PyTorch tensor计算edge overlap
        
        Args:
            gen_tensor (torch.Tensor): 生成的图片张量，形状(C, H, W)或(B, C, H, W)
            gt_tensor (torch.Tensor): 真值图片张量，形状(C, H, W)或(B, C, H, W)
            normalize_range (str): 输入归一化范围，'[-1,1]'或'[0,1]'，默认'[-1,1]'
            
        Returns:
            float: Edge Overlap，值域[0, 1]，值越大表示边缘重叠越多
            
        Note:
            如果输入是batch，返回batch的平均overlap
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
        
        # 归一化到[0, 1]
        gen_edge_normalized = (gen_edge + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        gt_edge_normalized = (gt_edge + 1.0) / 2.0    # [-1, 1] -> [0, 1]
        
        # 取第一个通道（edge map是灰度的）
        gen_edge_gray = gen_edge_normalized[:, 0, :, :]  # (B, H, W)
        gt_edge_gray = gt_edge_normalized[:, 0, :, :]
        
        # 二值化
        gen_edge_bin = (gen_edge_gray > 0.5).float()
        gt_edge_bin = (gt_edge_gray > 0.5).float()
        
        # 计算交集和GT边缘总数
        intersection = gen_edge_bin * gt_edge_bin
        gt_edge_count = torch.sum(gt_edge_bin)
        
        # 计算重叠率
        if gt_edge_count == 0:
            return 0.0
        
        overlap = torch.sum(intersection) / gt_edge_count
        
        return float(overlap.item())
    
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
            float: Edge Overlap
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
default_edge_overlap_calculator = EdgeOverlapCalculator()


def calculate_edge_overlap(
    gen_img: Union[str, np.ndarray, torch.Tensor],
    gt_img: Union[str, np.ndarray, torch.Tensor],
    **kwargs
) -> float:
    """
    便捷函数：使用默认参数计算edge overlap
    
    Args:
        gen_img: 生成图片（路径/numpy数组/tensor）
        gt_img: 真值图片（路径/numpy数组/tensor）
        **kwargs: 传递给EdgeOverlapCalculator的参数
        
    Returns:
        float: Edge Overlap (范围 [0, 1])
    """
    return default_edge_overlap_calculator(gen_img, gt_img, **kwargs)

