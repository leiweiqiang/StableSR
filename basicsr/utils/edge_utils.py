"""
Edge Map Generation Utilities
用于训练和推理的边缘图生成工具
"""

import cv2
import numpy as np
import torch


class EdgeMapGenerator:
    """
    边缘图生成器，使用Canny边缘检测算法
    
    在训练和推理中保持一致的edge map生成逻辑
    """
    
    def __init__(
        self, 
        gaussian_kernel_size=(5, 5),
        gaussian_sigma=1.4,
        canny_threshold_lower_factor=0.7,
        canny_threshold_upper_factor=1.3,
        morph_kernel_size=(3, 3),
        morph_kernel_shape=cv2.MORPH_ELLIPSE,
        device='cuda'
    ):
        """
        初始化边缘图生成器
        
        Args:
            gaussian_kernel_size (tuple): 高斯模糊核大小，默认(5, 5)
            gaussian_sigma (float): 高斯模糊标准差，默认1.4
            canny_threshold_lower_factor (float): Canny下阈值因子，默认0.7
            canny_threshold_upper_factor (float): Canny上阈值因子，默认1.3
            morph_kernel_size (tuple): 形态学操作核大小，默认(3, 3)
            morph_kernel_shape: 形态学核形状，默认cv2.MORPH_ELLIPSE
            device (str): 设备类型，默认'cuda'
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.canny_threshold_lower_factor = canny_threshold_lower_factor
        self.canny_threshold_upper_factor = canny_threshold_upper_factor
        self.morph_kernel_size = morph_kernel_size
        self.morph_kernel_shape = morph_kernel_shape
        self.device = device
        
        # 预先创建形态学核
        self.morph_kernel = cv2.getStructuringElement(
            self.morph_kernel_shape, 
            self.morph_kernel_size
        )
    
    def generate_from_numpy(self, img_np, input_format='BGR', normalize_input=True):
        """
        从numpy数组生成edge map
        
        Args:
            img_np (np.ndarray): 输入图像，形状为(H, W, C)
            input_format (str): 输入图像格式，'BGR'或'RGB'，默认'BGR'
            normalize_input (bool): 输入是否为[0,1]范围，默认True
                                   如果False，则认为是[0,255]范围
        
        Returns:
            np.ndarray: 边缘图，形状为(H, W, C)，值域[0, 1]，float32
        """
        # 确保输入是float32
        img = img_np.astype(np.float32)
        
        # 如果输入是[0,1]范围，转换为[0,255]
        if normalize_input:
            img = img * 255.0
        
        # 转换为灰度图
        if input_format == 'BGR':
            img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:  # RGB
            img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur (reduces noise, improves edge detection)
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
        
        # Apply Canny edge detector with fixed thresholds
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        
        # 转换为3通道
        if input_format == 'BGR':
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:  # RGB
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 转换回float32 [0, 1]
        edges_color = edges_color.astype(np.float32) / 255.0
        
        return edges_color
    
    def generate_from_tensor(self, image_tensor, input_format='RGB', normalize_range='[-1,1]'):
        """
        从PyTorch tensor生成edge map（支持batch处理）
        
        Args:
            image_tensor (torch.Tensor): 输入图像张量，形状为(B, C, H, W)或(C, H, W)
            input_format (str): 输入图像格式，'BGR'或'RGB'，默认'RGB'
            normalize_range (str): 输入归一化范围，'[-1,1]'或'[0,1]'，默认'[-1,1]'
        
        Returns:
            torch.Tensor: 边缘图张量，形状与输入相同，值域[-1, 1]
        """
        # 处理单张图像的情况
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            single_image = True
        else:
            single_image = False
        
        batch_size = image_tensor.size(0)
        edge_maps = []
        
        for i in range(batch_size):
            # 转换为numpy并归一化到[0, 1]
            img = image_tensor[i].cpu().numpy()
            
            if normalize_range == '[-1,1]':
                img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            # 如果是[0,1]则不需要转换
            
            # 转换为(H, W, C)格式
            img = np.transpose(img, (1, 2, 0))
            
            # 生成edge map
            edge_map_np = self.generate_from_numpy(
                img, 
                input_format=input_format, 
                normalize_input=True
            )
            
            # 转换回tensor格式 (C, H, W)
            edge_map_tensor = torch.from_numpy(edge_map_np).permute(2, 0, 1).float()
            
            # 归一化到[-1, 1]
            edge_map_tensor = edge_map_tensor * 2.0 - 1.0
            
            edge_maps.append(edge_map_tensor)
        
        # 堆叠为batch
        result = torch.stack(edge_maps).to(image_tensor.device)
        
        # 如果输入是单张图像，返回单张
        if single_image:
            result = result.squeeze(0)
        
        return result
    
    def __call__(self, image, **kwargs):
        """
        便捷调用方法，自动检测输入类型
        
        Args:
            image: numpy数组或PyTorch tensor
            **kwargs: 传递给相应生成函数的参数
        
        Returns:
            与输入相同类型的edge map
        """
        if isinstance(image, np.ndarray):
            return self.generate_from_numpy(image, **kwargs)
        elif isinstance(image, torch.Tensor):
            return self.generate_from_tensor(image, **kwargs)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


# 创建默认的全局实例
default_edge_generator = EdgeMapGenerator()


def generate_edge_map(image, **kwargs):
    """
    便捷函数：使用默认参数生成edge map
    
    Args:
        image: numpy数组或PyTorch tensor
        **kwargs: 传递给EdgeMapGenerator的参数
    
    Returns:
        edge map（与输入相同类型）
    """
    return default_edge_generator(image, **kwargs)

