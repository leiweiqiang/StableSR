# StableSR Edge Map 推理使用指南

## 概述

本指南详细说明如何在推理时使用edge map来提升StableSR的超分辨率质量。

## 1. 推理时Edge Map的使用流程

### 1.1 基本流程

推理时使用edge map的基本流程如下：

```
输入图像 → 生成Edge Map → 模型推理 → 输出超分辨率图像
```

### 1.2 详细步骤

1. **加载Edge模型**: 使用配置了`use_edge_processing: True`的模型
2. **生成Edge Map**: 从输入图像生成Canny边缘图
3. **准备条件**: 构建包含edge_map的条件字典
4. **执行推理**: 调用模型的sample方法进行推理

## 2. Edge Map生成

### 2.1 推理时Edge Map生成函数

```python
import torch
import cv2
import numpy as np

def generate_edge_map_for_inference(image: torch.Tensor) -> torch.Tensor:
    """
    推理时从输入图像生成edge map
    
    Args:
        image: 输入图像张量 [B, C, H, W]，值范围 [-1, 1]
        
    Returns:
        edge_map: 边缘图张量 [B, 3, H, W]，值范围 [-1, 1]
    """
    # 转换为numpy数组进行处理
    if image.dim() == 4:
        img_np = image[0].cpu().numpy()
    else:
        img_np = image.cpu().numpy()
        
    # 从 [-1, 1] 转换到 [0, 1]
    img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    
    # 转换维度从 [C, H, W] 到 [H, W, C]
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # 转换为uint8格式 [0, 255]
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # 转换为BGR格式（OpenCV使用BGR）
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    # 转换为灰度图进行边缘检测
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    
    # 应用Canny边缘检测
    edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
    
    # 转换为3通道BGR格式
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 转换回RGB格式
    edges_rgb = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB)
    
    # 转换为float32并归一化到 [0, 1]
    edges_float = edges_rgb.astype(np.float32) / 255.0
    
    # 转换维度从 [H, W, C] 到 [C, H, W]
    edges_tensor = np.transpose(edges_float, (2, 0, 1))
    
    # 转换到 [-1, 1] 范围
    edges_tensor = 2.0 * edges_tensor - 1.0
    
    # 添加batch维度
    if image.dim() == 4:
        edges_tensor = np.expand_dims(edges_tensor, axis=0)
    
    # 转换为torch张量并移动到相同设备
    edge_map = torch.from_numpy(edges_tensor).to(image.device)
    
    return edge_map
```

### 2.2 Edge Map生成参数

- **高斯模糊**: 5×5 kernel，σ=1.4
- **Canny阈值**: 低阈值=100，高阈值=200
- **输出格式**: 3通道RGB，值范围[-1, 1]

## 3. 推理代码实现

### 3.1 使用EdgeDDIMSampler的推理

```python
import torch
from ldm.models.diffusion.ddpm_with_edge import LatentDiffusionSRTextWTWithEdge
from tra_report import EdgeDDIMSampler

class StableSRInferenceWithEdge:
    def __init__(self, config_path, ckpt_path, device="cuda"):
        """
        初始化StableSR Edge推理器
        
        Args:
            config_path: 配置文件路径
            ckpt_path: 模型检查点路径
            device: 设备类型
        """
        self.device = device
        
        # 加载配置
        from omegaconf import OmegaConf
        self.config = OmegaConf.load(config_path)
        
        # 加载模型
        self.model = LatentDiffusionSRTextWTWithEdge.from_pretrained(
            ckpt_path, 
            config=self.config
        ).to(self.device)
        
        # 创建DDIM采样器
        self.sampler = EdgeDDIMSampler(self.model)
        
    def inference_with_edge(self, input_image, caption="", ddpm_steps=20, 
                          use_edge_detection=True, seed=42):
        """
        使用edge map进行推理
        
        Args:
            input_image: 输入图像张量 [1, 3, H, W]，值范围 [-1, 1]
            caption: 文本描述（可选）
            ddpm_steps: DDPM采样步数
            use_edge_detection: 是否使用edge检测
            seed: 随机种子
            
        Returns:
            output_image: 超分辨率输出图像
        """
        torch.manual_seed(seed)
        
        # 确保输入图像在正确设备上
        input_image = input_image.to(self.device)
        
        # 上采样输入图像到目标尺寸
        upscaled_image = self._upscale_image(input_image)
        
        # 获取文本条件
        cross_attn = self.model.get_learned_conditioning([caption])
        
        # 为Edge模型准备struct_cond和edge_map
        if hasattr(self.model, 'use_edge_processing') and self.model.use_edge_processing:
            # 将upscaled_image编码到潜在空间
            encoder_posterior = self.model.encode_first_stage(upscaled_image)
            z_upscaled = self.model.get_first_stage_encoding(encoder_posterior).detach()
            
            # 生成struct_cond
            struct_cond = self.model.structcond_stage_model(
                z_upscaled, 
                torch.zeros(1, device=self.device)
            )
            
            # 生成edge_map（如果启用边缘检测）
            if use_edge_detection:
                edge_map = self._generate_edge_map(upscaled_image)
                print(f"Generated edge_map with shape: {edge_map.shape}")
                
                conditioning = {
                    "c_concat": upscaled_image, 
                    "c_crossattn": cross_attn,
                    "struct_cond": struct_cond,
                    "edge_map": edge_map
                }
            else:
                print("Edge detection disabled, using struct_cond only")
                conditioning = {
                    "c_concat": upscaled_image, 
                    "c_crossattn": cross_attn,
                    "struct_cond": struct_cond
                }
        else:
            conditioning = {
                "c_concat": upscaled_image, 
                "c_crossattn": cross_attn
            }
        
        # 执行DDIM采样
        with torch.no_grad():
            samples, _ = self.sampler.sample(
                S=ddpm_steps,
                conditioning=conditioning,
                batch_size=1,
                shape=(4, 64, 64),  # latent shape
                verbose=False
            )
        
        # 解码到图像空间
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        return x_samples
    
    def _generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """生成edge map"""
        return generate_edge_map_for_inference(image)
    
    def _upscale_image(self, image: torch.Tensor) -> torch.Tensor:
        """上采样图像到目标尺寸"""
        # 这里可以使用双三次插值或其他上采样方法
        import torch.nn.functional as F
        
        # 假设目标尺寸为512x512
        target_size = 512
        current_size = min(image.shape[-2:])
        
        if current_size < target_size:
            scale_factor = target_size / current_size
            new_h = int(image.shape[-2] * scale_factor)
            new_w = int(image.shape[-1] * scale_factor)
            
            upscaled = F.interpolate(
                image, 
                size=(new_h, new_w), 
                mode='bicubic', 
                align_corners=False
            )
        else:
            upscaled = image
            
        return upscaled
```

### 3.2 简化的推理示例

```python
def simple_inference_with_edge():
    """
    简化的edge map推理示例
    """
    import torch
    from PIL import Image
    import numpy as np
    
    # 1. 加载模型
    config_path = "configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    ckpt_path = "/path/to/edge_model.ckpt"
    
    model = LatentDiffusionSRTextWTWithEdge.from_pretrained(
        ckpt_path, 
        config=config_path
    ).cuda()
    
    # 2. 加载输入图像
    image_path = "input.jpg"
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    image = (image - 0.5) / 0.5  # 归一化到[-1, 1]
    image = image.cuda()
    
    # 3. 生成edge map
    edge_map = generate_edge_map_for_inference(image)
    
    # 4. 准备条件
    caption = ""  # 空字符串表示无文本条件
    cross_attn = model.get_learned_conditioning([caption])
    
    # 上采样图像
    upscaled_image = F.interpolate(image, size=(512, 512), mode='bicubic')
    
    # 编码到潜在空间
    encoder_posterior = model.encode_first_stage(upscaled_image)
    z_upscaled = model.get_first_stage_encoding(encoder_posterior).detach()
    
    # 生成struct_cond
    struct_cond = model.structcond_stage_model(z_upscaled, torch.zeros(1, device='cuda'))
    
    # 构建条件字典
    conditioning = {
        "c_concat": upscaled_image,
        "c_crossattn": cross_attn,
        "struct_cond": struct_cond,
        "edge_map": edge_map
    }
    
    # 5. 执行推理
    with torch.no_grad():
        samples, _ = model.sample(
            cond=conditioning,
            struct_cond=struct_cond,
            edge_map=edge_map,
            batch_size=1,
            timesteps=20,  # DDPM步数
            time_replace=20,
            x_T=None,  # 随机噪声
            return_intermediates=True
        )
    
    # 6. 解码结果
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    # 7. 保存结果
    result_image = x_samples[0].cpu().permute(1, 2, 0).numpy()
    result_image = (result_image * 255).astype(np.uint8)
    Image.fromarray(result_image).save("output_with_edge.jpg")
    
    print("推理完成，结果保存为 output_with_edge.jpg")
```

## 4. 配置要求

### 4.1 模型配置

确保使用正确的配置文件：

```yaml
# configs/stableSRNew/v2-finetune_text_T_512_edge.yaml
model:
  target: ldm.models.diffusion.ddpm_with_edge.LatentDiffusionSRTextWTWithEdge
  params:
    use_edge_processing: True  # 启用edge处理
    edge_input_channels: 3
    
    unet_config:
      target: ldm.modules.diffusionmodules.unet_with_edge.UNetModelDualcondV2WithEdge
      params:
        use_edge_processing: True
        edge_input_channels: 3
```

### 4.2 检查点要求

- 使用经过edge map训练的模型检查点
- 确保模型包含edge处理模块的权重

## 5. 推理参数优化

### 5.1 关键参数

- **DDPM步数**: 推荐20-50步，步数越多质量越好但速度越慢
- **Edge检测**: 可以启用/禁用来对比效果
- **输入尺寸**: 建议512×512或更大
- **随机种子**: 固定种子确保结果可复现

### 5.2 性能优化

```python
# 优化推理性能的配置
inference_config = {
    "ddpm_steps": 20,           # 减少步数以提升速度
    "use_edge_detection": True, # 启用edge检测
    "precision": "autocast",    # 使用自动精度
    "tile_size": 512,          # 瓦片大小
    "tile_overlap": 32,        # 瓦片重叠
}
```

## 6. 常见问题和解决方案

### 6.1 模型兼容性

```python
# 检查模型是否支持edge处理
if hasattr(model, 'use_edge_processing') and model.use_edge_processing:
    print("模型支持edge处理")
else:
    print("模型不支持edge处理，将使用标准推理")
```

### 6.2 内存优化

```python
# 对于大图像，使用瓦片处理
if image.shape[-1] > 1024 or image.shape[-2] > 1024:
    # 使用sample_canvas方法进行瓦片处理
    samples, _ = model.sample_canvas(
        cond=conditioning,
        struct_cond=struct_cond,
        edge_map=edge_map,
        batch_size=1,
        timesteps=ddpm_steps,
        tile_size=512,
        tile_overlap=32
    )
```

### 6.3 Edge Map质量问题

```python
# 调整Canny参数以获得更好的edge map
def generate_edge_map_custom(image, low_threshold=50, high_threshold=150):
    """自定义Canny参数的edge map生成"""
    # ... 实现细节 ...
    edges = cv2.Canny(img_blurred, threshold1=low_threshold, threshold2=high_threshold)
    # ... 其余处理 ...
```

## 7. 完整推理脚本示例

```python
#!/usr/bin/env python3
"""
StableSR Edge Map 推理脚本
"""

import torch
import argparse
from PIL import Image
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description="StableSR Edge Map 推理")
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出图像路径")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--ckpt", required=True, help="模型检查点路径")
    parser.add_argument("--steps", type=int, default=20, help="DDPM采样步数")
    parser.add_argument("--no-edge", action="store_true", help="禁用edge检测")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载模型
    print("加载模型...")
    model = LatentDiffusionSRTextWTWithEdge.from_pretrained(
        args.ckpt, 
        config=args.config
    ).cuda()
    
    # 加载输入图像
    print("加载输入图像...")
    image = Image.open(args.input).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5
    image = image.cuda()
    
    # 执行推理
    print("开始推理...")
    inference_with_edge_map(
        model=model,
        input_image=image,
        output_path=args.output,
        ddpm_steps=args.steps,
        use_edge_detection=not args.no_edge
    )
    
    print(f"推理完成，结果保存为 {args.output}")

if __name__ == "__main__":
    main()
```

## 8. 总结

推理时使用edge map的关键点：

1. **模型要求**: 必须使用支持edge处理的模型和配置
2. **Edge Map生成**: 使用Canny边缘检测从输入图像生成
3. **条件构建**: 将edge_map包含在conditioning字典中
4. **采样器**: 使用EdgeDDIMSampler或支持edge的采样方法
5. **参数优化**: 根据质量和速度需求调整DDPM步数

通过正确使用edge map，可以显著提升StableSR在边缘细节保持方面的表现。
