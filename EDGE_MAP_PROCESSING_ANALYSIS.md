# StableSR Edge Map 处理逻辑分析

## 概述

StableSR Edge版本通过集成edge map处理来提升超分辨率质量。本文档详细分析了edge map的生成、处理和融合机制。

## 1. Edge Map 生成逻辑

### 1.1 Canny边缘检测实现

Edge map生成使用Canny边缘检测算法，具体实现如下：

```python
def _generate_edge_map(self, image: torch.Tensor) -> torch.Tensor:
    """
    从输入图像生成边缘图
    
    Args:
        image: 输入图像张量 [B, C, H, W]，值范围 [-1, 1]
        
    Returns:
        edge_map: 边缘图张量 [B, 3, H, W]，值范围 [-1, 1]
    """
    # 1. 转换为numpy数组进行处理
    if image.dim() == 4:
        img_np = image[0].cpu().numpy()
    else:
        img_np = image.cpu().numpy()
        
    # 2. 从 [-1, 1] 转换到 [0, 1]
    img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    
    # 3. 转换维度从 [C, H, W] 到 [H, W, C]
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # 4. 转换为uint8格式 [0, 255]
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # 5. 转换为BGR格式（OpenCV使用BGR）
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    # 6. 转换为灰度图进行边缘检测
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 7. 应用高斯模糊减少噪声
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    
    # 8. 应用Canny边缘检测
    edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
    
    # 9. 转换为3通道BGR格式
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 10. 转换回RGB格式
    edges_rgb = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB)
    
    # 11. 转换为float32并归一化到 [0, 1]
    edges_float = edges_rgb.astype(np.float32) / 255.0
    
    # 12. 转换维度从 [H, W, C] 到 [C, H, W]
    edges_tensor = np.transpose(edges_float, (2, 0, 1))
    
    # 13. 转换到 [-1, 1] 范围
    edges_tensor = 2.0 * edges_tensor - 1.0
    
    # 14. 添加batch维度
    if image.dim() == 4:
        edges_tensor = np.expand_dims(edges_tensor, axis=0)
    
    # 15. 转换为torch张量并移动到相同设备
    edge_map = torch.from_numpy(edges_tensor).to(image.device)
    
    return edge_map
```

### 1.2 关键参数

- **高斯模糊**: 5×5 kernel，σ=1.4，用于噪声减少
- **Canny阈值**: 低阈值=100，高阈值=200
- **输出格式**: 3通道RGB，值范围[-1, 1]

## 2. Edge Map 处理器架构

### 2.1 EdgeMapProcessor 结构

EdgeMapProcessor将edge map转换为64×64×4的潜在特征：

```python
class EdgeMapProcessor(nn.Module):
    """
    内存优化的Edge Map处理器，将edge map转换为64x64x4潜在特征
    
    架构:
    1. 初始下采样以减少内存使用
    2. 3x3卷积层，减少通道数：3层，每层256通道
    3. 4x4卷积层（步长=2）：4层，通道数[128, 64, 16, 4]
    """
    
    def __init__(self, input_channels=3, output_channels=4, target_size=64, use_checkpoint=False):
        # 初始下采样以减少内存使用（步长=2）
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 阶段1：3x3卷积层，减少通道数以提高内存效率
        self.stage1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if i == 0 else 256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for i in range(3)  # 从5层减少到3层
        ])
        
        # 阶段2：4x4卷积层，步长=2，减少通道数
        stage2_channels = [128, 64, 16, 4]  # 从[512, 256, 64, 16, 4]减少
        self.stage2_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 if i == 0 else stage2_channels[i-1], stage2_channels[i], 4, stride=2, padding=1),
                nn.BatchNorm2d(stage2_channels[i]),
                nn.ReLU(inplace=True) if i < len(stage2_channels) - 1 else nn.Identity()
            ) for i in range(4)  # 从5层减少到4层
        ])
        
        # 自适应池化以确保输出尺寸为目标尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_size, target_size))
```

### 2.2 前向传播流程

```python
def forward(self, edge_map):
    """
    前向传播，支持梯度检查点
    
    Args:
        edge_map: 输入edge map张量 [B, C, H, W]
        
    Returns:
        latent_features: 处理后的特征 [B, 4, 64, 64]
    """
    x = edge_map
    
    # 初始下采样
    x = self.initial_conv(x)
    
    # 阶段1：3x3卷积层
    for i, layer in enumerate(self.stage1_layers):
        x = layer(x)
    
    # 阶段2：4x4卷积层
    for i, layer in enumerate(self.stage2_layers):
        x = layer(x)
    
    # 自适应池化以确保输出尺寸
    x = self.adaptive_pool(x)
    
    return x
```

## 3. Edge Feature 融合机制

### 3.1 EdgeFusionModule 结构

```python
class EdgeFusionModule(nn.Module):
    """
    融合edge特征与U-Net输入特征的模块
    结合 64x64x4 (U-Net输入) + 64x64x4 (edge特征) = 64x64x8
    """
    
    def __init__(self, unet_channels=4, edge_channels=4, output_channels=8):
        # 融合层结合特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(unet_channels + edge_channels, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, unet_input, edge_features):
        """
        融合U-Net输入与edge特征
        
        Args:
            unet_input: U-Net输入特征 [B, 4, 64, 64]
            edge_features: Edge处理后的特征 [B, 4, 64, 64]
            
        Returns:
            fused_features: 融合后的特征 [B, 8, 64, 64]
        """
        # 沿通道维度连接
        combined = torch.cat([unet_input, edge_features], dim=1)
        
        # 应用融合卷积
        fused = self.fusion_conv(combined)
        
        return fused
```

### 3.2 融合流程

1. **输入**: U-Net输入特征 [B, 4, 64, 64] + Edge特征 [B, 4, 64, 64]
2. **连接**: 沿通道维度连接 → [B, 8, 64, 64]
3. **卷积**: 3×3卷积 + BatchNorm + ReLU → [B, 8, 64, 64]
4. **输出**: 融合后的特征 [B, 8, 64, 64]

## 4. U-Net 集成

### 4.1 UNetModelDualcondV2WithEdge

扩展的U-Net模型支持edge map处理：

```python
class UNetModelDualcondV2WithEdge(UNetModelDualcondV2):
    """
    支持edge map处理的扩展UNetModelDualcondV2
    
    关键特性:
    - 处理edge map生成64x64x4潜在特征
    - 融合edge特征与U-Net输入（64x64x4 + 64x64x4 = 64x64x8）
    - 保持与原UNetModelDualcondV2的兼容性
    """
    
    def forward(self, x, timesteps=None, context=None, struct_cond=None, edge_map=None, y=None, **kwargs):
        """
        支持edge map处理的前向传播
        
        Args:
            x: 输入张量 [B, C, H, W] - U-Net输入（通常是64x64x4）
            timesteps: 扩散时间步
            context: 文本条件
            struct_cond: 结构条件
            edge_map: Edge map张量 [B, 3, H, W] - 可选的edge map输入
            y: 类别条件
            **kwargs: 额外参数
            
        Returns:
            输出张量 [B, C, H, W]
        """
        # 如果提供了edge map且启用了edge处理，则处理edge map
        if self.use_edge_processing and edge_map is not None:
            # 处理edge map获得64x64x4特征
            edge_features = self.edge_processor(edge_map)
            
            # 融合U-Net输入与edge特征
            x = self.edge_fusion(x, edge_features)
        
        # 调用父类前向方法
        return super().forward(
            x=x,
            timesteps=timesteps,
            context=context,
            struct_cond=struct_cond,
            y=y,
            **kwargs
        )
```

### 4.2 架构修改

1. **输入层修改**: 将第一个输入块从4通道扩展到8通道
2. **权重初始化**: 为新增的4个通道初始化权重
3. **前向传播**: 在调用父类forward前进行edge处理

## 5. 扩散模型集成

### 5.1 LatentDiffusionSRTextWTWithEdge

扩展的扩散模型支持edge map：

```python
class LatentDiffusionSRTextWTWithEdge(LatentDiffusionSRTextWT):
    """
    支持edge map处理的扩展LatentDiffusionSRTextWT
    
    关键特性:
    - 处理数据集中的edge map
    - 融合edge特征与U-Net输入
    - 保持与原训练管道的兼容性
    """
    
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, val=False, text_cond=[''], return_gt=False, resize_lq=True):
        """
        重写get_input以处理edge map处理
        """
        # 调用父类get_input方法
        result = super().get_input(
            batch=batch,
            k=k,
            return_first_stage_outputs=return_first_stage_outputs,
            force_c_encode=force_c_encode,
            cond_key=cond_key,
            return_original_cond=return_original_cond,
            bs=bs,
            val=val,
            text_cond=text_cond,
            return_gt=return_gt,
            resize_lq=resize_lq
        )
        
        # 如果可用且启用了edge处理，则添加edge map到结果中
        if self.use_edge_processing and 'img_edge' in batch:
            edge_map = batch['img_edge'].cuda()
            edge_map = edge_map.to(memory_format=torch.contiguous_format).float()
            
            # 将edge map归一化到[-1, 1]范围
            edge_map = edge_map * 2.0 - 1.0
            
            # 确保edge_map需要梯度用于训练
            edge_map = edge_map.requires_grad_(True)
            
            if bs is not None:
                edge_map = edge_map[:bs]
            
            # 添加edge map到结果中
            if isinstance(result, list):
                result.append(edge_map)
            else:
                result = [result, edge_map]
        
        return result
```

### 5.2 训练集成

```python
def training_step(self, batch, batch_idx, optimizer_idx=0):
    """
    重写training_step以在训练中处理edge map
    """
    if optimizer_idx == 0:
        # 获取带edge map的输入
        if self.use_edge_processing and 'img_edge' in batch:
            result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
            z, c, z_gt, x, gt, xrec, edge_map = result
        else:
            result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
            z, c, z_gt, x, gt, xrec = result
            edge_map = None
        
        # 获取时间步
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        t_ori = t
        
        # 从潜在表示创建结构条件
        if self.test_gt:
            # 重新编码ground truth以进行训练
            encoder_posterior_gt = self.encode_first_stage(gt)
            z_gt_with_grad = self.get_first_stage_encoding(encoder_posterior_gt)
            struct_cond = self.structcond_stage_model(z_gt_with_grad, t_ori)
        else:
            # 重新编码低质量输入以进行训练
            encoder_posterior_x = self.encode_first_stage(x)
            z_with_grad = self.get_first_stage_encoding(encoder_posterior_x)
            struct_cond = self.structcond_stage_model(z_with_grad, t_ori)
        
        # 处理文本条件以从字符串转换为张量
        if isinstance(c, list) and len(c) > 0 and isinstance(c[0], str):
            c = self.cond_stage_model(c)
        
        # 使用edge map计算损失
        if self.use_edge_processing and edge_map is not None:
            loss, loss_dict = self.p_losses(z, c, struct_cond, t, t_ori, z_gt, edge_map=edge_map)
        else:
            loss, loss_dict = self.p_losses(z, c, struct_cond, t, t_ori, z_gt)
        
        return loss
```

## 6. 配置和启用

### 6.1 配置文件

```yaml
model:
  target: ldm.models.diffusion.ddpm_with_edge.LatentDiffusionSRTextWTWithEdge
  params:
    # Edge处理参数
    use_edge_processing: True
    edge_input_channels: 3
    
    unet_config:
      target: ldm.modules.diffusionmodules.unet_with_edge.UNetModelDualcondV2WithEdge
      params:
        use_edge_processing: True
        edge_input_channels: 3
```

### 6.2 数据集要求

数据集需要包含edge map，`RealESRGANDataset`自动生成：

- **自动生成**: 使用Canny边缘检测从ground truth图像生成
- **格式**: 3通道BGR edge map
- **键名**: `img_edge`

## 7. 内存优化

### 7.1 优化策略

1. **减少层数**: 从5层减少到3层（Stage 1）和4层（Stage 2）
2. **减少通道数**: 优化通道配置以减少内存使用
3. **初始下采样**: 使用stride=2减少初始分辨率
4. **梯度检查点**: 支持梯度检查点以减少内存使用

### 7.2 性能影响

- **内存使用**: 显著减少内存占用
- **计算效率**: 保持合理的计算复杂度
- **精度**: 保持edge处理的有效性

## 8. 测试和验证

### 8.1 测试脚本

项目包含多个测试脚本：

- `test_edge_processing.py`: 基础功能测试
- `test_edge_map_quick.py`: 快速测试
- `test_edge_map_comprehensive.py`: 综合测试
- `test_edge_map_performance.py`: 性能测试

### 8.2 验证方法

1. **形状验证**: 确保所有张量形状正确
2. **梯度验证**: 确保梯度正确传播
3. **内存验证**: 检查内存使用情况
4. **性能验证**: 测试处理速度

## 9. 总结

StableSR Edge版本的edge map处理逻辑包括：

1. **生成**: 使用Canny边缘检测从ground truth生成edge map
2. **处理**: 通过EdgeMapProcessor转换为64×64×4潜在特征
3. **融合**: 使用EdgeFusionModule融合edge特征与U-Net输入
4. **集成**: 在U-Net和扩散模型中集成edge处理
5. **训练**: 在训练过程中使用edge map提升超分辨率质量

这种设计既保持了与原始StableSR的兼容性，又通过edge信息显著提升了超分辨率质量。
