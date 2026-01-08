# 图像 0051 推理任务说明

## 需求分析

需要处理图像 0051，包含以下任务：

1. **Edge 4x**: 0051 带 edge map，4x 放大，w=0.0，epoch 111
2. **Noedge 4x**: 0051 不带 edge map，4x 放大，w=0.0，epoch 111 (已存在，将跳过)
3. **Edge 8x**: 0051 带 edge map，8x 放大，w=0.0，epoch 111
4. **Noedge 8x**: 0051 不带 edge map，8x 放大，w=0.0，epoch 111

## 已存在的输出

- `outputs/4x/0051_4x_noedge_w0.0_e111/` - 已存在，会自动跳过
- `outputs/4x/0051_4x_stablesr_w0.5/` - 不同的配置
- `outputs/8x/0051_8x_stablesr_w0.5/` - 不同的配置

## 需要的输出

1. `outputs/4x/0051_4x_w0.0_e111/` - 带 edge 的 4x 输出
2. `outputs/4x/0051_4x_noedge_w0.0_e111/` - 不带 edge 的 4x 输出 (已存在，跳过)
3. `outputs/8x/0051_8x_w0.0_e111/` - 带 edge 的 8x 输出
4. `outputs/8x/0051_8x_noedge_w0.0_e111/` - 不带 edge 的 8x 输出

## 使用方法

### 方法 1: 使用自动化脚本（推荐）

```bash
./run_0051_inference.sh
```

这个脚本会：
1. 运行 4x with edge
2. 跳过已存在的 4x noedge
3. 运行 8x with edge
4. 运行 8x noedge

### 方法 2: 手动运行

#### 步骤 1: 运行 4x with edge

```bash
# 备份脚本
cp batch_inference_blended_tile.sh batch_inference_blended_tile.sh.bak

# 修改 NOEDGE 为 false
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

# 临时隐藏 8x 文件
mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak

# 运行
./batch_inference_blended_tile.sh --only-image 0051

# 恢复文件
mv /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png.bak /stablesr_dataset/weiqiang/lr_x8_480x270/0051.png
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh
```

#### 步骤 2: 运行 8x with edge

```bash
# 备份脚本
cp batch_inference_blended_tile.sh batch_inference_blended_tile.sh.bak

# 修改 NOEDGE 为 false
sed -i 's/^NOEDGE=true$/NOEDGE=false/' batch_inference_blended_tile.sh

# 临时隐藏 4x 文件
mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak

# 运行
./batch_inference_blended_tile.sh --only-image 0051

# 恢复文件
mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png
sed -i 's/^NOEDGE=false$/NOEDGE=true/' batch_inference_blended_tile.sh
```

#### 步骤 3: 运行 8x noedge

```bash
# 临时隐藏 4x 文件
mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak

# 运行
./batch_inference_blended_tile.sh --only-image 0051 --noedge

# 恢复文件
mv /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png.bak /stablesr_dataset/weiqiang/lr_x4_960x540/0051.png
```

## 脚本修改说明

### 对 batch_inference_blended_tile.sh 的修改

1. 添加了 `--only-image` 参数，用于处理指定图像
2. 添加了自动跳过已存在输出的功能
3. 完善了输出目录存在检查

### 新增的功能

- `--only-image ID`: 只处理指定的图像 ID
- 自动跳过已存在的输出目录

## 输出目录命名规则

- `{image_id}_{scale}_{edge_mode}_{w=value}_e{epoch_value}`
- 示例：
  - `0051_4x_w0.0_e111` - 4x with edge
  - `0051_4x_noedge_w0.0_e111` - 4x without edge
  - `0051_8x_w0.0_e111` - 8x with edge
  - `0051_8x_noedge_w0.0_e111` - 8x without edge

## 注意事项

1. 确保有足够的磁盘空间
2. 推理可能需要较长时间，每个图像可能需要几分钟
3. 如果中断，已完成的输出会被跳过
4. 建议先测试小批量图像

## 检查结果

运行完成后，检查输出：

```bash
# 查看 4x 输出
ls -lah outputs/4x/0051*

# 查看 8x 输出  
ls -lah outputs/8x/0051*

# 查看输出图像
ls -lah outputs/4x/0051_4x_w0.0_e111/*.png
ls -lah outputs/8x/0051_8x_noedge_w0.0_e111/*.png
```

## 故障排除

如果遇到问题：

1. 检查输入文件是否存在
2. 检查 checkpoint 文件是否存在
3. 检查 VQGAN checkpoint 是否存在
4. 查看输出日志
5. 使用 `--help` 查看帮助信息
