# Resume训练功能说明

本项目现在支持从checkpoint恢复训练，有两种方式可以使用：

## 方式1: 使用 `train_canny_in.sh` (修改变量)

在脚本中设置 `RESUME_PATH` 变量：

```bash
# 编辑 train_canny_in.sh，修改第22行
RESUME_PATH="logs/stablesr_canny_in_20231015_123456/checkpoints/epoch-000010.ckpt"

# 或者指定日志目录，将自动使用 last.ckpt
RESUME_PATH="logs/stablesr_canny_in_20231015_123456"

# 然后运行脚本
./train_canny_in.sh
```

## 方式2: 使用 `train_canny_in_resume.sh` (命令行参数)

直接通过命令行参数指定resume路径：

### 新训练
```bash
./train_canny_in_resume.sh
```

### 从特定checkpoint恢复
```bash
./train_canny_in_resume.sh --resume logs/stablesr_canny_in_20231015_123456/checkpoints/epoch-000010.ckpt
```

### 从日志目录恢复 (使用last.ckpt)
```bash
./train_canny_in_resume.sh --resume logs/stablesr_canny_in_20231015_123456
```

### 其他选项
```bash
# 自定义配置文件
./train_canny_in_resume.sh --config path/to/config.yaml

# 自定义GPU
./train_canny_in_resume.sh --gpus "0,1,2,3"

# 组合使用
./train_canny_in_resume.sh --resume logs/xxx --gpus "0,1"

# 查看帮助
./train_canny_in_resume.sh --help
```

## Resume功能说明

### 支持的路径格式

1. **Checkpoint文件路径**
   ```
   logs/stablesr_canny_in_20231015_123456/checkpoints/epoch-000010.ckpt
   logs/stablesr_canny_in_20231015_123456/checkpoints/last.ckpt
   ```

2. **日志目录路径** (自动使用 `checkpoints/last.ckpt`)
   ```
   logs/stablesr_canny_in_20231015_123456
   ```

### Resume时的行为

当从checkpoint恢复训练时：
- 会自动加载模型权重
- 会恢复优化器状态
- 会恢复学习率调度器状态
- 会继续使用原来的训练配置
- 会继续使用原来的日志目录
- 训练会从上次中断的epoch/step继续

### 查找可用的checkpoints

```bash
# 列出所有训练日志
ls -lh logs/

# 查看特定训练的checkpoints
ls -lh logs/stablesr_canny_in_20231015_123456/checkpoints/

# 查找最近的训练日志
ls -lt logs/ | head -5
```

### 常见的checkpoint文件

- `last.ckpt` - 最新的checkpoint (自动保存)
- `epoch-000001.ckpt` - 特定epoch的checkpoint
- `epoch-000010.ckpt` - 第10个epoch的checkpoint

## 注意事项

1. **Resume和新训练的区别**
   - Resume: 会继续使用原来的日志目录和配置
   - 新训练: 会创建新的日志目录和使用当前配置

2. **配置文件**
   - Resume时会使用保存的配置文件，命令行的 `--base` 配置会被忽略
   - 如果需要修改配置，建议使用新训练而不是resume

3. **GPU设置**
   - Resume时可以修改GPU数量和ID
   - 但需要确保batch size和梯度累积适配新的GPU配置

4. **文件检查**
   - 脚本会自动检查resume路径是否存在
   - 如果是目录，会检查 `checkpoints/last.ckpt` 是否存在
   - 检查失败会终止训练并给出错误信息

## 示例工作流

### 场景1: 训练中断后恢复

```bash
# 1. 训练中断了
# 2. 找到最近的日志目录
ls -lt logs/ | head -5

# 3. 恢复训练
./train_canny_in_resume.sh --resume logs/stablesr_canny_in_20241019_120000
```

### 场景2: 从特定epoch继续训练

```bash
# 1. 查看可用的checkpoints
ls -lh logs/stablesr_canny_in_20241019_120000/checkpoints/

# 2. 选择想要的checkpoint
./train_canny_in_resume.sh --resume logs/stablesr_canny_in_20241019_120000/checkpoints/epoch-000005.ckpt
```

### 场景3: 在不同GPU上恢复训练

```bash
# 原来在8个GPU上训练，现在只有4个GPU可用
./train_canny_in_resume.sh --resume logs/xxx --gpus "0,1,2,3"
```

## 直接使用main.py

如果想更灵活地控制，可以直接使用main.py:

```bash
# Resume from checkpoint file
python main.py --base configs/xxx.yaml --train --gpus 0,1,2,3 --resume path/to/checkpoint.ckpt

# Resume from log directory
python main.py --base configs/xxx.yaml --train --gpus 0,1,2,3 --resume logs/xxx

# 注意: 使用--resume时不能同时使用--name参数
```

## 获取帮助

```bash
# 查看脚本帮助
./train_canny_in_resume.sh --help

# 查看main.py帮助
python main.py --help
```

