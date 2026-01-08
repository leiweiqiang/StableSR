# 并行推理脚本使用说明

## 概述

针对 0019 图像的 6 个推理任务，提供了两种运行方式：

1. **顺序运行** (`run_0019_inference.sh`) - 逐一执行任务
2. **并行运行** (`run_0019_inference_parallel.sh`) - 使用多GPU同时运行

## 并行推理的优势

### 时间对比

| 方式 | 预计总时间 | GPU利用率 |
|------|------------|-----------|
| 顺序运行 | 40-60 分钟 | 1块GPU |
| 并行运行 | **10-15 分钟** | **6块GPU** |

**时间节省：75-80%**

### GPU 分配策略

```
8块 NVIDIA RTX A6000 (每块 49GB)

GPU 0: Task 1 - Edge 4x
GPU 1: Task 2 - Noedge 4x  
GPU 2: Task 3 - Edge 8x
GPU 3: Task 4 - Noedge 8x
GPU 4: Task 5 - StableSR 4x
GPU 5: Task 6 - StableSR 8x
GPU 6-7: 空闲 (备用)
```

## 使用方法

### 方法 1: 并行运行 (推荐)

```bash
# 停止当前顺序运行的脚本
# 按 Ctrl+C 或查找并杀死进程

# 运行并行版本
./run_0019_inference_parallel.sh
```

### 方法 2: 继续顺序运行

```bash
# 让当前脚本继续运行
# 大约还需 30-40 分钟完成
```

## 并行脚本特性

### 1. 自动跳过已完成任务
```bash
# 脚本会自动检查输出目录
# 如果已有结果，自动跳过
```

### 2. GPU 隔离
```bash
# 每个任务独立运行在指定GPU
CUDA_VISIBLE_DEVICES=0 python ...  # GPU 0
CUDA_VISIBLE_DEVICES=1 python ...  # GPU 1
# 等等
```

### 3. 实时监控
```bash
# 监控所有GPU使用情况
watch -n 1 nvidia-smi
```

### 4. 后台运行记录
```bash
# 每个任务有独立日志
outputs/0019_inference/4x/0019_4x_w0.0_e111/inference_command.txt
```

## 当前状态

### 已完成的输出

- ✅ Task 1: Edge 4x - **已完成**
- ⏳ Task 2: Noedge 4x - **运行中** (约11分钟)

### 建议操作

由于当前 Task 2 已运行 11 分钟（接近完成），建议：

**选项 A**: 等待当前任务完成（约 5-10 分钟），然后运行并行脚本处理剩余任务

**选项 B**: 立即切换到并行模式
```bash
# 1. 停止当前进程
pkill -f run_0019_inference

# 2. 运行并行版本
./run_0019_inference_parallel.sh
# 已完成的 Task 1 会被跳过
# Task 2 将重新开始（但已在备份）
```

## 任务列表

| 任务 | 输出目录 | 跳过检查 |
|------|----------|----------|
| Task 1: Edge 4x | `outputs/0019_inference/4x/0019_4x_w0.0_e111/` | ✅ 已完成 |
| Task 2: Noedge 4x | `outputs/0019_inference/4x/0019_4x_noedge_w0.0_e111/` | ⏳ 运行中 |
| Task 3: Edge 8x | `outputs/0019_inference/8x/0019_8x_w0.0_e111/` | ❌ 未开始 |
| Task 4: Noedge 8x | `outputs/0019_inference/8x/0019_8x_noedge_w0.0_e111/` | ❌ 未开始 |
| Task 5: StableSR 4x | `outputs/0019_inference/4x/0019_4x_stablesr_w0.5/` | ❌ 未开始 |
| Task 6: StableSR 8x | `outputs/0019_inference/8x/0019_8x_stablesr_w0.5/` | ❌ 未开始 |

## 监控命令

### 查看 GPU 使用情况
```bash
nvidia-smi
```

### 查看进程状态
```bash
ps aux | grep run_0019
ps aux | grep python | grep inference
```

### 查看实时输出
```bash
tail -f outputs/0019_inference/*/inference_command.txt
```

## 注意事项

1. **内存使用**: 每块 GPU 需要约 18-20GB 显存
2. **CPU**: 需要足够的 CPU 核心支持并行任务
3. **I/O**: 确保磁盘 I/O 不会成为瓶颈
4. **网络**: 如果有网络访问checkpoint，确保带宽充足

## 故障恢复

如果某个任务失败：

```bash
# 查看失败的任务
ls -la outputs/0019_inference/*/ | grep -E "(png|error)"

# 重新运行失败的任务
# 只需要单独运行那个任务所在的GPU
CUDA_VISIBLE_DEVICES=X python scripts/...
```
