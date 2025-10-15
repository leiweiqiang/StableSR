# EdgeInference 环境配置说明

## ✅ 已解决的问题

### NumPy版本不兼容问题

**问题**: 
```
RuntimeError: Numpy is not available
```

**原因**: 
- PyTorch 1.12.1 是用 NumPy 1.x 编译的
- 环境中安装了 NumPy 2.0.2
- 导致 `torch.from_numpy()` 不兼容

**解决方案**: 
降级到兼容的版本组合

---

## 📦 正确的依赖版本

### 已验证的版本组合 ✅

```bash
NumPy: 1.23.5
OpenCV: 4.8.0
PyTorch: 1.12.1
```

这些版本已经过测试，完全兼容，可以正常工作。

---

## 🔧 环境配置步骤

### 方法1: 自动配置（推荐）

```bash
# 激活环境
conda activate sr_infer

# 安装兼容的版本
pip install "numpy==1.23.5" "opencv-python==4.8.0.74" --force-reinstall
```

### 方法2: 手动配置

```bash
# 1. 激活环境
conda activate sr_infer

# 2. 降级NumPy
pip install "numpy<2.0" --force-reinstall

# 3. 降级OpenCV  
pip install "opencv-python<4.9" --force-reinstall
```

---

## ✅ 验证环境

运行以下命令验证环境是否正确配置：

```bash
conda activate sr_infer

python << 'EOF'
import numpy as np
import cv2
import torch
from basicsr.utils.edge_utils import EdgeMapGenerator

print("✓ NumPy version:", np.__version__)
print("✓ OpenCV version:", cv2.__version__)
print("✓ PyTorch version:", torch.__version__)

# Test torch.from_numpy
x = np.array([1, 2, 3])
t = torch.from_numpy(x)
print("✓ torch.from_numpy works!")

# Test EdgeMapGenerator
edge_gen = EdgeMapGenerator()
print("✓ EdgeMapGenerator works!")

print("\n🎉 Environment is correctly configured!")
EOF
```

**预期输出**:
```
✓ NumPy version: 1.23.5
✓ OpenCV version: 4.8.0
✓ PyTorch version: 1.12.1
✓ torch.from_numpy works!
✓ EdgeMapGenerator works!

🎉 Environment is correctly configured!
```

---

## 📋 完整依赖列表

### 核心依赖

| 包 | 版本 | 说明 |
|---|------|------|
| Python | 3.9+ | 基础Python版本 |
| PyTorch | 1.12.1 | 深度学习框架 |
| NumPy | 1.23.5 | 数值计算（必须<2.0） |
| OpenCV | 4.8.0 | 图像处理 |

### 其他依赖

- pytorch-lightning
- omegaconf
- einops
- tqdm
- PIL/Pillow
- torchvision

---

## ⚠️ 常见问题

### Q1: 为什么不能使用NumPy 2.x？

**A**: PyTorch 1.12.1 是用 NumPy 1.x 的API编译的，与 NumPy 2.x 不兼容。升级到 NumPy 2.x 会导致：
```
RuntimeError: Numpy is not available
```

### Q2: 如果已经安装了NumPy 2.x怎么办？

**A**: 运行以下命令降级：
```bash
conda activate sr_infer
pip install "numpy==1.23.5" --force-reinstall
```

### Q3: OpenCV也需要降级吗？

**A**: 是的。新版OpenCV (4.10+) 要求 NumPy 2.x，会导致版本冲突。使用 OpenCV 4.8.x 可以兼容 NumPy 1.x。

### Q4: 能升级PyTorch吗？

**A**: 可以，但需要确保：
- 使用支持 NumPy 2.x 的 PyTorch 版本（>=2.0）
- 相应升级所有依赖
- 测试兼容性

---

## 🔍 故障排查

### 错误: "Numpy is not available"

**症状**:
```python
RuntimeError: Numpy is not available
```

**解决**:
```bash
pip install "numpy==1.23.5" --force-reinstall
```

### 错误: OpenCV导入失败

**症状**:
```python
ImportError: cannot import name 'cv2'
```

**解决**:
```bash
pip install "opencv-python==4.8.0.74" --force-reinstall
```

### 错误: torch.from_numpy不工作

**症状**:
```python
# UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

**解决**:
1. 检查NumPy版本：`python -c "import numpy; print(numpy.__version__)"`
2. 确保 < 2.0
3. 降级：`pip install "numpy<2.0" --force-reinstall`

---

## 📝 环境锁定

为了防止依赖自动升级，建议创建 `requirements.txt`:

```bash
# 导出当前环境
conda activate sr_infer
pip freeze | grep -E "(numpy|opencv|torch)" > edge_requirements.txt
```

**edge_requirements.txt 示例**:
```
numpy==1.23.5
opencv-python==4.8.0.74
torch==1.12.1
torchvision==0.13.1
```

恢复环境：
```bash
pip install -r edge_requirements.txt
```

---

## 🎯 快速修复命令

如果遇到NumPy问题，运行这个一键修复：

```bash
conda activate sr_infer && \
pip install "numpy==1.23.5" "opencv-python==4.8.0.74" --force-reinstall && \
python -c "from basicsr.utils.edge_utils import EdgeMapGenerator; print('✓ Fixed!')"
```

---

## 📚 相关文档

- [NumPy 1.x vs 2.x Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [OpenCV-Python Documentation](https://docs.opencv.org/4.8.0/)

---

**最后更新**: 2025-10-15  
**测试环境**: sr_infer (conda)  
**状态**: ✅ 已验证工作

