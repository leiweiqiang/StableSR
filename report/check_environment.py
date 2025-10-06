#!/usr/bin/env python3
"""
环境检查脚本
检查运行edge map测试所需的环境和依赖

使用方法:
python check_environment.py
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("  ✓ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("  ❌ Python版本过低，需要 >= 3.7")
        return False


def check_conda():
    """检查conda是否可用"""
    print("\n检查conda...")
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ conda可用: {result.stdout.strip()}")
            return True
        else:
            print("  ❌ conda不可用")
            return False
    except FileNotFoundError:
        print("  ❌ conda未安装或不在PATH中")
        return False


def check_conda_env(env_name="sr_edge"):
    """检查指定的conda环境"""
    print(f"\n检查conda环境 '{env_name}'...")
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            envs = result.stdout
            if env_name in envs:
                print(f"  ✓ 找到conda环境: {env_name}")
                return True
            else:
                print(f"  ❌ 未找到conda环境: {env_name}")
                print("  可用的环境:")
                for line in envs.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        print(f"    {line}")
                return False
        else:
            print("  ❌ 无法获取conda环境列表")
            return False
    except Exception as e:
        print(f"  ❌ 检查conda环境时出错: {e}")
        return False


def check_package(package_name, import_name=None):
    """检查Python包是否可用"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name}: 未安装")
        return False


def check_required_packages():
    """检查必需的Python包"""
    print("\n检查必需的Python包...")
    
    packages = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("matplotlib", "matplotlib"),
        ("pillow", "PIL"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]
    
    all_available = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_available = False
    
    return all_available


def check_optional_packages():
    """检查可选的Python包"""
    print("\n检查可选的Python包...")
    
    packages = [
        ("omegaconf", "omegaconf"),
        ("basicsr", "basicsr"),
    ]
    
    for package_name, import_name in packages:
        check_package(package_name, import_name)


def check_cuda():
    """检查CUDA是否可用"""
    print("\n检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ⚠️  CUDA不可用，将使用CPU")
            return False
    except ImportError:
        print("  ❌ PyTorch未安装，无法检查CUDA")
        return False


def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    required_files = [
        "ldm/modules/diffusionmodules/edge_processor.py",
        "ldm/modules/diffusionmodules/unet_with_edge.py",
        "ldm/models/diffusion/ddpm_with_edge.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path}: 文件不存在")
            all_exist = False
    
    return all_exist


def check_test_scripts():
    """检查测试脚本"""
    print("\n检查测试脚本...")
    
    current_dir = Path(__file__).parent
    
    test_scripts = [
        "test_edge_map_quick.py",
        "test_edge_map_comprehensive.py",
        "test_edge_map_real_images.py",
        "test_edge_map_performance.py",
        "debug_edge_test.py",
    ]
    
    all_exist = True
    for script in test_scripts:
        script_path = current_dir / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ❌ {script}: 文件不存在")
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print("Edge Map测试环境检查")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("conda", check_conda),
        ("conda环境", lambda: check_conda_env("sr_edge")),
        ("必需包", check_required_packages),
        ("可选包", check_optional_packages),
        ("CUDA", check_cuda),
        ("项目结构", check_project_structure),
        ("测试脚本", check_test_scripts),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ❌ 检查 {name} 时出错: {e}")
            results[name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("检查结果总结:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查都通过！可以开始运行edge map测试。")
        print("\n建议的运行方式:")
        print("1. 使用bash脚本: ./run_edge_test.sh")
        print("2. 使用Python脚本: python debug_edge_test.py")
        print("3. 直接运行: conda activate sr_edge && python test_edge_map_quick.py --synthetic")
    else:
        print("❌ 部分检查失败，请解决上述问题后再运行测试。")
        print("\n常见解决方案:")
        print("1. 创建conda环境: conda create -n sr_edge python=3.8")
        print("2. 激活环境: conda activate sr_edge")
        print("3. 安装依赖: pip install torch torchvision opencv-python matplotlib pillow numpy")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
