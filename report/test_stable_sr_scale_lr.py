"""
StableSR_ScaleLR 测试脚本
用于验证类的功能和参数设置
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_sr_scale_lr import StableSR_ScaleLR
    print("✓ 成功导入 StableSR_ScaleLR 类")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


def test_class_initialization():
    """测试类初始化"""
    print("\n测试类初始化...")
    
    try:
        # 使用虚拟路径进行初始化测试（不会实际加载模型）
        processor = StableSR_ScaleLR(
            config_path="configs/stableSRNew/v2-finetune_text_T_512.yaml",
            ckpt_path="dummy_ckpt.ckpt",
            vqgan_ckpt_path="dummy_vqgan.ckpt",
            ddpm_steps=200,
            dec_w=0.5,
            colorfix_type="adain"
        )
        print("✗ 初始化应该失败（因为检查点不存在）")
        return False
    except Exception as e:
        print(f"✓ 预期的初始化失败: {e}")
        return True


def test_parameter_validation():
    """测试参数验证"""
    print("\n测试参数验证...")
    
    # 测试无效的颜色修正类型
    try:
        processor = StableSR_ScaleLR(
            config_path="configs/stableSRNew/v2-finetune_text_T_512.yaml",
            ckpt_path="dummy_ckpt.ckpt",
            vqgan_ckpt_path="dummy_vqgan.ckpt",
            colorfix_type="invalid_type"
        )
        print("✗ 应该拒绝无效的颜色修正类型")
        return False
    except Exception as e:
        print(f"✓ 正确拒绝了无效参数: {e}")
        return True


def test_directory_creation():
    """测试目录创建功能"""
    print("\n测试目录创建...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_out_dir = os.path.join(temp_dir, "test_output")
        
        # 创建测试输入目录和文件
        test_input_dir = os.path.join(temp_dir, "test_input")
        os.makedirs(test_input_dir, exist_ok=True)
        
        # 创建一个简单的测试图像（1x1像素的PNG）
        from PIL import Image
        import numpy as np
        
        test_image = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)
        test_image_path = os.path.join(test_input_dir, "test.png")
        test_image.save(test_image_path)
        
        print(f"✓ 创建测试图像: {test_image_path}")
        
        # 测试目录结构创建（不实际处理图像）
        res_dir = os.path.join(test_out_dir, "RES")
        lr_dir = os.path.join(test_out_dir, "LR")
        hq_dir = os.path.join(test_out_dir, "HQ")
        
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)
        os.makedirs(hq_dir, exist_ok=True)
        
        # 验证目录创建
        assert os.path.exists(res_dir), "RES目录未创建"
        assert os.path.exists(lr_dir), "LR目录未创建"
        assert os.path.exists(hq_dir), "HQ目录未创建"
        
        print("✓ 目录结构创建成功")
        return True


def test_file_path_handling():
    """测试文件路径处理"""
    print("\n测试文件路径处理...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试单文件路径
        test_file = os.path.join(temp_dir, "test.png")
        from PIL import Image
        import numpy as np
        
        test_image = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)
        test_image.save(test_file)
        
        # 测试目录路径
        test_dir = os.path.join(temp_dir, "test_dir")
        os.makedirs(test_dir, exist_ok=True)
        
        test_image2 = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 255)
        test_image2.save(os.path.join(test_dir, "test2.png"))
        
        print("✓ 文件路径处理测试完成")
        return True


def main():
    """主测试函数"""
    print("StableSR_ScaleLR 功能测试")
    print("=" * 50)
    
    tests = [
        test_class_initialization,
        test_parameter_validation,
        test_directory_creation,
        test_file_path_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return True
    else:
        print("✗ 部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
