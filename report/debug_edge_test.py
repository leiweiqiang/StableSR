#!/usr/bin/env python3
"""
Edge Map测试调试脚本
自动激活conda环境并运行edge map测试

使用方法:
python debug_edge_test.py
python debug_edge_test.py --test_type quick
python debug_edge_test.py --test_type comprehensive
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_with_conda_env(script_path, args, conda_env="sr_edge"):
    """在conda环境中运行脚本"""
    # 构建完整的命令
    cmd_parts = [
        "conda", "run", "-n", conda_env,
        "python", str(script_path)
    ]
    cmd_parts.extend(args)
    
    cmd = " ".join(cmd_parts)
    print(f"执行命令: {cmd}")
    print("="*60)
    
    try:
        # 使用conda run来在指定环境中运行
        result = subprocess.run(cmd, shell=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def check_conda_env(env_name="sr_edge"):
    """检查conda环境是否存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            envs = result.stdout
            if env_name in envs:
                print(f"✓ 找到conda环境: {env_name}")
                return True
            else:
                print(f"❌ 未找到conda环境: {env_name}")
                print("可用的环境:")
                print(envs)
                return False
        else:
            print("❌ 无法获取conda环境列表")
            return False
    except Exception as e:
        print(f"❌ 检查conda环境时出错: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Edge Map测试调试脚本")
    
    parser.add_argument("--test_type", type=str, default="quick",
                       choices=["quick", "comprehensive", "real", "performance"],
                       help="测试类型")
    parser.add_argument("--conda_env", type=str, default="sr_edge",
                       help="conda环境名称")
    parser.add_argument("--input_image", type=str, default=None,
                       help="输入图像路径（用于real测试）")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="输入图像目录（用于real测试）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备")
    
    args = parser.parse_args()
    
    print("Edge Map测试调试脚本")
    print("="*60)
    print(f"目标conda环境: {args.conda_env}")
    print(f"测试类型: {args.test_type}")
    
    # 检查conda环境
    if not check_conda_env(args.conda_env):
        print("\n请先创建conda环境:")
        print(f"conda create -n {args.conda_env} python=3.8")
        print(f"conda activate {args.conda_env}")
        print("pip install torch torchvision opencv-python matplotlib pillow numpy")
        return 1
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 根据测试类型选择脚本和参数
    if args.test_type == "quick":
        script_path = script_dir / "test_edge_map_quick.py"
        script_args = ["--synthetic", "--test_processor"]
        if args.output_dir:
            script_args.extend(["--output_dir", args.output_dir])
            
    elif args.test_type == "comprehensive":
        script_path = script_dir / "test_edge_map_comprehensive.py"
        script_args = []
        if args.output_dir:
            script_args.extend(["--output_dir", args.output_dir])
        if args.device != "auto":
            script_args.extend(["--device", args.device])
            
    elif args.test_type == "real":
        script_path = script_dir / "test_edge_map_real_images.py"
        script_args = []
        
        if args.input_image:
            script_args.extend(["--input_image", args.input_image])
        elif args.input_dir:
            script_args.extend(["--input_dir", args.input_dir])
        else:
            print("❌ real测试需要指定 --input_image 或 --input_dir")
            return 1
            
        if args.output_dir:
            script_args.extend(["--output_dir", args.output_dir])
            
    elif args.test_type == "performance":
        script_path = script_dir / "test_edge_map_performance.py"
        script_args = []
        if args.output_dir:
            script_args.extend(["--output_dir", args.output_dir])
        if args.device != "auto":
            script_args.extend(["--device", args.device])
    
    # 检查脚本文件是否存在
    if not script_path.exists():
        print(f"❌ 脚本文件不存在: {script_path}")
        return 1
    
    print(f"使用脚本: {script_path}")
    print(f"脚本参数: {' '.join(script_args)}")
    
    # 在conda环境中运行脚本
    print("\n开始执行测试...")
    success = run_with_conda_env(script_path, script_args, args.conda_env)
    
    if success:
        print("\n🎉 测试完成!")
        if args.output_dir:
            print(f"结果保存在: {args.output_dir}")
        else:
            print("结果保存在默认目录")
    else:
        print("\n❌ 测试失败")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
