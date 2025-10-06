#!/usr/bin/env python3
"""
Edge Map测试脚本使用示例
演示如何使用各种测试脚本
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr:
                print("错误:")
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False


def main():
    """主函数 - 演示各种测试脚本的使用"""
    print("Edge Map测试脚本使用示例")
    print("="*60)
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 示例1: 快速测试（使用合成图像）
    print("\n示例1: 快速测试 - 使用合成图像")
    cmd1 = f"cd {script_dir} && python test_edge_map_quick.py --synthetic --output_dir quick_test_demo"
    run_command(cmd1, "快速测试 - 合成图像")
    
    # 示例2: 快速测试（包含edge处理器测试）
    print("\n示例2: 快速测试 - 包含edge处理器测试")
    cmd2 = f"cd {script_dir} && python test_edge_map_quick.py --synthetic --test_processor --output_dir quick_test_with_processor"
    run_command(cmd2, "快速测试 - 包含edge处理器")
    
    # 示例3: 性能测试
    print("\n示例3: 性能基准测试")
    cmd3 = f"cd {script_dir} && python test_edge_map_performance.py --output_dir performance_demo"
    run_command(cmd3, "性能基准测试")
    
    # 示例4: 综合测试（仅生成和可视化）
    print("\n示例4: 综合测试 - 生成和可视化")
    cmd4 = f"cd {script_dir} && python test_edge_map_comprehensive.py --test_type generation --output_dir comprehensive_demo"
    run_command(cmd4, "综合测试 - 生成和可视化")
    
    # 示例5: 综合测试（仅可视化）
    print("\n示例5: 综合测试 - 可视化")
    cmd5 = f"cd {script_dir} && python test_edge_map_comprehensive.py --test_type visualize --output_dir visualize_demo"
    run_command(cmd5, "综合测试 - 可视化")
    
    print("\n" + "="*60)
    print("示例演示完成!")
    print("="*60)
    
    print("\n生成的结果目录:")
    demo_dirs = [
        "quick_test_demo",
        "quick_test_with_processor", 
        "performance_demo",
        "comprehensive_demo",
        "visualize_demo"
    ]
    
    for demo_dir in demo_dirs:
        demo_path = script_dir / demo_dir
        if demo_path.exists():
            print(f"  ✓ {demo_dir}/")
            # 列出目录内容
            try:
                files = list(demo_path.rglob("*"))
                for file in files[:5]:  # 只显示前5个文件
                    rel_path = file.relative_to(demo_path)
                    print(f"    - {rel_path}")
                if len(files) > 5:
                    print(f"    ... 还有 {len(files) - 5} 个文件")
            except:
                pass
        else:
            print(f"  ❌ {demo_dir}/ (未生成)")
    
    print("\n使用说明:")
    print("1. 查看 README_edge_test.md 了解详细使用方法")
    print("2. 使用 --help 参数查看各脚本的详细选项")
    print("3. 准备真实图像进行测试:")
    print("   python test_edge_map_real_images.py --input_image /path/to/image.jpg")
    print("4. 进行完整的性能测试:")
    print("   python test_edge_map_performance.py --batch_sizes 1,2,4,8")


if __name__ == "__main__":
    main()