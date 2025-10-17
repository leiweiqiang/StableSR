#!/usr/bin/env python3
"""
Debug script for comparison grid generation
Checks all prerequisites and provides detailed diagnostics
"""

import os
import sys
from pathlib import Path

def check_comparison_requirements(base_path, gt_dir, epoch_num):
    """
    Check all requirements for generating comparison grids
    """
    print("=" * 60)
    print("对比图生成诊断工具")
    print("=" * 60)
    print()
    
    errors = []
    warnings = []
    
    # 1. Check base path
    print(f"1️⃣  检查基础路径: {base_path}")
    if not os.path.exists(base_path):
        errors.append(f"❌ 基础路径不存在: {base_path}")
        print(f"   ❌ 不存在")
    else:
        print(f"   ✅ 存在")
    print()
    
    # 2. Check GT directory
    print(f"2️⃣  检查GT目录: {gt_dir}")
    if not os.path.exists(gt_dir):
        errors.append(f"❌ GT目录不存在: {gt_dir}")
        print(f"   ❌ 不存在")
    else:
        gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))]
        print(f"   ✅ 存在")
        print(f"   📊 包含 {len(gt_files)} 个图片文件")
        if gt_files:
            print(f"   📝 示例: {gt_files[0]}")
    print()
    
    # 3. Check epoch number format
    print(f"3️⃣  检查epoch格式: {epoch_num}")
    try:
        epoch_int = int(epoch_num)
        print(f"   ✅ 格式正确（整数: {epoch_int}）")
        if epoch_num != str(epoch_int):
            print(f"   ℹ️  注意：带前导零 '{epoch_num}' 将被转换为 '{epoch_int}'")
    except ValueError:
        errors.append(f"❌ Epoch格式错误，应为整数: {epoch_num}")
        print(f"   ❌ 格式错误，应为整数")
        epoch_int = epoch_num  # fallback
    print()
    
    # 4. Check edge directory (use epoch_int to remove leading zeros)
    edge_dir = os.path.join(base_path, "edge", f"epochs_{epoch_int}")
    print(f"4️⃣  检查edge目录: {edge_dir}")
    if not os.path.exists(edge_dir):
        errors.append(f"❌ Edge目录不存在: {edge_dir}")
        print(f"   ❌ 不存在")
    else:
        edge_files = [f for f in os.listdir(edge_dir) 
                     if f.endswith('.png') and not os.path.isdir(os.path.join(edge_dir, f))]
        print(f"   ✅ 存在")
        print(f"   📊 包含 {len(edge_files)} 个PNG文件")
        if edge_files:
            print(f"   📝 示例: {edge_files[0]}")
        else:
            warnings.append("⚠️  Edge目录为空，无图片文件")
    print()
    
    # 5. Check other mode directories
    modes = [
        ("no_edge", "No-Edge模式"),
        ("dummy_edge", "Dummy-Edge模式"),
        ("stablesr", "StableSR基准")
    ]
    
    for mode, mode_name in modes:
        if mode == "stablesr":
            mode_dir = os.path.join(base_path, mode, "epochs_baseline")
            if not os.path.exists(mode_dir):
                mode_dir = os.path.join(base_path, mode, "baseline")
        else:
            mode_dir = os.path.join(base_path, mode, f"epochs_{epoch_int}")
        
        print(f"5️⃣  检查{mode_name}目录: {mode_dir}")
        if not os.path.exists(mode_dir):
            warnings.append(f"⚠️  {mode_name}目录不存在: {mode_dir}")
            print(f"   ⚠️  不存在（对比图中该列将为空）")
        else:
            mode_files = [f for f in os.listdir(mode_dir) 
                         if f.endswith('.png') and not os.path.isdir(os.path.join(mode_dir, f))]
            print(f"   ✅ 存在")
            print(f"   📊 包含 {len(mode_files)} 个PNG文件")
        print()
    
    # 6. Check metrics files
    print("6️⃣  检查metrics文件:")
    metrics_modes = [
        ("edge", f"epochs_{epoch_int}"),
        ("no_edge", f"epochs_{epoch_int}"),
        ("dummy_edge", f"epochs_{epoch_int}"),
        ("stablesr", "epochs_baseline")
    ]
    
    for mode, subdir in metrics_modes:
        if mode == "stablesr" and not os.path.exists(os.path.join(base_path, mode, subdir)):
            metrics_path = os.path.join(base_path, mode, "baseline", "metrics.json")
        else:
            metrics_path = os.path.join(base_path, mode, subdir, "metrics.json")
        
        if os.path.exists(metrics_path):
            print(f"   ✅ {mode}: {metrics_path}")
        else:
            warnings.append(f"⚠️  {mode} metrics文件不存在")
            print(f"   ⚠️  {mode}: 不存在（对比图中无metrics显示）")
    print()
    
    # 7. Check output directory
    comparison_dir = os.path.join(base_path, "comparisons", f"epochs_{epoch_int}")
    print(f"7️⃣  检查输出目录: {comparison_dir}")
    if os.path.exists(comparison_dir):
        existing_comparisons = [f for f in os.listdir(comparison_dir) 
                               if f.endswith('_comparison.png')]
        print(f"   ✅ 存在")
        print(f"   📊 已有 {len(existing_comparisons)} 个对比图")
    else:
        print(f"   ℹ️  不存在（将自动创建）")
    print()
    
    # Summary
    print("=" * 60)
    print("诊断结果汇总")
    print("=" * 60)
    
    if errors:
        print("\n❌ 严重错误（会导致生成失败）:")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print("\n⚠️  警告（不影响生成，但对比图可能不完整）:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not errors and not warnings:
        print("\n✅ 所有检查通过！可以生成对比图")
    elif not errors:
        print("\n✅ 基本检查通过，可以生成对比图（但可能不完整）")
    else:
        print("\n❌ 存在严重错误，无法生成对比图")
        print("\n建议修复步骤:")
        print("  1. 确保已运行推理生成edge模式的结果")
        print("  2. 检查路径配置是否正确")
        print("  3. 确保epoch号格式正确")
    
    print("=" * 60)
    return len(errors) == 0


def main():
    if len(sys.argv) != 4:
        print("用法: python debug_comparison.py <base_path> <gt_dir> <epoch_num>")
        print()
        print("示例:")
        print("  python scripts/debug_comparison.py \\")
        print("    ~/validation_results/2025-10-17T00-00-00_experiment \\")
        print("    ~/nas/test_dataset/512x512_White_GT \\")
        print("    47")
        sys.exit(1)
    
    base_path = os.path.expanduser(sys.argv[1])
    gt_dir = os.path.expanduser(sys.argv[2])
    epoch_num = sys.argv[3]
    
    success = check_comparison_requirements(base_path, gt_dir, epoch_num)
    
    if success:
        print("\n尝试生成对比图...")
        print("运行命令:")
        print(f"  python scripts/create_comparison_grid.py '{base_path}' '{gt_dir}' '{epoch_num}'")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

