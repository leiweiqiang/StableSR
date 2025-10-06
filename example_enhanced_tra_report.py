#!/usr/bin/env python3
"""
Enhanced TraReport使用示例
演示如何使用EnhancedTraReport进行StableSR Edge和StableSR Upscale模型的比较评估
"""

import os
import json
from pathlib import Path
from enhanced_tra_report import EnhancedTraReport


def example_basic_usage():
    """基本使用示例"""
    print("="*60)
    print("Enhanced TraReport - Basic Usage Example")
    print("="*60)
    
    # 设置路径
    gt_dir = "/path/to/your/gt/images"  # 高分辨率图片目录
    val_dir = "/path/to/your/val/images"  # 低分辨率图片目录
    
    # 模型路径
    stablesr_edge_model = "/path/to/stablesr_edge_model.ckpt"
    stablesr_upscale_model = "/path/to/stablesr_upscale_model.ckpt"
    
    # 配置文件路径
    edge_config = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
    upscale_config = "./configs/stableSRNew/v2-finetune_text_T_512.yaml"
    
    # 创建EnhancedTraReport实例
    enhanced_tra_report = EnhancedTraReport(
        gt_dir=gt_dir,
        val_dir=val_dir,
        stablesr_edge_model_path=stablesr_edge_model,
        stablesr_upscale_model_path=stablesr_upscale_model,
        stablesr_edge_config_path=edge_config,
        stablesr_upscale_config_path=upscale_config,
        device="cuda",
        ddpm_steps=200,
        upscale=4.0,
        colorfix_type="adain",
        seed=42
    )
    
    # 运行评估
    results = enhanced_tra_report.run_evaluation("comparison_results.json")
    
    # 打印结果
    print_results_summary(results)


def example_custom_parameters():
    """自定义参数示例"""
    print("="*60)
    print("Enhanced TraReport - Custom Parameters Example")
    print("="*60)
    
    # 使用自定义参数
    enhanced_tra_report = EnhancedTraReport(
        gt_dir="/path/to/gt",
        val_dir="/path/to/val",
        stablesr_edge_model_path="/path/to/edge_model.ckpt",
        stablesr_upscale_model_path="/path/to/upscale_model.ckpt",
        device="cuda",
        ddpm_steps=100,  # 减少采样步数以加快处理
        upscale=2.0,     # 2倍超分辨率
        colorfix_type="wavelet",  # 使用小波颜色修复
        seed=123
    )
    
    # 运行评估
    results = enhanced_tra_report.run_evaluation("custom_results.json")
    
    # 打印结果
    print_results_summary(results)


def example_batch_evaluation():
    """批量评估示例"""
    print("="*60)
    print("Enhanced TraReport - Batch Evaluation Example")
    print("="*60)
    
    # 多个数据集
    datasets = [
        {
            "name": "DIV2K",
            "gt_dir": "/path/to/DIV2K/HR",
            "val_dir": "/path/to/DIV2K/LR"
        },
        {
            "name": "Set5",
            "gt_dir": "/path/to/Set5/HR",
            "val_dir": "/path/to/Set5/LR"
        },
        {
            "name": "Set14",
            "gt_dir": "/path/to/Set14/HR",
            "val_dir": "/path/to/Set14/LR"
        }
    ]
    
    model_paths = {
        "edge": "/path/to/stablesr_edge_model.ckpt",
        "upscale": "/path/to/stablesr_upscale_model.ckpt"
    }
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\nEvaluating {dataset['name']} dataset...")
        
        enhanced_tra_report = EnhancedTraReport(
            gt_dir=dataset["gt_dir"],
            val_dir=dataset["val_dir"],
            stablesr_edge_model_path=model_paths["edge"],
            stablesr_upscale_model_path=model_paths["upscale"],
            ddpm_steps=200,
            upscale=4.0,
            colorfix_type="adain"
        )
        
        results = enhanced_tra_report.run_evaluation(f"{dataset['name']}_results.json")
        all_results[dataset["name"]] = results
        
        print(f"{dataset['name']} evaluation completed!")
        print_results_summary(results)
    
    # 保存所有结果
    with open("batch_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nBatch evaluation completed!")
    print("All results saved to batch_evaluation_results.json")


def example_model_comparison():
    """模型比较示例"""
    print("="*60)
    print("Enhanced TraReport - Model Comparison Example")
    print("="*60)
    
    # 比较不同的模型配置
    model_configs = [
        {
            "name": "StableSR Edge (200 steps)",
            "edge_model": "/path/to/edge_model.ckpt",
            "upscale_model": "/path/to/upscale_model.ckpt",
            "ddpm_steps": 200
        },
        {
            "name": "StableSR Edge (100 steps)",
            "edge_model": "/path/to/edge_model.ckpt",
            "upscale_model": "/path/to/upscale_model.ckpt",
            "ddpm_steps": 100
        },
        {
            "name": "StableSR Edge (50 steps)",
            "edge_model": "/path/to/edge_model.ckpt",
            "upscale_model": "/path/to/upscale_model.ckpt",
            "ddpm_steps": 50
        }
    ]
    
    gt_dir = "/path/to/gt"
    val_dir = "/path/to/val"
    
    comparison_results = {}
    
    for config in model_configs:
        print(f"\nTesting {config['name']}...")
        
        enhanced_tra_report = EnhancedTraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            stablesr_edge_model_path=config["edge_model"],
            stablesr_upscale_model_path=config["upscale_model"],
            ddpm_steps=config["ddpm_steps"],
            upscale=4.0,
            colorfix_type="adain"
        )
        
        results = enhanced_tra_report.run_evaluation(f"{config['name'].replace(' ', '_')}_results.json")
        comparison_results[config["name"]] = results
        
        print(f"{config['name']} completed!")
        print_results_summary(results)
    
    # 比较结果
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    for name, results in comparison_results.items():
        edge_psnr = results["summary"]["stablesr_edge"]["average_psnr"]
        upscale_psnr = results["summary"]["stablesr_upscale"]["average_psnr"]
        difference = results["summary"]["comparison"]["psnr_difference"]
        
        print(f"{name}:")
        print(f"  StableSR Edge PSNR: {edge_psnr:.4f}")
        print(f"  StableSR Upscale PSNR: {upscale_psnr:.4f}")
        print(f"  Difference: {difference:.4f}")
        print(f"  Better: {results['summary']['comparison']['better_model']}")
        print()


def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "-"*40)
    print("RESULTS SUMMARY")
    print("-"*40)
    print(f"Total files: {results['evaluation_info']['total_files']}")
    print(f"StableSR Edge Average PSNR: {results['summary']['stablesr_edge']['average_psnr']:.4f}")
    print(f"StableSR Upscale Average PSNR: {results['summary']['stablesr_upscale']['average_psnr']:.4f}")
    print(f"PSNR Difference: {results['summary']['comparison']['psnr_difference']:.4f}")
    print(f"Better Model: {results['summary']['comparison']['better_model']}")
    print(f"Improvement: {results['summary']['comparison']['improvement_percentage']:.2f}%")
    
    # 详细统计
    edge_stats = results['summary']['stablesr_edge']
    upscale_stats = results['summary']['stablesr_upscale']
    
    print(f"\nStableSR Edge Statistics:")
    print(f"  Min PSNR: {edge_stats['min_psnr']:.4f}")
    print(f"  Max PSNR: {edge_stats['max_psnr']:.4f}")
    print(f"  Std PSNR: {edge_stats['std_psnr']:.4f}")
    
    print(f"\nStableSR Upscale Statistics:")
    print(f"  Min PSNR: {upscale_stats['min_psnr']:.4f}")
    print(f"  Max PSNR: {upscale_stats['max_psnr']:.4f}")
    print(f"  Std PSNR: {upscale_stats['std_psnr']:.4f}")


def example_analysis_results():
    """分析结果示例"""
    print("="*60)
    print("Enhanced TraReport - Results Analysis Example")
    print("="*60)
    
    # 加载结果文件
    results_file = "comparison_results.json"
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Please run evaluation first.")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # 分析每个图片的结果
    print("Individual Image Analysis:")
    print("-"*40)
    
    for i, result in enumerate(results["results"]):
        val_name = Path(result["val_file"]).name
        edge_psnr = result["stablesr_edge"]["psnr"]
        upscale_psnr = result["stablesr_upscale"]["psnr"]
        difference = result["psnr_difference"]
        better = result["better_model"]
        
        print(f"Image {i+1} ({val_name}):")
        print(f"  StableSR Edge: {edge_psnr:.4f}")
        print(f"  StableSR Upscale: {upscale_psnr:.4f}")
        print(f"  Difference: {difference:.4f}")
        print(f"  Better: {better}")
        print()
    
    # 统计哪个模型更好
    edge_wins = sum(1 for r in results["results"] if r["better_model"] == "StableSR Edge")
    upscale_wins = sum(1 for r in results["results"] if r["better_model"] == "StableSR Upscale")
    total = len(results["results"])
    
    print("Model Performance Summary:")
    print("-"*40)
    print(f"StableSR Edge wins: {edge_wins}/{total} ({edge_wins/total*100:.1f}%)")
    print(f"StableSR Upscale wins: {upscale_wins}/{total} ({upscale_wins/total*100:.1f}%)")
    
    # 计算平均改进
    improvements = [r["psnr_difference"] for r in results["results"]]
    avg_improvement = sum(improvements) / len(improvements)
    print(f"Average PSNR improvement: {avg_improvement:.4f}")


def main():
    """主函数"""
    print("Enhanced TraReport Examples")
    print("="*60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Parameters", example_custom_parameters),
        ("Batch Evaluation", example_batch_evaluation),
        ("Model Comparison", example_model_comparison),
        ("Results Analysis", example_analysis_results),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nNote: Please update the file paths in the examples to match your setup.")
    print("The examples show the structure and usage patterns for Enhanced TraReport.")
    
    # 运行基本使用示例（注释掉实际执行，因为需要真实的模型文件）
    # example_basic_usage()


if __name__ == "__main__":
    main()
