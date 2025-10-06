#!/usr/bin/env python3
"""
TraReport使用示例
演示如何使用TraReport类进行超分辨率模型评估
"""

from tra_report import TraReport
import json

def example_usage():
    """TraReport使用示例"""
    
    # 示例参数
    gt_dir = "/path/to/gt/images"  # 真实高分辨率图片目录
    val_dir = "/path/to/val/images"  # 待处理的低分辨率图片目录
    model_path = "/path/to/model.ckpt"  # 模型权重文件路径
    config_path = "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"  # 配置文件路径
    
    # 创建TraReport实例
    tra_report = TraReport(
        gt_dir=gt_dir,
        val_dir=val_dir,
        model_path=model_path,
        config_path=config_path,
        ddpm_steps=200,  # DDPM采样步数
        upscale=4.0,     # 超分辨率倍数
        colorfix_type="adain",  # 颜色修复类型
        seed=42          # 随机种子
    )
    
    # 运行评估
    results = tra_report.run_evaluation(output_path="evaluation_results.json")
    
    # 打印结果摘要
    print("\n=== 评估结果摘要 ===")
    print(f"处理文件总数: {results['total_files']}")
    print(f"平均PSNR: {results['summary']['average_psnr']:.4f}")
    print(f"最小PSNR: {results['summary']['min_psnr']:.4f}")
    print(f"最大PSNR: {results['summary']['max_psnr']:.4f}")
    print(f"PSNR标准差: {results['summary']['std_psnr']:.4f}")
    
    # 打印前5个结果
    print("\n=== 前5个文件结果 ===")
    for i, result in enumerate(results['results'][:5]):
        print(f"{i+1}. {result['val_file']}: PSNR = {result['psnr']:.4f}")

def batch_evaluation():
    """批量评估多个模型"""
    
    # 模型配置列表
    model_configs = [
        {
            "name": "model_v1",
            "model_path": "/path/to/model_v1.ckpt",
            "config_path": "./configs/stableSRNew/v2-finetune_text_T_512_edge.yaml"
        },
        {
            "name": "model_v2", 
            "model_path": "/path/to/model_v2.ckpt",
            "config_path": "./configs/stableSRNew/v2-finetune_text_T_512.yaml"
        }
    ]
    
    gt_dir = "/path/to/gt/images"
    val_dir = "/path/to/val/images"
    
    batch_results = {}
    
    for config in model_configs:
        print(f"\n=== 评估模型: {config['name']} ===")
        
        tra_report = TraReport(
            gt_dir=gt_dir,
            val_dir=val_dir,
            model_path=config['model_path'],
            config_path=config['config_path'],
            ddpm_steps=200,
            upscale=4.0,
            colorfix_type="adain",
            seed=42
        )
        
        results = tra_report.evaluate()
        batch_results[config['name']] = results['summary']
        
        # 保存单个模型结果
        tra_report.save_results(results, f"results_{config['name']}.json")
    
    # 保存批量比较结果
    with open("batch_comparison.json", 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print("\n=== 批量评估结果比较 ===")
    for name, summary in batch_results.items():
        print(f"{name}: 平均PSNR = {summary['average_psnr']:.4f}")

if __name__ == "__main__":
    print("TraReport使用示例")
    print("请根据实际情况修改路径参数")
    
    # 运行示例
    # example_usage()
    
    # 批量评估示例
    # batch_evaluation()
