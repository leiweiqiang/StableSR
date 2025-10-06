#!/usr/bin/env python3
"""
简化版TraReport类，用于快速测试PSNR计算流程
不使用复杂的DDIM采样，直接使用双三次插值进行超分辨率
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

from basicsr.metrics.psnr_ssim import calculate_psnr


class SimpleTraReport:
    """简化版超分辨率模型性能评估类"""
    
    def __init__(self, gt_dir: str, val_dir: str, upscale: float = 4.0):
        """
        初始化SimpleTraReport类
        
        Args:
            gt_dir: 真实高分辨率图片目录
            val_dir: 待处理的低分辨率图片目录  
            upscale: 超分辨率倍数
        """
        self.gt_dir = Path(gt_dir)
        self.val_dir = Path(val_dir)
        self.upscale = upscale
        
        # 验证目录存在
        if not self.gt_dir.exists():
            raise ValueError(f"GT directory does not exist: {gt_dir}")
        if not self.val_dir.exists():
            raise ValueError(f"Val directory does not exist: {val_dir}")
            
    def _load_img(self, path: str) -> np.ndarray:
        """加载图片并预处理"""
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"Loaded image of size ({w}, {h}) from {path}")
        
        # 转换为numpy数组
        image = np.array(image).astype(np.float32)
        return image
        
    def _upscale_image_simple(self, image: np.ndarray) -> np.ndarray:
        """使用双三次插值进行简单的超分辨率处理"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * self.upscale), int(w * self.upscale)
        
        # 使用PIL进行双三次插值
        pil_image = Image.fromarray(image.astype(np.uint8))
        upscaled = pil_image.resize((new_w, new_h), Image.BICUBIC)
        
        return np.array(upscaled).astype(np.float32)
        
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图片之间的PSNR"""
        return calculate_psnr(img1, img2, crop_border=0, test_y_channel=False)
        
    def _find_matching_files(self) -> List[Tuple[str, str]]:
        """查找val和gt目录中匹配的文件对"""
        val_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        gt_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        val_files = []
        for ext in val_extensions:
            val_files.extend(glob.glob(str(self.val_dir / f"*{ext}")))
            val_files.extend(glob.glob(str(self.val_dir / f"*{ext.upper()}")))
            
        matching_pairs = []
        for val_file in val_files:
            val_name = Path(val_file).stem
            
            # 查找对应的gt文件
            for ext in gt_extensions:
                gt_file = self.gt_dir / f"{val_name}{ext}"
                if gt_file.exists():
                    matching_pairs.append((val_file, str(gt_file)))
                    break
                gt_file = self.gt_dir / f"{val_name}{ext.upper()}"
                if gt_file.exists():
                    matching_pairs.append((val_file, str(gt_file)))
                    break
                    
        return matching_pairs
        
    def evaluate(self) -> Dict:
        """执行评估并返回JSON结果"""
        # 查找匹配的文件对
        matching_pairs = self._find_matching_files()
        if not matching_pairs:
            raise ValueError("No matching files found between val and gt directories")
            
        print(f"Found {len(matching_pairs)} matching file pairs")
        
        results = {
            "method": "simple_bicubic_upscaling",
            "gt_dir": str(self.gt_dir),
            "val_dir": str(self.val_dir),
            "total_files": len(matching_pairs),
            "parameters": {
                "upscale": self.upscale
            },
            "results": [],
            "summary": {
                "average_psnr": 0.0,
                "min_psnr": float('inf'),
                "max_psnr": 0.0
            }
        }
        
        psnr_values = []
        
        # 处理每个文件对
        for val_file, gt_file in tqdm(matching_pairs, desc="Processing images"):
            try:
                # 加载图片
                val_image = self._load_img(val_file)
                gt_image = self._load_img(gt_file)
                
                # 简单的超分辨率处理（双三次插值）
                sr_image = self._upscale_image_simple(val_image)
                
                # 调整GT图片尺寸以匹配SR图片
                if sr_image.shape != gt_image.shape:
                    gt_pil = Image.fromarray(gt_image.astype(np.uint8))
                    gt_resized = gt_pil.resize((sr_image.shape[1], sr_image.shape[0]), Image.LANCZOS)
                    gt_image = np.array(gt_resized).astype(np.float32)
                
                # 计算PSNR
                psnr = self._calculate_psnr(sr_image, gt_image)
                psnr_values.append(psnr)
                
                # 记录结果
                result_entry = {
                    "val_file": val_file,
                    "gt_file": gt_file,
                    "psnr": float(psnr),
                    "sr_shape": sr_image.shape,
                    "gt_shape": gt_image.shape
                }
                results["results"].append(result_entry)
                
                print(f"Processed {Path(val_file).name}: PSNR = {psnr:.4f}")
                
            except Exception as e:
                print(f"Error processing {val_file}: {str(e)}")
                continue
                
        # 计算统计信息
        if psnr_values:
            results["summary"]["average_psnr"] = float(np.mean(psnr_values))
            results["summary"]["min_psnr"] = float(np.min(psnr_values))
            results["summary"]["max_psnr"] = float(np.max(psnr_values))
            results["summary"]["std_psnr"] = float(np.std(psnr_values))
            
        print(f"\nEvaluation completed!")
        print(f"Average PSNR: {results['summary']['average_psnr']:.4f}")
        print(f"Min PSNR: {results['summary']['min_psnr']:.4f}")
        print(f"Max PSNR: {results['summary']['max_psnr']:.4f}")
        
        return results
        
    def save_results(self, results: Dict, output_path: str):
        """保存结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
        
    def run_evaluation(self, output_path: Optional[str] = None) -> Dict:
        """运行完整的评估流程"""
        results = self.evaluate()
        
        if output_path is None:
            output_path = f"simple_tra_report_results.json"
            
        self.save_results(results, output_path)
        return results


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SimpleTraReport - Simple Super Resolution Model Evaluation")
    parser.add_argument("--gt_dir", type=str, required=True,
                       help="Ground truth high-resolution images directory")
    parser.add_argument("--val_dir", type=str, required=True,
                       help="Validation low-resolution images directory")
    parser.add_argument("--output", type=str, default="simple_results.json",
                       help="Output JSON file path")
    parser.add_argument("--upscale", type=float, default=4.0,
                       help="Upscale factor (default: 4.0)")
    
    args = parser.parse_args()
    
    # 创建SimpleTraReport实例
    tra_report = SimpleTraReport(
        gt_dir=args.gt_dir,
        val_dir=args.val_dir,
        upscale=args.upscale
    )
    
    # 运行评估
    results = tra_report.run_evaluation(output_path=args.output)
    
    return results


if __name__ == "__main__":
    main()
