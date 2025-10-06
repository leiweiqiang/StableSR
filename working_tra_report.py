#!/usr/bin/env python3
"""
工作版TraReport类，使用双三次插值进行超分辨率
这样可以快速验证整个PSNR计算流程
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


class WorkingTraReport:
    """工作版超分辨率模型性能评估类"""
    
    def __init__(self, gt_dir: str, val_dir: str, upscale: float = 4.0):
        """
        初始化WorkingTraReport类
        
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
    
    def _load_img(self, path: Union[str, Path]) -> np.ndarray:
        """加载图片并转换为numpy数组"""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _bicubic_upscale(self, img: np.ndarray, scale: float) -> np.ndarray:
        """使用双三次插值进行超分辨率"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_pil = Image.fromarray(img)
        upscaled_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(upscaled_pil)
    
    def _calculate_psnr_for_images(self, gt_img: np.ndarray, sr_img: np.ndarray) -> float:
        """计算两张图片之间的PSNR"""
        # 确保图片尺寸一致
        if gt_img.shape != sr_img.shape:
            # 裁剪到较小的尺寸
            min_h = min(gt_img.shape[0], sr_img.shape[0])
            min_w = min(gt_img.shape[1], sr_img.shape[1])
            gt_img = gt_img[:min_h, :min_w]
            sr_img = sr_img[:min_h, :min_w]
        
        # 计算PSNR
        psnr = calculate_psnr(
            img=gt_img,
            img2=sr_img,
            crop_border=0,
            input_order='HWC',
            test_y_channel=False
        )
        return float(psnr)
    
    def _match_files(self) -> List[Tuple[Path, Path]]:
        """匹配GT和Val目录中的对应文件"""
        # 获取所有图片文件
        gt_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        val_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        gt_files = []
        for ext in gt_extensions:
            gt_files.extend(self.gt_dir.glob(ext))
            gt_files.extend(self.gt_dir.glob(ext.upper()))
        
        val_files = []
        for ext in val_extensions:
            val_files.extend(self.val_dir.glob(ext))
            val_files.extend(self.val_dir.glob(ext.upper()))
        
        # 按文件名匹配
        matched_pairs = []
        gt_dict = {f.name: f for f in gt_files}
        
        for val_file in val_files:
            if val_file.name in gt_dict:
                matched_pairs.append((gt_dict[val_file.name], val_file))
        
        return matched_pairs
    
    def run_evaluation(self, output_json_path: str = "working_tra_report_results.json") -> Dict:
        """运行完整的评估流程"""
        print("开始评估...")
        print(f"GT目录: {self.gt_dir}")
        print(f"Val目录: {self.val_dir}")
        print(f"超分辨率倍数: {self.upscale}")
        
        # 匹配文件
        matched_pairs = self._match_files()
        if not matched_pairs:
            raise ValueError("没有找到匹配的文件对")
        
        print(f"找到 {len(matched_pairs)} 对匹配的文件")
        
        # 处理每对文件
        results = []
        psnr_values = []
        
        for gt_path, val_path in tqdm(matched_pairs, desc="处理图片"):
            try:
                # 加载图片
                gt_img = self._load_img(gt_path)
                val_img = self._load_img(val_path)
                
                # 超分辨率处理
                sr_img = self._bicubic_upscale(val_img, self.upscale)
                
                # 计算PSNR
                psnr = self._calculate_psnr_for_images(gt_img, sr_img)
                
                # 记录结果
                result = {
                    "filename": gt_path.name,
                    "gt_path": str(gt_path),
                    "val_path": str(val_path),
                    "gt_size": list(gt_img.shape[:2]),
                    "val_size": list(val_img.shape[:2]),
                    "sr_size": list(sr_img.shape[:2]),
                    "psnr": psnr
                }
                results.append(result)
                psnr_values.append(psnr)
                
                print(f"{gt_path.name}: PSNR = {psnr:.4f}")
                
            except Exception as e:
                print(f"处理文件 {gt_path.name} 时出错: {str(e)}")
                continue
        
        # 计算统计信息
        if psnr_values:
            summary = {
                "average_psnr": float(np.mean(psnr_values)),
                "min_psnr": float(np.min(psnr_values)),
                "max_psnr": float(np.max(psnr_values)),
                "std_psnr": float(np.std(psnr_values)),
                "total_files": len(results),
                "successful_files": len(psnr_values)
            }
        else:
            summary = {
                "average_psnr": 0.0,
                "min_psnr": 0.0,
                "max_psnr": 0.0,
                "std_psnr": 0.0,
                "total_files": len(matched_pairs),
                "successful_files": 0
            }
        
        # 构建最终结果
        final_results = {
            "gt_dir": str(self.gt_dir),
            "val_dir": str(self.val_dir),
            "upscale": self.upscale,
            "total_files": len(matched_pairs),
            "parameters": {
                "upscale": self.upscale,
                "method": "bicubic_interpolation"
            },
            "results": results,
            "summary": summary
        }
        
        # 保存结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 评估完成 ===")
        print(f"处理文件数: {len(matched_pairs)}")
        print(f"成功处理: {summary['successful_files']}")
        print(f"平均PSNR: {summary['average_psnr']:.4f}")
        print(f"最小PSNR: {summary['min_psnr']:.4f}")
        print(f"最大PSNR: {summary['max_psnr']:.4f}")
        print(f"PSNR标准差: {summary['std_psnr']:.4f}")
        print(f"结果已保存到: {output_json_path}")
        
        return final_results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="工作版TraReport超分辨率评估")
    parser.add_argument("--gt_dir", type=str, required=True, help="真实高分辨率图片目录")
    parser.add_argument("--val_dir", type=str, required=True, help="待处理的低分辨率图片目录")
    parser.add_argument("--upscale", type=float, default=4.0, help="超分辨率倍数")
    parser.add_argument("--output", type=str, default="working_tra_report_results.json", help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 创建评估器
    tra_report = WorkingTraReport(
        gt_dir=args.gt_dir,
        val_dir=args.val_dir,
        upscale=args.upscale
    )
    
    # 运行评估
    results = tra_report.run_evaluation(args.output)
    return results


if __name__ == "__main__":
    main()
