#!/usr/bin/env python3
"""
Generate CSV report from inference results metrics.
This script reads metrics.json files from edge and no_edge directories
and generates a comprehensive CSV report matching the table format.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def parse_metrics_file(metrics_file_path):
    """Parse a metrics.json file and return the data."""
    try:
        with open(metrics_file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {metrics_file_path}: {e}")
        return None

def extract_path_info(file_path, results_base_path):
    """Extract edge type and epoch information from file path."""
    rel_path = os.path.relpath(file_path, results_base_path)
    path_parts = rel_path.split(os.sep)
    
    if len(path_parts) < 3:
        return None, None
    
    edge_type = path_parts[-3]  # edge or no_edge
    epoch_dir = path_parts[-2]  # epochs_47, epochs_95, etc.
    
    return edge_type, epoch_dir

def get_column_name(edge_type, epoch_dir):
    """Get the column name based on edge type and epoch directory."""
    if edge_type == "edge":
        epoch_num = epoch_dir.replace("epochs_", "")
        return f"Epoch {epoch_num} (edge)"
    elif edge_type == "no_edge":
        epoch_num = epoch_dir.replace("epochs_", "")
        return f"Epoch {epoch_num} (no edge)"
    else:
        return None

def collect_all_metrics(results_path):
    """Collect all metrics from the results directory."""
    metrics_data = defaultdict(dict)  # {metric_type: {column: {filename: value}}}
    image_files = set()
    
    # Find all metrics.json files
    metrics_files = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file == "metrics.json":
                metrics_files.append(os.path.join(root, file))
    
    print(f"Found {len(metrics_files)} metrics.json files")
    
    for metrics_file in metrics_files:
        print(f"Processing: {metrics_file}")
        
        # Extract path information
        edge_type, epoch_dir = extract_path_info(metrics_file, results_path)
        if not edge_type or not epoch_dir:
            print(f"  ⚠ Skipping: Invalid path format")
            continue
        
        column_name = get_column_name(edge_type, epoch_dir)
        if not column_name:
            print(f"  ⚠ Skipping: Unknown column type")
            continue
        
        print(f"  Edge type: {edge_type}, Epoch: {epoch_dir} -> Column: {column_name}")
        
        # Parse metrics
        data = parse_metrics_file(metrics_file)
        if not data:
            continue
        
        # Extract average values
        avg_psnr = data.get('average_psnr')
        avg_ssim = data.get('average_ssim')
        avg_lpips = data.get('average_lpips')
        
        if avg_psnr is not None:
            metrics_data['PSNR'][column_name] = {'Average': avg_psnr}
        if avg_ssim is not None:
            metrics_data['SSIM'][column_name] = {'Average': avg_ssim}
        if avg_lpips is not None:
            metrics_data['LPIPS'][column_name] = {'Average': avg_lpips}
        
        # Extract individual image values
        for img_data in data.get('images', []):
            img_name = img_data.get('image_name')
            if not img_name:
                continue
            
            image_files.add(img_name)
            
            psnr_val = img_data.get('psnr')
            ssim_val = img_data.get('ssim')
            lpips_val = img_data.get('lpips')
            
            if psnr_val is not None:
                if 'PSNR' not in metrics_data:
                    metrics_data['PSNR'] = {}
                if column_name not in metrics_data['PSNR']:
                    metrics_data['PSNR'][column_name] = {}
                metrics_data['PSNR'][column_name][img_name] = psnr_val
            
            if ssim_val is not None:
                if 'SSIM' not in metrics_data:
                    metrics_data['SSIM'] = {}
                if column_name not in metrics_data['SSIM']:
                    metrics_data['SSIM'][column_name] = {}
                metrics_data['SSIM'][column_name][img_name] = ssim_val
            
            if lpips_val is not None:
                if 'LPIPS' not in metrics_data:
                    metrics_data['LPIPS'] = {}
                if column_name not in metrics_data['LPIPS']:
                    metrics_data['LPIPS'][column_name] = {}
                metrics_data['LPIPS'][column_name][img_name] = lpips_val
    
    return metrics_data, sorted(image_files)

def format_value(value):
    """Format numeric values to 2 decimal places."""
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return str(value)

def generate_csv_report(metrics_data, image_files, output_path):
    """Generate the final CSV report."""
    
    # Define column order (based on the image format)
    # Edge and no edge results for the same epoch should be adjacent
    column_order = [
        "StableSR",
        "Epoch 47 (edge)",
        "Epoch 47 (no edge)",
        "Epoch 95 (edge)", 
        "Epoch 95 (no edge)",
        "Epoch 142 (edge)",
        "Epoch 142 (no edge)",
        "Epoch 190 (edge)",
        "Epoch 190 (no edge)",
        "Epoch 238 (edge)",
        "Epoch 238 (no edge)",
        "Epoch 285 (edge)",
        "Epoch 285 (no edge)"
    ]
    
    # Find columns that actually have data
    columns_with_data = set()
    for metric_type in metrics_data:
        for column in metrics_data[metric_type]:
            # Check if this column has any non-empty values
            has_data = False
            if "Average" in metrics_data[metric_type][column]:
                avg_val = metrics_data[metric_type][column]["Average"]
                if avg_val != "" and avg_val is not None:
                    has_data = True
            
            if not has_data:
                # Check individual image data
                for img_file in image_files:
                    if img_file in metrics_data[metric_type][column]:
                        val = metrics_data[metric_type][column][img_file]
                        if val != "" and val is not None:
                            has_data = True
                            break
            
            if has_data:
                columns_with_data.add(column)
    
    # Build final column list based on predefined order, only including columns with data
    final_columns = []
    for col in column_order:
        if col in columns_with_data:
            final_columns.append(col)
            columns_with_data.remove(col)
    final_columns.extend(sorted(columns_with_data))
    
    # Create two-row header structure
    def create_header_rows():
        """Create two-row header for better readability."""
        # First row: epoch information
        first_row = ["Metric", "Filename"]
        # Second row: edge/no edge information
        second_row = ["", ""]
        
        for col in final_columns:
            if col == "StableSR":
                first_row.append("StableSR")
                second_row.append("")
            else:
                # Parse epoch and edge type
                if "(edge)" in col:
                    epoch_part = col.replace(" (edge)", "")
                    first_row.append(epoch_part)
                    second_row.append("edge")
                elif "(no edge)" in col:
                    epoch_part = col.replace(" (no edge)", "")
                    first_row.append(epoch_part)
                    second_row.append("no edge")
                else:
                    first_row.append(col)
                    second_row.append("")
        
        return first_row, second_row
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write two-row header
        first_header, second_header = create_header_rows()
        writer.writerow(first_header)
        writer.writerow(second_header)
        
        # Write data for each metric type
        metric_types = ["PSNR", "SSIM", "LPIPS"]
        for i, metric_type in enumerate(metric_types):
            if metric_type not in metrics_data:
                continue
            
            # Add empty row before each metric (except the first one)
            if i > 0:
                writer.writerow([])
            
            # Write average row - metric name only in first row
            avg_row = [metric_type, "Average"]
            for col in final_columns:
                value = ""
                if col in metrics_data[metric_type]:
                    value = format_value(metrics_data[metric_type][col].get("Average", ""))
                avg_row.append(value)
            writer.writerow(avg_row)
            
            # Write individual image rows - empty metric column for subsequent rows
            for img_file in image_files:
                img_row = ["", img_file]  # Empty metric name for image rows
                for col in final_columns:
                    value = ""
                    if col in metrics_data[metric_type]:
                        value = format_value(metrics_data[metric_type][col].get(img_file, ""))
                    img_row.append(value)
                writer.writerow(img_row)

def main():
    parser = argparse.ArgumentParser(description="Generate CSV report from inference results")
    parser.add_argument("results_path", help="Path to inference results directory")
    parser.add_argument("--output", "-o", help="Output CSV file path (default: inference_report.csv in results directory)")
    
    args = parser.parse_args()
    
    results_path = args.results_path
    if not os.path.exists(results_path):
        print(f"Error: Results directory does not exist: {results_path}")
        sys.exit(1)
    
    if args.output:
        output_path = args.output
    else:
        # Extract directory name for filename
        dir_name = os.path.basename(os.path.normpath(results_path))
        output_filename = f"{dir_name}_inference_report.csv"
        output_path = os.path.join(results_path, output_filename)
    
    print(f"Scanning results directory: {results_path}")
    print(f"Output will be saved to: {output_path}")
    print()
    
    # Collect all metrics
    metrics_data, image_files = collect_all_metrics(results_path)
    
    if not metrics_data:
        print("No metrics data found!")
        sys.exit(1)
    
    print(f"\nFound data for {len(image_files)} images:")
    for img in image_files:
        print(f"  - {img}")
    
    print(f"\nFound metrics columns:")
    for metric_type in metrics_data:
        print(f"  {metric_type}: {list(metrics_data[metric_type].keys())}")
    
    # Generate report
    print(f"\nGenerating CSV report...")
    generate_csv_report(metrics_data, image_files, output_path)
    
    print(f"✓ Report generated successfully: {output_path}")
    
    # Show summary
    print(f"\nReport summary:")
    print(f"- Metrics: {', '.join(metrics_data.keys())}")
    print(f"- Images: {len(image_files)}")
    print(f"- Columns: {len(set().union(*[metrics_data[m].keys() for m in metrics_data]))}")
    print(f"- Total rows: {len(metrics_data) * (1 + len(image_files))}")

if __name__ == "__main__":
    main()
