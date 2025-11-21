"""
评估模型输出的深度估计效果
Evaluate depth estimation models on NYU2 test dataset
Includes all metrics from BTS evaluation (d1-d3, RMSE, Log errors, Rel errors)
Fixed: Added JSON serialization for Numpy types
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import struct
import json
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """
    用于解决 json.dump 无法序列化 numpy 数据类型的问题
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def read_pfm(file_path):
    """
    读取PFM (Portable Float Map) 文件
    Returns: numpy array of depth values
    """
    with open(file_path, 'rb') as f:
        # Read header
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Read dimensions
        dim_match = f.readline().decode('utf-8')
        width, height = map(int, dim_match.split())

        # Read scale
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # Read data
        data = np.fromfile(f, endian + 'f')

    # Reshape
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)  # PFM stores bottom-to-top

    return data, scale


def compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=10.0):
    """
    计算完整的深度估计评估指标
    基于 example_eval.py 的实现
    """
    # Create valid mask
    valid_mask = (gt > min_depth) & (gt < max_depth)

    if valid_mask.sum() == 0:
        return None

    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # Align prediction to ground truth using scale+shift (matches training!)
    # Solve: scale*pred + shift = gt (least squares)
    A = np.stack([pred, np.ones_like(pred)], axis=1)  # [N, 2]
    b = gt[:, np.newaxis]  # [N, 1]
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        scale, shift = x[0, 0], x[1, 0]
        pred = scale * pred + shift
    except:
        # Fallback to median scaling if lstsq fails
        if np.median(pred) > 0:
            pred = pred * np.median(gt) / np.median(pred)
        else:
            return None

    # Filter out zero/negative values
    valid_pred_mask = pred > 0
    valid_gt_mask = gt > 0
    valid_both = valid_pred_mask & valid_gt_mask
    
    if valid_both.sum() == 0:
        return None

    pred = pred[valid_both]
    gt = gt[valid_both]

    # --- Metrics Calculation ---

    # Threshold metrics (d1, d2, d3)
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    # RMSE
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # RMSE log
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Absolute Relative Error
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    # Squared Relative Error
    sq_rel = np.mean(((gt - pred)**2) / gt)

    # SILog (Scale Invariant Logarithmic Error)
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    # log10 Error
    err_log10 = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err_log10)

    return {
        'd1': d1,
        'd2': d2,
        'd3': d3,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'log10': log10,
        'silog': silog
    }


def evaluate_model(model_name, output_dir, test_csv):
    """
    评估单个模型的性能
    """
    print(f"\n{'='*70}")
    print(f"评估模型: {model_name}")
    print(f"{'='*70}")

    # Read test CSV
    try:
        test_data = pd.read_csv(test_csv, header=None, names=['rgb', 'depth'])
    except Exception as e:
        print(f"无法读取测试CSV: {e}")
        return None

    # Model output directory
    model_output_dir = os.path.join(output_dir, model_name)

    if not os.path.exists(model_output_dir):
        print(f"错误: 模型输出目录不存在: {model_output_dir}")
        return None

    # Collect metrics
    all_metrics = []
    not_found = []
    errors = []

    print(f"开始评估 {len(test_data)} 个测试样本...")

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        rgb_path = row['rgb']
        depth_gt_path = row['depth']

        img_name = os.path.basename(rgb_path)
        img_idx = img_name.split('_')[0]

        pred_pfm_path = os.path.join(model_output_dir, f"{img_idx}_colors-{model_name}.pfm")

        if not os.path.exists(pred_pfm_path):
            not_found.append(img_idx)
            continue

        try:
            pred_depth, _ = read_pfm(pred_pfm_path)
            gt_depth = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED)

            if gt_depth is None:
                errors.append(f"{img_idx}: 无法读取真实深度图")
                continue

            gt_depth = gt_depth.astype(np.float32) / 1000.0

            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

            metrics = compute_depth_metrics(pred_depth, gt_depth)

            if metrics is not None:
                all_metrics.append(metrics)

        except Exception as e:
            errors.append(f"{img_idx}: {str(e)}")
            continue

    if len(all_metrics) == 0:
        print("\n❌ 没有成功评估的样本!")
        return None

    print(f"\n✓ 成功评估: {len(all_metrics)} 个样本")
    if not_found:
        print(f"⚠ 未找到预测: {len(not_found)} 个样本")
    if errors:
        print(f"⚠ 评估错误: {len(errors)} 个样本")

    # Calculate mean metrics
    mean_metrics = {}
    for key in all_metrics[0].keys():
        # 使用 np.mean 计算均值
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])

    return mean_metrics


def print_metrics(model_name, metrics):
    """打印评估指标"""
    if metrics is None:
        print(f"\n{model_name}: 评估失败")
        return

    print(f"\n{'='*70}")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'值':>15} {'说明':>20}")
    print("-" * 70)
    
    print(f"{'d1 (δ < 1.25)':<20} {metrics['d1']:>15.4f} {'(越高越好)':>20}")
    print(f"{'d2 (δ < 1.25^2)':<20} {metrics['d2']:>15.4f} {'(越高越好)':>20}")
    print(f"{'d3 (δ < 1.25^3)':<20} {metrics['d3']:>15.4f} {'(越高越好)':>20}")
    print("-" * 70)
    
    print(f"{'AbsRel':<20} {metrics['abs_rel']:>15.4f} {'(越低越好)':>20}")
    print(f"{'SqRel':<20} {metrics['sq_rel']:>15.4f} {'(越低越好)':>20}")
    print(f"{'RMSE':<20} {metrics['rmse']:>15.4f} {'(越低越好)':>20}")
    print(f"{'RMSElog':<20} {metrics['rmse_log']:>15.4f} {'(越低越好)':>20}")
    print(f"{'SILog':<20} {metrics['silog']:>15.4f} {'(越低越好)':>20}")
    print(f"{'log10':<20} {metrics['log10']:>15.4f} {'(越低越好)':>20}")
    print("=" * 70)


def main():
    # Paths
    output_dir = '../output'
    test_csv = '../input/nyu2_test.csv'

    if not os.path.exists(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}")
        sys.exit(1)

    model_dirs = [d for d in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith('.')]

    if len(model_dirs) == 0:
        print(f"错误: 在 {output_dir} 中没有找到模型输出目录")
        sys.exit(1)

    print(f"找到 {len(model_dirs)} 个模型: {', '.join(model_dirs)}")

    # Evaluate each model
    results = {}
    for model_name in model_dirs:
        metrics = evaluate_model(model_name, output_dir, test_csv)
        results[model_name] = metrics

    valid_models = {k: v for k, v in results.items() if v is not None}

    if len(valid_models) == 0:
        print("没有成功评估的模型")
        return

    # Print individual results
    for model_name, metrics in valid_models.items():
        print_metrics(model_name, metrics)

    # Print comparison table if multiple models
    if len(valid_models) > 1:
        print("\n\n" + "="*100)
        print("模型对比总结 (Model Comparison)")
        print("="*100)

        table_metrics = [
            ('d1', 'd1 (↑)', '{:.3f}'),
            ('abs_rel', 'AbsRel (↓)', '{:.3f}'),
            ('rmse', 'RMSE (↓)', '{:.3f}'),
            ('silog', 'SILog (↓)', '{:.3f}')
        ]

        header = f"{'Model':<20}"
        for _, name, _ in table_metrics:
            header += f"{name:>15}"
        print(header)
        print("-" * len(header))

        for model_name, metrics in valid_models.items():
            row = f"{model_name:<20}"
            for key, _, fmt in table_metrics:
                row += f"{fmt.format(metrics[key]):>15}"
            print(row)
        print("=" * len(header))
        print("↑: 越高越好 (Higher is better), ↓: 越低越好 (Lower is better)")

        best_model = max(valid_models.keys(), key=lambda k: valid_models[k]['d1'])
        print(f"\n综合最佳模型 (Based on d1): {best_model}")

    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"deep_evaluation_results_{timestamp}.json"

    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'test_csv': test_csv,
        'models': valid_models
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        # 修复: 添加 cls=NumpyEncoder 以处理 float32 错误
        json.dump(results_to_save, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()