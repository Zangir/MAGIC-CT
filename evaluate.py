#!/usr/bin/env python
"""
MAGIC-CT Model Evaluation Script
Evaluate segmentation models on the MAGIC-CT dataset.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

from monai.networks.nets import SwinUNETR, UNETR, SegResNet, DynUNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, \
    Orientationd, ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped
from monai.data import Dataset, DataLoader

# Model configurations
MODEL_CONFIGS = {
    'swinunetr': {
        'class': SwinUNETR,
        'params': {'img_size': (96,96,96), 'in_channels': 1, 'out_channels': 2, 'feature_size': 48},
        'paper_dice': 72.3,
        'pretrained_url': 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt'
    },
    'unetr': {
        'class': UNETR,
        'params': {'in_channels': 1, 'out_channels': 2, 'img_size': (96,96,96), 'feature_size': 16, 
                   'hidden_size': 768, 'mlp_dim': 3072, 'num_heads': 12},
        'paper_dice': 68.7,
    },
    'segresnet': {
        'class': SegResNet,
        'params': {'in_channels': 1, 'out_channels': 2, 'init_filters': 16},
        'paper_dice': 65.2,
    },
    'dynunet': {
        'class': DynUNet,
        'params': {
            'spatial_dims': 3, 'in_channels': 1, 'out_channels': 2,
            'kernel_size': [[3,3,3]]*6, 'strides': [[1,1,1]] + [[2,2,2]]*5,
            'upsample_kernel_size': [[2,2,2]]*5
        },
        'paper_dice': 67.1,
    }
}

CANCER_TYPES = ['liver_cancer', 'renal_cancer', 'liver_cyst', 'pancreas',
                'kidney_cyst', 'lung_metastases', 'lung_cancer', 'liver_metastases']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on MAGIC-CT')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODEL_CONFIGS.keys()) + ['all'],
                       help='Model to evaluate (or "all" for all models)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--cancer_types', nargs='+', default=None,
                       help='Cancer types to evaluate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--roi_size', nargs=3, type=int, default=[96,96,96],
                       help='ROI size')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    return parser.parse_args()


def load_model(model_name: str, device: str) -> nn.Module:
    """Load model with optional pretrained weights."""
    config = MODEL_CONFIGS[model_name]
    model = config['class'](**config['params'])
    
    # Try to load pretrained weights for SwinUNETR
    if model_name == 'swinunetr':
        pretrained_path = Path('pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
        if pretrained_path.exists():
            print(f"Loading pretrained weights from {pretrained_path}")
            weight = torch.load(pretrained_path, map_location='cpu')
            model.load_from(weights=weight)
        else:
            print(f"Pretrained weights not found. Download from:\n{config['pretrained_url']}")
            print("Place in pretrained/ folder for better results")
    
    model = model.to(device)
    model.eval()
    return model


def get_transforms(roi_size: Tuple[int, int, int]):
    """Get preprocessing transforms."""
    return Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys=['image'], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=['image', 'label'], spatial_size=roi_size, mode=('trilinear', 'nearest')),
        EnsureTyped(keys=['image', 'label']),
    ])


def build_dataset(data_dir: Path, cancer_types: List[str], transforms) -> List[Dict]:
    """Build dataset file list."""
    data_list = []
    scans_dir = data_dir / 'scans'
    seg_dir = data_dir / 'segmentations'
    
    for cancer_type in cancer_types:
        scan_folder = scans_dir / cancer_type
        seg_folder = seg_dir / cancer_type
        
        if not scan_folder.exists():
            print(f"Warning: {scan_folder} not found, skipping...")
            continue
        
        for scan_file in sorted(scan_folder.glob('*.nrrd')):
            patient_id = scan_file.stem
            seg_file = seg_folder / scan_file.name
            
            if not seg_file.exists():
                continue
            
            data_list.append({
                'patient_id': patient_id,
                'cancer_type': cancer_type,
                'image': str(scan_file),
                'label': str(seg_file),
            })
    
    return data_list


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    intersection = np.sum(pred & target)
    if pred.sum() + target.sum() == 0:
        return 1.0
    return 2 * intersection / (pred.sum() + target.sum())


def compute_hd95(pred: np.ndarray, target: np.ndarray, spacing=(1.5,1.5,2.0)) -> float:
    """Compute 95th percentile Hausdorff distance."""
    try:
        import surface_distance
        if not np.any(pred) or not np.any(target):
            return float('inf')
        surface_distances = surface_distance.compute_surface_distances(
            target.astype(bool), pred.astype(bool), spacing_mm=spacing
        )
        return surface_distance.compute_robust_hausdorff(surface_distances, 95)
    except:
        return float('nan')


def compute_metrics(pred: np.ndarray, target: np.ndarray, spacing=(1.5,1.5,2.0)) -> Dict:
    """Compute all metrics."""
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)
    
    tp = np.sum(pred_flat & target_flat)
    fp = np.sum(pred_flat & ~target_flat)
    fn = np.sum(~pred_flat & target_flat)
    tn = np.sum(~pred_flat & ~target_flat)
    
    return {
        'dice': compute_dice(pred, target),
        'hd95': compute_hd95(pred, target, spacing),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str, 
                  save_dir: Optional[Path] = None) -> List[Dict]:
    """Evaluate model on dataset."""
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label']
            
            start_time = time.time()
            preds = model(images)
            inference_time = time.time() - start_time
            
            if preds.shape[1] > 1:
                preds = torch.softmax(preds, dim=1)[:, 1:].sum(dim=1, keepdim=True)
            else:
                preds = torch.sigmoid(preds)
            
            for i in range(images.shape[0]):
                pred = preds[i, 0].cpu().numpy()
                label = labels[i, 0].numpy()
                
                metrics = compute_metrics(pred, label)
                metrics['patient_id'] = batch['patient_id'][i]
                metrics['cancer_type'] = batch['cancer_type'][i]
                metrics['inference_time'] = inference_time / images.shape[0]
                
                results.append(metrics)
                
                # Save prediction if requested
                if save_dir:
                    pred_binary = (pred > 0.5).astype(np.uint8)
                    img = sitk.GetImageFromArray(pred_binary)
                    sitk.WriteImage(img, str(save_dir / f"{batch['patient_id'][i]}_pred.nrrd"))
    
    return results


def print_results(results: List[Dict], model_name: str):
    """Print formatted results."""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"Results for {model_name.upper()}")
    print('='*70)
    
    # Overall metrics
    for metric in ['dice', 'hd95', 'sensitivity', 'specificity']:
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            if metric == 'hd95':
                print(f"{metric.upper():15s}: {mean:6.2f} ± {std:5.2f} mm")
            else:
                print(f"{metric.upper():15s}: {mean*100:6.2f} ± {std*100:5.2f} %")
    
    avg_time = df['inference_time'].mean()
    print(f"{'Inference Time':15s}: {avg_time:6.3f} s")
    
    # Per cancer type
    print(f"\n{'-'*70}")
    print("Per Cancer Type:")
    print('-'*70)
    for cancer_type in df['cancer_type'].unique():
        subset = df[df['cancer_type'] == cancer_type]
        dice = subset['dice'].mean()
        print(f"{cancer_type:20s}: Dice = {dice*100:5.1f}%, n = {len(subset)}")
    
    # Compare with paper
    paper_dice = MODEL_CONFIGS[model_name]['paper_dice']
    obtained_dice = df['dice'].mean() * 100
    print(f"\n{'-'*70}")
    print(f"Paper Dice:     {paper_dice:.1f}%")
    print(f"Obtained Dice:  {obtained_dice:.1f}%")
    print(f"Difference:     {obtained_dice - paper_dice:+.1f}%")
    print('='*70 + '\n')


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get cancer types
    cancer_types = args.cancer_types or CANCER_TYPES
    print(f"Evaluating cancer types: {cancer_types}")
    
    # Build dataset
    print("Building dataset...")
    transforms = get_transforms(tuple(args.roi_size))
    data_list = build_dataset(data_dir, cancer_types, transforms)
    print(f"Found {len(data_list)} samples")
    
    if len(data_list) == 0:
        print("ERROR: No data found. Check --data_dir path.")
        return
    
    # Create dataset and dataloader
    dataset = Dataset(data=data_list, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    # Determine models to evaluate
    models_to_eval = list(MODEL_CONFIGS.keys()) if args.model == 'all' else [args.model]
    
    # Evaluate each model
    all_results = {}
    for model_name in models_to_eval:
        print(f"\n{'#'*70}")
        print(f"# Evaluating: {model_name.upper()}")
        print('#'*70)
        
        # Load model
        model = load_model(model_name, device)
        
        # Setup save directory
        save_dir = None
        if args.save_predictions:
            save_dir = output_dir / model_name / 'predictions'
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        results = evaluate_model(model, dataloader, device, save_dir)
        
        # Save results
        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(model_output / 'results.csv', index=False)
        
        # Compute summary statistics
        summary = {}
        for metric in ['dice', 'hd95', 'sensitivity', 'specificity']:
            values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            summary[metric] = {'mean': float(values.mean()), 'std': float(values.std())}
        
        with open(model_output / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        all_results[model_name] = results
        
        # Print results
        print_results(results, model_name)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Create comparison table if multiple models
    if len(models_to_eval) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON (Table 2 from Paper)")
        print('='*70)
        
        comparison = []
        for model_name in models_to_eval:
            df = pd.DataFrame(all_results[model_name])
            dice = df['dice'].mean() * 100
            dice_std = df['dice'].std() * 100
            hd95_vals = df['hd95'].replace([np.inf, -np.inf], np.nan).dropna()
            hd95 = hd95_vals.mean() if len(hd95_vals) > 0 else 0
            hd95_std = hd95_vals.std() if len(hd95_vals) > 0 else 0
            
            comparison.append({
                'Model': model_name.upper(),
                'Dice (%)': f'{dice:.1f} ± {dice_std:.1f}',
                'HD95 (mm)': f'{hd95:.1f} ± {hd95_std:.1f}',
                'Paper Dice (%)': f"{MODEL_CONFIGS[model_name]['paper_dice']:.1f}"
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
        comp_df.to_csv(output_dir / 'comparison.csv', index=False)
        print(f"\n{'='*70}\n")
    
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
