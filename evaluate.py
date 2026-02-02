#!/usr/bin/env python
"""
MAGIC-CT Model Evaluation Script

Paper: MAGIC-CT: Multiorgan Annotation and Grounded Image Captioning in CT for Cancer
Authors: Popov et al., 2025
Dataset: https://zenodo.org/uploads/18389015
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Check critical dependencies
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    sys.exit(1)

try:
    from monai.networks.nets import SwinUNETR, UNETR, SegResNet, DynUNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        Orientationd, ScaleIntensityRanged, CropForegroundd,
        Resized, EnsureTyped
    )
    from monai.data import Dataset, DataLoader
except ImportError:
    print("ERROR: MONAI not installed. Run: pip install monai")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'swinunetr': {
        'class': SwinUNETR,
        'params': {
            'img_size': (96, 96, 96),
            'in_channels': 1,
            'out_channels': 2,
            'feature_size': 48,
            'use_checkpoint': True
        },
        'paper_dice': 72.3,
        'paper_hd95': 8.2,
        'paper_time': 2.8,
        'params_m': 61.9,
        'pretrained_url': 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt'
    },
    'unetr': {
        'class': UNETR,
        'params': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': (96, 96, 96),
            'feature_size': 16,
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'pos_embed': 'perceptron',
            'norm_name': 'instance',
            'res_block': True
        },
        'paper_dice': 68.7,
        'paper_hd95': 9.4,
        'paper_time': 2.5,
        'params_m': 102.4,
    },
    'segresnet': {
        'class': SegResNet,
        'params': {
            'blocks_down': [1, 2, 2, 4],
            'blocks_up': [1, 1, 1],
            'init_filters': 16,
            'in_channels': 1,
            'out_channels': 2,
            'dropout_prob': 0.2
        },
        'paper_dice': 65.2,
        'paper_hd95': 11.2,
        'paper_time': 1.2,
        'params_m': 15.7,
    },
    'dynunet': {
        'class': DynUNet,
        'params': {
            'spatial_dims': 3,
            'in_channels': 1,
            'out_channels': 2,
            'kernel_size': [[3, 3, 3]] * 6,
            'strides': [[1, 1, 1]] + [[2, 2, 2]] * 5,
            'upsample_kernel_size': [[2, 2, 2]] * 5,
            'norm_name': 'instance'
        },
        'paper_dice': 67.1,
        'paper_hd95': 10.1,
        'paper_time': 1.8,
        'params_m': 22.3,
    }
}

CANCER_TYPES = [
    'liver_cancer', 'renal_cancer', 'liver_cyst', 'pancreas',
    'kidney_cyst', 'lung_metastases', 'lung_cancer', 'liver_metastases'
]

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation models on MAGIC-CT dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python evaluate.py --model swinunetr --data_dir data/
  
  # Evaluate all models (reproduce paper)
  python evaluate.py --model all --data_dir data/ --device cuda
  
  # Specific cancer types
  python evaluate.py --model swinunetr --data_dir data/ \\
      --cancer_types liver_cancer lung_cancer
  
  # Save predictions
  python evaluate.py --model all --data_dir data/ --save_predictions

For more info: https://github.com/maxtrubetskoy/MagicCT
        """
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()) + ['all'],
        help='Model to evaluate (or "all" for all models)'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to dataset directory (containing scans/ and segmentations/)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory (default: results/)'
    )
    parser.add_argument(
        '--cancer_types', nargs='+', default=None,
        help=f'Cancer types to evaluate (default: all). Options: {", ".join(CANCER_TYPES)}'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--roi_size', nargs=3, type=int, default=[96, 96, 96],
        help='ROI size for resampling (default: 96 96 96)'
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save prediction masks as .nrrd files'
    )
    
    return parser.parse_args()

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: str, device: str) -> nn.Module:
    """Load model with optional pretrained weights."""
    print(f"\n{'='*70}")
    print(f"Loading model: {model_name.upper()}")
    print('='*70)
    
    config = MODEL_CONFIGS[model_name]
    
    try:
        model = config['class'](**config['params'])
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}")
        print(f"Error: {e}")
        sys.exit(1)
    
    # Try to load pretrained weights for SwinUNETR
    if model_name == 'swinunetr':
        pretrained_path = Path('pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
        if pretrained_path.exists():
            try:
                print(f"Loading pretrained weights from: {pretrained_path}")
                weight = torch.load(pretrained_path, map_location='cpu')
                model.load_from(weights=weight)
                print("✓ Pretrained weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Continuing with random initialization...")
        else:
            print(f"\nNote: Pretrained weights not found at {pretrained_path}")
            print("For better results, download from:")
            print(f"  {config['pretrained_url']}")
            print("and place in pretrained/ folder\n")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f}M")
    
    model = model.to(device)
    model.eval()
    
    return model

# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms(roi_size: Tuple[int, int, int]):
    """Get preprocessing transforms."""
    return Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        Spacingd(
            keys=['image', 'label'],
            pixdim=(1.5, 1.5, 2.0),
            mode=('bilinear', 'nearest')
        ),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(
            keys=['image', 'label'],
            spatial_size=roi_size,
            mode=('trilinear', 'nearest')
        ),
        EnsureTyped(keys=['image', 'label']),
    ])


def build_dataset(
    data_dir: Path,
    cancer_types: List[str],
    transforms
) -> List[Dict]:
    """Build dataset file list."""
    data_list = []
    scans_dir = data_dir / 'scans'
    seg_dir = data_dir / 'segmentations'
    
    # Verify directories exist
    if not scans_dir.exists():
        print(f"ERROR: Scans directory not found: {scans_dir}")
        print("Expected structure: data_dir/scans/cancer_type/*.nrrd")
        sys.exit(1)
    
    if not seg_dir.exists():
        print(f"ERROR: Segmentations directory not found: {seg_dir}")
        print("Expected structure: data_dir/segmentations/cancer_type/*.nrrd")
        sys.exit(1)
    
    for cancer_type in cancer_types:
        scan_folder = scans_dir / cancer_type
        seg_folder = seg_dir / cancer_type
        
        if not scan_folder.exists():
            print(f"Warning: {scan_folder} not found, skipping {cancer_type}...")
            continue
        
        if not seg_folder.exists():
            print(f"Warning: {seg_folder} not found, skipping {cancer_type}...")
            continue
        
        scan_files = sorted(scan_folder.glob('*.nrrd'))
        
        for scan_file in scan_files:
            patient_id = scan_file.stem
            seg_file = seg_folder / scan_file.name
            
            if not seg_file.exists():
                print(f"Warning: Segmentation not found for {patient_id}, skipping...")
                continue
            
            data_list.append({
                'patient_id': patient_id,
                'cancer_type': cancer_type,
                'image': str(scan_file),
                'label': str(seg_file),
            })
    
    return data_list

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient."""
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    
    intersection = np.sum(pred & target)
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    dice = 2 * intersection / union
    return float(dice)


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0)
) -> float:
    """Compute 95th percentile Hausdorff Distance."""
    try:
        import surface_distance
        
        if not np.any(pred) or not np.any(target):
            return float('inf')
        
        surface_distances = surface_distance.compute_surface_distances(
            target.astype(bool),
            pred.astype(bool),
            spacing_mm=spacing
        )
        hd95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
        return float(hd95)
    
    except Exception as e:
        # Fallback if surface_distance fails
        return float('nan')


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0)
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    # Binarize
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    # Flatten for classification metrics
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)
    
    # Confusion matrix components
    tp = np.sum(pred_flat & target_flat)
    fp = np.sum(pred_flat & ~target_flat)
    fn = np.sum(~pred_flat & target_flat)
    tn = np.sum(~pred_flat & ~target_flat)
    
    # Compute metrics
    metrics = {
        'dice': compute_dice(pred, target),
        'hd95': compute_hd95(pred, target, spacing),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
    }
    
    return metrics

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    save_dir: Optional[Path] = None
) -> List[Dict]:
    """Evaluate model on dataset."""
    results = []
    
    print("\nRunning inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', ncols=80):
            images = batch['image'].to(device)
            labels = batch['label']
            
            # Time inference
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            preds = model(images)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            
            # Apply activation
            if preds.shape[1] > 1:
                # Multi-class: softmax
                preds = torch.softmax(preds, dim=1)
                preds = preds[:, 1:].sum(dim=1, keepdim=True)
            else:
                # Binary: sigmoid
                preds = torch.sigmoid(preds)
            
            # Process each sample in batch
            for i in range(images.shape[0]):
                pred = preds[i, 0].cpu().numpy()
                label = labels[i, 0].numpy()
                
                # Compute metrics
                metrics = compute_metrics(pred, label, spacing=(1.5, 1.5, 2.0))
                
                # Add metadata
                metrics['patient_id'] = batch['patient_id'][i]
                metrics['cancer_type'] = batch['cancer_type'][i]
                metrics['inference_time'] = inference_time / images.shape[0]
                
                results.append(metrics)
                
                # Save prediction if requested
                if save_dir:
                    try:
                        pred_binary = (pred > 0.5).astype(np.uint8)
                        img = sitk.GetImageFromArray(pred_binary)
                        save_path = save_dir / f"{batch['patient_id'][i]}_pred.nrrd"
                        sitk.WriteImage(img, str(save_path))
                    except Exception as e:
                        print(f"Warning: Could not save prediction for {batch['patient_id'][i]}: {e}")
    
    return results

# ============================================================================
# RESULTS FORMATTING
# ============================================================================

def print_results(results: List[Dict], model_name: str):
    """Print formatted results."""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {model_name.upper()}")
    print('='*70)
    
    # Overall metrics
    print("\nOverall Performance:")
    print('-'*70)
    
    for metric in ['dice', 'hd95', 'sensitivity', 'specificity', 'precision']:
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(values) == 0:
            continue
        
        mean = values.mean()
        std = values.std()
        
        if metric == 'hd95':
            print(f"  {metric.upper():15s}: {mean:6.2f} ± {std:5.2f} mm")
        else:
            print(f"  {metric.upper():15s}: {mean*100:6.2f} ± {std*100:5.2f} %")
    
    # Inference time
    avg_time = df['inference_time'].mean()
    print(f"  {'Inference Time':15s}: {avg_time:6.3f} s/volume")
    
    # Per cancer type breakdown
    print(f"\n{'-'*70}")
    print("Performance by Cancer Type:")
    print('-'*70)
    
    for cancer_type in sorted(df['cancer_type'].unique()):
        subset = df[df['cancer_type'] == cancer_type]
        dice_mean = subset['dice'].mean() * 100
        dice_std = subset['dice'].std() * 100
        n = len(subset)
        print(f"  {cancer_type:20s}: {dice_mean:5.1f} ± {dice_std:4.1f} %  (n={n})")
    
    # Compare with paper
    config = MODEL_CONFIGS[model_name]
    paper_dice = config['paper_dice']
    obtained_dice = df['dice'].mean() * 100
    difference = obtained_dice - paper_dice
    
    print(f"\n{'-'*70}")
    print("Comparison with Paper:")
    print('-'*70)
    print(f"  Paper Dice:      {paper_dice:5.1f} %")
    print(f"  Obtained Dice:   {obtained_dice:5.1f} %")
    print(f"  Difference:      {difference:+5.1f} %")
    
    if abs(difference) < 3.0:
        print(f"  Status:          ✓ Results match paper (within 3%)")
    else:
        print(f"  Status:          ⚠ Deviation > 3% (check pretrained weights)")
    
    print('='*70 + '\n')


def save_results(results: List[Dict], output_dir: Path, model_name: str):
    """Save results to files."""
    # Create output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-case results
    df = pd.DataFrame(results)
    csv_path = model_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved per-case results to: {csv_path}")
    
    # Compute and save summary statistics
    summary = {}
    for metric in ['dice', 'hd95', 'sensitivity', 'specificity', 'precision']:
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            summary[metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            }
    
    summary['inference_time'] = {
        'mean': float(df['inference_time'].mean())
    }
    
    # Add paper comparison
    config = MODEL_CONFIGS[model_name]
    summary['paper_comparison'] = {
        'paper_dice': config['paper_dice'],
        'obtained_dice': float(df['dice'].mean() * 100),
        'difference': float(df['dice'].mean() * 100 - config['paper_dice'])
    }
    
    json_path = model_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to: {json_path}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "MAGIC-CT MODEL EVALUATION")
    print("="*70)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please check --data_dir path")
        sys.exit(1)
    
    # Get cancer types
    cancer_types = args.cancer_types or CANCER_TYPES
    print(f"\nCancer types: {', '.join(cancer_types)}")
    
    # Build dataset
    print(f"\nBuilding dataset from: {data_dir}")
    transforms = get_transforms(tuple(args.roi_size))
    data_list = build_dataset(data_dir, cancer_types, transforms)
    
    if len(data_list) == 0:
        print("\nERROR: No data found!")
        print("Expected directory structure:")
        print("  data_dir/")
        print("    scans/")
        print("      liver_cancer/")
        print("        patient_001.nrrd")
        print("        ...")
        print("    segmentations/")
        print("      liver_cancer/")
        print("        patient_001.nrrd")
        print("        ...")
        sys.exit(1)
    
    print(f"Found {len(data_list)} samples")
    
    # Create dataset and dataloader
    dataset = Dataset(data=data_list, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False
    )
    
    # Determine models to evaluate
    if args.model == 'all':
        models_to_eval = list(MODEL_CONFIGS.keys())
        print(f"\nEvaluating all {len(models_to_eval)} models")
    else:
        models_to_eval = [args.model]
        print(f"\nEvaluating: {args.model}")
    
    # Evaluate each model
    all_results = {}
    
    for idx, model_name in enumerate(models_to_eval):
        print(f"\n{'#'*70}")
        print(f"# MODEL {idx+1}/{len(models_to_eval)}: {model_name.upper()}")
        print('#'*70)
        
        # Load model
        model = load_model(model_name, device)
        
        # Setup save directory for predictions
        save_dir = None
        if args.save_predictions:
            save_dir = output_dir / model_name / 'predictions'
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Predictions will be saved to: {save_dir}")
        
        # Evaluate
        results = evaluate_model(model, dataloader, device, save_dir)
        
        # Save results
        save_results(results, output_dir, model_name)
        
        # Store for comparison
        all_results[model_name] = results
        
        # Print results
        print_results(results, model_name)
        
        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Create comparison table if multiple models evaluated
    if len(models_to_eval) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON (Table 2 from Paper)")
        print('='*70 + '\n')
        
        comparison = []
        for model_name in models_to_eval:
            df = pd.DataFrame(all_results[model_name])
            config = MODEL_CONFIGS[model_name]
            
            dice = df['dice'].mean() * 100
            dice_std = df['dice'].std() * 100
            
            hd95_vals = df['hd95'].replace([np.inf, -np.inf], np.nan).dropna()
            hd95 = hd95_vals.mean() if len(hd95_vals) > 0 else 0
            hd95_std = hd95_vals.std() if len(hd95_vals) > 0 else 0
            
            inf_time = df['inference_time'].mean()
            
            comparison.append({
                'Model': model_name.upper(),
                'Dice (%)': f'{dice:.1f} ± {dice_std:.1f}',
                'HD95 (mm)': f'{hd95:.1f} ± {hd95_std:.1f}',
                'Time (s)': f'{inf_time:.2f}',
                'Params (M)': f"{config.get('params_m', 'N/A')}",
                'Paper Dice (%)': f"{config['paper_dice']:.1f}"
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
        
        comp_path = output_dir / 'comparison.csv'
        comp_df.to_csv(comp_path, index=False)
        print(f"\n{'='*70}")
        print(f"Comparison table saved to: {comp_path}")
        print('='*70 + '\n')
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nOutput structure:")
    for model_name in models_to_eval:
        print(f"  {output_dir}/{model_name}/")
        print(f"    ├── results.csv       (per-case metrics)")
        print(f"    ├── summary.json      (aggregate statistics)")
        if args.save_predictions:
            print(f"    └── predictions/      (prediction masks)")
    
    if len(models_to_eval) > 1:
        print(f"  {output_dir}/comparison.csv   (model comparison)")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
