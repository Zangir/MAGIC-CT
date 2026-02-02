# MAGIC-CT GitHub Setup Guide

## Complete Setup Steps

### 1. Create GitHub Repository

```bash
# On GitHub, create new repository: MagicCT
# Don't initialize with README (we have our own)
```

### 2. Initialize and Push Code

```bash
cd magic-ct
git init
git add .
git commit -m "Initial commit: MAGIC-CT evaluation scripts"
git branch -M main
git remote add origin https://github.com/maxtrubetskoy/MagicCT.git
git push -u origin main
```

### 3. Users Can Clone and Setup

```bash
# Clone
git clone https://github.com/maxtrubetskoy/MagicCT.git
cd MagicCT

# Install dependencies
pip install -r requirements.txt

# Download dataset from Zenodo
# https://zenodo.org/uploads/18389015
# Extract to data/ folder
```

### 4. Running Evaluations

```bash
# Single model
python evaluate.py --model swinunetr --data_dir data/

# All models (reproduce paper)
python evaluate.py --model all --data_dir data/

# With GPU
python evaluate.py --model all --data_dir data/ --device cuda

# Specific cancer types
python evaluate.py --model swinunetr --data_dir data/ --cancer_types liver_cancer lung_cancer

# Save predictions
python evaluate.py --model swinunetr --data_dir data/ --save_predictions
```

## File Structure

```
MagicCT/
├── README.md              # Main documentation
├── requirements.txt       # Dependencies
├── evaluate.py           # Main evaluation script
├── .gitignore           # Git ignore rules
└── SETUP_GUIDE.md       # This file
```

## Key Features

✅ **Compact**: Only 4 essential files
✅ **Simple**: Single evaluation script
✅ **Complete**: Reproduces all paper results
✅ **Flexible**: Easy to extend for custom models
✅ **Fast**: Efficient data loading and processing

## Expected Output

```
results/
├── swinunetr/
│   ├── results.csv       # Per-case metrics
│   ├── summary.json      # Aggregate statistics
│   └── predictions/      # (if --save_predictions)
├── unetr/
├── segresnet/
├── dynunet/
└── comparison.csv        # Model comparison table
```

## Pretrained Weights (Optional)

For best results with SwinUNETR:

```bash
mkdir pretrained
cd pretrained
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
```

## Troubleshooting

**CUDA out of memory:**
```bash
python evaluate.py --model swinunetr --data_dir data/ --batch_size 1
```

**Can't find data:**
- Ensure data is extracted to `data/` with subfolders `scans/` and `segmentations/`
- Check that cancer type folders exist (e.g., `data/scans/liver_cancer/`)

**Missing dependencies:**
```bash
pip install torch monai nibabel SimpleITK scikit-learn surface-distance
```

## Contact

Questions? Contact: maxim.popov@nu.edu.kz
