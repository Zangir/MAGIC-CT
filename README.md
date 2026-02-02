# MAGIC-CT Evaluation

Evaluation scripts for MAGIC-CT: Multiorgan Annotation and Grounded Image Captioning in CT for Cancer.

## 📊 Dataset

Download the dataset from Zenodo: **https://zenodo.org/uploads/18389015**

Expected structure after extraction:
```
data/
├── scans/
│   ├── liver_cancer/
│   ├── kidney_cyst/
│   └── ...
└── segmentations/
    ├── liver_cancer/
    └── ...
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Evaluate Single Model

```bash
python evaluate.py --model swinunetr --data_dir data/ --output results/
```

### Evaluate All Models (Reproduce Paper Results)

```bash
python evaluate.py --model all --data_dir data/ --output results/
```

### Specific Cancer Types

```bash
python evaluate.py --model swinunetr --data_dir data/ --cancer_types liver_cancer lung_cancer
```

## 📋 Command-Line Options

```bash
python evaluate.py --help
```

**Main arguments:**
- `--model`: Model name (`swinunetr`, `unetr`, `segresnet`, `dynunet`, or `all`)
- `--data_dir`: Path to dataset directory
- `--output`: Output directory for results (default: `results/`)
- `--cancer_types`: Specific cancer types to evaluate (default: all)
- `--device`: Device (`cuda` or `cpu`, default: auto-detect)
- `--batch_size`: Batch size (default: 1)
- `--workers`: Number of data loading workers (default: 4)
- `--save_predictions`: Save prediction masks
- `--roi_size`: ROI size for input (default: 96 96 96)

## 📈 Expected Results (Table 2 from Paper)

| Model | Dice (%) | HD95 (mm) | Inference Time (s) | Params (M) |
|-------|----------|-----------|-------------------|------------|
| SwinUNETR | 72.3 ± 2.1 | 8.2 ± 2.1 | 2.8 | 61.9 |
| UNETR | 68.7 ± 2.3 | 9.4 ± 2.8 | 2.5 | 102.4 |
| SegResNet | 65.2 ± 1.9 | 11.2 ± 3.1 | 1.2 | 15.7 |
| DynUNet | 67.1 ± 2.0 | 10.1 ± 2.9 | 1.8 | 22.3 |

## 🔗 Model Weights

Models are loaded via MONAI with pretrained weights:
- **SwinUNETR**: [Download](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt)
- Place in `pretrained/` folder (optional, for better results)

## 📄 Citation

```bibtex
@article{popov2025magicct,
  title={MAGIC-CT: Multiorgan Annotation and Grounded Image Captioning in CT for Cancer},
  author={Popov, Maxim and Iklassov, Zangir and Baimagambet, Zhanas and others},
  year={2025},
  url={https://zenodo.org/uploads/18389015}
}
```

## 📝 License

- Code: MIT License
- Dataset: CC0 1.0 (Public Domain)

## 👥 Contact

For questions: maxim.popov@nu.edu.kz
