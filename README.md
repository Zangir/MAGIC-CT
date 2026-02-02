# MAGIC-CT: Multiorgan Annotation and Grounded Image Captioning in CT for Cancer

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange)](https://zenodo.org/uploads/18389015)
[![License: Code](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![License: Dataset](https://img.shields.io/badge/License-CC0%201.0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official evaluation scripts for the MAGIC-CT dataset.

## 📋 Overview

MAGIC-CT is a comprehensive multimodal dataset for abdominal oncology combining:
- **562 patients** with contrast-enhanced CT scans
- **~1,250 annotated lesions** across 8 pathologies (4 organs)
- **4,937 organ descriptions** in 3 languages (EN/RU/KZ)
- Expert-validated 3D segmentation masks
- Rich clinical narratives by radiologists

**Pathologies**: Liver cancer (HCC), renal cancer, lung cancer, pancreatic cancer, liver cysts, kidney cysts, lung metastases, liver metastases

## 📊 Dataset

Download from Zenodo: **https://zenodo.org/uploads/18389015**

Expected structure after extraction:
```
data/
├── scans/
│   ├── liver_cancer/
│   │   ├── patient_001.nrrd
│   │   └── ...
│   ├── kidney_cyst/
│   ├── renal_cancer/
│   └── ...
└── segmentations/
    ├── liver_cancer/
    │   ├── patient_001.nrrd
    │   └── ...
    └── ...
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/maxtrubetskoy/MagicCT.git
cd MagicCT

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 1.12+, MONAI 1.2+, CUDA 11.3+ (for GPU)

## ⚡ Quick Start

### Evaluate Single Model

```bash
python evaluate.py --model swinunetr --data_dir data/ --output results/
```

### Evaluate All Models (Reproduce Paper Results)

```bash
python evaluate.py --model all --data_dir data/ --output results/ --device cuda
```

### With GPU Acceleration

```bash
python evaluate.py --model all --data_dir data/ --device cuda
```

### Specific Cancer Types

```bash
python evaluate.py --model swinunetr --data_dir data/ \
    --cancer_types liver_cancer lung_cancer renal_cancer
```

### Save Prediction Masks

```bash
python evaluate.py --model swinunetr --data_dir data/ --save_predictions
```

## 📋 Command-Line Options

```bash
python evaluate.py --help
```

**Main arguments:**
- `--model`: Model name (`swinunetr`, `unetr`, `segresnet`, `dynunet`, or `all`)
- `--data_dir`: Path to dataset directory (containing `scans/` and `segmentations/`)
- `--output`: Output directory for results (default: `results/`)
- `--cancer_types`: Specific cancer types to evaluate (default: all 8 types)
- `--device`: Device (`auto`, `cuda`, or `cpu`; default: auto-detect)
- `--batch_size`: Batch size (default: 1)
- `--workers`: Number of data loading workers (default: 4)
- `--roi_size`: ROI size for input (default: `96 96 96`)
- `--save_predictions`: Save prediction masks as .nrrd files

## 📈 Expected Results

**Table 2 from Paper - Overall Model Performance:**

| Model | Dice (%) | HD95 (mm) | Sensitivity (%) | Specificity (%) | Inference Time (s) | Params (M) |
|-------|----------|-----------|-----------------|-----------------|-------------------|------------|
| **SwinUNETR** | **72.3 ± 2.1** | 8.2 ± 2.1 | 75.9 ± 5.1 | 98.9 ± 0.7 | 2.8 | 61.9 |
| UNETR | 68.7 ± 2.3 | 9.4 ± 2.8 | 73.4 ± 4.6 | 99.2 ± 0.6 | 2.5 | 102.4 |
| SegResNet | 65.2 ± 1.9 | 11.2 ± 3.1 | 67.3 ± 5.6 | 98.7 ± 0.8 | 1.2 | 15.7 |
| DynUNet | 67.1 ± 2.0 | 10.1 ± 2.9 | 69.6 ± 5.7 | 99.4 ± 0.5 | 1.8 | 22.3 |

**Table 3 from Paper - Cancer-Type Specific Performance (Dice %):**

| Cancer Type | SwinUNETR | UNETR | SegResNet | DynUNet |
|-------------|-----------|-------|-----------|---------|
| **Benign Lesions** ||||
| Liver Cysts | **84.3 ± 3.2** | 80.7 ± 3.8 | 76.9 ± 4.1 | 79.3 ± 4.5 |
| Kidney Cysts | **81.0 ± 3.8** | 77.6 ± 4.2 | 73.6 ± 4.6 | 76.0 ± 4.8 |
| **Primary Malignancies** ||||
| Hepatocellular Carcinoma | **78.1 ± 4.2** | 74.5 ± 4.6 | 70.5 ± 5.1 | 72.8 ± 5.3 |
| Lung Cancer | **74.9 ± 4.8** | 71.4 ± 5.2 | 67.7 ± 5.5 | 69.6 ± 5.7 |
| Renal Cancer | **71.6 ± 5.1** | 68.0 ± 5.4 | 64.4 ± 5.8 | 66.2 ± 6.0 |
| Pancreas Cancer | **53.1 ± 7.8** | 49.3 ± 8.1 | 46.8 ± 8.4 | 47.9 ± 8.6 |
| **Metastatic Disease** ||||
| Lung Metastases | **63.0 ± 6.2** | 59.4 ± 6.8 | 56.5 ± 7.1 | 57.9 ± 7.4 |

## 🔗 Model Weights

Models use MONAI framework with pretrained weights:

**SwinUNETR** (Recommended for best results):
- Download: [swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt)
- Place in: `pretrained/` folder
- Pre-trained on 5,050 CT scans with self-supervised learning

**Other models** (UNETR, SegResNet, DynUNet):
- Use MONAI's built-in architectures
- No additional weights required

```bash
# Optional: Download SwinUNETR pretrained weights for better results
mkdir -p pretrained
cd pretrained
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
cd ..
```

## 📁 Output Structure

```
results/
├── swinunetr/
│   ├── results.csv       # Per-case metrics (Dice, HD95, sensitivity, specificity)
│   ├── summary.json      # Aggregate statistics (mean ± std)
│   └── predictions/      # (optional) .nrrd prediction masks
├── unetr/
├── segresnet/
├── dynunet/
└── comparison.csv        # Model comparison table
```

## 🔬 Technical Details

**Dataset Statistics:**
- **Total patients**: 562 (492 with reports)
- **Age**: 63 ± 14 years (range: 19-92)
- **Gender**: 52% male, 46% female
- **Imaging**: Contrast-enhanced CT (Philips Ingenuity)
- **Contrast agents**: Ultravist 370, Gadovist
- **Radiation dose**: 7-15 mSv

**Annotations:**
- **Annotators**: 7 radiologists
- **Inter-annotator agreement**: Cohen's κ = 0.74 ("Substantial")
- **Software**: 3D Slicer
- **Format**: .nrrd (NRRD format with spatial metadata)

## 📖 Citation

If you use this dataset or code, please cite:

```bibtex
@article{popov2025magicct,
  title={{MAGIC-CT}: Multiorgan Annotation and Grounded Image Captioning in {CT} for Cancer},
  author={Popov, Maxim and Iklassov, Zangir and Baimagambet, Zhanas and Jakipov, Murat and Andreyeva, Xeniya and Akhtar, Muhammad and Tak\'{a}\v{c}, Martin and Jamwal, Prashant},
  year={2026},
  doi={10.5281/zenodo.17549293},
  url={https://zenodo.org/uploads/18389015}
}
```

## 👥 Authors

**Corresponding Author:** Maxim Popov (maxim.popov@nu.edu.kz)

1. **Maxim Popov** - Nazarbayev University, Kazakhstan  
2. **Zangir Iklassov** - Mohamed bin Zayed University of AI (MBZUAI), UAE  
3. **Zhanas Baimagambet** - Nazarbayev University, Kazakhstan  
4. **Murat Jakipov** - National Research Oncology Center (NROC), Kazakhstan  
5. **Xeniya Andreyeva** - National Research Oncology Center (NROC), Kazakhstan  
6. **Muhammad Akhtar** - Nazarbayev University, Kazakhstan  
7. **Martin Takáč** - Mohamed bin Zayed University of AI (MBZUAI), UAE  
8. **Prashant Jamwal** - Nazarbayev University, Kazakhstan (Team Lead)

## 🙏 Acknowledgments

This work was supported by:
- Collaborative Research Program of Nazarbayev University (Grant No. 111024CRP2007)
- National Research Oncology Center (NROC), Astana, Kazakhstan
- MONAI Consortium for framework support

## 📄 License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Dataset**: CC0 1.0 Universal (Public Domain)

## 🐛 Issues & Questions

- **Issues**: [GitHub Issues](https://github.com/maxtrubetskoy/MagicCT/issues)
- **Email**: zangir.iklassov@mbzuai.ac.ae
- **Dataset**: https://zenodo.org/uploads/18389015

## 🔗 Links

- **Paper**: Coming soon (under review)
- **Dataset**: https://zenodo.org/uploads/18389015
- **Code**: https://github.com/maxtrubetskoy/MagicCT
- **MONAI**: https://monai.io

---

**Last Updated**: February 2026
