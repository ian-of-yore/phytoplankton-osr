# Phytoplankton Open-Set Recognition (OSR)

This repository implements an open-set recognition (OSR) pipeline for phytoplankton image classification using a SYKE-PIC backbone and three OSR methods:

- **Maximum Logit Score (MLS)**
- **OpenMax**
- **Mahalanobis Distance**

The workflow consists of:

1. Environment setup  
2. Dataset preparation  
3. Closed-set training  
4. Open-set evaluation  

---

# Installation

## Requirements

- Windows 10/11  
- Python 3.9.x  
- Git  

⚠ Python 3.9 is required due to SYKE-PIC version constraints.

---

## 1. Clone the Repository

```bash
git clone https://github.com/ian-of-yore/phytoplankton-osr.git
cd phytoplankton-osr
```

---

## 2. Create Virtual Environment

```bash
py -3.9 -m venv venv
venv\Scripts\activate
```

Verify:

```bash
python --version
```

---

## 3. Install Dependencies (Order Matters)

Upgrade build tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch (CPU version – recommended default):

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
```

Install SYKE-PIC (pinned commit):

```bash
python -m pip install --no-deps "git+https://github.com/sykefi/syke-pic@bccae3bf20d476b444c7082dce77499f3397ac82"
```

Install remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Install this repository as a package:

```bash
python -m pip install -e .
```

---

## 4. Verify Installation

```bash
python -c "import torch, numpy, cv2, sykepic, phytosr; print('Environment OK')"
```

---

# CPU vs GPU Usage

The default installation uses the CPU build of PyTorch to ensure maximum reproducibility and hardware independence.

If a compatible NVIDIA GPU is available, a CUDA-enabled build of the same PyTorch version (2.2.2) may be installed instead.

Example (CUDA 11.8):

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

Using GPU acceleration does not change model behavior or results. It only improves training speed.

---

# Configuration

The project separates experiment configuration and machine-specific paths:

- `configs/exp.yaml` → experiment parameters  
- `configs/paths.yaml` → local file system paths  

Create the local paths file:

```bash
copy configs\paths.example.yaml configs\paths.yaml
```

Edit `configs/paths.yaml`:

```yaml
paths:
  dataset_root: C:\path\to\dataset
  out_base: C:\path\to\output
  model_out_dir: C:\path\to\output\models
  model_dir: ""
  sykepic_repo: ""
```

Path definitions:

- `dataset_root` — raw dataset location  
- `out_base` — prepared splits and intermediate outputs  
- `model_out_dir` — directory for trained models  

`paths.yaml` should remain local and not be committed.

---

# Data Preparation

Generate deterministic dataset splits:

```bash
python scripts/prepare_data.py --config configs/exp.yaml --paths configs/paths.yaml
```

---

# Training (Closed-Set)

Train the classifier on known classes:

```bash
python scripts/train.py --config configs/exp.yaml --paths configs/paths.yaml
```

Model artifacts are saved under:

```
output/models/<experiment_name>/
```

---

# Open-Set Evaluation

General format:

```bash
python scripts/run_osr.py --config configs/exp.yaml --paths configs/paths.yaml --method <method>
```

### Maximum Logit Score (MLS)

```bash
python scripts/run_osr.py --config configs/exp.yaml --paths configs/paths.yaml --method mls
```

### OpenMax

```bash
python scripts/run_osr.py --config configs/exp.yaml --paths configs/paths.yaml --method openmax
```

### Mahalanobis Distance

```bash
python scripts/run_osr.py --config configs/exp.yaml --paths configs/paths.yaml --method mahalanobis
```

Each method writes evaluation results to the corresponding trained model directory.

---

## OSR Comparison

A consolidated evaluation table can be generated from the saved confusion matrices:

```bash
python scripts/osr_compare.py --model-dir output/models/<experiment_name>
```
For example:
```bash
python scripts/osr_compare.py --model-dir output/models/resnet18_2
```

---

## Clustering (Colab)

The OpenMax-gated clustering stage is implemented in:

`phytoplankton-osr/notebooks/Clustering_OpenMax.ipynb`

The notebook loads a trained SYKE-PIC model, computes OpenMax unknownness, selects top-q unknown candidates, clusters them using HDBSCAN, and exports CSV + UMAP visualizations.

Clustering dependencies (`umap-learn`, `hdbscan`) are installed inside the notebook to avoid platform-specific issues.

---
# Reproducibility

Results are reproducible under:

- Python 3.9  
- Torch 2.2.2  
- NumPy < 2  
- Pinned SYKE-PIC commit  
- Fixed random seed in `exp.yaml`
