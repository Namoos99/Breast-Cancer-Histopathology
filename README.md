# Breast Cancer Histopathology Classification
### Multi-Magnification Deep Learning with Attention-Based Fusion + Grad-CAM

A deep learning pipeline for classifying breast cancer histopathology images as **benign or malignant** using multi-scale fusion, transfer learning, and model interpretability on the BreaKHis dataset.

Built as a final project for COMP4531: Deep Learning at the University of Denver.

---

## Overview

Most existing approaches to histopathology classification treat each magnification level independently or combine predictions with simple voting. This project takes a different approach: we train models across multiple magnification levels and use **attention-based fusion** that learns which magnifications are most important for a given classification — rather than weighting them equally.

The core hypothesis is that different magnification levels capture complementary information that a single-scale model misses:

- **40X** — broad tissue architecture and structural organization
- **200X** — fine-grained cellular morphology and nuclear detail

---

## Dataset

**BreaKHis (Breast Cancer Histopathological Database)**  
Federal University of Paraná — freely available for research

- 9,109 microscopic images from 82 patients
- 2,480 benign / 5,429 malignant (class imbalance addressed in training)
- 8 tumor subtypes across 4 magnification levels: 40X, 100X, 200X, 400X

Dataset: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/  
The notebook downloads the dataset automatically — no manual setup required.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet50 — 40X only | 91.33% | 95.45% | 91.75% | 93.56% |
| ResNet50 — 200X only | 94.70% | 96.17% | 96.17% | 96.17% |
| **Fusion (40X + 200X)** | **96.38%** | **95.90%** | **98.94%** | **97.40%** |

Fusion improved accuracy by **+1.67%** over the best single-magnification model. Most critically, recall on malignant cases reached **98.94%** — meaning the model correctly identified nearly all true positives, minimizing dangerous false negatives in a clinical context.

---

## Approach

### 1. Multi-Scale Fusion Architecture
Rather than training independent models per magnification, we designed a fusion approach with:
- **Late fusion** of magnification-specific CNN features
- **Attention mechanisms** that dynamically weight magnification levels based on learned importance
- **Shared encoder with magnification-specific decoder heads** to capture both common and unique features

### 2. Comprehensive Model Comparison
We evaluated both pre-trained and from-scratch approaches to understand transfer learning effectiveness for histopathological images, which differ substantially from natural images:
- **Pre-trained models** — ResNet50, EfficientNet, DenseNet, Vision Transformer (ViT) fine-tuned from ImageNet
- **Custom CNN** — architecture designed specifically for histopathological texture and cellular patterns

### 3. Handling Class Imbalance
- Focal loss and class weighting to handle the imbalance between benign (31.3%) and malignant (68.7%) samples
- Stratified train/val/test splits (70/15/15) to preserve class distribution
- Cross-magnification consistency checks to ensure stable predictions across zoom levels for the same patient

### 4. Model Interpretability
Grad-CAM heatmaps visualize what each model focuses on at different magnifications — a clinically important component, since pathologists need to understand *why* a model makes a prediction before trusting it in a diagnostic setting.

---

## Tech Stack

- **PyTorch** + **torchvision** — model training, transfer learning, data pipelines
- **ResNet50, EfficientNet, DenseNet, ViT** — pretrained architectures evaluated
- **Grad-CAM** (`pytorch-grad-cam`) — interpretability and heatmap visualization
- **Scikit-learn** — metrics, stratified splitting
- **Matplotlib** + **Seaborn** — visualizations
- Google Colab with GPU acceleration (T4)

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Namoos99/breast-cancer-histopathology.git
cd breast-cancer-histopathology

# Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pillow tqdm grad-cam

# Open the notebook
jupyter notebook Project_notebook.ipynb
```

> The notebook downloads the BreaKHis dataset automatically. A GPU is strongly recommended.

---

## Project Structure

```
breast-cancer-histopathology/
├── Project_notebook.ipynb   # Full pipeline — data loading, training, evaluation, Grad-CAM
└── README.md
```

---

## Key Takeaways

- Fusion outperforms both single-magnification models — confirming that 40X and 200X capture complementary features
- **98.94% recall** on malignant cases minimizes false negatives, which is the most critical metric in cancer detection
- Grad-CAM confirmed 40X models focus on tissue structure while 200X models target cellular detail
- Pre-trained ImageNet weights transfer meaningfully to histopathological images despite the domain gap

---

## Author

**Namoos Haider**
MS Applied Data Science & AI — University of Denver  
[LinkedIn](https://linkedin.com/in/namoos-haider-75a949209/) · [GitHub](https://github.com/Namoos99)
