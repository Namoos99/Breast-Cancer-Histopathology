# Breast Cancer Histopathology Classification
### Multi-Magnification Deep Learning with ResNet50 + Grad-CAM

A deep learning pipeline for classifying breast cancer histopathology images as **benign or malignant** using transfer learning and multi-magnification fusion on the BreaKHis dataset.

---

## Overview

Standard approaches to histopathology classification use a single magnification level. This project explores whether fusing features from **two magnification levels (40X and 200X)** improves classification performance — the intuition being that different scales capture complementary information:

- **40X** — captures broad tissue architecture and structural organization
- **200X** — captures fine-grained cellular morphology and nuclear detail

A separate ResNet50 model is trained for each magnification, and the predictions are fused for the final classification decision.

---

## Dataset

**BreaKHis (Breast Cancer Histopathological Database)**  
- 7,909 microscopic images from 82 patients
- Two classes: benign and malignant
- Multiple magnification factors: 40X, 100X, 200X, 400X
- This project uses 40X and 200X

Dataset: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz  
The notebook downloads it automatically — no manual setup required.

---

## Approach

1. **Data Loading** — Custom PyTorch `Dataset` class with stratified 70/15/15 train/val/test split
2. **Preprocessing** — Resize to 224x224, ImageNet normalization, data augmentation (random flips, color jitter)
3. **Baseline Models** — Fine-tuned ResNet50 (frozen backbone, custom classification head with dropout) trained independently on 40X and 200X
4. **Fusion Model** — Combines predictions from both magnification models for final classification
5. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix
6. **Interpretability** — Grad-CAM heatmaps to visualize what each model focuses on

---

## Results

| Model | Notes |
|-------|-------|
| ResNet50 (40X) | Tissue architecture features |
| ResNet50 (200X) | Cellular morphology features |
| Fusion Model | Combined multi-scale predictions |

*Grad-CAM analysis confirmed the two models focus on distinct visual features, validating the multi-magnification approach.*

---

## Tech Stack

- **PyTorch** + **torchvision** — model training and transfer learning
- **ResNet50** — pretrained on ImageNet, fine-tuned for binary classification
- **Grad-CAM** (`pytorch-grad-cam`) — model interpretability and heatmap visualization
- **Scikit-learn** — metrics and train/val/test splitting
- **Matplotlib** + **Seaborn** — visualizations

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

> The notebook downloads the BreaKHis dataset automatically. A GPU is strongly recommended — training on CPU will be slow.

---

## Project Structure

```
breast-cancer-histopathology/
├── Project_notebook.ipynb   # Main notebook — data loading, training, evaluation, Grad-CAM
└── README.md
```

---

## Key Takeaways

- Multi-magnification fusion leverages complementary visual information that single-scale models miss
- Grad-CAM confirmed the 40X model focuses on tissue structure while the 200X model targets cellular detail
- Proper evaluation with precision/recall/F1 is critical in medical imaging — accuracy alone is misleading due to class imbalance

---

## Author

**Namoos Haider**  
MS Applied Data Science & AI — University of Denver  
[LinkedIn](https://linkedin.com/in/namoos-haider-75a949209/) · [GitHub](https://github.com/Namoos99)
