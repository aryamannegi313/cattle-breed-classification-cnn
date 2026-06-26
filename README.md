# 🐄 Cattle Breed Classification Using CNNs

> Comparative analysis of a Custom CNN, MobileNetV2, and ResNet50 for automated cattle breed identification — with Explainable AI via Grad-CAM and SHAP.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Outputs](#outputs)

---

## Overview

This project trains and evaluates three deep-learning image classifiers to identify **five cattle breeds** from photographs. It benchmarks a lightweight custom CNN built from scratch against two ImageNet pre-trained transfer-learning models (MobileNetV2 and ResNet50), and applies Explainable AI techniques (Grad-CAM and SHAP) to interpret how each model makes its predictions.

The notebook is designed to run end-to-end on **Kaggle** with GPU acceleration and produces all evaluation tables, visualisations, and a downloadable results archive automatically.

**Key questions addressed:**
- How much accuracy does transfer learning gain over a small custom CNN on a modest dataset?
- Which breed categories are most confused by each model?
- Can Grad-CAM and SHAP reveal what visual features the models attend to?

---

## Dataset

The dataset follows a standard **folder-per-breed** structure and is auto-detected at runtime. It is compatible with the [`anandkumarsahu09/cattle-breeds-dataset`](https://www.kaggle.com/datasets/anandkumarsahu09/cattle-breeds-dataset) Kaggle dataset or any other folder-per-class cattle image set.

| Breed | Images |
|---|---|
| Ayrshire cattle | 260 |
| Brown Swiss cattle | 238 |
| Holstein Friesian cattle | 254 |
| Jersey cattle | 252 |
| Red Dane cattle | 204 |
| **Total** | **1,208** |

All images are resized to **224×224 RGB** before being fed to the models. The dataset is split into **train / validation / test** sets (70% / 15% / 15%) using a fixed random seed (`42`) for full reproducibility.

**Data augmentation** (applied to training only): horizontal flip, small rotation (±8°), zoom (±12%), and contrast jitter (±10%).

---

## Models

Three architectures are trained and compared. Preprocessing is baked into each model so all three share the same raw pixel pipeline.

| Model | Total Params | Trainable Params | Frozen Params |
|---|---|---|---|
| Custom CNN | 93,893 | 93,893 | 0 |
| MobileNetV2 | 2,264,389 | 6,405 | 2,257,984 |
| ResNet50 | 23,597,957 | 10,245 | 23,587,712 |

**Custom CNN** — Three convolutional blocks (`Conv2D → MaxPool`) with 32, 64, and 128 filters, followed by Global Average Pooling, Dropout (0.4), and a softmax Dense head. Trained entirely from scratch with `Rescaling(1/255)` normalisation.

**MobileNetV2** — Frozen ImageNet backbone with `[-1, 1]` rescaling, plus a GAP → Dropout (0.3) → Dense head. Only the 6,405 head parameters are trained.

**ResNet50** — Frozen ImageNet backbone with `resnet50.preprocess_input`, plus the same lightweight head. Only the 10,245 head parameters are trained.

All models are compiled with `Adam`, `sparse_categorical_crossentropy`, and use **early stopping** (patience 6) and **learning-rate reduction on plateau** (patience 3, factor 0.4). Class weights are applied to compensate for the modest class imbalance.

---

## Results

### Overall Performance

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) | Train Time (s) |
|---|---|---|---|---|---|---|
| Custom CNN | 64.1% | 65.7% | 64.1% | 63.8% | 63.5% | 146.8 |
| MobileNetV2 | 85.6% | 87.3% | 85.6% | 85.7% | 85.2% | 118.4 |
| **ResNet50** | **95.0%** | **95.3%** | **95.0%** | **95.0%** | **94.9%** | 169.8 |

ResNet50 achieves the highest accuracy and F1-score across all metrics. MobileNetV2 offers a strong accuracy-to-training-time trade-off, while the Custom CNN serves as a baseline demonstrating the value of pre-trained feature extraction.

### Per-Class F1 Scores

| Breed | Custom CNN | MobileNetV2 | ResNet50 |
|---|---|---|---|
| Ayrshire cattle | 0.620 | 0.877 | 0.933 |
| Brown Swiss cattle | 0.571 | 0.860 | 0.957 |
| Holstein Friesian cattle | 0.822 | 0.950 | 0.963 |
| Jersey cattle | 0.493 | 0.767 | 0.943 |
| Red Dane cattle | 0.667 | 0.806 | 0.947 |

Holstein Friesian was the easiest class across all models, likely due to its distinctive black-and-white colouring. Jersey cattle was the most challenging for the weaker models.

### Summary

| Model | Accuracy | F1-Score | Grad-CAM Quality | SHAP Interpretability | Overall |
|---|---|---|---|---|---|
| Custom CNN | 64.1% | 63.8% | Low | Low | Weak baseline; lowest accuracy and F1 |
| MobileNetV2 | 85.6% | 85.7% | Medium | Medium | Good lightweight option; fast training |
| ResNet50 | 95.0% | 95.0% | High | High | Best overall; highest accuracy, most parameters |

---

## Explainability

### Grad-CAM
Class Activation Maps are generated for each model's last convolutional layer. The `grad_model` captures the final convolutional feature tensor at build time — avoiding graph-disconnect issues common with nested transfer-learning bases — and overlays a heatmap on the original image for both correct predictions and misclassifications.

### SHAP
SHAP (SHapley Additive exPlanations) values highlight which image regions contribute positively or negatively to each breed prediction, providing a complementary pixel-level view alongside Grad-CAM.

Both explainability outputs are saved as figures (PDF + PNG) in the `figures/` output directory.

---

## Project Structure

```
cattle-breed-classification/
│
├── cattle-breed-classification.ipynb   # Main notebook (all code, figures, tables)
│
├── outputs/                            # Generated automatically on run
│   ├── figures/                        # Grad-CAM, SHAP, training curves, etc. (PDF + PNG)
│   └── tables/                         # All CSV outputs below
│
├── dataset_description.csv             # Breed names, image counts, image spec
├── model_architecture.csv              # Total / trainable / frozen params per model
├── evaluation_metrics.csv              # Accuracy, precision, recall, F1, train time
├── classification_report_Custom_CNN.csv
├── classification_report_MobileNetV2.csv
├── classification_report_ResNet50.csv
├── confusion_matrix_Custom_CNN.csv
├── confusion_matrix_MobileNetV2.csv
├── confusion_matrix_ResNet50.csv
└── final_comparison.csv                # Side-by-side model comparison with findings
```

---

## Getting Started

### Running on Kaggle (Recommended)

1. Open the notebook in [Kaggle](https://www.kaggle.com).
2. **Add dataset**: *Add Input* → search for `anandkumarsahu09/cattle-breeds-dataset` and attach it. Any other folder-per-breed cattle image dataset also works.
3. **Enable GPU**: *Settings → Accelerator → GPU T4 x2* (or P100).
4. **Enable Internet**: *Settings → Internet → On* (required to download ImageNet weights for MobileNetV2 and ResNet50).
5. Click **Run All**.
6. When complete, download `cattle_breed_outputs.zip` from the **Output** tab.

> **Note:** If Internet access is unavailable, the notebook will fall back to random weight initialisation for the transfer-learning models and make their bases fully trainable. Results will be significantly weaker.

### Running Locally

```bash
# Clone or download the repository
git clone <repo-url>
cd cattle-breed-classification

# Install dependencies
pip install -r requirements.txt

# Place your dataset in a folder-per-breed structure, e.g.:
# data/
#   Ayrshire cattle/
#   Brown Swiss cattle/
#   ...

# Update DATA_DIR in the notebook if not running on Kaggle, then run:
jupyter notebook cattle-breed-classification.ipynb
```

---

## Requirements

| Package | Purpose |
|---|---|
| `tensorflow >= 2.10` | Model building, training, Grad-CAM |
| `numpy` | Numerical operations |
| `pandas` | Table construction and CSV export |
| `matplotlib` | Figures and training curves |
| `Pillow` | Image loading |
| `scikit-learn` | Metrics, classification reports |
| `shap` | SHAP explainability |

A GPU is strongly recommended. Training all three models on CPU will be significantly slower.

---

## Outputs

After running the notebook, all outputs are zipped into `cattle_breed_outputs.zip` and include:

- **Figure 1** — Sample images per breed
- **Figure 2** — Model design workflow diagram
- **Figure 3** — Training & validation accuracy/loss curves (all three models)
- **Figure 4** — Confusion matrices
- **Figure 5** — Grad-CAM heatmaps (correct predictions)
- **Figure 6** — Grad-CAM heatmaps (misclassified examples)
- **Figure 7** — SHAP visualisations
- **Figure 8** — Final model comparison bar chart
- All CSV tables listed in [Project Structure](#project-structure)

---

## Key Findings

- **Transfer learning dramatically outperforms** a small custom CNN on this dataset size (~1,200 images). MobileNetV2 gains +21 percentage points and ResNet50 gains +31 percentage points in accuracy over the baseline.
- **ResNet50 is the clear winner** at 95.0% accuracy and F1, though it requires 250× more parameters than the Custom CNN.
- **MobileNetV2 is the best efficiency trade-off**: 85.6% accuracy, the shortest training time (118s), and only 2.3M parameters — making it practical for edge deployment.
- **Holstein Friesian** was the most reliably classified breed across all models, consistent with its visually distinctive colouring.
- **Jersey cattle** was the hardest to classify for weaker models, likely due to visual similarity with Ayrshire.
- Grad-CAM and SHAP quality correlate with model accuracy — ResNet50 produces the most coherent and focused attention maps.
