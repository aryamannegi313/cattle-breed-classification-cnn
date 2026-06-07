# Cattle Breed Classification Using CNNs

Comparative analysis of a **Custom CNN**, **MobileNetV2**, and **ResNet50** for classifying **cattle breeds** from images, with **Grad-CAM** and **SHAP** explainability.

Machine Learning course project (Phase II): proposal, code, and technical implementation.

---

## Problem

Given an image of a cow, classify the animal into one of several **cattle breeds**. The dataset is organised **one folder per breed**, and the notebook **auto-detects the breed folders and their names at runtime** — so it works with the recommended dataset or any cattle-breed image set laid out the same way.

## Dataset

- **Recommended (Kaggle):** https://www.kaggle.com/datasets/anandkumarsahu09/cattle-breeds-dataset (5 dairy breeds, folder-per-breed)
- **Larger alternative (Indian breeds):** https://www.kaggle.com/datasets/atharvadarpude/indian-cattle-image-dataset
- **Alternative:** https://www.kaggle.com/datasets/zaidworks0508/cow-breed-classification-dataset

Layout expected (the notebook finds this automatically, even if wrapped in an extra parent folder):

```
<dataset>/
├── <Breed A>/   *.jpg
├── <Breed B>/   *.jpg
└── ...
```

## Method

| Model | Preprocessing (baked into the model) | Base | Head |
|---|---|---|---|
| Custom CNN | `Rescaling(1/255)` | 3 conv blocks (32/64/128) | GAP → Dropout(0.4) → Dense |
| MobileNetV2 | `Rescaling(1/127.5, offset=-1)` | ImageNet, frozen | GAP → Dropout → Dense |
| ResNet50 | `resnet50.preprocess_input` | ImageNet, frozen | GAP → Dropout → Dense |

- Images resized to **224×224**, raw `[0,255]` pipeline (normalisation inside each model so Grad-CAM/SHAP overlays stay interpretable).
- Split **70 / 15 / 15** (train / val / test), fixed seed `42`.
- Training-only augmentation: horizontal flip, small rotation, zoom, contrast (helps on small datasets).
- Adam (lr 1e-3), sparse categorical cross-entropy, **class weights**, **early stopping** (patience 5, best weights restored), up to 25 epochs.
- Evaluation: accuracy, precision/recall/F1 (weighted + macro), confusion matrix, classification report, training time, parameter count.
- Explainability: **Grad-CAM** (last-conv feature tensor captured at build time) on correct and misclassified images; **SHAP** GradientExplainer on selected images. Figures render inline and are saved as PDF + PNG.

## Repository structure

```
.
├── Cattle_Breed_Classification.ipynb   # full pipeline (Kaggle-ready)
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── outputs/
    ├── figures/    # Figure1..Figure7 (PDF + PNG) — populated after a run
    └── tables/     # CSV tables — populated after a run
```

## How to run

### Option A — Kaggle (recommended)

1. Open the notebook in a new Kaggle Notebook.
2. **Add Input** → attach a cattle-breeds dataset (e.g. `anandkumarsahu09/cattle-breeds-dataset`).
3. **Settings → Accelerator → GPU**, and **Settings → Internet → On** (needed to download the ImageNet weights). If Internet is off, the notebook still runs but MobileNetV2/ResNet50 start from random weights and become fully trainable.
4. **Run All.**
5. Download the results from the **Output** tab: `cattle_breed_outputs.zip`, and copy the figures/tables into `outputs/` here.

### Option B — Local / other environment

```bash
pip install -r requirements.txt
jupyter notebook Cattle_Breed_Classification.ipynb
```

Download the dataset from the Kaggle link and place it where the notebook can find it (it scans the working directory and `/kaggle/input`). A GPU is recommended.

## Outputs

Figures (PDF + PNG) and tables (CSV) are written to `outputs/` and zipped on Kaggle. After running, embed the key figures here, e.g.:

```
![Training curves](outputs/figures/Figure3_training_curves.png)
![Confusion matrices](outputs/figures/Figure4_confusion_matrices.png)
![Grad-CAM (correct)](outputs/figures/Figure5_gradcam_correct.png)
```

## Research questions

RQ1 custom-CNN accuracy · RQ2 transfer-learning improvement · RQ3 performance/cost trade-off · RQ4–RQ5 Grad-CAM attention · RQ6 SHAP evidence · RQ7 explainability for error diagnosis. See the notebook's final section for how each output maps to a research question.

## Note on dataset size

The recommended dataset is small (~200 images across 5 breeds), so the from-scratch Custom CNN is expected to underperform and the transfer-learning models to do well — itself a useful finding for the discussion. For a larger study, the Indian-cattle alternative works with the same notebook unchanged.

## License

Released under the MIT License — see `LICENSE`.
