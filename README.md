# Endoscopy Image Classification — EfficientNetB0 + Channel Attention (SE)

> Lightweight endoscopy image classifier using EfficientNetB0 + Squeeze-and-Excitation (SE) attention and regularization.  
> Single-file release: **`model.ipynb`** (Colab-friendly).

## 🔍 Overview

This project builds a **computationally efficient** CNN-based classifier for GI endoscopy images. The final model integrates:

- **EfficientNetB0** (ImageNet pre-trained) as the feature extractor
- **Channel attention (SE block)** to reweight informative channels
- **Regularization** (L2 + Dropout) to reduce overfitting

The notebook trains/evaluates on four classes: **normal**, **esophagitis**, **ulcerative colitis**, **polyps**.

---

## 📦 Contents

- `efficientNetB0_attention_regularization(proposed_model).ipynb` — the full pipeline (data, model, training, metrics, plots).
- `README.md` — you are here.

Optional (recommended):

- `.gitignore` — ignore datasets, notebook checkpoints, etc.
- `LICENSE` — choose one (e.g., MIT).

---

## 🗂️ Dataset

- **Name**: WCE curated colon disease dataset (Kaggle, 2021).
- **Classes used here (4)**: `normal`, `esophagitis`, `ulcerative_colitis`, `polyps`
- **Images**: 6,000 (≈1,500 per class), various resolutions
- **Split (used in code)**: `train` ≈ 53%, `val` ≈ 33%, `test` ≈ 13%
- **Input size**: 224×224

### Expected directory layout

```
dataset_root/
├─ train/
│  ├─ normal/ ...images
│  ├─ esophagitis/ ...
│  ├─ ulcerative_colitis/ ...
│  └─ polyps/ ...
├─ val/
│  ├─ normal/ ...
│  ├─ esophagitis/ ...
│  ├─ ulcerative_colitis/ ...
│  └─ polyps/ ...
└─ test/
   ├─ normal/ ...
   ├─ esophagitis/ ...
   ├─ ulcerative_colitis/ ...
   └─ polyps/ ...
```

> If your data is elsewhere, update the notebook’s `flow_from_directory()` paths.

---

## 🧠 Model (Proposed architecture)

- **Backbone**: EfficientNetB0 (ImageNet weights)
- **Attention**: Squeeze-and-Excitation (SE) block
  - GAP → Dense(C/4, ReLU) → Dense(C, Sigmoid) → Reshape(1,1,C) → Multiply with feature map
- **Head**: GAP → Dense(256, ReLU, L2=0.01) → Dropout(0.5) → Dense(num_classes, Softmax, L2=0.01)

### Why SE?

SE learns channel-wise importance, letting the network **focus on diagnostic cues** and suppress less relevant channels with almost no compute overhead.

---

## ⚙️ Training Setup (as in the notebook)

- **Framework**: TensorFlow / Keras (Colab-ready)
- **Image size**: 224×224, **batch**: 32
- **Augmentation**: rotation, shifts, shear, zoom, horizontal flip, rescale=1/255
- **Optimizer**: Adam, **learning rate**: `1e-5`
- **Loss**: categorical cross-entropy
- **Regularization**: L2=0.01 (dense + output), Dropout=0.5
- **Epochs**: 20

---

## 📊 Results (from the notebook & report)

- **Test accuracy**: ≈ **93.7%** (EfficientNetB0 + SE + regularization)
- Baselines observed:
  - EfficientNetB0 (no attention): ~82%
  - - SE (no regularization): ~85%

**Precision–Recall** and **ROC** curves indicate stronger performance for **normal** and **esophagitis**; comparatively weaker on **ulcerative colitis** and **polyps** (scope for improvement).

> See the notebook’s final section for classification report, weighted F1, ROC and PR curves.

---

## 🚀 Quickstart

### Option A — Run on Google Colab

1. Open the notebook in Colab.
2. Mount your drive (first code cell does this).
3. Place the dataset at the expected path (or update the `flow_from_directory()` paths).
4. Run all cells end‑to‑end.

**Colab badge (replace with your repo path after you push):**

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/efficientNetB0_attention_regularization(proposed_model).ipynb)
```

### Option B — Local (GPU recommended)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install "tensorflow==2.16.1" "keras" "scikit-learn" "matplotlib" "numpy"
# Start Jupyter
pip install notebook
jupyter notebook
# Open the .ipynb and update dataset paths
```

---

## 🧪 Reproducibility Tips

- Keep the same folder split and augmentations.
- Use the same batch size and image size.
- If results jitter, try running for 25–30 epochs or tune lr in {1e-5, 3e-5, 1e-4}.

---

## 📉 Known Limitations & Future Work

- Limited dataset diversity may affect generalization to rare/flat polyps.
- Consider **spatial attention** (in addition to channel attention) and/or a small **multi-view ensemble** for harder classes.
- Real‑time, edge deployment would benefit from TensorFlow Lite export and post‑training quantization (int8).

---

## ✍️ Author

- Amrita Sinha Roy

If you use this work, please cite the project/report accordingly.

---

## 🧾 License


MIT --- see [LICENSE](./LICENSE)

MIT License 

---

## 🙌 Acknowledgements

- EfficientNet (Tan & Le)
- Squeeze-and-Excitation (Hu et al.)
- WCE curated colon disease dataset (Kaggle)
