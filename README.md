# 🧬 EpitopeNet

> **Backpropagation-free prototype learning inspired by B-cell epitope recognition for interpretable mammography classification**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Millimono/EpitopeNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Millimono/EpitopeNet/actions)

---

## 📋 Overview

**EpitopeNet** is a novel gradient-free learning framework for medical image classification, inspired by the selective dynamics of B-cell lymphocytes. Instead of backpropagation, EpitopeNet maintains a population of **B = 2,133 independent prototypes** that progressively specialize on discriminant local patches via a **Learning Vector Quantization (LVQ)** rule.

### Key properties

| Property | EpitopeNet | CNN (baseline) |
|---|---|---|
| Backpropagation | ❌ None | ✅ Required |
| Native interpretability | ✅ Exact | ❌ Post-hoc only |
| Gradient computation | ❌ None | ✅ Required |
| Decision explanation | ✅ Patch-level | ❌ Approximate |
| Training stability | ✅ No vanishing gradient | ⚠️ Possible |

---

## 📸 How it works

### Pipeline Architecture

```
Input image (256×256)
        │
        ▼
[Preprocessing]
Grayscale → Otsu masking → Crop → Resize 256×256 → Normalize [0,1]
        │
        ▼
[Patch Extraction]
57,121 patches of 18×18 via sliding window (stride=1)
Each patch z-score normalized → ẽ_k ∈ ℝ³²⁴
        │
        ▼
[Prototype Activation]
B=2133 prototypes compute Gaussian similarity s_i(x)
Prototype activated if s_i(x) ≥ θ=0.2
        │
        ▼
[LVQ Update (training only)]
Activated prototypes: attract if correct class, repel otherwise
Class assignment via normalized frequency counters
        │
        ▼
[Weighted Vote (inference)]
ŷ = argmax_k Σ f_i^(k) · w_i  over active prototypes
        │
        ▼
[Native Interpretability]
Decisive patches directly identifiable — no post-hoc method needed
```

---

## 📊 Results on MiniDDSM (Cancer vs Normal)

Benchmarked on MiniDDSM — 4,816 mammography images (1,924 Cancer + 1,924 Normal train / 484+484 val):

| Method | Accuracy | F1 | AUC | Time |
|---|---|---|---|---|
| kNN (k=5) | 80.99% | 0.8099 | 0.8960 | <1s |
| MLP (256-128) | 84.19% | 0.8410 | 0.9166 | 383s |
| SVM (RBF) | 87.40% | 0.8739 | 0.9343 | 1470s |
| **EpitopeNet (ours)** | **85.64 ± 2.10%** | **0.856 ± 0.021** | **— ** | ~19 min/run |

> EpitopeNet outperforms kNN and MLP **without any backpropagation** and with **exact native interpretability** — a property no baseline can provide. SVM achieves higher accuracy at the cost of 1470s training time and zero interpretability.

### Ablation study — key results

| Configuration | Accuracy |
|---|---|
| patch 10×10 | 70.14% |
| patch 28×28 | 79.34% |
| **patch 18×18 (retained)** | **85.64%** |
| θ = 0.1 | 83.68% |
| **θ = 0.2 (retained)** | **85.64%** |
| θ = 0.5 | 83.16% |
| θ = 0.7 | 77.27% |

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
# Pull the image
docker pull millimono/epitopenet:1.0.0

# Train on MiniDDSM
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python run.py --data_path /data/miniddsm --B 2133 --patch_size 18 \
  --theta 0.2 --epochs 30 --seed 42

# Run baselines
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python baselines.py --data_path /data/miniddsm

# Visualize interpretability
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python interpretability.py --model_path /data/best_model.pt
```

> **Windows users:** replace `$(pwd)` with `%cd%`

### Option 2 — Local installation

```bash
git clone https://github.com/Millimono/EpitopeNet.git
cd EpitopeNet
pip install -r requirements.txt
python run.py --data_path /path/to/miniddsm
```

---

## 🗂️ Project Structure

```
EpitopeNet/
├── model.py              # EpitopeNet model — prototypes, LVQ, weighted vote
├── train.py              # Training loop with early stopping + restoration
├── run.py                # Main entry point — train + evaluate
├── data.py               # MiniDDSM loading, Otsu preprocessing, patch extraction
├── baselines.py          # kNN, SVM, MLP baselines
├── interpretability.py   # Patch-level decision visualization
├── combine_results.py    # Multi-seed results aggregation
├── save_load.py          # Model save/load utilities
│
├── Ablation1_crop_multi_echelle/  # Patch size & multi-scale ablation results
├── Ablation3_Intensity_True/      # Intensity feature ablation results
├── ablation_2_Intensity/          # Intensity influence study
├── theta_ablation/                # Activation threshold ablation results
│
├── containers/
│   └── Dockerfile        # Reproducible environment
├── requirements.txt      # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI
└── README.md
```

---

## ⚙️ Model Parameters

| Hyperparameter | Value | Description |
|---|---|---|
| B | 2,133 | Number of prototypes |
| p | 18 | Patch size (18×18 pixels) |
| θ | 0.2 | Activation threshold |
| η₀ | 0.001 | Initial learning rate |
| lr decay | ×0.95/epoch | Exponential decay |
| K | 1 | Top-K patches per prototype |
| Patience | 7 epochs | Early stopping |
| Init | 50 images | Prototype initialization |
| Epochs | 30 max | Training epochs |

---

## 🔬 Native Interpretability

EpitopeNet provides **exact, native interpretability** — no post-hoc method (Grad-CAM, LIME, SHAP) needed.

For any classified image, the decisive patches are directly identifiable:

```python
from interpretability import explain_decision

# Get the patches responsible for classification
patches, weights, prototype_ids = explain_decision(
    model=epitopenet,
    image=x,
    top_k=5  # show top-5 most influential patches
)
# patches: actual image regions that drove the decision
# weights: exclusivity weight of each active prototype
# prototype_ids: which prototypes were activated
```

This is fundamentally different from post-hoc methods:
- **Grad-CAM**: approximates saliency after the fact
- **LIME**: perturbs input to estimate local behavior
- **EpitopeNet**: the decision mechanism IS the explanation

---

## 🛠️ Technical Stack

| Category | Tools |
|---|---|
| **Deep Learning** | PyTorch 2.0+ |
| **Numerical** | NumPy, SciPy |
| **Image processing** | Pillow, scikit-image (Otsu) |
| **Baselines** | scikit-learn (kNN, SVM, MLP) |
| **Visualization** | matplotlib, seaborn |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

---

## 🧪 Dataset

**MiniDDSM** — subset of the Digital Database for Screening Mammography (DDSM):

| Split | Cancer | Normal | Total |
|---|---|---|---|
| Train | 1,924 | 1,924 | 3,848 |
| Validation | 484 | 484 | 968 |
| **Total** | **2,408** | **2,408** | **4,816** |

Preprocessing pipeline:
1. Convert to grayscale
2. Otsu thresholding → extract breast tissue bounding box
3. Crop + resize to 256×256
4. Normalize to [0,1]
5. Extract 57,121 patches of 18×18 (stride=1)
6. Z-score normalize each patch independently

---

## 🔗 Link to Research

This repository contains the implementation of **EpitopeNet**, submitted for publication in 2026:

> Millimono, S., Bellarbi, L., Rhalem, W. **EpitopeNet: Patch-Based Prototype Learning Inspired by B Cell Epitope Recognition.** *(Under review, 2026)*

Related work:
- **HAtt-CNN** — Adaptive visual attention supervision with heuristic masks for CNN interpretability. *(Under review, 2026)*
- **MalariaScan** — AI detection of malaria via microscopy. Prix Jean-Marc Léger, UdeM 2025.
- **OmicsFlow** — Modular NGS pipeline. [github.com/Millimono/OmicsFlow](https://github.com/Millimono/OmicsFlow)

---

## 📄 Citation

```bibtex
@software{millimono2026epitopenet,
  author    = {Millimono, Sory and Bellarbi, Larbi and Rhalem, Wajih},
  title     = {EpitopeNet: Patch-Based Prototype Learning Inspired by B Cell Epitope Recognition},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Millimono/EpitopeNet}
}
```

---

## 👤 Author

**Sory Millimono**
PhD Candidate in AI · Bioinformatician
Université de Montréal & Mohammed V University – ENSIAS · E2SN Research Team

- 📧 millimono64.sm@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/sory-millimono-ai-searcher-820314162)
- 🎓 [Google Scholar](https://scholar.google.com/citations?user=5M-zcxYAAAAJ) — h-index 1 · 24 citations
- 🔬 [ORCID: 0009-0005-1960-9136](https://orcid.org/0009-0005-1960-9136)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
