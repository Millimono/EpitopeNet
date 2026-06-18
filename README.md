# 🧬 EpitopeNet

> **A backpropagation-free prototype learning framework inspired by B-cell epitope recognition for interpretable image classification**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Millimono/EpitopeNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Millimono/EpitopeNet/actions)

---

## 📋 Overview

**EpitopeNet** is a novel gradient-free learning framework for image classification, inspired by the selective dynamics of B-cell lymphocytes in the immune system.

The core idea: instead of backpropagation, a population of **B independent prototypes** progressively specializes on discriminant local patches through a **Learning Vector Quantization (LVQ)** rule — attracting toward patches of the correct class, repelling otherwise. Classification is performed by a **weighted population vote**, where each prototype contributes proportionally to its class exclusivity.

### Why EpitopeNet?

| Property | EpitopeNet | Standard CNN |
|---|---|---|
| Backpropagation | ❌ Not required | ✅ Required |
| Gradient computation | ❌ None | ✅ Required |
| Native interpretability | ✅ Exact, patch-level | ❌ Post-hoc approximation |
| Decision explanation | ✅ Directly from prototypes | ❌ Grad-CAM, LIME, SHAP |
| Vanishing gradient | ❌ Impossible | ⚠️ Possible |
| Biological inspiration | ✅ B-cell dynamics | ❌ None |

EpitopeNet is particularly suited for applications requiring **trustworthy, auditable decisions** — medical imaging, regulatory contexts, or any domain where "why did the model decide this?" is as important as the decision itself.

---

## 🔬 Framework

### Core algorithm

```
Input image x
      │
      ▼
[Preprocessing]
Grayscale → ROI extraction → Resize → Normalize [0,1]
      │
      ▼
[Patch Extraction]
P = (H-p+1)² patches of size p×p via sliding window (stride=1)
Each patch z-score normalized → ẽ_k ∈ ℝ^(p²)
      │
      ▼
[Prototype Activation]
For each prototype b_i:
  s_i(x) = exp(-min_k ||ẽ_k - b_i||² / √D)
  Activated if s_i(x) ≥ θ
      │
      ▼
[LVQ Update — training only]
  Attract: b_i ← b_i + η(z_i - b_i)  if class matches
  Repel:   b_i ← b_i - η(z_i - b_i)  otherwise
  Class assigned by normalized frequency counters c_i
      │
      ▼
[Weighted Vote — inference]
  ŷ = argmax_k Σ_{i∈A(x)} f_i^(k) · w_i
  where w_i = class exclusivity weight ∈ [0,1]
      │
      ▼
[Native Interpretability]
  Decisive patches directly identifiable — no post-hoc method needed
```

### Biological analogy

| EpitopeNet | B-cell immune system |
|---|---|
| Prototype b_i | B-cell receptor |
| Patch activation s_i(x) ≥ θ | Antigen recognition |
| LVQ attract/repel | Clonal selection |
| Counter c_i update | Affinity maturation |
| End-of-epoch reassignment | Clonal expansion → specialization |
| Weighted vote | Polyclonal immune response |

---

## 📊 Demonstration on MiniDDSM

As a proof of concept, EpitopeNet is evaluated on **MiniDDSM** — mammography classification (Cancer vs Normal) — a domain where interpretability is critical.

> **Note:** The hyperparameters below (B=2133, p=18, θ=0.2) are optimal for MiniDDSM 256×256. EpitopeNet is a general framework — these values should be tuned for other datasets and resolutions.

### Results

| Method | Accuracy | F1 | AUC | Time | Interpretable? |
|---|---|---|---|---|---|
| kNN (k=5) | 80.99% | 0.8099 | 0.8960 | <1s | ❌ |
| MLP (256-128) | 84.19% | 0.8410 | 0.9166 | 383s | ❌ |
| SVM (RBF) | 87.40% | 0.8739 | 0.9343 | 1470s | ❌ |
| **EpitopeNet (ours)** | **85.64 ± 2.10%** | **0.856 ± 0.021** | — | ~19 min/run | ✅ **Native** |

> EpitopeNet outperforms kNN and MLP **without any backpropagation**, and is the **only method providing exact, native interpretability**. SVM achieves higher accuracy at the cost of 1470s training and zero interpretability.

### Ablation — key findings

**Patch size** (θ=0.2, B=2133):

| Configuration | Accuracy |
|---|---|
| 10×10 | 70.14% |
| 28×28 | 79.34% |
| 10×10 + 18×18 | 76.76% |
| **18×18 (retained)** | **85.64%** |

**Activation threshold θ** (p=18, B=2133):

| θ | Accuracy |
|---|---|
| 0.1 | 83.68% |
| **0.2 (retained)** | **85.64%** |
| 0.3 | 85.33% |
| 0.5 | 83.16% |
| 0.7 | 77.27% |

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
# Pull the image
docker pull millimono/epitopenet:1.0.0

# Train on your dataset
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python run.py --data_path /data/your_dataset \
  --B 2133 --patch_size 18 --theta 0.2 \
  --epochs 30 --seed 42

# Run baselines comparison
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python baselines.py --data_path /data/your_dataset

# Explain a prediction
docker run --rm -v $(pwd)/data:/data millimono/epitopenet:1.0.0 \
  python interpretability.py --model_path /data/best_model.pt \
  --image_path /data/image.png
```

> **Windows:** replace `$(pwd)` with `%cd%`

### Option 2 — Local

```bash
git clone https://github.com/Millimono/EpitopeNet.git
cd EpitopeNet
pip install -r requirements.txt
python run.py --data_path /path/to/data --B 2133 --patch_size 18 --theta 0.2
```

---

## 🔬 Native Interpretability

EpitopeNet provides **exact, native interpretability** — fundamentally different from post-hoc methods:

```python
from interpretability import explain_decision

# Get the patches responsible for the classification decision
patches, weights, prototype_ids = explain_decision(
    model=epitopenet,
    image=x,
    top_k=5  # top-5 most influential prototypes
)
# patches       → actual image regions that drove the decision
# weights       → exclusivity weight w_i of each active prototype
# prototype_ids → which prototypes were activated
```

| Method | How it works | Exact? |
|---|---|---|
| Grad-CAM | Approximates saliency via gradients | ❌ Approximate |
| LIME | Perturbs input, fits local linear model | ❌ Approximate |
| SHAP | Computes Shapley values | ❌ Approximate |
| **EpitopeNet** | **Decision mechanism = explanation** | ✅ **Exact** |

---

## 🗂️ Project Structure

```
EpitopeNet/
├── model.py              # Core — prototypes, LVQ rule, weighted vote
├── train.py              # Training loop, early stopping, state restoration
├── run.py                # Main entry point — train + evaluate
├── data.py               # Data loading, Otsu preprocessing, patch extraction
├── baselines.py          # kNN, SVM, MLP comparison
├── interpretability.py   # Patch-level decision visualization
├── combine_results.py    # Multi-seed results aggregation
├── save_load.py          # Model save/load utilities
│
├── Ablation1_crop_multi_echelle/  # Patch size & multi-scale ablation
├── Ablation3_Intensity_True/      # Intensity feature ablation
├── ablation_2_Intensity/          # Intensity influence study
├── theta_ablation/                # Activation threshold ablation
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

## ⚙️ Key Parameters

| Parameter | MiniDDSM value | Description |
|---|---|---|
| B | 2,133 | Number of prototypes — scale with dataset size |
| p | 18 | Patch size in pixels — tune per resolution |
| θ | 0.2 | Activation threshold — controls selectivity |
| η₀ | 0.001 | Initial learning rate |
| lr decay | ×0.95/epoch | Exponential decay |
| K | 1 | Top-K nearest patches per prototype |
| Patience | 7 epochs | Early stopping |
| Init | 50 images | Prototype initialization images |

---

## 🛠️ Technical Stack

| Category | Tools |
|---|---|
| **Core framework** | PyTorch 2.0+ (unfold, index_add_) |
| **Numerical** | NumPy, SciPy |
| **Image processing** | Pillow, scikit-image (Otsu) |
| **Baselines** | scikit-learn (kNN, SVM, MLP) |
| **Visualization** | matplotlib, seaborn |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

---

## 🔗 Research

This repository is the official implementation of:

> Millimono, S. et al. **EpitopeNet: A Backpropagation-Free Prototype Learning Framework Inspired by B-Cell Epitope Recognition.** *(Under review, 2026)*

Related work:
- **HAtt-CNN** — Adaptive visual attention supervision for CNN interpretability. *(Under review, 2026)*
- **MalariaScan** — AI-based malaria detection via microscopy. Prix Jean-Marc Léger 2025.
- **OmicsFlow** — Modular NGS pipeline. [github.com/Millimono/OmicsFlow](https://github.com/Millimono/OmicsFlow)

---

## 📄 Citation

```bibtex
@software{millimono2026epitopenet,
  author    = {Millimono, Sory},
  title     = {EpitopeNet: A Backpropagation-Free Prototype Learning Framework},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Millimono/EpitopeNet}
}
```

---

## 👤 Author

**Sory Millimono**
PhD Candidate in AI · Mohammed V University – ENSIAS · E2SN Research Team

- 📧 millimono64.sm@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/sory-millimono-ai-searcher-820314162)
- 🎓 [Google Scholar](https://scholar.google.com/citations?user=5M-zcxYAAAAJ)
- 🔬 [ORCID: 0009-0005-1960-9136](https://orcid.org/0009-0005-1960-9136)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.