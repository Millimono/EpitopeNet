# ============================================================
# run_ablation_multiscale.py — Ablation study COMPLET
# ============================================================

import torch
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES, LR

ABLATION_SEED   = 42
ABLATION_EPOCHS = 15  # Plus d'epochs pour multi-échelle
BASE_CONFIG = {
    'num_cells': 6400,
    'patch_sizes': [(10, 10), (18, 18), (28, 28)],
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

# ============================================================
if __name__ == "__main__":
    print("=== ABLATION STUDY MULTI-SCALE ===\n")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    set_seed(ABLATION_SEED)
    
    # Charger données
    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, 
        use_mask=True,
        crop_roi=True  # ✅ CROP ROI comme dans ton meilleur run
    )

    results = []

    # ============================================================
    # 1. ABLATION Feature Intensité (CRITIQUE)
    # ============================================================
    print("\n=== 1. ABLATION use_intensity ===")
    for use_int in [False, True]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"intensity={use_int}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = use_int
        )
        print(f"  use_intensity={use_int} → {acc:.4f}")
        results.append({"param": "use_intensity", "value": str(use_int), "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 2. ABLATION Tailles patches (CRITIQUE)
    # ============================================================
    print("\n=== 2. ABLATION patch_sizes ===")
    patch_configs = [
        [(5, 5), (9, 9), (13, 13)],      # Petits (original 128×128)
        [(7, 7), (13, 13), (19, 19)],    # Intermédiaire
        [(10, 10), (18, 18), (28, 28)],  # Adapté 256×256 (baseline)
        [(8, 8), (16, 16), (24, 24)],    # Alternative
        [(12, 12), (20, 20), (30, 30)],  # Plus grands
    ]
    
    for patches in patch_configs:
        set_seed(ABLATION_SEED)
        patch_str = str(patches)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"patches={patch_str}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = patches,
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  patches={patch_str} → {acc:.4f}")
        results.append({"param": "patch_sizes", "value": patch_str, "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 3. ABLATION Seuil θ (IMPORTANT)
    # ============================================================
    print("\n=== 3. ABLATION theta ===")
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"theta={theta}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = theta,
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  theta={theta} → {acc:.4f}")
        results.append({"param": "theta", "value": theta, "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 4. ABLATION Nombre prototypes (SCALABILITÉ)
    # ============================================================
    print("\n=== 4. ABLATION num_cells ===")
    for num_cells in [3200, 6400, 9600, 12800]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"num_cells={num_cells}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = num_cells,
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  num_cells={num_cells} → {acc:.4f}")
        results.append({"param": "num_cells", "value": num_cells, "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 5. ABLATION Learning rate (CONVERGENCE)
    # ============================================================
    print("\n=== 5. ABLATION learning_rate ===")
    for lr in [0.05, 0.1, 0.15, 0.2]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"lr={lr}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = lr,
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  lr={lr} → {acc:.4f}")
        results.append({"param": "learning_rate", "value": lr, "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 6. ABLATION K neighbors (ACTIVATION)
    # ============================================================
    print("\n=== 6. ABLATION K ===")
    for K in [1, 2, 3, 5]:
        set_seed(ABLATION_SEED)
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name        = f"K={K}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = K,
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  K={K} → {acc:.4f}")
        results.append({"param": "K", "value": K, "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # 7. ABLATION CROP ROI (CRITIQUE pour démontrer importance)
    # ============================================================
    print("\n=== 7. ABLATION crop_roi ===")
    for crop in [False, True]:
        set_seed(ABLATION_SEED)
        
        # Recharger données avec/sans crop
        train_imgs, train_lbls, val_imgs, val_lbls = load_ddsm(
            TRAIN_DIR, VAL_DIR, 
            use_mask=True,
            crop_roi=crop
        )
        
        acc, _, _, _ = run_experiment(
            train_imgs, train_lbls, val_imgs, val_lbls,
            name        = f"crop_roi={crop}",
            num_classes = NUM_CLASSES,
            epochs      = ABLATION_EPOCHS,
            lr          = BASE_CONFIG['lr'],
            num_cells   = BASE_CONFIG['num_cells'],
            patch_sizes = BASE_CONFIG['patch_sizes'],
            theta_init  = BASE_CONFIG['theta_init'],
            device      = DEVICE,
            K           = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  crop_roi={crop} → {acc:.4f}")
        results.append({"param": "crop_roi", "value": str(crop), "acc": acc})
        torch.cuda.empty_cache()

    # ============================================================
    # RÉSUMÉ + SAUVEGARDE
    # ============================================================
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ ABLATION STUDY")
    print(f"{'='*60}")
    
    for param in df["param"].unique():
        sub = df[df["param"] == param].copy()
        # Convertir value en numérique si possible pour tri
        try:
            sub['value_num'] = pd.to_numeric(sub['value'])
            sub = sub.sort_values('value_num')
        except:
            pass
        
        print(f"\n{param}:")
        best_acc = sub["acc"].max()
        for _, row in sub.iterrows():
            marker = " ✅ BEST" if row["acc"] == best_acc else ""
            delta = row["acc"] - sub["acc"].min()
            print(f"  {str(row['value']):>25} → {row['acc']:.4f} (Δ={delta:+.4f}){marker}")

    # Sauvegarder CSV
    df.to_csv("figs/ablation_multiscale_results.csv", index=False)
    print(f"\n[OK] Résultats sauvés : figs/ablation_multiscale_results.csv")

    # ============================================================
    # FIGURE VISUALISATION
    # ============================================================
    params = df["param"].unique()
    n_params = len(params)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        sub = df[df["param"] == param].copy()
        
        # Trier si numérique
        try:
            sub['value_num'] = pd.to_numeric(sub['value'])
            sub = sub.sort_values('value_num')
        except:
            pass
        
        best_val = sub.loc[sub["acc"].idxmax(), "value"]
        colors = ["#4CAF50" if str(v) == str(best_val) else "#2196F3"
                  for v in sub["value"]]
        
        x_labels = [str(v) for v in sub["value"]]
        ax.bar(range(len(sub)), sub["acc"], color=colors, alpha=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title(f"{param}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0.5, 0.9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=BASE_CONFIG.get('baseline_acc', 0.825), 
                   color='red', linestyle='--', linewidth=1, 
                   label='Best config')

    # Cacher axes non utilisés
    for idx in range(len(params), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        f"Ablation Study Multi-Scale LVQ — MiniDDSM 256×256\n"
        f"({ABLATION_EPOCHS} epochs, seed={ABLATION_SEED})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("figs/ablation_multiscale_study.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("[OK] Figure sauvée : figs/ablation_multiscale_study.png")

    # ============================================================
    # TABLE LATEX (pour article)
    # ============================================================
    print("\n=== TABLE LATEX ===\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation study results on MiniDDSM (256×256)}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Parameter & Configuration & Accuracy \\\\")
    print("\\hline")
    
    for param in df["param"].unique():
        sub = df[df["param"] == param]
        best_acc = sub["acc"].max()
        for _, row in sub.iterrows():
            marker = "$^{\\star}$" if row["acc"] == best_acc else ""
            print(f"{param} & {row['value']} & {row['acc']:.4f}{marker} \\\\")
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\label{tab:ablation}")
    print("\\end{table}")

    print("\n✅ Ablation study terminée !")