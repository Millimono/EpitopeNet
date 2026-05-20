# ============================================================
# run_ablation_part1.py — ABLATIONS CRITIQUES
# ============================================================

import torch
import gc
import os
import numpy as np
import pandas as pd
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

ABLATION_SEED   = 42
ABLATION_EPOCHS = 15
BASE_CONFIG = {
    'num_cells': 6400,
    'patch_sizes': [(10, 10), (18, 18), (28, 28)],
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

if __name__ == "__main__":
    print("=== ABLATION PART 1 : PARAMÈTRES CRITIQUES ===\n")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    results = []

    # ========================================================
    # 1. use_intensity (CRITIQUE)
    # ========================================================
    print("\n=== 1. ABLATION use_intensity ===")
    for use_int in [False, True]:
        set_seed(ABLATION_SEED)
        
        train_images, train_labels, val_images, val_labels = load_ddsm(
            TRAIN_DIR, VAL_DIR, use_mask=True, crop_roi=True
        )
        
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
        print(f"  ✅ use_intensity={use_int} → {acc:.4f}")
        results.append({"param": "use_intensity", "value": str(use_int), "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # 2. patch_sizes (CRITIQUE)
    # ========================================================
    print("\n=== 2. ABLATION patch_sizes ===")
    patch_configs = [
        [(5, 5), (9, 9), (13, 13)],
        [(7, 7), (13, 13), (19, 19)],
        [(10, 10), (18, 18), (28, 28)],  # Baseline
        [(8, 8), (16, 16), (24, 24)],
        [(12, 12), (20, 20), (30, 30)],
    ]
    
    for patches in patch_configs:
        set_seed(ABLATION_SEED)
        
        train_images, train_labels, val_images, val_labels = load_ddsm(
            TRAIN_DIR, VAL_DIR, use_mask=True, crop_roi=True
        )
        
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
        print(f"  ✅ patches={patch_str} → {acc:.4f}")
        results.append({"param": "patch_sizes", "value": patch_str, "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # 3. crop_roi (CRITIQUE pour démontrer importance)
    # ========================================================
    print("\n=== 3. ABLATION crop_roi ===")
    for crop in [False, True]:
        set_seed(ABLATION_SEED)
        
        train_imgs, train_lbls, val_imgs, val_lbls = load_ddsm(
            TRAIN_DIR, VAL_DIR, use_mask=True, crop_roi=crop
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
        print(f"  ✅ crop_roi={crop} → {acc:.4f}")
        results.append({"param": "crop_roi", "value": str(crop), "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # SAUVEGARDE PART 1
    # ========================================================
    df = pd.DataFrame(results)
    df.to_csv("figs/ablation_part1_critical.csv", index=False)
    
    # Afficher résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ PART 1")
    print(f"{'='*60}")
    for param in df["param"].unique():
        sub = df[df["param"] == param]
        print(f"\n{param}:")
        best_acc = sub["acc"].max()
        for _, row in sub.iterrows():
            marker = " ✅ BEST" if row["acc"] == best_acc else ""
            print(f"  {str(row['value']):>30} → {row['acc']:.4f}{marker}")

    # ========================================================
    # TÉLÉCHARGEMENT AUTOMATIQUE (COLAB)
    # ========================================================
    print("\n📥 TÉLÉCHARGEMENT AUTOMATIQUE...")
    
    try:
        from google.colab import files
        
        # Télécharger le CSV
        files.download("figs/ablation_part1_critical.csv")
        print("✅ CSV téléchargé !")
        
        # Copier aussi sur Drive (backup)
        import shutil
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(
            "figs/ablation_part1_critical.csv",
            f"{drive_path}ablation_part1_critical.csv"
        )
        print(f"✅ Backup sur Drive : {drive_path}")
        
    except ImportError:
        print("⚠️  Pas sur Colab, fichier sauvé localement")
    
    print("\n" + "="*60)
    print("✅ PART 1 TERMINÉE !")
    print("="*60)