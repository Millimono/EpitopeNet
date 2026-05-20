# ============================================================
# run_ablation_part3.py — SCALABILITÉ
# ============================================================

import os

import torch
import gc
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
    print("=== ABLATION PART 3 : SCALABILITÉ ===\n")
    torch.cuda.empty_cache()
    gc.collect()

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, use_mask=True, crop_roi=True
    )

    results = []

    # ========================================================
    # 1. num_cells
    # ========================================================
    print("\n=== 1. ABLATION num_cells ===")
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
        print(f"  ✅ num_cells={num_cells} → {acc:.4f}")
        results.append({"param": "num_cells", "value": num_cells, "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # 2. K
    # ========================================================
    print("\n=== 2. ABLATION K ===")
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
        print(f"  ✅ K={K} → {acc:.4f}")
        results.append({"param": "K", "value": K, "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # SAUVEGARDE PART 3
    # ========================================================
    df = pd.DataFrame(results)
    df.to_csv("figs/ablation_part3_scalability.csv", index=False)
    print(f"   Résultats : figs/ablation_part3_scalability.csv")

    # ========================================================
    # TÉLÉCHARGEMENT AUTOMATIQUE (COLAB)
    # ========================================================
    print("\n📥 TÉLÉCHARGEMENT AUTOMATIQUE...")
    
    try:
        from google.colab import files
        
        files.download("figs/ablation_part3_scalability.csv")
        print("✅ CSV téléchargé !")
        
        import shutil
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(
            "figs/ablation_part3_scalability.csv",
            f"{drive_path}ablation_part3_scalability.csv"
        )
        print(f"✅ Backup sur Drive : {drive_path}")
        
    except ImportError:
        print("⚠️  Pas sur Colab, fichier sauvé localement")
    
    print("\n" + "="*60)
    print("✅ PART 3 TERMINÉE !")
    print("="*60)