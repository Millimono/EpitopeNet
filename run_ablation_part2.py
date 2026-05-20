# ============================================================
# run_ablation_part2.py — HYPERPARAMÈTRES
# ============================================================

import torch
import gc
import os
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
    print("=== ABLATION PART 2 : HYPERPARAMÈTRES ===\n")
    torch.cuda.empty_cache()
    gc.collect()

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, use_mask=True, crop_roi=True
    )

    results = []

    # ========================================================
    # 1. theta
    # ========================================================
    print("\n=== 1. ABLATION theta ===")
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
        print(f"  ✅ theta={theta} → {acc:.4f}")
        results.append({"param": "theta", "value": theta, "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # 2. learning_rate
    # ========================================================
    print("\n=== 2. ABLATION learning_rate ===")
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
        print(f"  ✅ lr={lr} → {acc:.4f}")
        results.append({"param": "learning_rate", "value": lr, "acc": acc})
        torch.cuda.empty_cache()

    # ========================================================
    # SAUVEGARDE PART 2
    # ========================================================
    df = pd.DataFrame(results)
    df.to_csv("figs/ablation_part2_hyperparams.csv", index=False)
    print(f"   Résultats : figs/ablation_part2_hyperparams.csv")

    # ========================================================
    # TÉLÉCHARGEMENT AUTOMATIQUE (COLAB)
    # ========================================================
    print("\n📥 TÉLÉCHARGEMENT AUTOMATIQUE...")
    
    try:
        from google.colab import files
        
        files.download("figs/ablation_part2_hyperparams.csv")
        print("✅ CSV téléchargé !")
        
        import shutil
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(
            "figs/ablation_part2_hyperparams.csv",
            f"{drive_path}ablation_part2_hyperparams.csv"
        )
        print(f"✅ Backup sur Drive : {drive_path}")
        
    except ImportError:
        print("⚠️  Pas sur Colab, fichier sauvé localement")
    
    print("\n" + "="*60)
    print("✅ PART 2 TERMINÉE !")
    print("="*60)