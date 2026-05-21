# ============================================================
# run_ablation_part1a.py — use_intensity + crop_roi
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
    print("=== ABLATION PART 1a : use_intensity + crop_roi ===\n")
    print("⏱️  Durée estimée : 2 heures")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    results = []

    # ========================================================
    # 1. use_intensity
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
    # 2. crop_roi
    # ========================================================
    print("\n=== 2. ABLATION crop_roi ===")
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
    # SAUVEGARDE
    # ========================================================
    df = pd.DataFrame(results)
    df.to_csv("figs/ablation_part1a.csv", index=False)
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ PART 1a")
    print(f"{'='*60}")
    for param in df["param"].unique():
        sub = df[df["param"] == param]
        print(f"\n{param}:")
        best_acc = sub["acc"].max()
        for _, row in sub.iterrows():
            marker = " ✅ BEST" if row["acc"] == best_acc else ""
            print(f"  {str(row['value']):>30} → {row['acc']:.4f}{marker}")

    # TÉLÉCHARGEMENT
    print("\n📥 TÉLÉCHARGEMENT AUTOMATIQUE...")
    try:
        from google.colab import files
        files.download("figs/ablation_part1a.csv")
        print("✅ CSV téléchargé !")
        
        import shutil
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy("figs/ablation_part1a.csv", f"{drive_path}ablation_part1a.csv")
        print(f"✅ Backup sur Drive : {drive_path}")
    except ImportError:
        print("⚠️  Pas sur Colab, fichier sauvé localement")
    
    print("\n" + "="*60)
    print("✅ PART 1a TERMINÉE !")
    print("="*60)