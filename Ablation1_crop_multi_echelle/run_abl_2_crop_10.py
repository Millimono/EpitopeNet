import sys
sys.path.append('/content/population-CBT-learning')

import torch, gc, os, pandas as pd, json
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEED, EPOCHS = 42, 15
BASE = {'num_cells': 6400, 'theta_init': 0.5, 'lr': 0.1, 'K': 1, 'use_intensity': True}

if __name__ == "__main__":
    print("=== RUN 2 : Avec CROP, patch (10,10) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=True  # ← CROP
    )

    acc, _, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='crop_patch10', num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(10, 10)],  # ← 1 SEULE ÉCHELLE
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ Avec CROP, (10,10) → {acc:.4f}")
    result = [{"run": 2, "crop": True, "patches": "[(10,10)]", "acc": acc}]

    pd.DataFrame(result).to_csv("figs/abl_2_crop_10.csv", index=False)
    with open("figs/abl_2_crop_10.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        import shutil
        from google.colab import files
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy("figs/abl_2_crop_10.csv",  f"{drive_path}abl_2_crop_10.csv")
        shutil.copy("figs/abl_2_crop_10.json", f"{drive_path}abl_2_crop_10.json")
        files.download("figs/abl_2_crop_10.csv")
        files.download("figs/abl_2_crop_10.json")
        print("✅ Sauvegardé + téléchargé !")
    except ImportError:
        pass

    print("\n✅ RUN 2 TERMINÉ !")