import importlib
import sys
sys.path.append('/content/population-CBT-learning')
import train
importlib.reload(train)

import torch, gc, os, pandas as pd, json, shutil
from data      import load_ddsm
from train     import run_experiment
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEED, EPOCHS = 42, 30  # ✅ 30 epochs
BASE = {
    'num_cells': 2133,
    'theta_init': 0.5,
    'lr': 0.001,  # ✅
    'K': 1,
    'use_intensity': False  # ✅
}

if __name__ == "__main__":
    print("=== RUN 1 : NOCROP, patch (10,10), lr=0.001 ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, trainer, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch10_lr0001',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(10, 10)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ NOCROP, (10,10), lr=0.001 → {acc:.4f}")

    # Sauvegarder
    result = [{"run": 1, "crop": False, "patches": "[(10,10)]",
               "lr": 0.001, "intensity": False, "acc": acc}]
    pd.DataFrame(result).to_csv("figs/run1_nocrop_10.csv", index=False)
    with open("figs/run1_nocrop_10.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path="figs/model_run1_nocrop_10.pt")

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)
    shutil.copy("figs/run1_nocrop_10.csv",       f"{drive_path}run1_nocrop_10.csv")
    shutil.copy("figs/run1_nocrop_10.json",      f"{drive_path}run1_nocrop_10.json")
    shutil.copy("figs/model_run1_nocrop_10.pt",  f"{drive_path}model_run1_nocrop_10.pt")
    print("✅ Sauvegardé sur Drive !")

    print("\n✅ RUN 1 TERMINÉ !")