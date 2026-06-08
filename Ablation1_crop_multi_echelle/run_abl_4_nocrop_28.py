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

SEED, EPOCHS = 42, 30
BASE = {
    'num_cells': 2133,
    'theta_init': 0.5,
    'lr': 0.001,
    'K': 1,
    'use_intensity': False
}

if __name__ == "__main__":
    print("=== RUN 4 : NOCROP, patch (28,28) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch28_lr0001',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(28, 28)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ NOCROP, (28,28), lr=0.001 → {acc:.4f}")

    result = [{"run": 4, "crop": False, "patches": "[(28,28)]",
               "lr": 0.001, "intensity": False, "acc": acc}]

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)

    pd.DataFrame(result).to_csv("figs/run4_nocrop_28.csv", index=False)
    with open("figs/run4_nocrop_28.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path="figs/model_run4_nocrop_28.pt")

    shutil.copy("figs/run4_nocrop_28.csv",      f"{drive_path}run4_nocrop_28.csv")
    shutil.copy("figs/run4_nocrop_28.json",     f"{drive_path}run4_nocrop_28.json")
    shutil.copy("figs/model_run4_nocrop_28.pt", f"{drive_path}model_run4_nocrop_28.pt")
    print("✅ Sauvegardé sur Drive !")

    print("\n✅ RUN 4 TERMINÉ !")