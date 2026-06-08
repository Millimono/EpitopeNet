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
    'num_cells': 4266,  # 2133×2
    'theta_init': 0.5,
    'lr': 0.001,
    'K': 1,
    'use_intensity': False
}

if __name__ == "__main__":
    print("=== RUN 5 : NOCROP, patches (10,10)+(18,18) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch10_18_lr0001',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(10, 10), (18, 18)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ NOCROP, (10,10)+(18,18), lr=0.001 → {acc:.4f}")

    result = [{"run": 5, "crop": False, "patches": "[(10,10),(18,18)]",
               "lr": 0.001, "intensity": False, "acc": acc}]

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)

    pd.DataFrame(result).to_csv("figs/run5_nocrop_10_18.csv", index=False)
    with open("figs/run5_nocrop_10_18.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path="figs/model_run5_nocrop_10_18.pt")

    shutil.copy("figs/run5_nocrop_10_18.csv",       f"{drive_path}run5_nocrop_10_18.csv")
    shutil.copy("figs/run5_nocrop_10_18.json",      f"{drive_path}run5_nocrop_10_18.json")
    shutil.copy("figs/model_run5_nocrop_10_18.pt",  f"{drive_path}model_run5_nocrop_10_18.pt")
    print("✅ Sauvegardé sur Drive !")

    print("\n✅ RUN 5 TERMINÉ !")