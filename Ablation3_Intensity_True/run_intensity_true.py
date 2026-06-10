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
    'use_intensity': True  # ← TRUE
}

if __name__ == "__main__":
    print("=== ABLATION intensity=True, patch (18,18), lr=0.001 ===\n")
    
    os.makedirs("figs", exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch18_intensity_true_lr0001',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(18, 18)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ intensity=True, (18,18), lr=0.001 → {acc:.4f}")

    result = [{"config": "intensity_true", "patches": "[(18,18)]",
               "lr": 0.001, "intensity": True, "acc": acc}]

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)

    pd.DataFrame(result).to_csv("figs/ablation_intensity_true.csv", index=False)
    with open("figs/ablation_intensity_true.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path="figs/model_intensity_true.pt")

    shutil.copy("figs/ablation_intensity_true.csv",  f"{drive_path}ablation_intensity_true.csv")
    shutil.copy("figs/ablation_intensity_true.json", f"{drive_path}ablation_intensity_true.json")
    shutil.copy("figs/model_intensity_true.pt",      f"{drive_path}model_intensity_true.pt")
    print("✅ Sauvegardé sur Drive !")

    print("\n✅ ABLATION INTENSITY TRUE TERMINÉE !")