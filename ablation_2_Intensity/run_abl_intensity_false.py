import sys
sys.path.append('/content/population-CBT-learning')

import torch, gc, os, pandas as pd, json, shutil
from data      import load_ddsm
from train     import run_experiment
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEED, EPOCHS = 42, 15
BASE = {
    'num_cells': 2133,
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': False
}

if __name__ == "__main__":
    print("=== ABLATION intensity=False, patch (18,18) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, trainer, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch18_nointensity',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(18, 18)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=False
    )

    print(f"\n✅ intensity=False, (18,18) → {acc:.4f}")
    result = [{"param": "use_intensity", "value": "False",
               "patches": "[(18,18)]", "crop": False, "acc": acc}]

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)

    # CSV + JSON
    pd.DataFrame(result).to_csv("figs/abl_intensity_false.csv", index=False)
    with open("figs/abl_intensity_false.json", "w") as f:
        json.dump(result, f, indent=2)

    # ✅ MODÈLE
    save_model(pop, path="figs/model_abl_intensity_false.pt")

    # Backup Drive
    shutil.copy("figs/abl_intensity_false.csv",       f"{drive_path}abl_intensity_false.csv")
    shutil.copy("figs/abl_intensity_false.json",      f"{drive_path}abl_intensity_false.json")
    shutil.copy("figs/model_abl_intensity_false.pt",  f"{drive_path}model_abl_intensity_false.pt")
    print("✅ Tout sauvegardé sur Drive !")

    print("\n✅ ABLATION INTENSITY TERMINÉE !")