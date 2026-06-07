import sys
sys.path.append('/content/population-CBT-learning')

import torch, gc, os, pandas as pd, json
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEED, EPOCHS = 42, 15
BASE = {'num_cells': 4266, 'theta_init': 0.5, 'lr': 0.1, 'K': 1, 'use_intensity': True}  # 2133×2

if __name__ == "__main__":
    print("=== RUN 5 : NOCROP, patches (10,10)+(18,18) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False  # ← NOCROP
    )

    acc, _, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch10_18', num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(10, 10), (18, 18)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ NOCROP, (10,10)+(18,18) → {acc:.4f}")
    result = [{"run": 5, "crop": False, "patches": "[(10,10),(18,18)]", "acc": acc}]

    pd.DataFrame(result).to_csv("figs/abl_5_nocrop_10_18.csv", index=False)
    with open("figs/abl_5_nocrop_10_18.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        import shutil
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy("figs/abl_5_nocrop_10_18.csv",  f"{drive_path}abl_5_nocrop_10_18.csv")
        shutil.copy("figs/abl_5_nocrop_10_18.json", f"{drive_path}abl_5_nocrop_10_18.json")
        print("✅ Sauvegardé sur Drive !")
    except:
        pass

    print("\n✅ RUN 5 TERMINÉ !")