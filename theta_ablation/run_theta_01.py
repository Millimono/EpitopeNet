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
THETA = 0.1  # ← CHANGER ICI

if __name__ == "__main__":
    print(f"=== ABLATION theta={THETA}, patch (18,18), lr=0.001 ===\n")
    os.makedirs("figs", exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, _, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name=f'nocrop_patch18_theta{THETA}_lr0001',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=0.001, num_cells=2133,
        patch_sizes=[(18, 18)],
        theta_init=THETA,
        device=DEVICE, K=1, use_intensity=False
    )

    print(f"\n✅ theta={THETA} → {acc:.4f}")
    result = [{"param": "theta", "value": THETA, "patches": "[(18,18)]",
               "lr": 0.001, "intensity": False, "acc": acc}]

    drive_path = "/content/drive/MyDrive/ablation_results/"
    os.makedirs(drive_path, exist_ok=True)

    pd.DataFrame(result).to_csv(f"figs/ablation_theta_{str(THETA).replace('.','')}.csv", index=False)
    with open(f"figs/ablation_theta_{str(THETA).replace('.','')}.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path=f"figs/model_theta_{str(THETA).replace('.','')}.pt")

    for ext in ['.csv', '.json']:
        fname = f"ablation_theta_{str(THETA).replace('.',''  )}{ext}"
        shutil.copy(f"figs/{fname}", f"{drive_path}{fname}")
    shutil.copy(f"figs/model_theta_{str(THETA).replace('.','')}.pt",
                f"{drive_path}model_theta_{str(THETA).replace('.','')}.pt")
    print("✅ Sauvegardé sur Drive !")
    print(f"\n✅ THETA {THETA} TERMINÉ !")