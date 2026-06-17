import importlib
import sys
sys.path.append('/content/population-CBT-learning')
import train
importlib.reload(train)

import torch, gc, os, pandas as pd, json, shutil
from data      import load_ddsm
from train     import run_experiment
from save_load import save_model
from run       import TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEEDS  = [42, 123, 456, 789, 1024]
EPOCHS = 30

# ── Config finale ──────────────────────────────────────────
CFG = {
    'num_cells'    : 2133,
    'patch_sizes'  : [(18, 18)],
    'theta_init'   : 0.2,
    'lr'           : 0.001,
    'K'            : 1,
    'use_intensity': False,
}

drive_path = "/content/drive/MyDrive/ablation_results/final_5seeds/"
os.makedirs("figs", exist_ok=True)
os.makedirs(drive_path, exist_ok=True)

# ── Charger données une seule fois ─────────────────────────
train_images, train_labels, val_images, val_labels = load_ddsm(
    TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
)

all_results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"=== RUN seed={seed} ===")
    print(f"{'='*60}\n")

    torch.cuda.empty_cache(); gc.collect()
    torch.manual_seed(seed)

    acc, pop, _, history = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name=f'final_seed{seed}',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=CFG['lr'], num_cells=CFG['num_cells'],
        patch_sizes=CFG['patch_sizes'],
        theta_init=CFG['theta_init'],
        device=DEVICE, K=CFG['K'],
        use_intensity=CFG['use_intensity']
    )

    print(f"\n✅ seed={seed} → {acc:.4f}")

    result = {
        "seed"      : seed,
        "acc"       : acc,
        "history"   : history,
        "patch"     : "[(18,18)]",
        "theta"     : CFG['theta_init'],
        "lr"        : CFG['lr'],
        "intensity" : CFG['use_intensity'],
        "num_cells" : CFG['num_cells'],
    }
    all_results.append(result)

    # ── Sauvegarder ce run ────────────────────────────────
    fname_base = f"final_seed{seed}"

    pd.DataFrame([result]).to_csv(
        f"figs/{fname_base}.csv", index=False)
    with open(f"figs/{fname_base}.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path=f"figs/model_{fname_base}.pt")

    shutil.copy(f"figs/{fname_base}.csv",
                f"{drive_path}{fname_base}.csv")
    shutil.copy(f"figs/{fname_base}.json",
                f"{drive_path}{fname_base}.json")
    shutil.copy(f"figs/model_{fname_base}.pt",
                f"{drive_path}model_{fname_base}.pt")

    # ── Télécharger immédiatement ─────────────────────────
    try:
        from google.colab import files
        files.download(f"figs/{fname_base}.csv")
        files.download(f"figs/{fname_base}.json")
        print(f"✅ seed={seed} téléchargé !")
    except:
        pass

# ── Résumé final ──────────────────────────────────────────
accs = [r["acc"] for r in all_results]
print(f"\n{'='*60}")
print(f"RÉSUMÉ 5 SEEDS")
print(f"{'='*60}")
for r in all_results:
    print(f"  seed={r['seed']} → {r['acc']:.4f}")
print(f"\n  Mean : {pd.Series(accs).mean():.4f}")
print(f"  Std  : {pd.Series(accs).std():.4f}")
print(f"  Best : {max(accs):.4f}")

# Sauvegarder résumé
summary = {
    "mean": pd.Series(accs).mean(),
    "std" : pd.Series(accs).std(),
    "best": max(accs),
    "runs": all_results
}
with open("figs/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
shutil.copy("figs/final_summary.json",
            f"{drive_path}final_summary.json")
print("✅ Résumé sauvegardé sur Drive !")