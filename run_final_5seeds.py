import importlib
import sys
sys.path.append('/content/population-CBT-learning')
import train
importlib.reload(train)

import torch, gc, os, time, pandas as pd, json, shutil
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from data      import load_ddsm
from train     import run_experiment
from save_load import save_model
from run       import TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES, set_seed

SEEDS  = [42, 123, 456, 789, 1024]
EPOCHS = 30

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

all_results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"=== RUN seed={seed} ===")
    print(f"{'='*60}\n")

    torch.cuda.empty_cache(); gc.collect()

    # ✅ set_seed AVANT load_ddsm
    set_seed(seed)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    start_time = time.time()

    acc, pop, trainer, history = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name=f'final_seed{seed}',
        num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=CFG['lr'], num_cells=CFG['num_cells'],
        patch_sizes=CFG['patch_sizes'],
        theta_init=CFG['theta_init'],
        device=DEVICE, K=CFG['K'],
        use_intensity=CFG['use_intensity']
    )

    elapsed_min = (time.time() - start_time) / 60

    # ── Métriques détaillées ──────────────────────────────
    from model import TrainerMultiScale
    preds = trainer.predict_batch(val_images, batch_size=4)

    y_true = []
    y_pred = []
    for p, l in zip(preds, val_labels):
        if p is not None:
            y_true.append(l)
            y_pred.append(p)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1  = f1_score(y_true, y_pred, average='macro')
    acc_final = (y_true == y_pred).mean()

    # Best epoch
    best_epoch = int(np.argmax(history)) + 1
    n_epochs   = len(history)

    print(f"\n✅ seed={seed} → Acc: {acc:.4f} | F1: {f1:.4f} | "
          f"Best epoch: {best_epoch}/{n_epochs} | "
          f"Temps: {elapsed_min:.1f} min")

    result = {
        "seed"       : seed,
        "acc"        : float(acc),
        "f1_macro"   : float(f1),
        "best_epoch" : best_epoch,
        "n_epochs"   : n_epochs,
        "time_min"   : float(elapsed_min),
        "history"    : [float(h) for h in history],
        "patch"      : "[(18,18)]",
        "theta"      : CFG['theta_init'],
        "lr"         : CFG['lr'],
        "intensity"  : CFG['use_intensity'],
        "num_cells"  : CFG['num_cells'],
        "K"          : CFG['K'],
        "epochs_max" : EPOCHS,
    }
    all_results.append(result)

    # ── Sauvegarder ──────────────────────────────────────
    fname_base = f"final_seed{seed}"

    pd.DataFrame([result]).to_csv(f"figs/{fname_base}.csv", index=False)
    with open(f"figs/{fname_base}.json", "w") as f:
        json.dump(result, f, indent=2)

    save_model(pop, path=f"figs/model_{fname_base}.pt")

    shutil.copy(f"figs/{fname_base}.csv",      f"{drive_path}{fname_base}.csv")
    shutil.copy(f"figs/{fname_base}.json",     f"{drive_path}{fname_base}.json")
    shutil.copy(f"figs/model_{fname_base}.pt", f"{drive_path}model_{fname_base}.pt")

    try:
        from google.colab import files
        files.download(f"figs/{fname_base}.csv")
        files.download(f"figs/{fname_base}.json")
        print(f"✅ seed={seed} téléchargé !")
    except:
        pass

# ── Résumé final ──────────────────────────────────────────
accs   = [r["acc"]      for r in all_results]
f1s    = [r["f1_macro"] for r in all_results]
times  = [r["time_min"] for r in all_results]
epochs = [r["best_epoch"] for r in all_results]

print(f"\n{'='*60}")
print(f"RÉSUMÉ 5 SEEDS")
print(f"{'='*60}")
print(f"{'Seed':<8} {'Acc':>8} {'F1':>8} {'Best Ep':>10} {'Temps':>10}")
print("-"*50)
for r in all_results:
    print(f"  {r['seed']:<6} {r['acc']:>8.4f} {r['f1_macro']:>8.4f} "
          f"{r['best_epoch']:>10} {r['time_min']:>9.1f}m")

print(f"\n  Acc  : {pd.Series(accs).mean():.4f} ± {pd.Series(accs).std():.4f}")
print(f"  F1   : {pd.Series(f1s).mean():.4f} ± {pd.Series(f1s).std():.4f}")
print(f"  Best : {max(accs):.4f} (seed={all_results[accs.index(max(accs))]['seed']})")
print(f"  Time : {pd.Series(times).mean():.1f} min/run")

summary = {
    "acc_mean"  : float(pd.Series(accs).mean()),
    "acc_std"   : float(pd.Series(accs).std()),
    "acc_best"  : float(max(accs)),
    "f1_mean"   : float(pd.Series(f1s).mean()),
    "f1_std"    : float(pd.Series(f1s).std()),
    "time_mean" : float(pd.Series(times).mean()),
    "best_seed" : all_results[accs.index(max(accs))]['seed'],
    "runs"      : all_results
}

with open("figs/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
shutil.copy("figs/final_summary.json", f"{drive_path}final_summary.json")

try:
    from google.colab import files
    files.download("figs/final_summary.json")
except:
    pass

print("\n✅ Résumé sauvegardé sur Drive !")