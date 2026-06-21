# ============================================================
# run_experiment_cli.py — Fidèle à 100% à train.run_experiment
# ============================================================

import sys, argparse, importlib

sys.path.insert(0, '/content/population-CBT-learning')
for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()

import torch, gc, os, json, shutil, time
from data      import load_ddsm
from train     import run_experiment
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/unified_runs/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="custom")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patch", type=str, default="18")
    p.add_argument("--theta", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_cells", type=int, default=2133)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--intensity", type=str, default="false", choices=["true", "false"])
    p.add_argument("--name", type=str, default=None)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


def run(args):
    patch_list  = [int(x) for x in args.patch.split(",")]
    patch_sizes = [(p, p) for p in patch_list]
    use_intensity = args.intensity == "true"

    if args.name:
        run_name = args.name
    else:
        patch_str = "_".join(str(p) for p in patch_list)
        run_name = (f"{args.mode}_seed{args.seed}_patch{patch_str}"
                    f"_theta{str(args.theta).replace('.','')}"
                    f"_lr{str(args.lr).replace('.','')}_int{use_intensity}")

    print(f"\n{'='*70}\nRUN : {run_name}\n{'='*70}\n")
    os.makedirs("figs", exist_ok=True)
    os.makedirs(DRIVE_PATH, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()

    # ── TRAIN : cache existant (avec augmentation) — INCHANGÉ ──
    set_seed(args.seed)
    train_images, train_labels, _, _ = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    # ── VAL : SEUL changement, on remplace par le cache propre ──
    data_clean = torch.load(CACHE_CLEAN)
    val_images = data_clean["val_images"]
    val_labels = data_clean["val_labels"]
    print(f"Train : {len(train_images)} images")
    print(f"Val propre : {len(val_images)} images "
          f"({sum(1 for l in val_labels if l==0)} Cancer, "
          f"{sum(1 for l in val_labels if l==1)} Normal)\n")

    # ── APPEL STRICTEMENT IDENTIQUE À TON SCRIPT ORIGINAL ────────
    
    start_time = time.time()

    acc, pop, _, history = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name=run_name,
        num_classes=NUM_CLASSES, epochs=args.epochs,
        lr=args.lr, num_cells=args.num_cells,
        patch_sizes=patch_sizes,
        theta_init=args.theta, device=DEVICE,
        K=args.K, use_intensity=use_intensity
    )

    elapsed_min = (time.time() - start_time) / 60
    best_epoch  = int(history.index(max(history))) + 1

    print(f"\n✅ {run_name} → {acc:.4f} | Best epoch: {best_epoch}/{len(history)} "
          f"| Temps: {elapsed_min:.1f} min")

    # ── Sauvegarde — même format que tes scripts originaux ────────
    result = [{
        "run_name": run_name, "mode": args.mode, "seed": args.seed,
        "patch_sizes": str(patch_sizes), "theta": args.theta, "lr": args.lr,
        "num_cells": args.num_cells, "K": args.K,
        "intensity": use_intensity,
        "acc": acc, "best_epoch": best_epoch, "n_epochs": len(history),
        "time_min": elapsed_min, "val_set": "clean_773",
    }]

    import pandas as pd
    pd.DataFrame(result).to_csv(f"figs/{run_name}.csv", index=False)
    with open(f"figs/{run_name}.json", "w") as f:
        json.dump({**result[0], "history": history}, f, indent=2)

    save_model(pop, path=f"figs/model_{run_name}.pt")

    shutil.copy(f"figs/{run_name}.csv",  f"{DRIVE_PATH}{run_name}.csv")
    shutil.copy(f"figs/{run_name}.json", f"{DRIVE_PATH}{run_name}.json")
    shutil.copy(f"figs/model_{run_name}.pt", f"{DRIVE_PATH}model_{run_name}.pt")

    try:
        from google.colab import files
        files.download(f"figs/{run_name}.json")
    except:
        pass

    print(f"✅ Sauvegardé sur Drive : {run_name}")
    return acc, pop, history


if __name__ == "__main__":
    run(parse_args())