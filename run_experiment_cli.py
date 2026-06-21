# ============================================================
# run_experiment_cli.py — Script unifié pour toutes les expériences
# Usage:
#   python run_experiment_cli.py --mode seed --seed 123
#   python run_experiment_cli.py --mode theta --theta 0.3 --seed 42
#   python run_experiment_cli.py --mode patch --patch 10 --seed 42
#   python run_experiment_cli.py --mode patch --patch 10,18 --seed 42
#   python run_experiment_cli.py --mode intensity --intensity true --seed 42
#   python run_experiment_cli.py --mode custom --patch 18 --theta 0.2 --lr 0.001 --seed 42
# ============================================================

import sys, argparse, importlib

sys.path.insert(0, '/content/population-CBT-learning')
for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()

import torch, gc, os, json, shutil, time
import numpy as np
from sklearn.metrics import f1_score
from data      import load_ddsm
from model     import PopulationBMultiScale, TrainerMultiScale
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/unified_runs/"


# ============================================================
# PARSER D'ARGUMENTS
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="EpitopeNet — runs unifiés")
    p.add_argument("--mode", type=str, default="custom",
                    choices=["seed", "theta", "patch", "intensity", "custom"],
                    help="Type d'expérience (juste pour nommer le run)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patch", type=str, default="18",
                    help="Taille(s) de patch séparées par virgule, ex: '18' ou '10,18' ou '10,18,28'")
    p.add_argument("--theta", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_cells", type=int, default=2133)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--intensity", type=str, default="false",
                    choices=["true", "false"])
    p.add_argument("--n_samples", type=int, default=50,
                    help="Nb images pour initialisation des prototypes")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--name", type=str, default=None,
                    help="Nom custom du run (sinon auto-généré)")
    # Pour usage notebook (pas de vrais argv)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


# ============================================================
# INIT PROTOTYPES
# ============================================================
def init_prototypes_from_data(population, images, labels, device, n_samples=50):
    all_patches_per_scale = [[] for _ in range(population.n_scales)]
    for i in range(0, min(n_samples, len(images)), 10):
        batch = images[i:i+10]
        imgs_batch = torch.stack(batch).to(device)
        for scale_idx, patch_size in enumerate(population.patch_sizes):
            patches = population.extract_patches_batch(imgs_batch, patch_size)
            patches_norm = population.preprocess_patches(patches, keep_intensity=True)
            all_patches_per_scale[scale_idx].append(
                patches_norm.reshape(-1, patches_norm.shape[-1]).cpu())
        del imgs_batch
        torch.cuda.empty_cache()
    for scale_idx in range(population.n_scales):
        all_patches = torch.cat(all_patches_per_scale[scale_idx], dim=0)
        B_scale = population.B_per_scale[scale_idx]
        idx = torch.randperm(all_patches.shape[0])[:B_scale]
        population.prototypes[scale_idx] = all_patches[idx].to(device)


# ============================================================
# RUN PRINCIPAL
# ============================================================
def run(args):
    # ── Parsing patch_sizes ───────────────────────────────
    patch_list = [int(x) for x in args.patch.split(",")]
    patch_sizes = [(p, p) for p in patch_list]
    use_intensity = args.intensity == "true"

    # ── Nom du run ─────────────────────────────────────────
    if args.name:
        run_name = args.name
    else:
        patch_str = "_".join(str(p) for p in patch_list)
        run_name = (f"{args.mode}_seed{args.seed}_patch{patch_str}"
                    f"_theta{str(args.theta).replace('.','')}"
                    f"_lr{str(args.lr).replace('.','')}"
                    f"_int{use_intensity}")

    print(f"\n{'='*70}")
    print(f"RUN : {run_name}")
    print(f"  mode={args.mode} | seed={args.seed} | patch={patch_sizes} | "
          f"theta={args.theta} | lr={args.lr} | num_cells={args.num_cells} | "
          f"K={args.K} | intensity={use_intensity} | n_samples={args.n_samples}")
    print(f"{'='*70}\n")

    os.makedirs("figs", exist_ok=True)
    os.makedirs(DRIVE_PATH, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()

    # ── TRAIN : cache existant (avec augmentation) ─────────
    set_seed(args.seed)
    train_images, train_labels, _, _ = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    # ── VAL : cache propre (sans augmentation) ─────────────
    data_clean = torch.load(CACHE_CLEAN)
    val_images = data_clean["val_images"]
    val_labels = data_clean["val_labels"]
    print(f"Train : {len(train_images)} images")
    print(f"Val propre : {len(val_images)} images "
          f"({sum(1 for l in val_labels if l==0)} Cancer, "
          f"{sum(1 for l in val_labels if l==1)} Normal)\n")

    # ── Construction du modèle ──────────────────────────────
    set_seed(args.seed)
    pop = PopulationBMultiScale(
        num_cells     = args.num_cells,
        patch_sizes   = patch_sizes,
        theta_init    = args.theta,
        beta          = 5.0,
        num_classes   = NUM_CLASSES,
        K             = args.K,
        use_intensity = use_intensity,
        device        = DEVICE
    )
    trainer = TrainerMultiScale(population=pop, num_classes=NUM_CLASSES, device=DEVICE)
    init_prototypes_from_data(pop, train_images, train_labels, DEVICE,
                               n_samples=args.n_samples)

    # ── Boucle d'entraînement ───────────────────────────────
    best_acc     = 0.0
    best_protos  = [p.clone() for p in pop.prototypes]
    best_counts  = [c.clone() for c in pop.class_counts]
    best_classes = [c.clone() for c in pop.proto_class]
    patience = 0
    history = []
    final_preds = None

    start_time = time.time()

    for epoch in range(args.epochs):
        lr_epoch = args.lr * (0.95 ** epoch)
        trainer.train_batch(train_images, train_labels, batch_size=2, lr=lr_epoch)
        pop.reassign_proto_class(train_images, train_labels, DEVICE, batch_size=2)

        preds = trainer.predict_batch(val_images, batch_size=4)
        correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
        acc = correct / len(val_images)
        history.append(acc)

        if acc > best_acc:
            best_acc     = acc
            best_protos  = [p.clone() for p in pop.prototypes]
            best_counts  = [c.clone() for c in pop.class_counts]
            best_classes = [c.clone() for c in pop.proto_class]
            patience     = 0
            final_preds  = preds
            marker = "✅"
        elif acc < best_acc - 0.05:
            pop.prototypes   = [p.clone() for p in best_protos]
            pop.class_counts = [c.clone() for c in best_counts]
            pop.proto_class  = [c.clone() for c in best_classes]
            patience += 1
            marker = f"restauré (patience {patience}/{args.patience})"
        else:
            patience += 1
            marker = f"  (patience {patience}/{args.patience})"

        print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
              f"lr: {lr_epoch:.4f} {marker}")

        if patience >= args.patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}")
            break

    elapsed_min = (time.time() - start_time) / 60

    pop.prototypes   = [p.clone() for p in best_protos]
    pop.class_counts = [c.clone() for c in best_counts]
    pop.proto_class  = [c.clone() for c in best_classes]

    best_epoch = int(history.index(best_acc)) + 1
    n_epochs   = len(history)

    y_true = [l for p, l in zip(final_preds, val_labels) if p is not None]
    y_pred = [p for p in final_preds if p is not None]
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n>>> BEST ACCURACY (val propre): {best_acc:.4f} | F1: {f1:.4f} "
          f"(epoch {best_epoch}/{n_epochs}) | Temps: {elapsed_min:.1f} min")

    # ── Sauvegarde ───────────────────────────────────────────
    result = {
        "run_name"     : run_name,
        "mode"         : args.mode,
        "seed"         : args.seed,
        "patch_sizes"  : patch_list,
        "theta"        : args.theta,
        "lr"           : args.lr,
        "num_cells"    : args.num_cells,
        "K"            : args.K,
        "use_intensity": use_intensity,
        "n_samples"    : args.n_samples,
        "acc"          : float(best_acc),
        "f1_macro"     : float(f1),
        "best_epoch"   : best_epoch,
        "n_epochs"     : n_epochs,
        "time_min"     : float(elapsed_min),
        "history"      : [float(h) for h in history],
        "val_set"      : "clean_773",
    }

    with open(f"figs/{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    save_model(pop, path=f"figs/model_{run_name}.pt")

    shutil.copy(f"figs/{run_name}.json", f"{DRIVE_PATH}{run_name}.json")
    shutil.copy(f"figs/model_{run_name}.pt", f"{DRIVE_PATH}model_{run_name}.pt")

    try:
        from google.colab import files
        files.download(f"figs/{run_name}.json")
    except:
        pass

    print(f"✅ Sauvegardé : {DRIVE_PATH}{run_name}.json / model_{run_name}.pt")
    return result


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    run(args)