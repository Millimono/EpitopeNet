# run_experiment_gabor_cli.py
import sys, argparse, importlib, torch, gc, os, json, shutil, time

sys.path.insert(0, '/content/population-CBT-learning')

for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'model_cnn', 
                    'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()

from data      import load_ddsm
from train     import TrainerMultiScale
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES
from model_cnn import PopulationBMultiScaleGabor

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/gabor_runs/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--patch",         type=str,   default="18")
    p.add_argument("--theta",         type=float, default=0.2)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--num_cells",     type=int,   default=2133)
    p.add_argument("--K",             type=int,   default=1)
    p.add_argument("--n_orientations",type=int,   default=8)
    p.add_argument("--n_scales",      type=int,   default=4)
    p.add_argument("--name",          type=str,   default=None)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


def run(args):
    patch_list  = [int(x) for x in args.patch.split(",")]
    patch_sizes = [(p, p) for p in patch_list]

    run_name = args.name or (
        f"gabor_seed{args.seed}_patch{args.patch}"
        f"_theta{str(args.theta).replace('.','')}"
        f"_or{args.n_orientations}_sc{args.n_scales}"
    )

    print(f"\n{'='*70}\nRUN : {run_name}\n{'='*70}\n")
    os.makedirs("figs", exist_ok=True)
    os.makedirs(DRIVE_PATH, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()

    # ── Données ──────────────────────────────────────────────
    set_seed(args.seed)
    train_images, train_labels, _, _ = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, 
        use_mask=True, crop_roi=False
    )
    data_clean = torch.load(CACHE_CLEAN)
    val_images = data_clean["val_images"]
    val_labels = data_clean["val_labels"]

    print(f"Train : {len(train_images)} images")
    print(f"Val   : {len(val_images)} images\n")

    # ── Modèle Gabor ─────────────────────────────────────────
    pop = PopulationBMultiScaleGabor(
        num_cells     = args.num_cells,
        patch_sizes   = patch_sizes,
        theta_init    = args.theta,
        beta          = 5.0,
        num_classes   = NUM_CLASSES,
        K             = args.K,
        use_intensity = False,
        device        = DEVICE,
        n_orientations= args.n_orientations,
        n_scales      = args.n_scales,
    )
    trainer = TrainerMultiScale(
        population  = pop,
        num_classes = NUM_CLASSES,
        device      = DEVICE
    )

    # ── Initialisation depuis patches Gabor réels ────────────
    print("Initialisation prototypes depuis patches Gabor...")
    all_patches = []
    with torch.no_grad():
        for i in range(0, min(100, len(train_images)), 5):
            batch = torch.stack(
                train_images[i:i+5]).to(DEVICE)
            for scale_idx, ps in enumerate(pop.patch_sizes):
                patches     = pop.extract_patches_batch(batch, ps)
                patches_std = pop.preprocess_patches(patches)
                flat        = patches_std.reshape(
                    -1, patches_std.shape[-1])
                idx         = torch.randperm(
                    flat.shape[0])[:500]
                all_patches.append(flat[idx].cpu())

    all_p = torch.cat(all_patches, dim=0)
    print(f"Total patches : {all_p.shape[0]}")
    print(f"Min/Max : {all_p.min():.4f} / {all_p.max():.4f}")

    for scale_idx in range(pop.n_scales):
        B   = pop.B_per_scale[scale_idx]
        idx = torch.randperm(all_p.shape[0])[:B]
        pop.prototypes[scale_idx] = all_p[idx].to(DEVICE)

    # Vérifier activations
    images_t   = torch.stack(train_images[:4]).to(DEVICE)
    all_act, _ = pop.process_batch(images_t)
    print(f"Protos activés : {all_act[0].float().mean():.4f}\n")

    # ── Entraînement ─────────────────────────────────────────
    start_time   = time.time()
    best_acc     = 0.0
    best_protos  = [p.clone() for p in pop.prototypes]
    best_counts  = [c.clone() for c in pop.class_counts]
    best_class   = [c.clone() for c in pop.proto_class]
    patience     = 0
    max_patience = 7
    history      = []

    for epoch in range(args.epochs):
        lr_epoch = args.lr * (0.95 ** epoch)
        trainer.train_batch(
            train_images, train_labels,
            batch_size=2, lr=lr_epoch
        )
        pop.reassign_proto_class(
            train_images, train_labels, DEVICE, batch_size=2)

        preds   = trainer.predict_batch(val_images, batch_size=4)
        correct = sum(p == l for p, l in zip(preds, val_labels)
                      if p is not None)
        acc     = correct / len(val_images)
        history.append(acc)

        if acc > best_acc:
            best_acc    = acc
            best_protos = [p.clone() for p in pop.prototypes]
            best_counts = [c.clone() for c in pop.class_counts]
            best_class  = [c.clone() for c in pop.proto_class]
            patience    = 0
            marker      = "✅"
        else:
            patience += 1
            marker    = f"  (patience {patience}/{max_patience})"

        print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | "
              f"Best: {best_acc:.4f} | lr: {lr_epoch:.4f} {marker}")

        if patience >= max_patience:
            print(f"\n  Early stopping à l'epoch {epoch+1}")
            break

    pop.prototypes   = best_protos
    pop.class_counts = best_counts
    pop.proto_class  = best_class

    elapsed_min = (time.time() - start_time) / 60
    best_epoch  = int(history.index(max(history))) + 1

    print(f"\n✅ {run_name} → {best_acc:.4f} | "
          f"Best epoch: {best_epoch}/{len(history)} | "
          f"Temps: {elapsed_min:.1f} min")

    # ── Sauvegarde ───────────────────────────────────────────
    result = {
        "run_name"      : run_name,
        "model"         : "EpitopeNet-Gabor",
        "n_orientations": args.n_orientations,
        "n_scales"      : args.n_scales,
        "seed"          : args.seed,
        "patch_sizes"   : str(patch_sizes),
        "theta"         : args.theta,
        "lr"            : args.lr,
        "num_cells"     : args.num_cells,
        "K"             : args.K,
        "acc"           : best_acc,
        "best_epoch"    : best_epoch,
        "n_epochs"      : len(history),
        "time_min"      : elapsed_min,
        "val_set"       : "clean_773",
        "history"       : history,
    }

    with open(f"figs/{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    save_model(pop, path=f"figs/model_{run_name}.pt")
    shutil.copy(f"figs/{run_name}.json",
                f"{DRIVE_PATH}{run_name}.json")
    shutil.copy(f"figs/model_{run_name}.pt",
                f"{DRIVE_PATH}model_{run_name}.pt")

    print(f"✅ Sauvegardé : {DRIVE_PATH}")
    return best_acc, pop, history


if __name__ == "__main__":
    run(parse_args())