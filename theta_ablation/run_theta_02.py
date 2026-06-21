import sys, importlib

for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()
sys.path.insert(0, '/content/population-CBT-learning')

import torch, gc, os, json, shutil, time
from data      import load_ddsm
from model     import PopulationBMultiScale, TrainerMultiScale
from save_load import save_model
from run       import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

SEED, EPOCHS = 42, 30

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

# ============================================================
# CHARGER TRAIN (avec augmentation, comme à l'origine)
# ============================================================
set_seed(SEED)
train_images, train_labels, _, _ = load_ddsm(
    TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
)

# ============================================================
# CHARGER VAL PROPRE (sans aug) — utilisé DÈS l'entraînement
# ============================================================
CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
data_clean  = torch.load(CACHE_CLEAN)
val_images  = data_clean["val_images"]
val_labels  = data_clean["val_labels"]

print(f"Train : {len(train_images)} images")
print(f"Val propre : {len(val_images)} images "
      f"({sum(1 for l in val_labels if l==0)} Cancer, "
      f"{sum(1 for l in val_labels if l==1)} Normal)")

# ============================================================
# RÉENTRAÎNEMENT MANUEL (copie de run_experiment, val propre)
# ============================================================
def init_prototypes_from_data(population, images, labels, device, n_samples=50):
    all_patches_per_scale = [[] for _ in range(population.n_scales)]
    for i in range(0, min(n_samples, len(images)), 10):
        batch = images[i:i+10]
        imgs_batch = torch.stack(batch).to(device)
        for scale_idx, patch_size in enumerate(population.patch_sizes):
            patches = population.extract_patches_batch(imgs_batch, patch_size)
            patches_norm = population.preprocess_patches(patches, keep_intensity=True)
            all_patches_per_scale[scale_idx].append(patches_norm.reshape(-1, patches_norm.shape[-1]).cpu())
        del imgs_batch
        torch.cuda.empty_cache()
    for scale_idx in range(population.n_scales):
        all_patches = torch.cat(all_patches_per_scale[scale_idx], dim=0)
        B_scale = population.B_per_scale[scale_idx]
        idx = torch.randperm(all_patches.shape[0])[:B_scale]
        population.prototypes[scale_idx] = all_patches[idx].to(device)

torch.cuda.empty_cache(); gc.collect()
set_seed(SEED)

pop = PopulationBMultiScale(
    num_cells     = CFG['num_cells'],
    patch_sizes   = CFG['patch_sizes'],
    theta_init    = CFG['theta_init'],
    beta          = 5.0,
    num_classes   = NUM_CLASSES,
    K             = CFG['K'],
    use_intensity = CFG['use_intensity'],
    device        = DEVICE
)
trainer = TrainerMultiScale(population=pop, num_classes=NUM_CLASSES, device=DEVICE)
init_prototypes_from_data(pop, train_images, train_labels, DEVICE, n_samples=50)

best_acc     = 0.0
best_protos  = [p.clone() for p in pop.prototypes]
best_counts  = [c.clone() for c in pop.class_counts]
best_classes = [c.clone() for c in pop.proto_class]
patience, max_patience = 0, 7
history = []

start_time = time.time()

for epoch in range(EPOCHS):
    lr_epoch = CFG['lr'] * (0.95 ** epoch)

    trainer.train_batch(train_images, train_labels, batch_size=2, lr=lr_epoch)
    pop.reassign_proto_class(train_images, train_labels, DEVICE, batch_size=2)

    # ✅ ÉVALUATION SUR LE VAL PROPRE
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
        marker       = "✅"
    elif acc < best_acc - 0.05:
        pop.prototypes   = [p.clone() for p in best_protos]
        pop.class_counts = [c.clone() for c in best_counts]
        pop.proto_class  = [c.clone() for c in best_classes]
        patience        += 1
        marker           = f"restauré (patience {patience}/{max_patience})"
    else:
        patience += 1
        marker    = f"  (patience {patience}/{max_patience})"

    print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | "
          f"lr: {lr_epoch:.4f} {marker}")

    if patience >= max_patience:
        print(f"\n  Early stopping à l'epoch {epoch+1}")
        break

elapsed_min = (time.time() - start_time) / 60

pop.prototypes   = [p.clone() for p in best_protos]
pop.class_counts = [c.clone() for c in best_counts]
pop.proto_class  = [c.clone() for c in best_classes]

best_epoch = int(history.index(best_acc)) + 1
n_epochs   = len(history)

print(f"\n>>> BEST ACCURACY (val propre): {best_acc:.4f} "
      f"(epoch {best_epoch}/{n_epochs}) | Temps: {elapsed_min:.1f} min")

# ============================================================
# SAUVEGARDER
# ============================================================
result = {
    "seed"       : SEED,
    "acc"        : float(best_acc),
    "best_epoch" : best_epoch,
    "n_epochs"   : n_epochs,
    "time_min"   : float(elapsed_min),
    "history"    : [float(h) for h in history],
    "val_set"    : "clean_773",
    **CFG
}

fname_base = f"final_seed{SEED}_cleanval"
with open(f"figs/{fname_base}.json", "w") as f:
    json.dump(result, f, indent=2)

save_model(pop, path=f"figs/model_{fname_base}.pt")

shutil.copy(f"figs/{fname_base}.json", f"{drive_path}{fname_base}.json")
shutil.copy(f"figs/model_{fname_base}.pt", f"{drive_path}model_{fname_base}.pt")

try:
    from google.colab import files
    files.download(f"figs/{fname_base}.json")
except:
    pass

print("✅ Sauvegardé sur Drive !")