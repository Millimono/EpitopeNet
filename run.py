# ============================================================
# run.py — Version LVQ avec visualisations
# ============================================================

import os
import gc
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from data  import load_ddsm
from train import run_experiment
from save_load import save_model
from interpretability import visualize_best_model

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR   = ""
VAL_DIR     = ""
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
EPOCHS      = 40
LR          = 0.1
NUM_CELLS   = 6400
PATCH_SIZES = [(5, 5), (9, 9), (13, 13)]
THETA_INIT  = 0.3
SEEDS       = [42]
K           = 1
USE_INTENSITY = True


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_learning_curve(history, name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(history) + 1)
    ax.plot(epochs, history, color="#2196F3", linewidth=2, label="Multi-scale")
    ax.axhline(y=max(history), color="#2196F3", linestyle=":",
               linewidth=1.5, label=f"Best ({max(history):.2%})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (validation)", fontsize=12)
    ax.set_title(name, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.0)
    if len(history) > 1:
        ax.set_xlim(1, len(history))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Courbe sauvegardée : {save_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)
    print(f"Device: {DEVICE}")

    accs = []
    f1s = []
    aucs = []
    train_times = []
    best_acc = 0.0
    best_pop = None
    best_history = None
    best_trainer = None
    best_val_labels = None
    best_val_images = None

    print("=== EVALUATION MULTI-SCALE LVQ ===\n")
    print(f"Patch sizes : {PATCH_SIZES}")
    print(f"Feature intensité : {USE_INTENSITY}\n")
    
    for seed in SEEDS:
        set_seed(seed)
        torch.cuda.empty_cache()
        gc.collect()
        
        # train_images, train_labels, val_images, val_labels = load_ddsm(
        #     TRAIN_DIR, VAL_DIR, use_mask=True
        # )

        # APRÈS
        train_images, train_labels, val_images, val_labels = load_ddsm(
            TRAIN_DIR, VAL_DIR, 
            use_mask=True,
            mask_background_flag=True  # ✅ ACTIVER le masquage
        )

        start_time = time.time()

        acc, pop, trainer, history = run_experiment(
            train_images, train_labels,
            val_images,   val_labels,
            name          = f"MiniDDSM Multi-scale LVQ — seed={seed}",
            num_classes   = NUM_CLASSES,
            epochs        = EPOCHS,
            lr            = LR,
            num_cells     = NUM_CELLS,
            patch_sizes   = PATCH_SIZES,
            theta_init    = THETA_INIT,
            device        = DEVICE,
            K             = K,
            use_intensity = USE_INTENSITY
        )

        elapsed = time.time() - start_time
        train_times.append(elapsed)
        accs.append(acc)

        # Métriques
        preds = trainer.predict_batch(val_images, batch_size=4)
        preds_clean = [p if p is not None else 0 for p in preds]
        f1  = f1_score(val_labels, preds_clean, average="macro")
        auc = roc_auc_score(val_labels, preds_clean)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Seed {seed:5d} → Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"AUC: {auc:.4f} | Temps: {elapsed/60:.1f} min\n")

        if acc > best_acc:
            best_acc = acc
            best_pop = pop
            best_history = history
            best_trainer = trainer
            best_val_labels = val_labels
            best_val_images = val_images

    # Résumé
    print(f"{'='*60}")
    print(f"RÉSUMÉ MULTI-SCALE LVQ")
    print(f"{'='*60}")
    print(f"Patch sizes : {PATCH_SIZES}")
    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 macro  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Temps/run : {np.mean(train_times)/60:.1f} min")
    print(f"{'='*60}")

    # Rapport détaillé
    if best_trainer is not None:
        print("\n=== RAPPORT DÉTAILLÉ (meilleur modèle) ===")
        preds_best  = best_trainer.predict_batch(best_val_images, batch_size=4)
        preds_clean = [p if p is not None else 0 for p in preds_best]
        print(classification_report(
            best_val_labels, preds_clean,
            target_names=["Cancer", "Normal"]))

    # Sauvegarder
    if best_pop is not None:
        save_model(best_pop, path="figs/model_multiscale_lvq_best.pt")

    # Courbe
    if best_history is not None:
        plot_learning_curve(
            history   = best_history,
            name      = f"Multi-scale LVQ (best={best_acc:.2%})",
            save_path = "figs/learning_curve_lvq.png"
        )

    # ✅ VISUALISATIONS
    if best_pop is not None and best_trainer is not None:
        print("\n=== GÉNÉRATION VISUALISATIONS ===")
        visualize_best_model(
            pop=best_pop,
            trainer=best_trainer,
            val_images=best_val_images,
            val_labels=best_val_labels,
            class_names=["Cancer", "Normal"],
            save_dir="figs/best_model_viz",
            n_examples=16
        )

    print("\n[OK] Terminé !")
    print(f"{'='*60}")
    print(f"Modèle sauvegardé : figs/model_multiscale_lvq_best.pt")
    print(f"Visualisations    : figs/best_model_viz/")
    print(f"{'='*60}")