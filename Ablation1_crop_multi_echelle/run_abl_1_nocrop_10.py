import sys
sys.path.append('/content/population-CBT-learning')

import torch, gc, os, pandas as pd, json
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES
from save_load import save_model
from interpretability import visualize_best_model

SEED, EPOCHS = 42, 15
BASE = {
    'num_cells': 2133,  # ✅ 1 seule échelle
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

if __name__ == "__main__":
    print("=== RUN 1 : Sans CROP, patch (10,10) ===\n")
    torch.cuda.empty_cache(); gc.collect(); os.makedirs("figs", exist_ok=True)
    set_seed(SEED)

    train_images, train_labels, val_images, val_labels = load_ddsm(
        TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
    )

    acc, pop, trainer, _ = run_experiment(
        train_images, train_labels, val_images, val_labels,
        name='nocrop_patch10', num_classes=NUM_CLASSES, epochs=EPOCHS,
        lr=BASE['lr'], num_cells=BASE['num_cells'],
        patch_sizes=[(10, 10)],
        theta_init=BASE['theta_init'], device=DEVICE,
        K=BASE['K'], use_intensity=BASE['use_intensity']
    )

    print(f"\n✅ Sans CROP, (10,10) → {acc:.4f}")

    # ✅ SAUVEGARDER MODÈLE
    save_model(pop, path="figs/model_abl_1_nocrop_10.pt")

    # ✅ GÉNÉRER VISUALISATIONS
    print("\n=== GÉNÉRATION VISUALISATIONS ===")
    visualize_best_model(
        pop=pop,
        trainer=trainer,
        val_images=val_images,
        val_labels=val_labels,
        class_names=["Cancer", "Normal"],
        save_dir="figs/viz_abl_1_nocrop_10",
        n_examples=8
    )

    # ✅ SAUVEGARDER RÉSULTATS
    result = [{"run": 1, "crop": False, "patches": "[(10,10)]", "acc": acc}]
    pd.DataFrame(result).to_csv("figs/abl_1_nocrop_10.csv", index=False)
    with open("figs/abl_1_nocrop_10.json", "w") as f:
        json.dump(result, f, indent=2)

    # ✅ ZIPPER VISUALISATIONS
    import shutil
    shutil.make_archive("figs/viz_abl_1_nocrop_10", 'zip', "figs/viz_abl_1_nocrop_10")

    # ✅ TÉLÉCHARGER TOUT
    try:
        from google.colab import files
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)

        # CSV + JSON
        shutil.copy("figs/abl_1_nocrop_10.csv",  f"{drive_path}abl_1_nocrop_10.csv")
        shutil.copy("figs/abl_1_nocrop_10.json", f"{drive_path}abl_1_nocrop_10.json")

        # Modèle
        shutil.copy("figs/model_abl_1_nocrop_10.pt", f"{drive_path}model_abl_1_nocrop_10.pt")

        # Visualisations ZIP
        shutil.copy("figs/viz_abl_1_nocrop_10.zip", f"{drive_path}viz_abl_1_nocrop_10.zip")

        files.download("figs/abl_1_nocrop_10.csv")
        files.download("figs/abl_1_nocrop_10.json")
        files.download("figs/viz_abl_1_nocrop_10.zip")
        print("✅ Tout téléchargé !")
    except ImportError:
        pass

    print("\n✅ RUN 1 TERMINÉ !")