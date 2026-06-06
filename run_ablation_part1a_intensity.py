import torch, gc, os, pandas as pd
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

ABLATION_SEED   = 42
ABLATION_EPOCHS = 15
BASE_CONFIG = {
    'num_cells': 6400,
    'patch_sizes': [(10, 10), (18, 18), (28, 28)],
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

if __name__ == "__main__":
    print("=== ABLATION : use_intensity ===\n")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    results = []

    for use_int in [False, True]:
        set_seed(ABLATION_SEED)
        
        train_images, train_labels, val_images, val_labels = load_ddsm(
            TRAIN_DIR, VAL_DIR,
            img_size=256,
            use_mask=True,
            crop_roi=True
        )
        
        acc, _, _, _ = run_experiment(
            train_images, train_labels, val_images, val_labels,
            name          = f"intensity={use_int}",
            num_classes   = NUM_CLASSES,
            epochs        = ABLATION_EPOCHS,
            lr            = BASE_CONFIG['lr'],
            num_cells     = BASE_CONFIG['num_cells'],
            patch_sizes   = BASE_CONFIG['patch_sizes'],
            theta_init    = BASE_CONFIG['theta_init'],
            device        = DEVICE,
            K             = BASE_CONFIG['K'],
            use_intensity = use_int
        )
        print(f"  ✅ use_intensity={use_int} → {acc:.4f}")
        results.append({"param": "use_intensity", "value": str(use_int), "acc": acc})
        torch.cuda.empty_cache()

    # Sauvegarder JSON + CSV
    import json
    df = pd.DataFrame(results)
    df.to_csv("figs/ablation_intensity.csv", index=False)
    with open("figs/ablation_intensity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    best_acc = df["acc"].max()
    for _, row in df.iterrows():
        marker = " ✅ BEST" if row["acc"] == best_acc else ""
        print(f"  use_intensity={row['value']:>5} → {row['acc']:.4f}{marker}")

    try:
        from google.colab import files
        import shutil
        files.download("figs/ablation_intensity.csv")
        files.download("figs/ablation_intensity.json")
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy("figs/ablation_intensity.csv",  f"{drive_path}ablation_intensity.csv")
        shutil.copy("figs/ablation_intensity.json", f"{drive_path}ablation_intensity.json")
        print("✅ CSV + JSON téléchargés + backup Drive !")
    except ImportError:
        pass

    print("\n✅ ABLATION use_intensity TERMINÉE !")