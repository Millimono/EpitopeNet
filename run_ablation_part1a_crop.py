import torch, gc, os, pandas as pd, json
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

ABLATION_SEED   = 42
ABLATION_EPOCHS = 10  # ✅ Réduit
BASE_CONFIG = {
    'num_cells': 6400,
    'patch_sizes': [(10, 10), (18, 18), (28, 28)],
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

if __name__ == "__main__":
    print("=== ABLATION : crop_roi ===\n")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    results = []

    for crop in [False, True]:
        set_seed(ABLATION_SEED)
        
        train_imgs, train_lbls, val_imgs, val_lbls = load_ddsm(
            TRAIN_DIR, VAL_DIR,
            img_size=256,
            use_mask=True,
            crop_roi=crop
        )
        
        acc, _, _, _ = run_experiment(
            train_imgs, train_lbls, val_imgs, val_lbls,
            name          = f"crop_roi={crop}",
            num_classes   = NUM_CLASSES,
            epochs        = ABLATION_EPOCHS,
            lr            = BASE_CONFIG['lr'],
            num_cells     = BASE_CONFIG['num_cells'],
            patch_sizes   = BASE_CONFIG['patch_sizes'],
            theta_init    = BASE_CONFIG['theta_init'],
            device        = DEVICE,
            K             = BASE_CONFIG['K'],
            use_intensity = BASE_CONFIG['use_intensity']
        )
        print(f"  ✅ crop_roi={crop} → {acc:.4f}")
        results.append({"param": "crop_roi", "value": str(crop), "acc": acc})
        
        # ✅ SAUVEGARDE APRÈS CHAQUE CONFIG (pas à la fin)
        df_temp = pd.DataFrame(results)
        df_temp.to_csv("figs/ablation_crop.csv", index=False)
        with open("figs/ablation_crop.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # ✅ BACKUP DRIVE APRÈS CHAQUE CONFIG
        try:
            import shutil
            drive_path = "/content/drive/MyDrive/ablation_results/"
            os.makedirs(drive_path, exist_ok=True)
            shutil.copy("figs/ablation_crop.csv",  f"{drive_path}ablation_crop.csv")
            shutil.copy("figs/ablation_crop.json", f"{drive_path}ablation_crop.json")
            print(f"  ✅ Backup Drive sauvegardé après crop_roi={crop}")
        except:
            pass
        
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    best_acc = df["acc"].max()
    for _, row in df.iterrows():
        marker = " ✅ BEST" if row["acc"] == best_acc else ""
        print(f"  crop_roi={row['value']:>5} → {row['acc']:.4f}{marker}")

    try:
        from google.colab import files
        files.download("figs/ablation_crop.csv")
        files.download("figs/ablation_crop.json")
        print("✅ Téléchargés !")
    except ImportError:
        pass

    print("\n✅ ABLATION crop_roi TERMINÉE !")