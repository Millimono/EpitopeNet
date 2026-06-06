import torch, gc, os, pandas as pd, json
from data  import load_ddsm
from train import run_experiment
from run   import set_seed, TRAIN_DIR, VAL_DIR, DEVICE, NUM_CLASSES

ABLATION_SEED   = 42
ABLATION_EPOCHS = 10
BASE_CONFIG = {
    'num_cells': 6400,
    'patch_sizes': [(10, 10), (18, 18), (28, 28)],
    'theta_init': 0.5,
    'lr': 0.1,
    'K': 1,
    'use_intensity': True
}

if __name__ == "__main__":
    print("=== ABLATION : crop_roi=False SEULEMENT ===\n")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs("figs", exist_ok=True)

    set_seed(ABLATION_SEED)
    
    train_imgs, train_lbls, val_imgs, val_lbls = load_ddsm(
        TRAIN_DIR, VAL_DIR,
        img_size=256,
        use_mask=True,
        crop_roi=False  # ← JUSTE False
    )
    
    acc, _, _, _ = run_experiment(
        train_imgs, train_lbls, val_imgs, val_lbls,
        name          = "crop_roi=False",
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
    
    print(f"\n✅ crop_roi=False → {acc:.4f}")
    
    result = [{"param": "crop_roi", "value": "False", "acc": acc}]
    
    pd.DataFrame(result).to_csv("figs/ablation_crop_false.csv", index=False)
    with open("figs/ablation_crop_false.json", "w") as f:
        json.dump(result, f, indent=2)
    
    try:
        import shutil
        from google.colab import files
        drive_path = "/content/drive/MyDrive/ablation_results/"
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy("figs/ablation_crop_false.csv",  f"{drive_path}ablation_crop_false.csv")
        shutil.copy("figs/ablation_crop_false.json", f"{drive_path}ablation_crop_false.json")
        files.download("figs/ablation_crop_false.csv")
        files.download("figs/ablation_crop_false.json")
        print("✅ Sauvegardé + téléchargé !")
    except ImportError:
        pass

    print("\n✅ TERMINÉ !")