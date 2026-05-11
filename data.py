# ============================================================
# data.py — Chargement cache 128×128 AVEC MASQUAGE
# ============================================================

import os
import torch

CACHE_PATH = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_128.pt"


def mask_background(img, threshold=0.1):
    """
    Remplace l'arrière-plan noir par gris neutre.
    
    Args:
        img: Tensor (128, 128), valeurs [0, 1]
        threshold: Pixels < threshold = arrière-plan
    
    Returns:
        img_masked: Tensor (128, 128), arrière-plan = 0.5
    """
    img_masked = img.clone()
    img_masked[img < threshold] = 0.5
    return img_masked


def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False, 
              mask_background_flag=False):  # ✅ NOUVEAU paramètre
    """Charge depuis cache MiniDDSM 128×128 avec initialisation équilibrée."""
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache non trouvé : {CACHE_PATH}")
    
    print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
    data = torch.load(CACHE_PATH)
    
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images   = data["val_images"]
    val_labels   = data["val_labels"]
    
    # ✅ MASQUER arrière-plan SI demandé (AVANT l'entrelacement)
    if mask_background_flag:
        print("[MASQUAGE] Remplacement arrière-plan noir par gris neutre (0.5)...")
        train_images = [mask_background(img, threshold=0.1) for img in train_images]
        val_images = [mask_background(img, threshold=0.1) for img in val_images]
        print("[OK] Arrière-plan masqué sur train + val")
    
    # Séparer par classe
    cancer_idx = [i for i, l in enumerate(train_labels) if l == 0]
    normal_idx = [i for i, l in enumerate(train_labels) if l == 1]
    
    # Mélanger CHAQUE classe séparément
    cancer_perm = torch.randperm(len(cancer_idx))
    normal_perm = torch.randperm(len(normal_idx))
    
    cancer_shuffled = [cancer_idx[i] for i in cancer_perm]
    normal_shuffled = [normal_idx[i] for i in normal_perm]
    
    # Entrelacer Cancer/Normal pour alternance parfaite
    train_images_balanced = []
    train_labels_balanced = []
    
    max_len = max(len(cancer_shuffled), len(normal_shuffled))
    for i in range(max_len):
        if i < len(cancer_shuffled):
            train_images_balanced.append(train_images[cancer_shuffled[i]])
            train_labels_balanced.append(0)
        if i < len(normal_shuffled):
            train_images_balanced.append(train_images[normal_shuffled[i]])
            train_labels_balanced.append(1)
    
    train_images = train_images_balanced
    train_labels = train_labels_balanced
    
    # Stats
    n_train_cancer = sum(1 for l in train_labels if l == 0)
    n_train_normal = sum(1 for l in train_labels if l == 1)
    n_val_cancer   = sum(1 for l in val_labels if l == 0)
    n_val_normal   = sum(1 for l in val_labels if l == 1)
    
    print(f"Train: {len(train_images)} ({n_train_cancer} Cancer, {n_train_normal} Normal) [entrelacé]")
    print(f"Val  : {len(val_images)} ({n_val_cancer} Cancer, {n_val_normal} Normal)")
    print(f"Shape: {train_images[0].shape}")
    
    return train_images, train_labels, val_images, val_labels


load_cbis_ddsm = load_ddsm