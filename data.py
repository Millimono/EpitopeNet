# ============================================================
# data.py — CROP ROI mammaire
# ============================================================

import os
import torch
import cv2
import numpy as np

CACHE_PATH = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_128.pt"


def crop_breast_roi(img_tensor, target_size=128):
    """
    Détecte la ROI mammaire et crop/resize.
    
    Args:
        img_tensor: Tensor (128, 128), valeurs [0, 1]
        target_size: Taille de sortie
    
    Returns:
        Tensor (target_size, target_size) croppé sur le sein
    """
    # Convertir en numpy uint8
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    
    # Binariser pour détecter le sein (seuil à 25/255 ≈ 0.1)
    _, binary = cv2.threshold(img_np, 25, 255, cv2.THRESH_BINARY)
    
    # Trouver contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback : retourner image originale
        return img_tensor
    
    # Plus grand contour = sein
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Ajouter marge 5%
    margin = int(0.05 * max(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_np.shape[1] - x, w + 2 * margin)
    h = min(img_np.shape[0] - y, h + 2 * margin)
    
    # Crop
    cropped = img_np[y:y+h, x:x+w]
    
    # Resize à target_size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Reconvertir en tensor [0, 1]
    return torch.from_numpy(resized.astype(np.float32) / 255.0)


def load_ddsm(train_dir=None, val_dir=None, img_size=128, use_mask=False, 
              crop_roi=False):
    """Charge depuis cache MiniDDSM avec option CROP ROI."""
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache non trouvé : {CACHE_PATH}")
    
    print(f"[OK] Chargement depuis cache : {CACHE_PATH}")
    data = torch.load(CACHE_PATH)
    
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images   = data["val_images"]
    val_labels   = data["val_labels"]
    
    # ✅ CROP ROI SI demandé
    if crop_roi:
        print("[CROP ROI] Extraction région mammaire...")
        train_images = [crop_breast_roi(img, target_size=img_size) for img in train_images]
        val_images = [crop_breast_roi(img, target_size=img_size) for img in val_images]
        print("[OK] ROI extraite sur train + val")
    
    # Séparer par classe
    cancer_idx = [i for i, l in enumerate(train_labels) if l == 0]
    normal_idx = [i for i, l in enumerate(train_labels) if l == 1]
    
    # Mélanger
    cancer_perm = torch.randperm(len(cancer_idx))
    normal_perm = torch.randperm(len(normal_idx))
    
    cancer_shuffled = [cancer_idx[i] for i in cancer_perm]
    normal_shuffled = [normal_idx[i] for i in normal_perm]
    
    # Entrelacer
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