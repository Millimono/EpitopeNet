# ============================================================
# interpretability.py — Multi-scale visualization
# ============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def visualize_multiscale_prediction(pop, img_tensor, true_label, pred_label,
                                     class_names=["Cancer", "Normal"],
                                     save_path=None):
    """Visualise activations multi-échelles pour UNE image."""
    device = pop.prototypes[0].device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    all_activated, _ = pop.process_batch(img_tensor)
    
    n_scales = pop.n_scales
    fig, axes = plt.subplots(1, n_scales + 1, figsize=(4 * (n_scales + 1), 4))
    
    # Image originale
    ax = axes[0]
    img_np = img_tensor.squeeze().cpu().numpy()
    ax.imshow(img_np, cmap='gray')
    
    status = "✅ CORRECT" if pred_label == true_label else "❌ INCORRECT"
    ax.set_title(f"Original\nTrue: {class_names[true_label]}\n"
                 f"Pred: {class_names[pred_label]}\n{status}", 
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Pour chaque échelle
    H, W = img_np.shape
    
    for scale_idx in range(n_scales):
        ax = axes[scale_idx + 1]
        patch_size = pop.patch_sizes[scale_idx]
        activated = all_activated[scale_idx][0]
        
        ph, pw = patch_size
        n_patches_h = H - ph + 1
        n_patches_w = W - pw + 1
        
        # Heatmaps par classe
        heatmap_cancer = np.zeros((n_patches_h, n_patches_w))
        heatmap_normal = np.zeros((n_patches_h, n_patches_w))
        
        # Compter activations par classe
        for proto_idx in torch.where(activated)[0]:
            proto_class = pop.proto_class[scale_idx][proto_idx].item()
            if proto_class == 0:  # Cancer
                heatmap_cancer += 1
            elif proto_class == 1:  # Normal
                heatmap_normal += 1
        
        # Afficher image avec overlay
        ax.imshow(img_np, cmap='gray', alpha=0.7)
        
        # Resize heatmaps à taille image
        if heatmap_cancer.max() > 0:
            scale_h = H / n_patches_h
            scale_w = W / n_patches_w
            heatmap_resized = zoom(heatmap_cancer / heatmap_cancer.max(), 
                                  (scale_h, scale_w), order=1)
            ax.imshow(heatmap_resized, cmap='Reds', alpha=0.4, 
                     extent=[0, W, H, 0])
        
        if heatmap_normal.max() > 0:
            scale_h = H / n_patches_h
            scale_w = W / n_patches_w
            heatmap_resized = zoom(heatmap_normal / heatmap_normal.max(),
                                  (scale_h, scale_w), order=1)
            ax.imshow(heatmap_resized, cmap='Blues', alpha=0.4,
                     extent=[0, W, H, 0])
        
        # Stats
        n_cancer = (pop.proto_class[scale_idx][activated] == 0).sum().item()
        n_normal = (pop.proto_class[scale_idx][activated] == 1).sum().item()
        
        ax.set_title(f"Patch {ph}×{pw}\n🔴 Cancer: {n_cancer}\n🔵 Normal: {n_normal}", 
                     fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    return fig


def visualize_best_model(pop, trainer, val_images, val_labels, 
                        class_names=["Cancer", "Normal"],
                        save_dir="figs/best_model_viz",
                        n_examples=16):
    """Visualise le meilleur modèle sur plusieurs exemples."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prédire
    print(f"  Prédiction sur {min(50, len(val_images))} images...")
    preds = trainer.predict_batch(val_images[:min(50, len(val_images))], batch_size=4)
    
    # Sélectionner meilleurs exemples
    correct_cancer = []
    correct_normal = []
    error_cancer = []
    error_normal = []
    
    for i, (pred, true) in enumerate(zip(preds[:50], val_labels[:50])):
        if pred is None:
            continue
        
        if pred == 0 and true == 0:
            correct_cancer.append(i)
        elif pred == 1 and true == 1:
            correct_normal.append(i)
        elif pred == 1 and true == 0:
            error_cancer.append(i)
        elif pred == 0 and true == 1:
            error_normal.append(i)
    
    # Prendre les premiers de chaque catégorie
    selected_indices = (
        correct_cancer[:n_examples//4] +
        correct_normal[:n_examples//4] +
        error_cancer[:n_examples//4] +
        error_normal[:n_examples//4]
    )
    
    print(f"  Génération {len(selected_indices)} visualisations...")
    
    for idx in selected_indices:
        pred = preds[idx]
        true = val_labels[idx]
        
        if pred == true:
            status = "correct"
        else:
            status = "error"
        
        save_path = os.path.join(
            save_dir,
            f"img{idx:03d}_{status}_true{class_names[true]}_pred{class_names[pred]}.png"
        )
        
        visualize_multiscale_prediction(
            pop=pop,
            img_tensor=val_images[idx],
            true_label=true,
            pred_label=pred,
            class_names=class_names,
            save_path=save_path
        )
    
    print(f"  ✅ {len(selected_indices)} visualisations dans {save_dir}/")