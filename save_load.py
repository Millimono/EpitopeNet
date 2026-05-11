# ============================================================
# save_load.py — Multi-scale avec intensité (CORRIGÉ)
# ============================================================

import torch
import os


def save_model(population, path="model_multiscale.pt"):
    """Sauvegarde modèle multi-échelle avec flag intensité."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    state = {
        'B': population.B,
        'patch_sizes': population.patch_sizes,
        'n_scales': population.n_scales,
        'B_per_scale': population.B_per_scale,
        'theta_init': population.theta,  # ✅ FIX: attribut s'appelle theta
        'beta': population.beta,
        'num_classes': population.num_classes,
        'K': population.K,
        'use_intensity': population.use_intensity,
        'prototypes': [p.cpu() for p in population.prototypes],
        'proto_class': [c.cpu() for c in population.proto_class],
        'class_counts': [c.cpu() for c in population.class_counts],
    }
    
    torch.save(state, path)
    print(f"[OK] Modèle sauvegardé : {path}")


def load_model(path, device="cuda"):
    """Charge modèle multi-échelle."""
    from model import PopulationBMultiScale
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    
    state = torch.load(path, map_location=device)
    
    pop = PopulationBMultiScale(
        num_cells=state['B'],
        patch_sizes=state['patch_sizes'],
        theta_init=state['theta_init'],
        beta=state['beta'],
        num_classes=state['num_classes'],
        K=state['K'],
        use_intensity=state.get('use_intensity', False),
        device=device
    )
    
    pop.prototypes = [p.to(device) for p in state['prototypes']]
    pop.proto_class = [c.to(device) for c in state['proto_class']]
    pop.class_counts = [c.to(device) for c in state['class_counts']]
    
    print(f"[OK] Modèle chargé : {path}")
    print(f"     Échelles : {pop.patch_sizes}")
    print(f"     Intensité : {pop.use_intensity}")
    
    # Afficher stats prototypes
    for scale_idx in range(pop.n_scales):
        assigned = (pop.proto_class[scale_idx] >= 0).sum().item()
        total = pop.B_per_scale[scale_idx]
        ps = pop.patch_sizes[scale_idx]
        print(f"     Échelle {scale_idx} ({ps[0]}×{ps[1]}): {assigned}/{total} protos assignés")
    
    return pop