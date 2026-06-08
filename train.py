# ============================================================
# train.py — LVQ GPU conforme à l'article
# ============================================================

import torch
from model import PopulationBMultiScale, TrainerMultiScale


def init_prototypes_from_data(population, images, labels, device, n_samples=200):
    """Initialisation aléatoire depuis patches réels (article: n=200)."""
    print(f"Initialisation depuis {n_samples} premières images...")
    
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
    
    # Échantillonner
    for scale_idx in range(population.n_scales):
        all_patches = torch.cat(all_patches_per_scale[scale_idx], dim=0)
        B_scale = population.B_per_scale[scale_idx]
        
        idx = torch.randperm(all_patches.shape[0])[:B_scale]
        population.prototypes[scale_idx] = all_patches[idx].to(device)
        
        print(f"  Échelle {scale_idx} : {B_scale} prototypes initialisés")


def run_experiment(train_images, train_labels, val_images, val_labels,
                   name, num_classes, epochs=40, lr=0.1,
                   num_cells=6400, patch_sizes=[(5,5), (9,9), (13,13)],
                   theta_init=0.5, K=1, device="cuda",
                   use_intensity=True):
    """Training LVQ GPU conforme à l'article."""
    
    print(f"\n{'='*50}")
    print(f"EXPÉRIENCE : {name}")
    print(f"{'='*50}")

    pop = PopulationBMultiScale(
        num_cells     = num_cells,
        patch_sizes   = patch_sizes,
        theta_init    = theta_init,
        beta          = 5.0,
        num_classes   = num_classes,
        K             = K,
        use_intensity = use_intensity,
        device        = device
    )
    trainer = TrainerMultiScale(population=pop, num_classes=num_classes, device=device)
    init_prototypes_from_data(pop, train_images, train_labels, device, n_samples=50)

    best_acc     = 0.0
    best_protos  = [p.clone() for p in pop.prototypes]
    best_counts  = [c.clone() for c in pop.class_counts]
    best_classes = [c.clone() for c in pop.proto_class]
    patience, max_patience = 0, 7
    history = []

    for epoch in range(epochs):
        lr_epoch = lr * (0.95 ** epoch)
        
        # ✅ CHANGEMENT : Appel GPU
        trainer.train_batch(train_images, train_labels, batch_size=2, lr=lr_epoch)
        pop.reassign_proto_class(train_images, train_labels, device, batch_size=2)
        preds = trainer.predict_batch(val_images, batch_size=4)
        
        correct = sum(p == l for p, l in zip(preds, val_labels) if p is not None)
        acc     = correct / len(val_images)
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

    pop.prototypes   = [p.clone() for p in best_protos]
    pop.class_counts = [c.clone() for c in best_counts]
    pop.proto_class  = [c.clone() for c in best_classes]
    print(f"\n>>> BEST ACCURACY [{name}]: {best_acc:.4f}")
    return best_acc, pop, trainer, history