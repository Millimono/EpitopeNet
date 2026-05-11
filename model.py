# ============================================================
# model.py — LVQ FULL GPU vectorisé
# ============================================================

import torch
import torch.nn.functional as F


class PopulationBMultiScale:
    """
    Population multi-échelle LVQ 100% GPU.
    Toutes les opérations vectorisées, zéro boucle Python sur prototypes.
    """
    
    def __init__(self, num_cells, patch_sizes, theta_init, beta, 
                 num_classes, K, use_intensity, device):
        self.B = num_cells
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)
        self.theta = theta_init
        self.beta = beta
        self.num_classes = num_classes
        self.K = K
        self.use_intensity = use_intensity
        self.device = device
        
        # Répartition équitable
        self.B_per_scale = [num_cells // self.n_scales] * self.n_scales
        remainder = num_cells % self.n_scales
        for i in range(remainder):
            self.B_per_scale[i] += 1
        
        # Initialisation prototypes
        self.prototypes = []
        self.class_counts = []
        self.proto_class = []
        
        for i, (ph, pw) in enumerate(patch_sizes):
            D = ph * pw
            if use_intensity:
                D += 1
            
            B_scale = self.B_per_scale[i]
            
            self.prototypes.append(
                torch.randn(B_scale, D, device=device) * 0.1
            )
            self.class_counts.append(
                torch.zeros(B_scale, num_classes, device=device)
            )
            self.proto_class.append(
                torch.full((B_scale,), -1, dtype=torch.long, device=device)
            )
        
        print(f"[Multi-scale LVQ GPU] {self.n_scales} échelles (intensité: {use_intensity}):")
        for i, ps in enumerate(patch_sizes):
            D_feat = (ps[0] * ps[1] + 1) if use_intensity else (ps[0] * ps[1])
            print(f"  Échelle {i}: {ps[0]}×{ps[1]} → {self.B_per_scale[i]} protos, {D_feat} features")
    
    def extract_patches_batch(self, images, patch_size):
        """Extraction patches par convolution."""
        patches = F.unfold(
            images.unsqueeze(1),
            kernel_size=patch_size,
            stride=1
        )
        return patches.transpose(1, 2)
    
    def preprocess_patches(self, patches, keep_intensity=True):
        """Normalisation patches + feature intensité optionnelle."""
        if not self.use_intensity or not keep_intensity:
            mean = patches.mean(dim=-1, keepdim=True)
            std = patches.std(dim=-1, keepdim=True).clamp(min=1e-8)
            return (patches - mean) / std
        
        intensity = patches.mean(dim=-1, keepdim=True)
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True).clamp(min=1e-8)
        patches_norm = (patches - mean) / std
        
        return torch.cat([patches_norm, intensity], dim=-1)
    
    def process_batch(self, images):
        """Traite un batch d'images pour toutes les échelles."""
        images = images.to(self.device)
        all_activated = []
        all_z = []
        
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            patches = self.extract_patches_batch(images, patch_size)
            patches_std = self.preprocess_patches(patches, keep_intensity=True)
            protos = self.prototypes[scale_idx]
            
            N, P, D = patches_std.shape
            B_scale = protos.shape[0]
            
            # Distances
            patches_sq = (patches_std ** 2).sum(dim=-1)
            protos_sq = (protos ** 2).sum(dim=-1)
            dot = torch.einsum("npd,bd->nbp", patches_std, protos)
            dists_sq = (patches_sq.unsqueeze(1) + 
                       protos_sq.view(1, B_scale, 1) - 2 * dot).clamp(min=0)
            
            # Top-K
            topk_dists, topk_idx = dists_sq.topk(self.K, dim=2, largest=False)
            sim = torch.exp(-topk_dists.mean(dim=2) / D ** 0.5)
            activated = (sim >= self.theta).bool()
            
            # Agréger patches capturés
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            patches_exp = patches_std.unsqueeze(1).expand(-1, B_scale, -1, -1)
            z = patches_exp.gather(2, topk_idx_exp).mean(dim=2)
            
            all_activated.append(activated)
            all_z.append(z)
        
        return all_activated, all_z
    
    def update_batch_lvq_gpu(self, all_activated, all_z, labels, lr=0.1):
        """
        LVQ 100% GPU vectorisé.
        Pas de boucle Python sur prototypes.
        """
        N = len(labels)
        labels_t = torch.tensor(labels, device=self.device, dtype=torch.long)
        
        for scale_idx in range(self.n_scales):
            activated = all_activated[scale_idx]  # (N, B_scale)
            z = all_z[scale_idx]                   # (N, B_scale, D)
            
            # ✅ Mettre à jour compteurs (vectorisé)
            for i in range(N):
                lbl = labels_t[i].item()
                act = activated[i]
                if not act.any():
                    continue
                
                self.class_counts[scale_idx][act] *= 0.99
                self.class_counts[scale_idx][act, lbl] += 1
            
            # Réassigner classes
            self.proto_class[scale_idx] = self.class_counts[scale_idx].argmax(dim=1)
            self.proto_class[scale_idx][self.class_counts[scale_idx].sum(dim=1) == 0] = -1
            
            # ✅ LVQ UPDATE VECTORISÉ (pas de boucle sur prototypes)
            proto_updates = torch.zeros_like(self.prototypes[scale_idx])
            
            for i in range(N):
                lbl = labels_t[i].item()
                act_i = activated[i]  # (B_scale,) booléen
                
                if not act_i.any():
                    continue
                
                # Patches capturés
                z_active = z[i][act_i]  # (n_active, D)
                protos_active = self.prototypes[scale_idx][act_i]  # (n_active, D)
                classes_active = self.proto_class[scale_idx][act_i]  # (n_active,)
                
                # Différence
                diff = z_active - protos_active  # (n_active, D)
                
                # Masques
                same_class = (classes_active == lbl) & (classes_active >= 0)
                diff_class = (classes_active != lbl) & (classes_active >= 0)
                
                # Gradients
                grads_active = torch.zeros_like(diff)
                grads_active[same_class] = diff[same_class]   # Rapprocher
                grads_active[diff_class] = -diff[diff_class]  # Éloigner
                
                # Accumuler
                active_indices = torch.where(act_i)[0]
                proto_updates.index_add_(0, active_indices, grads_active)
            
            # ✅ Appliquer updates
            self.prototypes[scale_idx] += lr * proto_updates
            self.prototypes[scale_idx].clamp_(-5.0, 5.0)
    
    def reassign_proto_class(self, train_images, train_labels, device, batch_size=2):
        """Réassignation classes en fin d'epoch (normalisation fréquence)."""
        for scale_idx in range(self.n_scales):
            self.class_counts[scale_idx].zero_()
        
        images_t = torch.stack(train_images).to(device)
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, _ = self.process_batch(images_t[start:end])
            lbls_b = train_labels[start:end]
            
            for scale_idx in range(self.n_scales):
                activated = all_activated[scale_idx]
                for i in range(end - start):
                    lbl = lbls_b[i] if isinstance(lbls_b[i], int) else lbls_b[i].item()
                    self.class_counts[scale_idx][activated[i], lbl] += 1
        
        # Normaliser par fréquence classe
        for scale_idx in range(self.n_scales):
            assigned = self.class_counts[scale_idx].sum(dim=1) > 0
            n_assigned = assigned.sum().item()
            
            class_freq = self.class_counts[scale_idx].sum(dim=0).clamp(min=1)
            counts_norm = self.class_counts[scale_idx] / class_freq.unsqueeze(0)
            self.proto_class[scale_idx][assigned] = counts_norm[assigned].argmax(dim=1)
            self.proto_class[scale_idx][~assigned] = -1
            
            ps = self.patch_sizes[scale_idx]
            print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} protos")
    
    def get_vote_weights(self, scale_idx):
        """Poids exclusivité pour vote pondéré."""
        total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
        freq = self.class_counts[scale_idx] / total
        max_freq = freq.max(dim=1).values
        mean_freq = freq.mean(dim=1)
        weights = (max_freq - mean_freq) * 2
        return weights, freq


class TrainerMultiScale:
    """Trainer multi-échelle."""
    
    def __init__(self, population, num_classes, device):
        self.population = population
        self.device = device
        self.num_classes = num_classes
    
    def train_batch(self, images, labels, batch_size=2, lr=0.1):
        """Training LVQ GPU."""
        images_t = torch.stack(images).to(self.device)
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, all_z = self.population.process_batch(images_t[start:end])
            
            if not any(a.any() for a in all_activated):
                continue
            
            self.population.update_batch_lvq_gpu(
                all_activated, all_z, labels[start:end], lr
            )
    
    def predict_batch(self, images, batch_size=4):
        """Prédiction par vote pondéré."""
        images_t = torch.stack(images).to(self.device)
        all_preds = []
        
        for start in range(0, len(images_t), batch_size):
            end = min(start + batch_size, len(images_t))
            all_activated, _ = self.population.process_batch(images_t[start:end])
            
            for i in range(end - start):
                total_votes = torch.zeros(self.num_classes, device=self.device)
                
                for scale_idx in range(self.population.n_scales):
                    act_i = all_activated[scale_idx][i]
                    valid = act_i & (self.population.proto_class[scale_idx] >= 0)
                    
                    if not valid.any():
                        continue
                    
                    weights, freq = self.population.get_vote_weights(scale_idx)
                    active_freq = freq[valid]
                    active_weights = weights[valid]
                    votes = (active_freq * active_weights.unsqueeze(1)).sum(dim=0)
                    total_votes += votes
                
                if total_votes.sum() == 0:
                    all_preds.append(None)
                else:
                    all_preds.append(total_votes.argmax().item())
        
        return all_preds