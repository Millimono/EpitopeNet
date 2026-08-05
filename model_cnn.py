# ============================================================
# model_cnn.py — EpitopeNet + CNN figé comme extracteur
# Modifications minimales par rapport à model.py
# ============================================================

import torch
import torch.nn.functional as F
import torchvision.models as models



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
            #print(f"    [Reassign {ps[0]}×{ps[1]}] {n_assigned}/{self.B_per_scale[scale_idx]} protos")
    
    def get_vote_weights(self, scale_idx):
        """Poids exclusivité pour vote pondéré."""
        total = self.class_counts[scale_idx].sum(dim=1, keepdim=True).clamp(min=1)
        freq = self.class_counts[scale_idx] / total
        max_freq = freq.max(dim=1).values
        mean_freq = freq.mean(dim=1)
        weights = (max_freq - mean_freq) * 2
        return weights, freq


# Dans model_cnn.py — remplace la classe PopulationBMultiScaleCNN
# par PopulationBMultiScaleGabor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaborEncoder(nn.Module):
    """Banc de filtres de Gabor fixes pour extraction de features."""
    
    def __init__(self, n_orientations=8, n_scales=4, 
                 kernel_size=11, device='cpu'):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales       = n_scales
        self.kernel_size    = kernel_size
        self.n_filters      = n_orientations * n_scales
        
        # Créer filtres Gabor
        filters = self._create_gabor_filters()
        
        # Conv fixe (pas de gradient)
        self.conv = nn.Conv2d(
            1, self.n_filters, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.conv.weight.data = filters
        for p in self.conv.parameters():
            p.requires_grad = False
        self.conv = self.conv.to(device)
    
    def _create_gabor_filters(self):
        """Crée une banque de filtres Gabor."""
        filters = []
        k = self.kernel_size
        x = np.linspace(-k//2, k//2, k)
        y = np.linspace(-k//2, k//2, k)
        xx, yy = np.meshgrid(x, y)
        
        # Fréquences spatiales
        frequencies = [0.1, 0.2, 0.3, 0.4][:self.n_scales]
        
        for freq in frequencies:
            sigma = 1.0 / (2 * np.pi * freq * 0.5)
            for theta in np.linspace(0, np.pi, 
                                     self.n_orientations, 
                                     endpoint=False):
                # Rotation
                x_rot =  xx * np.cos(theta) + yy * np.sin(theta)
                y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
                
                # Filtre Gabor
                gaussian = np.exp(-(x_rot**2 + y_rot**2) / 
                                  (2 * sigma**2))
                sinusoid = np.cos(2 * np.pi * freq * x_rot)
                gabor    = gaussian * sinusoid
                
                # Normaliser
                gabor -= gabor.mean()
                gabor_std = gabor.std()
                if gabor_std > 0:
                    gabor /= gabor_std
                
                filters.append(gabor)
        
        filters = np.stack(filters)  # (n_filters, k, k)
        return torch.FloatTensor(filters).unsqueeze(1)
        # shape : (n_filters, 1, k, k)
    
    def forward(self, x):
        """
        x : (N, H, W) ou (N, 1, H, W)
        → (N, n_filters, H, W)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.conv(x)
        return torch.relu(features)


class PopulationBMultiScaleGabor(PopulationBMultiScale):
    """
    EpitopeNet avec banc de filtres Gabor fixes.
    Pas de pré-entraînement, paradigme sans gradient préservé.
    """
    
    def __init__(self, num_cells, patch_sizes, theta_init, beta,
                 num_classes, K, use_intensity, device,
                 n_orientations=8, n_scales=4, gabor_kernel=11):
        
        self.n_gabor_filters = n_orientations * n_scales
        
        # ── Gabor Encoder (fixe) ────────────────────────────
        self.encoder = GaborEncoder(
            n_orientations=n_orientations,
            n_scales=n_scales,
            kernel_size=gabor_kernel,
            device=device
        )
        
        # ── Adapter patch_sizes ─────────────────────────────
        # Image 256×256 → feature map 256×256 (padding=same)
        # Patch sur feature map : même taille que l'original
        
        print(f"[Gabor Encoder] {self.n_gabor_filters} filtres "
              f"({n_orientations} orientations × {n_scales} échelles)")
        
        # ── Init parent ──────────────────────────────────────
        super().__init__(
            num_cells=num_cells,
            patch_sizes=patch_sizes,  # Mêmes patch_sizes !
            theta_init=theta_init,
            beta=beta,
            num_classes=num_classes,
            K=K,
            use_intensity=False,
            device=device
        )
        
        # Override D : n_gabor_filters × ph × pw
        for i, (ph, pw) in enumerate(patch_sizes):
            D_new   = self.n_gabor_filters * ph * pw
            B_scale = self.B_per_scale[i]
            self.prototypes[i] = torch.randn(
                B_scale, D_new, device=device) * 0.1
            self.class_counts[i] = torch.zeros(
                B_scale, num_classes, device=device)
            print(f"  Échelle {i}: {ph}×{pw} Gabor patch → "
                  f"{B_scale} protos, {D_new} features "
                  f"({self.n_gabor_filters} canaux)")
    
    def extract_patches_batch(self, images, patch_size):
        """
        Extraire patches sur feature map Gabor.
        """
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        with torch.no_grad():
            features = self.encoder(images)
        # features : (N, n_filters, H, W)
        
        patches = F.unfold(
            features,
            kernel_size=patch_size,
            stride=1
        )
        return patches.transpose(1, 2)
        # shape : (N, P, n_filters × ph × pw)
    
    def preprocess_patches(self, patches, keep_intensity=True):
        """Z-score par patch sur features Gabor."""
        mean = patches.mean(dim=-1, keepdim=True)
        std  = patches.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (patches - mean) / std

#