# ============================================================
# model.py — LVQ simple conforme à l'article
# ============================================================

import torch
import torch.nn.functional as F


class PopulationBMultiScale:
    """
    Population multi-échelle conforme à l'article.
    Apprentissage LVQ simple sans gradient descent.
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
        D = patch_sizes[0][0] * patch_sizes[0][1]
        if use_intensity:
            D += 1
        
        self.prototypes = [
            torch.randn(B_scale, D, device=device) * 0.1
            for B_scale in self.B_per_scale
        ]
        
        self.class_counts = [
            torch.zeros(B_scale, num_classes, device=device)
            for B_scale in self.B_per_scale
        ]
        
        self.proto_class = [
            torch.full((B_scale,), -1, dtype=torch.long, device=device)
            for B_scale in self.B_per_scale
        ]
    
    def extract_patches_batch(self, images, patch_size):
        """Extraction patches par convolution."""
        B, H, W = images.shape
        p_h, p_w = patch_size
        
        patches = F.unfold(
            images.unsqueeze(1),
            kernel_size=(p_h, p_w),
            stride=1
        )
        
        patches = patches.transpose(1, 2)
        return patches
    
    def preprocess_patches(self, patches, keep_intensity=True):
        """
        Normalisation patches + feature intensité optionnelle.
        
        patches: (B, P, D) où D = p*p
        """
        B, P, D = patches.shape
        
        # Normalisation Z-score
        mean = patches.mean(dim=2, keepdim=True)
        std = patches.std(dim=2, keepdim=True).clamp(min=1e-8)
        patches_norm = (patches - mean) / std
        
        if keep_intensity and self.use_intensity:
            # Feature intensité = moyenne pré-normalisation
            intensity = mean.squeeze(2)  # (B, P)
            patches_final = torch.cat([patches_norm, intensity.unsqueeze(2)], dim=2)
        else:
            patches_final = patches_norm
        
        return patches_final
    
    def compute_activations(self, patches, scale_idx):
        """
        Calcul similarités gaussiennes.
        
        patches: (B, P, D)
        prototypes: (B_scale, D)
        
        Returns:
            activated: (B, B_scale) booléen
            z: (B, B_scale, D) patches capturés
        """
        B, P, D = patches.shape
        B_scale = self.prototypes[scale_idx].shape[0]
        
        # Distance euclidienne²
        patches_flat = patches.reshape(B * P, D)
        protos = self.prototypes[scale_idx]
        
        dist_sq = torch.cdist(patches_flat, protos, p=2) ** 2
        dist_sq = dist_sq.reshape(B, P, B_scale)
        
        # Top-K patches
        topk_dist, topk_idx = dist_sq.topk(self.K, dim=1, largest=False)
        
        # Similarité gaussienne
        p_size = self.patch_sizes[scale_idx][0] * self.patch_sizes[scale_idx][1]
        if self.use_intensity:
            p_size += 1
        
        similarities = torch.exp(-topk_dist.mean(dim=1) / p_size)  # (B, B_scale)
        
        # Activation binaire
        activated = similarities >= self.theta
        
        # Patches capturés (closest)
        closest_idx = topk_idx[:, 0, :]  # (B, B_scale)
        z = torch.gather(
            patches,
            1,
            closest_idx.unsqueeze(2).expand(-1, -1, D)
        )
        
        return activated, z
    
    def update_batch_lvq(self, all_activated, all_z, labels, lr=0.1):
        """
        LVQ simple conforme à l'article.
        
        Règle :
        - Même classe : rapprocher
        - Classe différente : éloigner
        """
        N = len(labels)
        
        for scale_idx in range(self.n_scales):
            activated = all_activated[scale_idx]  # (N, B_scale)
            z = all_z[scale_idx]                   # (N, B_scale, D)
            
            # Mettre à jour compteurs
            for i in range(N):
                lbl = labels[i].item()
                act = activated[i]
                
                if not act.any():
                    continue
                
                # Décroissance + incrément
                self.class_counts[scale_idx][act] *= 0.99
                self.class_counts[scale_idx][act, lbl] += 1
            
            # Réassigner classes
            self.proto_class[scale_idx] = self.class_counts[scale_idx].argmax(dim=1)
            self.proto_class[scale_idx][self.class_counts[scale_idx].sum(dim=1) == 0] = -1
            
            # LVQ update
            for i in range(N):
                lbl = labels[i].item()
                act_i = activated[i]
                
                if not act_i.any():
                    continue
                
                for proto_idx in torch.where(act_i)[0]:
                    proto_class = self.proto_class[scale_idx][proto_idx].item()
                    
                    if proto_class < 0:
                        continue
                    
                    patch_captured = z[i, proto_idx]
                    
                    if proto_class == lbl:
                        # Rapprocher
                        self.prototypes[scale_idx][proto_idx] += lr * (
                            patch_captured - self.prototypes[scale_idx][proto_idx]
                        )
                    else:
                        # Éloigner
                        self.prototypes[scale_idx][proto_idx] -= lr * (
                            patch_captured - self.prototypes[scale_idx][proto_idx]
                        )
            
            # Clamp
            self.prototypes[scale_idx].clamp_(-5.0, 5.0)
    
    def reassign_proto_class(self, images, labels, device, batch_size=4):
        """Réassignation classes en fin d'epoch (normalisation fréquence)."""
        for scale_idx in range(self.n_scales):
            self.class_counts[scale_idx].zero_()
        
        # Accumuler compteurs
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            batch_lbls = labels[i:i+batch_size]
            
            imgs_batch = torch.stack(batch_imgs).to(device)
            
            for scale_idx, patch_size in enumerate(self.patch_sizes):
                patches = self.extract_patches_batch(imgs_batch, patch_size)
                patches = self.preprocess_patches(patches, keep_intensity=True)
                
                activated, _ = self.compute_activations(patches, scale_idx)
                
                for j, lbl in enumerate(batch_lbls):
                    act = activated[j]
                    self.class_counts[scale_idx][act, lbl] += 1
            
            del imgs_batch
            torch.cuda.empty_cache()
        
        # Normaliser par fréquence classe
        class_freq = torch.zeros(self.num_classes, device=device)
        for lbl in labels:
            class_freq[lbl] += 1
        
        for scale_idx in range(self.n_scales):
            counts_norm = self.class_counts[scale_idx] / (class_freq.unsqueeze(0) + 1e-8)
            self.proto_class[scale_idx] = counts_norm.argmax(dim=1)
            self.proto_class[scale_idx][self.class_counts[scale_idx].sum(dim=1) == 0] = -1
    
    def get_vote_weights(self, scale_idx):
        """Poids exclusivité pour vote pondéré."""
        freq = self.class_counts[scale_idx] / (
            self.class_counts[scale_idx].sum(dim=1, keepdim=True) + 1e-8
        )
        
        exclusivity = 2 * (
            freq.max(dim=1).values - freq.mean(dim=1)
        )
        exclusivity = exclusivity.clamp(0, 1)
        
        return exclusivity, freq


class TrainerMultiScale:
    """Trainer multi-échelle."""
    
    def __init__(self, population, num_classes, device):
        self.pop = population
        self.num_classes = num_classes
        self.device = device
    
    def train_batch(self, images, labels, batch_size=2, lr=0.1):
        """Training LVQ simple."""
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            batch_lbls = labels[i:i+batch_size]
            
            imgs_batch = torch.stack(batch_imgs).to(self.device)
            
            all_activated = []
            all_z = []
            
            for scale_idx, patch_size in enumerate(self.pop.patch_sizes):
                patches = self.pop.extract_patches_batch(imgs_batch, patch_size)
                patches = self.pop.preprocess_patches(patches, keep_intensity=True)
                
                activated, z = self.pop.compute_activations(patches, scale_idx)
                
                all_activated.append(activated)
                all_z.append(z)
            
            self.pop.update_batch_lvq(all_activated, all_z, batch_lbls, lr=lr)
            
            del imgs_batch
            torch.cuda.empty_cache()
    
    def predict_batch(self, images, batch_size=4):
        """Prédiction par vote pondéré."""
        predictions = []
        
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            imgs_batch = torch.stack(batch_imgs).to(self.device)
            
            votes = torch.zeros(len(batch_imgs), self.num_classes, device=self.device)
            
            for scale_idx, patch_size in enumerate(self.pop.patch_sizes):
                patches = self.pop.extract_patches_batch(imgs_batch, patch_size)
                patches = self.pop.preprocess_patches(patches, keep_intensity=True)
                
                activated, _ = self.pop.compute_activations(patches, scale_idx)
                
                exclusivity, freq = self.pop.get_vote_weights(scale_idx)
                
                for j in range(len(batch_imgs)):
                    act_j = activated[j] & (self.pop.proto_class[scale_idx] >= 0)
                    
                    if not act_j.any():
                        continue
                    
                    weights = exclusivity[act_j].unsqueeze(1) * freq[act_j]
                    votes[j] += weights.sum(dim=0)
            
            for j in range(len(batch_imgs)):
                if votes[j].sum() > 0:
                    predictions.append(votes[j].argmax().item())
                else:
                    predictions.append(None)
            
            del imgs_batch
            torch.cuda.empty_cache()
        
        return predictions