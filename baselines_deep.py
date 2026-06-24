# ============================================================
# baselines_deep.py — ResNet18 fine-tuning sur MiniDDSM
# ============================================================
import sys, importlib

for mod_name in list(sys.modules.keys()):
    if mod_name in ['data', 'run', 'model', 'train', 'save_load']:
        del sys.modules[mod_name]
importlib.invalidate_caches()
sys.path.insert(0, '/content/population-CBT-learning')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time, json, shutil, os

from data import load_ddsm
from run  import set_seed, TRAIN_DIR, VAL_DIR

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/baselines/"
CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-4

os.makedirs("figs", exist_ok=True)
os.makedirs(DRIVE_PATH, exist_ok=True)

# ── Data ───────────────────────────────────────────────────
set_seed(42)
train_images, train_labels, _, _ = load_ddsm(
    TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
)
data_clean = torch.load(CACHE_CLEAN)
val_images = data_clean["val_images"]
val_labels = data_clean["val_labels"]

print(f"Train : {len(train_images)} images")
print(f"Val   : {len(val_images)} images (propre)")

# ── Convertir en 3 canaux (ResNet attend RGB) ──────────────
def to_3ch(images):
    tensors = []
    for img in images:
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        tensors.append(img)
    return torch.stack(tensors)

X_tr = to_3ch(train_images)
X_vl = to_3ch(val_images)
y_tr = torch.tensor(train_labels, dtype=torch.long)
y_vl = torch.tensor(val_labels,   dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_vl, y_vl),
                           batch_size=BATCH_SIZE, shuffle=False)

# ── ResNet18 fine-tuning ────────────────────────────────────
print("\n=== FINE-TUNING ResNet18 (pretrained=ImageNet) ===\n")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_acc   = 0.0
best_state = None
history    = []
start_time = time.time()

for epoch in range(EPOCHS):
    # Train
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Val
    model.eval()
    all_preds, all_true, all_proba = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            out   = model(X_batch.to(DEVICE))
            proba = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
            all_proba.extend(proba)

    acc = accuracy_score(all_true, all_preds)
    history.append(acc)
    scheduler.step()

    if acc > best_acc:
        best_acc   = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        marker = "✅"
    else:
        marker = ""

    print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | "
          f"Best: {best_acc:.4f} {marker}")

elapsed_min = (time.time() - start_time) / 60

# ── Évaluation finale ──────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_preds, all_true, all_proba = [], [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        out   = model(X_batch.to(DEVICE))
        proba = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())
        all_proba.extend(proba)

all_true  = np.array(all_true)
all_preds = np.array(all_preds)
all_proba = np.array(all_proba)

y_bin = np.array([1 if l == 0 else 0 for l in all_true])

acc = accuracy_score(all_true, all_preds)
f1  = f1_score(all_true, all_preds, average='macro')
auc = roc_auc_score(y_bin, all_proba)

print(f"\n>>> ResNet18 RÉSULTATS FINAUX :")
print(f"    Accuracy : {acc:.4f}")
print(f"    F1 macro : {f1:.4f}")
print(f"    AUC      : {auc:.4f}")
print(f"    Temps    : {elapsed_min:.1f} min")

# ── Sauvegarder ───────────────────────────────────────────
result = {
    "method"  : "ResNet18 (fine-tuned, ImageNet)",
    "acc"     : float(acc),
    "f1"      : float(f1),
    "auc"     : float(auc),
    "time_min": float(elapsed_min),
    "epochs"  : EPOCHS,
    "history" : history,
    "scores"  : all_proba.tolist(),  # pour courbe ROC
}

with open("figs/resnet18_results.json", "w") as f:
    json.dump(result, f, indent=2)
torch.save(model.state_dict(), "figs/model_resnet18.pt")

shutil.copy("figs/resnet18_results.json",
            f"{DRIVE_PATH}resnet18_results.json")
shutil.copy("figs/model_resnet18.pt",
            f"{DRIVE_PATH}model_resnet18.pt")

try:
    from google.colab import files
    files.download("figs/resnet18_results.json")
except:
    pass

print(f"\n✅ ResNet18 sauvegardé sur Drive !")
print(f"✅ Scores ROC disponibles dans resnet18_results.json")
print(f"\n✅ RESNET18 TERMINÉ !")