import importlib
import sys
sys.path.append('/content/population-CBT-learning')

import torch, os, json, shutil, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data import load_ddsm
from run  import TRAIN_DIR, VAL_DIR

drive_path = "/content/drive/MyDrive/ablation_results/baselines/"
os.makedirs("figs", exist_ok=True)
os.makedirs(drive_path, exist_ok=True)

# ── Charger données ────────────────────────────────────────
train_images, train_labels, val_images, val_labels = load_ddsm(
    TRAIN_DIR, VAL_DIR, img_size=256, use_mask=True, crop_roi=False
)

# ── Aplatir 256×256 → 65536 features ─────────────────────
print("Aplatissement des images 256×256...")
X_tr = torch.stack(train_images).numpy().reshape(len(train_images), -1)
X_vl = torch.stack(val_images).numpy().reshape(len(val_images), -1)
y_tr = np.array(train_labels)
y_vl = np.array(val_labels)

print(f"X_tr shape : {X_tr.shape}")  # (3848, 65536)
print(f"X_vl shape : {X_vl.shape}")  # (968, 65536)

# ── Standardisation ───────────────────────────────────────
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_vl_s   = scaler.transform(X_vl)

# ── Baselines ─────────────────────────────────────────────
baselines = {
    "kNN (k=5)" : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM (RBF)" : SVC(kernel="rbf", C=1.0, random_state=42,
                      probability=True),
    "MLP"       : MLPClassifier(hidden_layer_sizes=(256, 128),
                                max_iter=50, random_state=42),
}

print("\n=== BASELINES 256×256 ===\n")
results = []

for name, clf in baselines.items():
    print(f"Training {name}...")
    start = time.time()
    clf.fit(X_tr_s, y_tr)
    elapsed = time.time() - start

    preds     = clf.predict(X_vl_s)
    proba     = clf.predict_proba(X_vl_s)[:, 1]
    acc       = accuracy_score(y_vl, preds)
    f1        = f1_score(y_vl, preds, average="macro")
    auc       = roc_auc_score(y_vl, proba)

    results.append({
        "method" : name,
        "acc"    : acc,
        "f1"     : f1,
        "auc"    : auc,
        "time_s" : elapsed
    })

    print(f"  {name:15s} → Acc: {acc:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f} | "
          f"Temps: {elapsed:.1f}s")

# ── Sauvegarder ───────────────────────────────────────────
df = pd.DataFrame(results)
print(f"\n{df.to_string(index=False)}")

df.to_csv("figs/baselines_256.csv", index=False)
with open("figs/baselines_256.json", "w") as f:
    json.dump(results, f, indent=2)

shutil.copy("figs/baselines_256.csv",  f"{drive_path}baselines_256.csv")
shutil.copy("figs/baselines_256.json", f"{drive_path}baselines_256.json")

try:
    from google.colab import files
    files.download("figs/baselines_256.csv")
    files.download("figs/baselines_256.json")
    print("✅ Téléchargé !")
except:
    pass

print("\n✅ BASELINES TERMINÉES !")