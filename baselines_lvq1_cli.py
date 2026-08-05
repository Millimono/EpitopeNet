# ============================================================
# baselines_lvq1_cli.py — LVQ1 classique baseline
# Même style que run_experiment_cli.py
# ============================================================

import sys, argparse, json, shutil, os, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, '/content/population-CBT-learning')

CACHE_CLEAN = "/content/drive/MyDrive/MiniDDSM/miniddsm_val_clean_256.pt"
CACHE_TRAIN = "/content/drive/MyDrive/MiniDDSM/miniddsm_cache_256.pt"
DRIVE_PATH  = "/content/drive/MyDrive/ablation_results/baselines/"


# ── LVQ1 Classique ──────────────────────────────────────────
class LVQ1Baseline:
    """LVQ1 classique sur pixels aplatis — baseline de comparaison."""

    def __init__(self, n_prototypes_per_class=10,
                 lr=0.01, epochs=30, seed=42):
        self.n_proto = n_prototypes_per_class
        self.lr      = lr
        self.epochs  = epochs
        np.random.seed(seed)

    def fit(self, X, y, X_val=None, y_val=None):
        classes = np.unique(y)
        self.prototypes   = []
        self.proto_labels = []

        for cls in classes:
            X_cls = X[y == cls]
            idx   = np.random.choice(len(X_cls), self.n_proto,
                                     replace=False)
            self.prototypes.extend(X_cls[idx])
            self.proto_labels.extend([cls] * self.n_proto)

        self.prototypes   = np.array(self.prototypes, dtype=np.float32)
        self.proto_labels = np.array(self.proto_labels)

        best_acc    = 0.0
        best_protos = self.prototypes.copy()
        history     = []
        idx_all     = np.arange(len(X))

        for epoch in range(self.epochs):
            np.random.shuffle(idx_all)
            lr_epoch = self.lr * (0.95 ** epoch)

            for i in idx_all:
                xi, yi = X[i], y[i]
                dists  = np.linalg.norm(
                    self.prototypes - xi, axis=1)
                j      = np.argmin(dists)

                if self.proto_labels[j] == yi:
                    self.prototypes[j] += lr_epoch * (
                        xi - self.prototypes[j])
                else:
                    self.prototypes[j] -= lr_epoch * (
                        xi - self.prototypes[j])

            if X_val is not None:
                preds = self.predict(X_val)
                acc   = accuracy_score(y_val, preds)
                history.append(acc)

                if acc > best_acc:
                    best_acc    = acc
                    best_protos = self.prototypes.copy()
                    marker      = "✅"
                else:
                    marker      = ""

                print(f"  Epoch {epoch+1:2d} | Acc: {acc:.4f} | "
                      f"Best: {best_acc:.4f} | lr: {lr_epoch:.4f} {marker}")

        self.prototypes = best_protos
        return history, best_acc

    def predict(self, X):
        preds = []
        for xi in X:
            dists = np.linalg.norm(self.prototypes - xi, axis=1)
            preds.append(self.proto_labels[np.argmin(dists)])
        return np.array(preds)

    def predict_scores(self, X):
        """Score softmax basé sur distances inverses."""
        scores = []
        for xi in X:
            dists     = np.linalg.norm(self.prototypes - xi, axis=1)
            sim       = 1 / (dists + 1e-8)
            # Score classe 1 = somme similarités protos classe 1
            score_1   = sim[self.proto_labels == 1].sum()
            score_0   = sim[self.proto_labels == 0].sum()
            scores.append(score_1 / (score_0 + score_1 + 1e-8))
        return np.array(scores)


# ── Parse args ───────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--epochs",  type=int,   default=30)
    p.add_argument("--n_proto", type=int,   default=10)
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--name",    type=str,   default=None)
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()


# ── Run ──────────────────────────────────────────────────────
def run(args):
    run_name = args.name or \
        f"lvq1_np{args.n_proto}_lr{str(args.lr).replace('.','')}_seed{args.seed}"

    print(f"\n{'='*70}\nRUN : {run_name}\n{'='*70}\n")
    os.makedirs("figs", exist_ok=True)
    os.makedirs(DRIVE_PATH, exist_ok=True)

    # ── Charger données ──────────────────────────────────────
    data_train = torch.load(CACHE_TRAIN)
    data_clean = torch.load(CACHE_CLEAN)

    X_train = np.array([img.numpy().flatten()
                        for img in data_train["train_images"]])
    y_train = np.array(data_train["train_labels"])
    X_val   = np.array([img.numpy().flatten()
                        for img in data_clean["val_images"]])
    y_val   = np.array(data_clean["val_labels"])

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}\n")

    # ── LVQ1 ────────────────────────────────────────────────
    start_time = time.time()
    np.random.seed(args.seed)

    lvq     = LVQ1Baseline(
        n_prototypes_per_class=args.n_proto,
        lr=args.lr, epochs=args.epochs, seed=args.seed
    )
    history, best_acc = lvq.fit(X_train, y_train, X_val, y_val)
    elapsed_min = (time.time() - start_time) / 60

    # ── Métriques finales ────────────────────────────────────
    preds  = lvq.predict(X_val)
    scores = lvq.predict_scores(X_val)

    acc    = accuracy_score(y_val, preds)
    f1     = f1_score(y_val, preds, average='macro')
    f1s    = f1_score(y_val, preds, average=None)
    auc    = roc_auc_score(y_val, scores)

    print(f"\n✅ {run_name} :")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 macro  : {f1:.4f}")
    print(f"   F1 Cancer : {f1s[0]:.4f}")
    print(f"   F1 Normal : {f1s[1]:.4f}")
    print(f"   AUC       : {auc:.4f}")
    print(f"   Temps     : {elapsed_min:.1f} min")

    # ── Sauvegarder ─────────────────────────────────────────
    result = {
        "run_name" : run_name,
        "method"   : "LVQ1",
        "n_proto"  : args.n_proto,
        "lr"       : args.lr,
        "epochs"   : args.epochs,
        "seed"     : args.seed,
        "acc"      : float(acc),
        "f1_macro" : float(f1),
        "f1_cancer": float(f1s[0]),
        "f1_normal": float(f1s[1]),
        "auc"      : float(auc),
        "time_min" : elapsed_min,
        "history"  : [float(h) for h in history],
        "val_set"  : "clean_773",
    }

    with open(f"figs/{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    shutil.copy(f"figs/{run_name}.json",
                f"{DRIVE_PATH}{run_name}.json")

    print(f"✅ Sauvegardé : {DRIVE_PATH}{run_name}.json")
    return acc, history


if __name__ == "__main__":
    run(parse_args())