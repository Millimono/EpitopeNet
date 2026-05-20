# ============================================================
# combine_results.py — Combiner les 3 ablations
# ============================================================

import pandas as pd
import os

print("🔗 COMBINAISON DES RÉSULTATS...")

# Charger les 3 CSV
try:
    df1 = pd.read_csv("figs/ablation_part1_critical.csv")
    print(f"✅ Part 1 chargée : {len(df1)} configs")
except:
    print("❌ Part 1 non trouvée")
    df1 = pd.DataFrame()

try:
    df2 = pd.read_csv("figs/ablation_part2_hyperparams.csv")
    print(f"✅ Part 2 chargée : {len(df2)} configs")
except:
    print("❌ Part 2 non trouvée")
    df2 = pd.DataFrame()

try:
    df3 = pd.read_csv("figs/ablation_part3_scalability.csv")
    print(f"✅ Part 3 chargée : {len(df3)} configs")
except:
    print("❌ Part 3 non trouvée")
    df3 = pd.DataFrame()

# Combiner
df_all = pd.concat([df1, df2, df3], ignore_index=True)
df_all.to_csv("figs/ablation_complete.csv", index=False)

print(f"\n✅ Résultats combinés : {len(df_all)} configurations totales")
print(f"   Fichier : figs/ablation_complete.csv")

# Afficher résumé
print("\n" + "="*60)
print("RÉSUMÉ COMPLET")
print("="*60)

for param in df_all["param"].unique():
    sub = df_all[df_all["param"] == param]
    print(f"\n📊 {param}:")
    best_acc = sub["acc"].max()
    for _, row in sub.iterrows():
        marker = " ✅" if row["acc"] == best_acc else ""
        print(f"  {str(row['value']):>30} → {row['acc']:.4f}{marker}")

# Télécharger
try:
    from google.colab import files
    files.download("figs/ablation_complete.csv")
    print("\n📥 Fichier combiné téléchargé !")
except:
    print("\n⚠️  Fichier sauvé localement")