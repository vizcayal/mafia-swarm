"""
Il Selezionatore — Worker de Feature Selection
Uso: python agents/workers/selezionatore/run.py [opciones]

Lee las features generadas por L'Artigiano, aplica selección y guarda el subconjunto.
"""

import argparse
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from pipeline.features import get_feature_cols

BANNER = """
╔══════════════════════════════════════════╗
║  IL SELEZIONATORE — Feature Selection    ║
╚══════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description='Il Selezionatore: feature selection')
    p.add_argument('--data',       default='data/serie_features.csv', help='CSV con features (salida de L\'Artigiano)')
    p.add_argument('--output',     default='data/serie_selected.csv', help='CSV con features seleccionadas')
    p.add_argument('--method',     default='importance',              help='importance | mutual_info | correlation | all')
    p.add_argument('--top-k',      type=int, default=20,              help='Top-K features a conservar')
    p.add_argument('--corr-threshold', type=float, default=0.95,      help='Umbral de correlación para eliminar redundantes')
    return p.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    df = pd.read_csv(os.path.join(ROOT, args.data), parse_dates=['ds'])
    feature_cols = get_feature_cols(df)
    print(f"Features disponibles: {len(feature_cols)}")
    print(f"Filas: {len(df)}")

    X = df[feature_cols].fillna(0).values
    y = df['y'].values

    importance_scores = np.zeros(len(feature_cols))

    # --- Método 1: Feature importance (Random Forest) ---
    if args.method in ('importance', 'all'):
        print("\n🔍 Calculando feature importance (RandomForest)...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        fi = rf.feature_importances_
        # Normalizar
        fi = fi / fi.sum()
        importance_scores += fi
        print(f"   Top-5: {sorted(zip(feature_cols, fi), key=lambda x: -x[1])[:5]}")

    # --- Método 2: Mutual Information ---
    if args.method in ('mutual_info', 'all'):
        print("\n🔍 Calculando mutual information...")
        mi = mutual_info_regression(X, y, random_state=42)
        mi_norm = mi / mi.sum() if mi.sum() > 0 else mi
        importance_scores += mi_norm
        print(f"   Top-5: {sorted(zip(feature_cols, mi), key=lambda x: -x[1])[:5]}")

    # --- Ranking final ---
    ranking = sorted(zip(feature_cols, importance_scores), key=lambda x: -x[1])
    print(f"\n📊 Ranking completo:")
    for i, (col, score) in enumerate(ranking):
        print(f"   {i+1:2d}. {col:<35} {score:.6f}")

    # --- Eliminar correlacionadas ---
    top_k_cols = [col for col, _ in ranking[:args.top_k]]
    X_top = df[top_k_cols].fillna(0)
    corr_matrix = X_top.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > args.corr_threshold)]

    if to_drop:
        print(f"\n🗑️  Eliminando {len(to_drop)} features redundantes (corr > {args.corr_threshold}): {to_drop}")
        top_k_cols = [c for c in top_k_cols if c not in to_drop]

    print(f"\n✅ Features seleccionadas ({len(top_k_cols)}): {top_k_cols}")

    # --- Guardar ---
    df_selected = df[['ds', 'y'] + top_k_cols]
    out_path = os.path.join(ROOT, args.output)
    df_selected.to_csv(out_path, index=False)
    print(f"\n💾 Guardado en: {args.output}")

    # Guardar lista de features
    feat_path = os.path.join(ROOT, 'data', 'selected_features.json')
    with open(feat_path, 'w') as f:
        json.dump({'selected_features': top_k_cols, 'method': args.method, 'top_k': args.top_k}, f, indent=2)
    print(f"   Features: data/selected_features.json")


if __name__ == '__main__':
    main()
