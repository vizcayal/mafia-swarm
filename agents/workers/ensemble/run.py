"""
Worker Ensemble — Blending de top-K modelos
Uso: python agents/workers/ensemble/run.py [opciones]

Lee Il Libro, selecciona los mejores modelos y los combina.
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pipeline.backtest import rolling_origin_backtest
from pipeline.features import build_features, get_feature_cols
from pipeline.evaluate import IlLibro

BANNER = """
╔══════════════════════════════════════╗
║   WORKER ENSEMBLE — Top-K Blending  ║
╚══════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description='Worker Ensemble: blend de modelos')
    p.add_argument('--top-k',   type=int, default=3,              help='Top-K modelos a combinar')
    p.add_argument('--method',  default='weighted',               help='simple | weighted | stacking')
    p.add_argument('--horizon', type=int, default=12,             help='Horizonte de predicción')
    p.add_argument('--folds',   type=int, default=5,              help='Número de folds')
    p.add_argument('--libro',   default='il_libro.json',          help='Path a Il Libro')
    p.add_argument('--data',    default='data/serie_features.csv', help='CSV con features')
    return p.parse_args()


def rebuild_model(exp: dict):
    """Reconstruye el modelo desde la config guardada en Il Libro."""
    model_name = exp['model']
    config = exp.get('config', {})

    if model_name == 'lightgbm':
        import lightgbm as lgb
        class W:
            def __init__(self): self.m = lgb.LGBMRegressor(**config, verbose=-1)
            def fit(self, X, y): self.m.fit(X, y)
            def predict(self, X): return self.m.predict(X)
        return W()
    elif model_name == 'xgboost':
        import xgboost as xgb
        class W:
            def __init__(self): self.m = xgb.XGBRegressor(**config, verbosity=0)
            def fit(self, X, y): self.m.fit(X, y)
            def predict(self, X): return self.m.predict(X)
        return W()
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        class W:
            def __init__(self): self.m = RandomForestRegressor(**config, n_jobs=-1)
            def fit(self, X, y): self.m.fit(X, y)
            def predict(self, X): return self.m.predict(X)
        return W()
    elif model_name == 'seasonal_naive':
        from pipeline.backtest import SeasonalNaiveModel
        return SeasonalNaiveModel(period=config.get('period', 12))
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado para ensemble")


class EnsembleModel:
    def __init__(self, base_models, weights=None, method='weighted'):
        self.base_models = base_models
        self.weights = weights
        self.method = method
        self._trained = []

    def fit(self, X, y):
        self._trained = []
        for m in self.base_models:
            m.fit(X, y)
            self._trained.append(m)

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self._trained])
        if self.method == 'simple':
            return preds.mean(axis=0)
        elif self.method == 'weighted' and self.weights is not None:
            w = np.array(self.weights).reshape(-1, 1)
            return (preds * w).sum(axis=0)
        return preds.mean(axis=0)


def main():
    print(BANNER)
    args = parse_args()

    libro = IlLibro(os.path.join(ROOT, args.libro))
    top_k = libro.get_top_k(k=args.top_k, only_beats_baseline=True)
    # Evitar recursión: no incluir otros ensembles como modelos base
    top_k = [e for e in top_k if not e['model'].startswith('ensemble')]

    if len(top_k) < 2:
        print(f"❌ No hay suficientes modelos que superen el baseline en Il Libro (se necesitan ≥ 2, hay {len(top_k)})")
        print("   Corre primero más workers de modelos.")
        sys.exit(1)

    print(f"Top-{args.top_k} modelos seleccionados de Il Libro:")
    for i, exp in enumerate(top_k):
        print(f"  {i+1}. {exp['model']:20s}  MAE={exp['mae_mean']:.4f}  ({exp['id']})")

    # Cargar datos
    data_path = os.path.join(ROOT, args.data)
    if not os.path.exists(data_path):
        # Si no hay features, usar serie base
        data_path = os.path.join(ROOT, 'data/serie.csv')
    df = pd.read_csv(data_path, parse_dates=['ds'])

    # Construir modelos base
    base_models = [rebuild_model(exp) for exp in top_k]
    mae_values  = [exp['mae_mean'] for exp in top_k]

    # Pesos inversos al MAE
    inv = [1.0 / m for m in mae_values]
    total = sum(inv)
    weights = [w / total for w in inv]

    print(f"\n⚖️  Pesos (inverso MAE): {[round(w, 3) for w in weights]}")
    print(f"🔧 Método: {args.method}")

    ensemble = EnsembleModel(base_models, weights=weights, method=args.method)

    print(f"\n🏁 Backtest ensemble (horizon={args.horizon}, folds={args.folds}):")
    result = rolling_origin_backtest(ensemble, df, horizon=args.horizon, n_folds=args.folds)

    baseline_mae = libro.data.get('best_mae') or mae_values[0]
    seasonal_mae = next(
        (e['mae_mean'] for e in libro.data['experiments'] if e['model'] == 'seasonal_naive'),
        baseline_mae
    )

    print(f"\n{'='*50}")
    print(f"  MAE por fold:     {[round(x, 2) for x in result['mae_per_fold']]}")
    print(f"  MAE ensemble:     {result['mae_mean']:.4f}")
    print(f"  MAE mejor single: {mae_values[0]:.4f}")
    mejora = (mae_values[0] - result['mae_mean']) / mae_values[0] * 100
    print(f"  {'✅ MEJORA' if result['mae_mean'] < mae_values[0] else '⚠️  NO MEJORA'} vs mejor modelo ({mejora:+.1f}%)")
    print(f"{'='*50}")

    exp_id = libro.register(
        model_name=f'ensemble_{args.method}',
        agent='worker_ensemble',
        mae_per_fold=result['mae_per_fold'],
        mae_seasonal_naive=seasonal_mae,
        config={
            'method': args.method,
            'top_k': args.top_k,
            'base_models': [e['id'] for e in top_k],
            'weights': weights,
        },
        horizon=args.horizon,
        extra={'improvement_vs_best_single': f'{mejora:+.1f}%'},
    )
    print(f"\n📖 Registrado en Il Libro: {exp_id}")

    libro.print_leaderboard()


if __name__ == '__main__':
    main()
