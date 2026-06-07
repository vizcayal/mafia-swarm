"""
Worker Modelos — Entrenamiento, HPO y backtest
Uso: python agents/workers/modelos/run.py --model lightgbm [opciones]

Entrena un modelo, corre HPO, hace backtest rolling-origin y registra en Il Libro.
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
from pipeline.backtest import rolling_origin_backtest, NaiveModel, SeasonalNaiveModel
from pipeline.features import get_feature_cols
from pipeline.evaluate import IlLibro

BANNER = """
╔══════════════════════════════════════╗
║   WORKER MODELOS — HPO + Backtest   ║
╚══════════════════════════════════════╝
"""

SUPPORTED_MODELS = ['naive', 'seasonal_naive', 'lightgbm', 'xgboost', 'random_forest']


def parse_args():
    p = argparse.ArgumentParser(description='Worker Modelos: HPO + backtest')
    p.add_argument('--model',    required=True,                       help=f'Modelo: {SUPPORTED_MODELS}')
    p.add_argument('--data',     default='data/serie_features.csv',   help='CSV con features')
    p.add_argument('--horizon',  type=int, default=12,                help='Horizonte de predicción')
    p.add_argument('--folds',    type=int, default=5,                 help='Número de folds')
    p.add_argument('--trials',   type=int, default=30,                help='Trials de HPO (Optuna)')
    p.add_argument('--period',   type=int, default=12,                help='Período para Seasonal Naive')
    p.add_argument('--libro',    default='il_libro.json',             help='Path al Il Libro')
    p.add_argument('--baseline-mae', type=float, default=None,        help='MAE del baseline a superar')
    return p.parse_args()


def get_baseline_mae(libro_path):
    """Lee el mejor MAE de Seasonal Naive de Il Libro."""
    libro = IlLibro(libro_path)
    for exp in sorted(libro.data['experiments'], key=lambda x: x['mae_mean']):
        if exp['model'] == 'seasonal_naive':
            return exp['mae_mean']
    return None


# ── Wrappers de modelos ──────────────────────────────────────────────────────

class LightGBMModel:
    def __init__(self, params):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(**params, verbose=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel:
    def __init__(self, params):
        import xgboost as xgb
        self.model = xgb.XGBRegressor(**params, verbosity=0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel:
    def __init__(self, params):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**params, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# ── HPO por modelo ───────────────────────────────────────────────────────────

def run_hpo_lightgbm(df, horizon, folds, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 8, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        res = rolling_origin_backtest(LightGBMModel(params), df, horizon=horizon, n_folds=folds)
        return res['mae_mean']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def run_hpo_xgboost(df, horizon, folds, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':     trial.suggest_int('max_depth', 3, 10),
            'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        res = rolling_origin_backtest(XGBoostModel(params), df, horizon=horizon, n_folds=folds)
        return res['mae_mean']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def run_hpo_random_forest(df, horizon, folds, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth':    trial.suggest_int('max_depth', 3, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
        res = rolling_origin_backtest(RandomForestModel(params), df, horizon=horizon, n_folds=folds)
        return res['mae_mean']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    args = parse_args()

    libro_path = os.path.join(ROOT, args.libro)
    data_path  = os.path.join(ROOT, args.data)

    print(f"🤖 Modelo:   {args.model}")
    print(f"📂 Datos:    {args.data}")
    print(f"📏 Horizon:  {args.horizon} | Folds: {args.folds} | Trials HPO: {args.trials}")

    # Obtener MAE baseline de Il Libro
    baseline_mae = args.baseline_mae or get_baseline_mae(libro_path) or float('inf')
    print(f"🎯 Baseline (Seasonal Naive): {baseline_mae:.4f}")
    print()

    libro = IlLibro(libro_path)

    # ── Baselines (sin features) ──
    if args.model in ('naive', 'seasonal_naive'):
        df_base = pd.read_csv(os.path.join(ROOT, 'data/serie.csv'), parse_dates=['ds'])
        if args.model == 'naive':
            model = NaiveModel()
        else:
            model = SeasonalNaiveModel(period=args.period)
        result = rolling_origin_backtest(model, df_base[['ds', 'y']], horizon=args.horizon, n_folds=args.folds)
        best_params = {'period': args.period} if args.model == 'seasonal_naive' else {}

    # ── Modelos ML con HPO ──
    else:
        df = pd.read_csv(data_path, parse_dates=['ds'])
        print(f"Features: {len(get_feature_cols(df))}")

        print(f"⏳ Corriendo HPO ({args.trials} trials)...")
        if args.model == 'lightgbm':
            best_params, best_hpo_mae = run_hpo_lightgbm(df, args.horizon, args.folds, args.trials)
            final_model = LightGBMModel(best_params)
        elif args.model == 'xgboost':
            best_params, best_hpo_mae = run_hpo_xgboost(df, args.horizon, args.folds, args.trials)
            final_model = XGBoostModel(best_params)
        elif args.model == 'random_forest':
            best_params, best_hpo_mae = run_hpo_random_forest(df, args.horizon, args.folds, args.trials)
            final_model = RandomForestModel(best_params)
        else:
            print(f"❌ Modelo '{args.model}' no soportado. Opciones: {SUPPORTED_MODELS}")
            sys.exit(1)

        print(f"   Mejor MAE HPO: {best_hpo_mae:.4f}")
        print(f"   Mejores params: {best_params}")
        print(f"\n🏁 Backtest final con mejores params:")
        result = rolling_origin_backtest(final_model, df, horizon=args.horizon, n_folds=args.folds)

    # ── Resultados ──
    mae_mean = result['mae_mean']
    print(f"\n{'='*50}")
    print(f"  MAE por fold:  {[round(x, 2) for x in result['mae_per_fold']]}")
    print(f"  MAE promedio:  {mae_mean:.4f}")
    print(f"  Baseline:      {baseline_mae:.4f}")
    mejora = (baseline_mae - mae_mean) / baseline_mae * 100
    print(f"  {'✅ SUPERA' if mae_mean < baseline_mae else '❌ NO SUPERA'} el baseline ({mejora:+.1f}%)")
    print(f"{'='*50}")

    # ── Registrar en Il Libro ──
    exp_id = libro.register(
        model_name=args.model,
        agent='worker_modelos',
        mae_per_fold=result['mae_per_fold'],
        mae_seasonal_naive=baseline_mae,
        config=best_params,
        features_pipeline=args.data,
        horizon=args.horizon,
    )
    print(f"\n📖 Registrado en Il Libro: {exp_id}")


if __name__ == '__main__':
    main()
