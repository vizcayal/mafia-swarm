"""
pipeline/backtest.py
Rolling-origin backtest para forecasting univariado.
Regla: SIEMPRE rolling-origin, NUNCA K-fold aleatorio.
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Optional
import json
from pathlib import Path


def rolling_origin_backtest(
    model,
    df: pd.DataFrame,
    horizon: int,
    n_folds: int = 5,
    min_train_size: Optional[int] = None,
    target_col: str = "y",
    date_col: str = "ds",
) -> dict:
    """
    Backtest rolling-origin (time series cross-validation).

    Args:
        model: Modelo con métodos fit(X, y) y predict(X) o equivalente.
               Para modelos clásicos (ARIMA, etc.) debe tener fit(series) + predict(h).
        df: DataFrame con columnas [date_col, target_col] y features.
        horizon: Pasos a predecir hacia adelante.
        n_folds: Número de folds.
        min_train_size: Mínimo de observaciones para el primer entrenamiento.
                        Si None, usa 60% de los datos.
        target_col: Nombre de la columna objetivo.
        date_col: Nombre de la columna de fecha.

    Returns:
        dict con mae_per_fold, mae_mean, predictions_per_fold
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    if min_train_size is None:
        min_train_size = int(n * 0.6)

    # Calcular puntos de corte para cada fold
    # Cada fold tiene horizon más observaciones que el anterior
    step = (n - min_train_size - horizon) // n_folds
    if step < 1:
        step = 1

    cutoffs = [
        min_train_size + i * step
        for i in range(n_folds)
    ]
    # Filtrar cutoffs válidos
    cutoffs = [c for c in cutoffs if c + horizon <= n]

    if len(cutoffs) == 0:
        raise ValueError(
            f"No hay suficientes datos para {n_folds} folds con horizon={horizon}. "
            f"n={n}, min_train_size={min_train_size}"
        )

    mae_per_fold = []
    predictions_per_fold = []

    for fold_idx, cutoff in enumerate(cutoffs):
        train = df.iloc[:cutoff].copy()
        test  = df.iloc[cutoff:cutoff + horizon].copy()

        y_train = train[target_col].values
        y_test  = test[target_col].values

        # Separar features si existen
        feature_cols = [c for c in df.columns if c not in [date_col, target_col]]

        if feature_cols:
            X_train = train[feature_cols].values
            X_test  = test[feature_cols].values
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # Modelo clásico: espera serie temporal
            model.fit(y_train)
            y_pred = model.predict(horizon)

        mae = mean_absolute_error(y_test, y_pred)
        mae_per_fold.append(float(mae))
        predictions_per_fold.append({
            "fold": fold_idx,
            "cutoff_idx": cutoff,
            "cutoff_date": str(train[date_col].iloc[-1]),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "mae": float(mae),
        })

        print(f"  Fold {fold_idx+1}/{len(cutoffs)} | cutoff={train[date_col].iloc[-1].date()} | MAE={mae:.4f}")

    result = {
        "mae_per_fold": mae_per_fold,
        "mae_mean": float(np.mean(mae_per_fold)),
        "mae_std": float(np.std(mae_per_fold)),
        "n_folds": len(cutoffs),
        "horizon": horizon,
        "predictions_per_fold": predictions_per_fold,
    }
    return result


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ---------------------------------------------------------------------------
# Baselines obligatorios
# ---------------------------------------------------------------------------

class NaiveModel:
    """Predice el último valor observado."""
    def fit(self, y: np.ndarray):
        self.last_value = y[-1]

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)


class SeasonalNaiveModel:
    """Predice el valor del mismo período de la temporada anterior."""
    def __init__(self, period: int = 7):
        self.period = period

    def fit(self, y: np.ndarray):
        self.history = y

    def predict(self, horizon: int) -> np.ndarray:
        preds = []
        for h in range(1, horizon + 1):
            idx = len(self.history) - self.period + ((h - 1) % self.period)
            preds.append(self.history[idx])
        return np.array(preds)


# ---------------------------------------------------------------------------
# CLI simple
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest rolling-origin")
    parser.add_argument("--data",    default="data/serie.csv",  help="Path al CSV (ds, y)")
    parser.add_argument("--model",   default="seasonal_naive",  help="naive | seasonal_naive")
    parser.add_argument("--horizon", type=int, default=14,       help="Horizonte de predicción")
    parser.add_argument("--folds",   type=int, default=5,        help="Número de folds")
    parser.add_argument("--period",  type=int, default=7,        help="Período para Seasonal Naive")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["ds"])
    print(f"Datos cargados: {len(df)} filas, desde {df['ds'].min().date()} hasta {df['ds'].max().date()}")

    if args.model == "naive":
        model = NaiveModel()
    elif args.model == "seasonal_naive":
        model = SeasonalNaiveModel(period=args.period)
    else:
        raise ValueError(f"Modelo '{args.model}' no reconocido en CLI. Implementa en backtest.py.")

    print(f"\nCorriendo backtest [{args.model}] | horizon={args.horizon} | folds={args.folds}")
    result = rolling_origin_backtest(
        model=model,
        df=df[["ds", "y"]],   # sin features para baselines clásicos
        horizon=args.horizon,
        n_folds=args.folds,
    )

    print(f"\nResultados:")
    print(f"  MAE por fold: {result['mae_per_fold']}")
    print(f"  MAE promedio: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")
