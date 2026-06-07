"""
pipeline/features.py
Feature engineering para forecasting univariado.
Todas las features usan shift(1) mínimo para evitar leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def build_features(
    df: pd.DataFrame,
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    diff_periods: List[int] = None,
    calendar: bool = True,
    fourier_configs: List[dict] = None,
    target_col: str = "y",
    date_col: str = "ds",
) -> pd.DataFrame:
    """
    Construye features para forecasting univariado.

    Args:
        df: DataFrame con columnas [date_col, target_col], ordenado por fecha.
        lags: Lista de lags a crear. Default: [1, 2, 3, 7, 14, 28].
        rolling_windows: Ventanas para rolling stats. Default: [7, 14, 28].
        diff_periods: Períodos para diferenciación. Default: [1, 7].
        calendar: Si True, agrega features de calendario.
        fourier_configs: Lista de dicts con {period, K}. Ej: [{'period': 7, 'K': 3}].
        target_col: Columna objetivo.
        date_col: Columna de fecha.

    Returns:
        DataFrame con features adicionales (sin NaN rows al inicio).
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    if lags is None:
        lags = [1, 2, 3, 7, 14, 28]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]
    if diff_periods is None:
        diff_periods = [1, 7]

    # --- Lags ---
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # --- Rolling statistics (shift(1) para evitar leakage) ---
    shifted = df[target_col].shift(1)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
        df[f"rolling_min_{w}"]  = shifted.rolling(w).min()
        df[f"rolling_max_{w}"]  = shifted.rolling(w).max()

    # --- Diferencias ---
    for p in diff_periods:
        df[f"diff_{p}"] = df[target_col].diff(p)

    # --- Calendar features ---
    if calendar:
        df[date_col] = pd.to_datetime(df[date_col])
        df["day_of_week"]  = df[date_col].dt.dayofweek
        df["day_of_month"] = df[date_col].dt.day
        df["day_of_year"]  = df[date_col].dt.dayofyear
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
        df["month"]        = df[date_col].dt.month
        df["quarter"]      = df[date_col].dt.quarter
        df["is_weekend"]   = (df[date_col].dt.dayofweek >= 5).astype(int)

    # --- Fourier terms ---
    if fourier_configs:
        t = np.arange(len(df))
        for config in fourier_configs:
            period = config["period"]
            K = config["K"]
            df = add_fourier_terms(df, t=t, period=period, K=K)

    # Eliminar filas con NaN al inicio (por lags y rolling)
    max_lag = max(lags) if lags else 0
    max_window = max(rolling_windows) if rolling_windows else 0
    n_drop = max(max_lag, max_window)
    df = df.iloc[n_drop:].reset_index(drop=True)

    n_nan = df.isnull().sum()
    cols_with_nan = n_nan[n_nan > 0]
    if len(cols_with_nan) > 0:
        pct = cols_with_nan / len(df) * 100
        for col, p in pct.items():
            if p > 30:
                print(f"⚠️  ADVERTENCIA: '{col}' tiene {p:.1f}% NaN — considerar eliminar")

    return df


def add_fourier_terms(
    df: pd.DataFrame,
    t: np.ndarray,
    period: float,
    K: int,
) -> pd.DataFrame:
    """Agrega K componentes de Fourier para un período dado."""
    for k in range(1, K + 1):
        df[f"fourier_sin_p{int(period)}_k{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"fourier_cos_p{int(period)}_k{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def get_feature_cols(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """Retorna lista de columnas de features (excluye ds, y y columnas indicadas)."""
    if exclude is None:
        exclude = []
    base_exclude = ["ds", "y"] + exclude
    return [c for c in df.columns if c not in base_exclude]


# ---------------------------------------------------------------------------
# CLI simple
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--data",   default="data/serie.csv", help="Path al CSV")
    parser.add_argument("--output", default="data/serie_features.csv", help="Output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["ds"])
    print(f"Datos: {len(df)} filas")

    df_feat = build_features(
        df,
        lags=[1, 2, 3, 7, 14, 28],
        rolling_windows=[7, 14, 28],
        diff_periods=[1, 7],
        calendar=True,
        fourier_configs=[{"period": 7, "K": 3}],
    )

    feature_cols = get_feature_cols(df_feat)
    print(f"Features generadas ({len(feature_cols)}): {feature_cols}")
    print(f"Filas tras eliminar NaN: {len(df_feat)}")

    df_feat.to_csv(args.output, index=False)
    print(f"Guardado en {args.output}")
