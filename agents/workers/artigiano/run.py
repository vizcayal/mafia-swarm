"""
L'Artigiano — Worker de Feature Engineering
Uso: python agents/workers/artigiano/run.py [opciones]

Genera features sobre la serie temporal y guarda el resultado en data/.
"""

import argparse
import sys
import os

# Agregar raíz del proyecto al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, ROOT)

import pandas as pd
from pipeline.features import build_features, get_feature_cols

BANNER = """
╔══════════════════════════════════════╗
║  L'ARTIGIANO — Feature Engineering  ║
╚══════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description="L'Artigiano: feature engineering")
    p.add_argument('--data',      default='data/serie.csv',          help='CSV de entrada (ds, y)')
    p.add_argument('--output',    default='data/serie_features.csv', help='CSV de salida con features')
    p.add_argument('--lags',      default='1,2,3,6,12,24',           help='Lags separados por coma')
    p.add_argument('--windows',   default='3,6,12',                  help='Ventanas rolling separadas por coma')
    p.add_argument('--diffs',     default='1,12',                    help='Períodos de diferenciación')
    p.add_argument('--fourier',   default='12:4',                    help='period:K para Fourier (ej: 12:4)')
    p.add_argument('--no-calendar', action='store_true',             help='No agregar features de calendario')
    return p.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    # Parsear argumentos
    lags    = [int(x) for x in args.lags.split(',')]
    windows = [int(x) for x in args.windows.split(',')]
    diffs   = [int(x) for x in args.diffs.split(',')]

    fourier_configs = []
    if args.fourier:
        for token in args.fourier.split(';'):
            period, K = token.split(':')
            fourier_configs.append({'period': int(period), 'K': int(K)})

    print(f"📂 Datos:    {args.data}")
    print(f"📤 Output:   {args.output}")
    print(f"⚙️  Lags:     {lags}")
    print(f"⚙️  Windows:  {windows}")
    print(f"⚙️  Diffs:    {diffs}")
    print(f"⚙️  Fourier:  {fourier_configs}")
    print(f"⚙️  Calendar: {not args.no_calendar}")
    print()

    # Cargar datos
    df = pd.read_csv(os.path.join(ROOT, args.data), parse_dates=['ds'])
    print(f"Filas cargadas: {len(df)} | rango: {df.ds.min().date()} → {df.ds.max().date()}")

    # Generar features
    df_feat = build_features(
        df,
        lags=lags,
        rolling_windows=windows,
        diff_periods=diffs,
        calendar=not args.no_calendar,
        fourier_configs=fourier_configs,
    )

    feature_cols = get_feature_cols(df_feat)
    print(f"\n✅ Features generadas: {len(feature_cols)}")
    for col in feature_cols:
        print(f"   {col}")

    # Guardar
    out_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f"\n💾 Guardado en: {args.output}")
    print(f"   Filas: {len(df_feat)} (originales: {len(df)}, eliminadas por NaN: {len(df) - len(df_feat)})")


if __name__ == '__main__':
    main()
