"""
pipeline/loop.py — El Karpathy Loop para mafia-swarm

Patrón de karpathy/autoresearch adaptado a forecasting univariado:
  - train.py      → pipeline/variant.py  (el agente edita este archivo)
  - val_bpb       → MAE (rolling-origin, folds fijos)
  - 5 min budget  → N iteraciones con backtest de budget fijo
  - program.md    → agents/workers/{worker}/program.md

Uso:
  python pipeline/loop.py --worker modelos --model lightgbm --budget 10
  python pipeline/loop.py --worker artigiano --budget 5

Modo agente (Claude Code / Ruflo):
  Apuntar el agente a agents/workers/{worker}/program.md
  El agente edita pipeline/variant.py en cada iteración
  Este script corre el backtest y decide keep/discard
"""

import argparse
import shutil
import json
import sys
import os
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from pipeline.evaluate import IlLibro, mae
from pipeline.backtest import rolling_origin_backtest

VARIANT_PATH = ROOT / "pipeline" / "variant.py"
BACKTEST_PATH = ROOT / "pipeline" / "backtest.py"


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def get_best_mae(libro: IlLibro) -> float:
    """Obtiene el mejor MAE actual de Il Libro."""
    if libro.data["best_mae"] is not None:
        return libro.data["best_mae"]
    return float("inf")


def run_backtest_from_variant(data_path: str, horizon: int, folds: int) -> dict:
    """
    Importa pipeline/variant.py dinámicamente y corre el backtest.
    variant.py debe exponer: get_model() → objeto con fit(X,y) y predict(X)
    """
    import pandas as pd

    # Recargar variant.py en cada iteración
    spec = importlib.util.spec_from_file_location("variant", VARIANT_PATH)
    variant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variant)

    df = pd.read_csv(data_path, parse_dates=["ds"])
    model = variant.get_model()
    result = rolling_origin_backtest(model, df, horizon=horizon, n_folds=folds)
    return result


def save_variant_snapshot(iteration: int) -> Path:
    """Guarda una copia de variant.py antes de cada modificación."""
    snapshot_dir = ROOT / "pipeline" / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snap_path = snapshot_dir / f"variant_iter_{iteration:04d}.py"
    shutil.copy(VARIANT_PATH, snap_path)
    return snap_path


def revert_variant(snapshot_path: Path):
    """Revierte variant.py al snapshot anterior."""
    shutil.copy(snapshot_path, VARIANT_PATH)
    log(f"↩️  Revertido a {snapshot_path.name}")


def parse_args():
    p = argparse.ArgumentParser(description="El Karpathy Loop para mafia-swarm")
    p.add_argument("--worker",   required=True,                         help="Nombre del worker (modelos, artigiano, etc.)")
    p.add_argument("--budget",   type=int, default=10,                  help="Número máximo de iteraciones")
    p.add_argument("--horizon",  type=int, default=12,                  help="Horizonte de predicción")
    p.add_argument("--folds",    type=int, default=5,                   help="Folds de backtest por iteración")
    p.add_argument("--data",     default="data/serie_features.csv",     help="CSV de datos con features")
    p.add_argument("--libro",    default="il_libro.json",               help="Path a Il Libro")
    p.add_argument("--init",     action="store_true",                   help="Inicializar variant.py desde backtest.py y salir")
    return p.parse_args()


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          EL KARPATHY LOOP — mafia-swarm                      ║
║   editar → backtest → MAE → conservar/descartar → repetir   ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    args = parse_args()

    data_path = str(ROOT / args.data)
    libro_path = str(ROOT / args.libro)
    program_path = ROOT / "agents" / "workers" / args.worker / "program.md"

    # ── Init: copiar backtest.py → variant.py ──────────────────────────────
    if not VARIANT_PATH.exists() or args.init:
        shutil.copy(BACKTEST_PATH, VARIANT_PATH)
        log(f"✅ variant.py inicializado desde backtest.py")

        # Agregar función get_model() al final si no existe
        content = VARIANT_PATH.read_text()
        if "def get_model" not in content:
            VARIANT_PATH.write_text(content + """

# ── KARPATHY LOOP INTERFACE ─────────────────────────────────────────────────
# El agente modifica get_model() en cada iteración.
# Puede cambiar el modelo, los hiperparámetros, o cualquier otra cosa.

import lightgbm as lgb

def get_model():
    \"\"\"
    Retorna el modelo a evaluar en esta iteración.
    EDITA ESTA FUNCIÓN para proponer cambios.
    \"\"\"
    return type('Model', (), {
        'fit':     lambda self, X, y: setattr(self, 'm', lgb.LGBMRegressor(
                       n_estimators=300, learning_rate=0.05, num_leaves=64, verbose=-1
                   ).fit(X, y)) or self,
        'predict': lambda self, X: self.m.predict(X),
    })()
""")
        log(f"✅ get_model() agregado a variant.py")

        if args.init:
            log(f"Listo. Edita pipeline/variant.py → get_model() para proponer cambios.")
            log(f"Luego corre: python pipeline/loop.py --worker {args.worker} --budget {args.budget}")
            return

    # ── Verificar que variant.py tiene get_model() ─────────────────────────
    if "def get_model" not in VARIANT_PATH.read_text():
        log("❌ variant.py no tiene get_model(). Corre con --init primero.")
        sys.exit(1)

    libro = IlLibro(libro_path)
    best_mae = get_best_mae(libro)
    seasonal_mae = next(
        (e["mae_mean"] for e in libro.data["experiments"] if e["model"] == "seasonal_naive"),
        float("inf")
    )

    log(f"Worker: {args.worker}")
    log(f"Budget: {args.budget} iteraciones | Horizon: {args.horizon} | Folds: {args.folds}")
    log(f"MAE actual (Il Libro): {best_mae:.4f}")
    log(f"Baseline (Seasonal Naive): {seasonal_mae:.4f}")
    if program_path.exists():
        log(f"program.md: {program_path}")
    print()

    kept = 0
    discarded = 0

    for iteration in range(1, args.budget + 1):
        print(f"\n{'─'*60}")
        log(f"ITERACIÓN {iteration}/{args.budget}")

        # Guardar snapshot antes de que el agente modifique
        snapshot = save_variant_snapshot(iteration)
        log(f"📸 Snapshot guardado: {snapshot.name}")

        # ── PUNTO DE EDICIÓN DEL AGENTE ────────────────────────────────────
        # En modo agente (Claude Code / Ruflo), el agente edita variant.py aquí.
        # En modo manual, el humano edita variant.py y presiona Enter.
        if sys.stdin.isatty():
            log(f"✏️  Edita pipeline/variant.py → get_model() con tu propuesta.")
            log(f"   (Referencia: agents/workers/{args.worker}/program.md)")
            input("   Presiona Enter cuando estés listo... ")
        # ──────────────────────────────────────────────────────────────────

        # Correr backtest con el variant actual
        log(f"⏳ Corriendo backtest (folds={args.folds}, horizon={args.horizon})...")
        try:
            result = run_backtest_from_variant(data_path, args.horizon, args.folds)
            current_mae = result["mae_mean"]
        except Exception as e:
            log(f"❌ Error en backtest: {e}")
            revert_variant(snapshot)
            discarded += 1
            libro.register(
                model_name=f"variant_iter_{iteration}",
                agent=f"worker_{args.worker}",
                mae_per_fold=[999.0] * args.folds,
                mae_seasonal_naive=seasonal_mae,
                config={"iteration": iteration, "status": "error", "error": str(e)},
                horizon=args.horizon,
            )
            continue

        log(f"   MAE: {current_mae:.4f}  |  Mejor: {best_mae:.4f}")
        log(f"   Por fold: {[round(x, 2) for x in result['mae_per_fold']]}")

        # ── Decisión: keep o discard ───────────────────────────────────────
        if current_mae < best_mae:
            mejora_pct = (best_mae - current_mae) / best_mae * 100
            log(f"✅ CONSERVADO (+{mejora_pct:.1f}% mejora) — nuevo best_mae: {current_mae:.4f}")
            best_mae = current_mae
            kept += 1
            status = "kept"
        else:
            log(f"❌ DESCARTADO — sin mejora vs {best_mae:.4f}")
            revert_variant(snapshot)
            discarded += 1
            status = "discarded"

        # Registrar en Il Libro siempre
        libro.register(
            model_name=f"variant_iter_{iteration}",
            agent=f"worker_{args.worker}_loop",
            mae_per_fold=result["mae_per_fold"],
            mae_seasonal_naive=seasonal_mae,
            config={"iteration": iteration, "status": status},
            horizon=args.horizon,
            extra={"loop_status": status, "loop_iteration": iteration},
        )

    # ── Resumen ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    log(f"LOOP COMPLETADO")
    log(f"  Iteraciones: {args.budget}")
    log(f"  Conservados: {kept} | Descartados: {discarded}")
    log(f"  MAE inicial: {get_best_mae(IlLibro(libro_path)):.4f}")
    log(f"  MAE final:   {best_mae:.4f}")
    mejora_total = (get_best_mae(IlLibro(libro_path)) - best_mae)
    log(f"  Mejora total: {mejora_total:.4f}")
    libro.print_leaderboard(top_n=5)


if __name__ == "__main__":
    main()
