"""
pipeline/evaluate.py
Métricas de evaluación y utilidades para Il Libro.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


# ---------------------------------------------------------------------------
# Il Libro — leaderboard
# ---------------------------------------------------------------------------

class IlLibro:
    """
    Registro central de experimentos. Persiste en il_libro.json.
    """

    def __init__(self, path: str = "il_libro.json"):
        self.path = Path(path)
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "experiments": [],
                "best_mae": None,
                "best_experiment_id": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def register(
        self,
        model_name: str,
        agent: str,
        mae_per_fold: List[float],
        mae_seasonal_naive: float,
        config: dict = None,
        features_pipeline: Optional[str] = None,
        horizon: int = None,
        extra: dict = None,
    ) -> str:
        """
        Registra un experimento en Il Libro.

        Returns:
            ID del experimento registrado.
        """
        exp_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        mae_mean = float(np.mean(mae_per_fold))
        beats_baseline = mae_mean < mae_seasonal_naive

        experiment = {
            "id": exp_id,
            "agent": agent,
            "model": model_name,
            "config": config or {},
            "features_pipeline": features_pipeline,
            "horizon": horizon,
            "mae_per_fold": [float(x) for x in mae_per_fold],
            "mae_mean": mae_mean,
            "mae_std": float(np.std(mae_per_fold)),
            "mae_seasonal_naive": float(mae_seasonal_naive),
            "beats_baseline": beats_baseline,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            experiment.update(extra)

        self.data["experiments"].append(experiment)

        # Actualizar mejor experimento
        if self.data["best_mae"] is None or mae_mean < self.data["best_mae"]:
            self.data["best_mae"] = mae_mean
            self.data["best_experiment_id"] = exp_id

        self._save()
        print(f"✅ Registrado: {exp_id} | MAE={mae_mean:.4f} | beats_baseline={beats_baseline}")
        return exp_id

    def get_top_k(self, k: int = 5, only_beats_baseline: bool = True) -> list:
        """Retorna los top-K experimentos por MAE."""
        exps = self.data["experiments"]
        if only_beats_baseline:
            exps = [e for e in exps if e.get("beats_baseline", False)]
        return sorted(exps, key=lambda x: x["mae_mean"])[:k]

    def summary(self) -> pd.DataFrame:
        """Retorna un DataFrame resumen de todos los experimentos."""
        rows = []
        for e in self.data["experiments"]:
            rows.append({
                "id": e["id"],
                "model": e["model"],
                "agent": e["agent"],
                "mae_mean": e["mae_mean"],
                "mae_std": e.get("mae_std", None),
                "beats_baseline": e.get("beats_baseline", None),
                "timestamp": e["timestamp"],
            })
        return pd.DataFrame(rows).sort_values("mae_mean")

    def print_leaderboard(self, top_n: int = 10):
        df = self.summary().head(top_n)
        print("\n📊 IL LIBRO — Leaderboard")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\nMejor MAE: {self.data['best_mae']:.4f} ({self.data['best_experiment_id']})")

    def append_results_tsv(self, tsv_path: str, batch_num: int,
                           selected_ids: List[str]) -> None:
        """
        Append one row per experiment dispatched in this batch to a flat TSV log.
        Schema: batch \t timestamp \t exp_id \t model \t agent \t mae_mean \t mae_std \t beats_baseline \t description
        Inspired by Karpathy's autoresearch results.tsv pattern.
        """
        header = "batch\ttimestamp\texp_id\tmodel\tagent\tmae_mean\tmae_std\tbeats_baseline\tdescription\n"
        path = Path(tsv_path)
        write_header = not path.exists() or path.stat().st_size == 0

        # Resolve only the experiments registered in this batch (by id match)
        rows = []
        ids_set = set(selected_ids)
        for e in self.data.get("experiments", []):
            # match either explicit id mapping or recently-added experiments via timestamp ordering
            if e["id"] in ids_set:
                desc = (e.get("config", {}) or {})
                desc_str = " ".join(f"{k}={v}" for k, v in list(desc.items())[:4]).replace("\t", " ")
                rows.append((
                    batch_num,
                    e.get("timestamp", ""),
                    e["id"],
                    e.get("model", ""),
                    e.get("agent", ""),
                    f"{e.get('mae_mean', 0):.6f}",
                    f"{e.get('mae_std', 0):.4f}",
                    "1" if e.get("beats_baseline") else "0",
                    desc_str,
                ))

        # If none matched (Contabile proposal id != experiment id), fall back to N most recent
        if not rows:
            recent = self.data.get("experiments", [])[-len(selected_ids):]
            for e in recent:
                desc = (e.get("config", {}) or {})
                desc_str = " ".join(f"{k}={v}" for k, v in list(desc.items())[:4]).replace("\t", " ")
                rows.append((
                    batch_num,
                    e.get("timestamp", ""),
                    e["id"],
                    e.get("model", ""),
                    e.get("agent", ""),
                    f"{e.get('mae_mean', 0):.6f}",
                    f"{e.get('mae_std', 0):.4f}",
                    "1" if e.get("beats_baseline") else "0",
                    desc_str,
                ))

        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    libro = IlLibro()
    libro.print_leaderboard()
