# 🤌 La Mafia — Agent Swarm para Forecasting

> *"En esta familia, el que no mejora el MAE, desaparece."*

Sistema de agent swarm jerárquico con temática mafiosa italiana para **forecasting de series temporales univariadas**. Objetivo único: **minimizar MAE**.

---

## Arquitectura

```
                        ┌─────────────────┐
                        │   EL PATRÓN     │  ← coordinador / Queen
                        │  (coordinator)  │     lanza y mata workers
                        └────────┬────────┘     según presupuesto
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐  ┌──────────────┐
     │  L'EMISSARIO   │ │  IL CONTABILE  │  │  IL LIBRO    │
     │  (researcher)  │ │  (analyzer)    │  │  (leaderboard│
     │  busca nuevas  │ │  analiza errores│  │   + memoria) │
     │  técnicas      │ │  y propone dirs│  └──────────────┘
     └────────┬───────┘ └───────┬────────┘
              └────────┬────────┘
                       ▼  Cola de propuestas
                ┌──────────────┐
                │   EL PATRÓN  │ rankea: MAE_esperado × confianza ÷ costo
                └──────┬───────┘
                       │ lanza workers en paralelo
       ┌───────────────┼───────────────┬──────────────┐
       ▼               ▼               ▼              ▼
┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│L'ARTIGIANO │ │IL SELEZION.  │ │ MODELOS  │ │  ENSEMBLE    │
│ (features) │ │(feat.select.)│ │(HPO+back)│ │  (Top-K      │
│lags, stats │ │importancia,  │ │ARIMA,ETS,│ │   blending)  │
│Fourier,cal.│ │mutual info   │ │LGBM,NBEATS│ └──────────────┘
└────────────┘ └──────────────┘ └──────────┘
```

Todos los workers reportan a **Il Libro** con MAE por fold. El Patrón cierra el loop.

---

## Reglas de evaluación (no negociables)

| Regla | Detalle |
|-------|---------|
| Validación | **Rolling-origin** — nunca K-fold aleatorio |
| Baseline | Siempre comparar contra **Seasonal Naive** |
| Reporte | **MAE por fold** + MAE promedio |
| Registro | Toda config y resultado en `il_libro.json` |

---

## Instalación

```bash
# Clonar / crear carpeta
git clone <repo> mafia-swarm && cd mafia-swarm

# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Orquestación con Ruflo
npx ruflo init
```

---

## Uso rápido

```bash
# 1. Coloca tu serie temporal en data/serie.csv
#    Columnas requeridas: ds (fecha), y (valor)

# 2. Correr baselines
python pipeline/backtest.py --model naive --model seasonal_naive

# 3. Correr feature engineering
python pipeline/features.py

# 4. Correr un modelo
python pipeline/backtest.py --model lightgbm

# 5. Ver leaderboard
cat il_libro.json
```

---

## Formato de datos

```csv
ds,y
2023-01-01,142.5
2023-01-02,138.0
2023-01-03,155.2
...
```

- `ds`: fecha en formato ISO 8601
- `y`: valor numérico de la serie

---

## Il Libro (leaderboard)

`il_libro.json` registra cada experimento:

```json
{
  "experiments": [
    {
      "id": "exp_001",
      "agent": "worker_modelos",
      "model": "lightgbm",
      "config": { "n_estimators": 300, "learning_rate": 0.05 },
      "mae_per_fold": [12.3, 11.8, 13.1, 12.0, 11.5],
      "mae_mean": 12.14,
      "beats_baseline": true,
      "timestamp": "2026-06-06T10:00:00Z"
    }
  ],
  "best_mae": 12.14,
  "best_experiment_id": "exp_001"
}
```

---

## Influencias técnicas

- **[Ruflo](https://github.com/ruvnet/ruflo)** — capa de orquestación multi-agente para Claude Code (swarm Queen-led, memoria vectorial HNSW via AgentDB)
- **[autoresearch](https://github.com/karpathy/autoresearch)** — patrón del loop iterativo: editar → backtest → medir → conservar/descartar
