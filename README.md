# 🤌 La Mafia — Agent Swarm for Forecasting

> *"In this family, if you don't improve the MAE, you disappear."*

A hierarchical agent swarm system with an Italian mafia theme for **univariate time series forecasting**. Single objective: **minimize MAE**.

---

## Architecture

```
                        ┌─────────────────┐
                        │   EL PATRÓN     │  ← coordinator / Queen
                        │  (coordinator)  │     launches & kills workers
                        └────────┬────────┘     based on budget
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐  ┌──────────────┐
     │  L'EMISSARIO   │ │  IL CONTABILE  │  │  IL LIBRO    │
     │  (researcher)  │ │  (analyzer)    │  │  (leaderboard│
     │  discovers new │ │  analyzes error│  │   + memory)  │
     │  techniques    │ │  & proposes    │  └──────────────┘
     └────────┬───────┘ └───────┬────────┘
              └────────┬────────┘
                       ▼  Proposal queue
                ┌──────────────┐
                │   EL PATRÓN  │ ranks: expected_MAE × confidence ÷ cost
                └──────┬───────┘
                       │ launches workers in parallel
       ┌───────────────┼───────────────┬──────────────┐
       ▼               ▼               ▼              ▼
┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│L'ARTIGIANO │ │IL SELEZION.  │ │ MODELOS  │ │  ENSEMBLE    │
│ (features) │ │(feat.select.)│ │(HPO+back)│ │  (Top-K      │
│lags, stats │ │importance,   │ │ARIMA,ETS,│ │   blending)  │
│Fourier,cal.│ │mutual info   │ │LGBM,NBEATS│ └──────────────┘
└────────────┘ └──────────────┘ └──────────┘
```

All workers report to **Il Libro** with per-fold MAE. El Patrón closes the loop.

---

## Evaluation Rules (non-negotiable)

| Rule | Details |
|------|---------|
| Validation | **Rolling-origin** — never random K-fold |
| Baseline | Always compare against **Seasonal Naive** |
| Reporting | **MAE per fold** + mean MAE |
| Logging | Every config and result saved in `il_libro.json` |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/vizcayal/mafia-swarm.git && cd mafia-swarm

# Install dependencies
pip install -r requirements.txt

# (Optional) Orchestration with Ruflo
npx ruflo init
```

---

## Quick Start

```bash
# 1. Place your time series in data/serie.csv
#    Required columns: ds (date), y (value)

# 2. Run baselines
python pipeline/backtest.py --model naive --model seasonal_naive

# 3. Run feature engineering
python pipeline/features.py

# 4. Train a model
python pipeline/backtest.py --model lightgbm

# 5. View leaderboard
cat il_libro.json
```

### Full Orchestration (recommended)

```bash
# Run the full autonomous swarm loop
python agents/patron/run.py --budget 50 --paralelo 8 --trials 30

# First time (bootstrap baselines + features automatically):
python agents/patron/run.py --budget 50 --paralelo 8 --bootstrap

# Launch the live dashboard (http://localhost:5050):
python dashboard/app.py
```

---

## Data Format

```csv
ds,y
2023-01-01,142.5
2023-01-02,138.0
2023-01-03,155.2
...
```

- `ds`: date in ISO 8601 format
- `y`: numeric value of the series

---

## Il Libro (leaderboard)

`il_libro.json` records every experiment:

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

## Technical Influences

- **[Ruflo](https://github.com/ruvnet/ruflo)** — multi-agent orchestration layer for Claude Code (Queen-led swarm, HNSW vector memory via AgentDB)
- **[autoresearch](https://github.com/karpathy/autoresearch)** — iterative improvement loop pattern: edit → backtest → measure → keep/discard
