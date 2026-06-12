"""
Il Contabile — AI Analyst Agent
Uso: python agents/contabile/run.py [opciones]

Reads Il Libro and calls Claude to generate ranked experiment proposals.
Claude reasons freely about what has and hasn't worked, then returns
structured JSON that El Patrón can act on.
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from agents.ai_client import call_claude, parse_json_response, MODEL

BANNER = """
╔══════════════════════════════════════════════════╗
║   IL CONTABILE — AI Analyst (Claude)             ║
╚══════════════════════════════════════════════════╝
"""

OUTPUT_FORMAT = """\

## Output format
Return ONLY a JSON object — no prose before or after:
{
  "reasoning": "2-3 sentence summary of what you observed in the data",
  "propuestas": [
    {
      "id": "unique_snake_case_id",
      "descripcion": "Short English description",
      "worker": "modelos|ensemble|artigiano|selezionatore",
      "accion": "run_model|run_ensemble|run_features|run_selection",
      "args_worker": { ... },
      "mae_delta_esperado": -5.0,
      "confianza": 0.75,
      "costo": "bajo|medio|alto",
      "score": 1.875,
      "evidencia": "Why this is the best next step"
    }
  ]
}

Score = abs(mae_delta_esperado) * confianza / cost_num  (bajo=1, medio=2, alto=4).
Return the top proposals sorted by score descending.
"""

SYSTEM_PROMPT_BASE = """\
You are Il Contabile, the strategic analyst of La Mafia — an autonomous ML \
forecasting agent swarm optimizing MAE on univariate time series.

Your research program is defined below in program.md. Follow its directions, \
prioritize the unexplored angles it identifies, and avoid repeating what has \
already failed.

"""


def load_system_prompt(program_path: str) -> str:
    """Build system prompt by injecting program.md content."""
    if os.path.exists(program_path):
        with open(program_path, encoding='utf-8') as f:
            program_content = f.read()
        return SYSTEM_PROMPT_BASE + program_content + OUTPUT_FORMAT
    # Fallback: minimal prompt without program.md
    return SYSTEM_PROMPT_BASE + OUTPUT_FORMAT


def build_user_prompt(libro: dict, budget_restante: int, historia: list) -> str:
    experiments = libro.get('experiments', [])
    best_mae    = libro.get('best_mae')
    best_id     = libro.get('best_experiment_id')

    baseline_mae = next(
        (e['mae_mean'] for e in experiments if e.get('model') == 'seasonal_naive'),
        None
    )
    models_tried = list({e.get('model') for e in experiments})
    beats = [e for e in experiments if e.get('beats_baseline')]

    sorted_exp = sorted(experiments, key=lambda x: x.get('mae_mean', 9999))
    # Send only top 15 + last 3 recent experiments to keep the prompt manageable
    recent = [e for e in experiments[-3:] if e not in sorted_exp[:15]]
    top_exp = sorted_exp[:15] + recent

    exp_summary = []
    for e in top_exp:
        exp_summary.append({
            'id':              e['id'],
            'model':           e.get('model'),
            'mae_mean':        round(e.get('mae_mean', 0), 4),
            'mae_per_fold':    [round(x, 2) for x in e.get('mae_per_fold', [])],
            'beats_baseline':  e.get('beats_baseline'),
            'config':          e.get('config', {}),
            'features':        e.get('features_pipeline'),
        })

    prompt = f"""## Il Libro — Current State

Best MAE: {best_mae} (experiment: {best_id})
Baseline MAE (Seasonal Naive): {baseline_mae}
Total experiments: {len(experiments)} (showing top 15 + 3 most recent)
Models tried: {models_tried}
Experiments beating baseline: {len(beats)}
Budget remaining: {budget_restante} experiments

## Experiment history (sorted by MAE, best first)

{json.dumps(exp_summary, indent=2)}
"""

    if historia:
        prompt += f"""
## El Patrón's decision history (last 3 decisions)

{json.dumps(historia[-3:], indent=2)}
"""

    prompt += """
## Your task

Analyze this experiment history carefully. What patterns do you see?
What has not been tried yet? What hyperparameter directions look promising?
What features or ensembles might help?

Propose the most impactful next experiments. Be specific and evidence-based.
"""
    return prompt


def parse_args():
    p = argparse.ArgumentParser(description='Il Contabile AI Agent')
    p.add_argument('--libro',        default='il_libro.json')
    p.add_argument('--output',       default='agents/contabile/propuestas.json')
    p.add_argument('--top-k',        type=int, default=6)
    p.add_argument('--budget-left',  type=int, default=99, help='Budget remaining (passed by El Patrón)')
    p.add_argument('--historia',     default='agents/patron/historia.json', help='Path to patron decision history')
    p.add_argument('--program',      default='program.md', help='Research program markdown file')
    p.add_argument('--verbose',      action='store_true')
    return p.parse_args()


def generate_fallback_proposals(libro: dict) -> dict:
    """
    Expert rule-based fallback proposal generator when the AI client is unavailable.
    Analyzes il_libro.json and suggests next logical experiments.
    """
    experiments = libro.get('experiments', [])
    models_run = [e.get('model') for e in experiments]
    
    propuestas = []
    
    # 1. Proposal: Ensemble with the new best models (since we have xgboost now)
    if 'xgboost' in models_run and 'lightgbm' in models_run:
        propuestas.append({
            "id": "ensemble_top_models_with_xgb",
            "descripcion": "Weighted Ensemble of best XGBoost and LightGBM models",
            "worker": "ensemble",
            "accion": "run_ensemble",
            "args_worker": {
                "top_k": 3,
                "method": "weighted"
            },
            "mae_delta_esperado": -1.2,
            "confianza": 0.85,
            "costo": "medio",
            "evidencia": "XGBoost and LightGBM are both beating the baseline. Ensembling them should yield further MAE reductions."
        })

    # 2. Proposal: HPO on LightGBM with 100 trials (to match XGBoost HPO)
    propuestas.append({
        "id": "lightgbm_hpo_100_trials",
        "descripcion": "Aggressive HPO (100 trials) for LightGBM",
        "worker": "modelos",
        "accion": "run_model",
        "args_worker": {
            "model": "lightgbm",
            "trials": 100
        },
        "mae_delta_esperado": -0.8,
        "confianza": 0.75,
        "costo": "medio",
        "evidencia": "LightGBM was previously run with fewer trials. 100 trials of HPO could close the gap to XGBoost."
    })

    # 3. Proposal: Feature selection reduction
    propuestas.append({
        "id": "feature_selection_top_15",
        "descripcion": "Reduce selected features to top 15 using Mutual Info",
        "worker": "selezionatore",
        "accion": "run_selection",
        "args_worker": {
            "method": "mutual_info",
            "top_k": 15
        },
        "mae_delta_esperado": -0.5,
        "confianza": 0.65,
        "costo": "bajo",
        "evidencia": "Reducing the feature count from 20 to 15 can reduce overfitting, especially in tree-based models."
    })

    # 4. Proposal: Artigiano feature expansion
    propuestas.append({
        "id": "artigiano_expand_rolling_windows",
        "descripcion": "Expand rolling windows to include larger horizons (e.g. 24, 48)",
        "worker": "artigiano",
        "accion": "run_features",
        "args_worker": {
            "lags": "1,2,3,6,12,24",
            "windows": "3,6,12,24,48",
            "diffs": "1,12",
            "fourier": "12:6"
        },
        "mae_delta_esperado": -0.6,
        "confianza": 0.70,
        "costo": "medio",
        "evidencia": "Adding longer rolling averages will capture longer-term trends in the series."
    })

    # Calculate scores: abs(mae_delta) * confianza / cost_num
    cost_map = {"bajo": 1, "medio": 2, "alto": 4}
    for p in propuestas:
        cost_num = cost_map.get(p["costo"], 2)
        p["score"] = float(abs(p["mae_delta_esperado"]) * p["confianza"] / cost_num)

    # Sort by score descending
    propuestas = sorted(propuestas, key=lambda x: x["score"], reverse=True)

    return {
        "reasoning": "Fallback Expert analysis: XGBoost is leading. Suggesting ensemble of top models, LightGBM HPO, and feature space tuning.",
        "propuestas": propuestas
    }


def main():
    print(BANNER)
    args = parse_args()

    libro_path = os.path.join(ROOT, args.libro)
    with open(libro_path) as f:
        libro = json.load(f)

    historia = []
    hist_path = os.path.join(ROOT, args.historia)
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            historia = json.load(f)

    n_exp = len(libro.get('experiments', []))
    print(f"📖 Il Libro: {n_exp} experiments | Best MAE: {libro.get('best_mae')}")
    print(f"🤖 Calling {MODEL}...")
    print()

    user_prompt = build_user_prompt(libro, args.budget_left, historia)

    if args.verbose:
        print("── User prompt ──────────────────────")
        print(user_prompt[:800])
        print("─────────────────────────────────────\n")

    program_path = os.path.join(ROOT, args.program)
    system_prompt = load_system_prompt(program_path)
    if args.verbose:
        using = args.program if os.path.exists(program_path) else "built-in prompt (program.md not found)"
        print(f"📋 Using research program: {using}\n")

    try:
        raw = call_claude(system_prompt, user_prompt, max_tokens=4096)
        if args.verbose:
            print("── Claude response ──────────────────")
            print(raw[:1200])
            print("─────────────────────────────────────\n")
        parsed = parse_json_response(raw)
    except Exception as e:
        print(f"⚠️  AI client unavailable ({e}). Running Expert Fallback Analyst...")
        parsed = generate_fallback_proposals(libro)

    reasoning  = parsed.get('reasoning', '')
    propuestas = parsed.get('propuestas', parsed if isinstance(parsed, list) else [])
    propuestas = propuestas[:args.top_k]

    for p in propuestas:
        p['generado_en'] = datetime.now(timezone.utc).isoformat()

    print(f"💡 Reasoning: {reasoning}\n")
    print(f"📋 {len(propuestas)} proposals:\n")
    for i, p in enumerate(propuestas):
        print(f"  {i+1}. [{p.get('score', 0):.3f}] {p.get('descripcion', '')}")
        print(f"      worker={p.get('worker')} | cost={p.get('costo')} | conf={p.get('confianza', 0)*100:.0f}%")
        print(f"      {p.get('evidencia', '')}\n")

    out_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'reasoning':   reasoning,
            'propuestas':  propuestas,
            'model':       MODEL,
            'generado_en': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"💾 Saved to {args.output}")
    return propuestas


if __name__ == '__main__':
    main()
