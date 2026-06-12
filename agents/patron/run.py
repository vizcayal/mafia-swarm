"""
El Patrón — AI Orchestrator
Uso: python agents/patron/run.py [opciones]

Strategic decision loop:
  Each batch → Il Contabile (Claude) analyzes Il Libro → El Patrón (Claude)
  decides which proposals to execute → Python dispatches workers in parallel
  → results written to Il Libro → repeat.

Intelligence lives in Claude. Execution lives in Python.
"""

import argparse
import sys
import os
import json
import re
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Reconfigure stdout/stderr to UTF-8 to prevent encoding crashes on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from pipeline.evaluate import IlLibro
from agents.ai_client import call_claude, parse_json_response, MODEL
from agents.status import update_status

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          EL PATRÓN — AI Orchestrator (Claude)            ║
║   "Niente si muove senza il mio permesso."               ║
╚══════════════════════════════════════════════════════════╝
"""

WORKER_SCRIPTS = {
    'artigiano':     'agents/workers/artigiano/run.py',
    'selezionatore': 'agents/workers/selezionatore/run.py',
    'modelos':       'agents/workers/modelos/run.py',
    'ensemble':      'agents/workers/ensemble/run.py',
    'contabile':     'agents/contabile/run.py',
}

HISTORIA_PATH = os.path.join(ROOT, 'agents', 'patron', 'historia.json')

PATRON_SYSTEM = """\
You are El Patrón, the strategic boss of La Mafia — an autonomous ML \
forecasting swarm. Il Contabile has analyzed the experiment history and \
provided ranked proposals. Your job is to decide which proposals to execute \
in the next batch.

## Your authority
- You can execute 1 to N proposals in parallel (N = paralelo).
- You can STOP the swarm early if you judge further experiments are unlikely \
  to improve MAE meaningfully.
- You can skip a proposal if you believe it's redundant or low-value.
- You must respect the remaining budget.

## Decision rules
- Prefer proposals with high score AND diverse approaches in the same batch \
  (don't run two nearly-identical models simultaneously).
- If the last K batches showed no MAE improvement, consider stopping.
- If the best MAE is already very close to zero or you've exhausted all \
  interesting proposals, stop.
- Always justify your decisions briefly.

## Output format — return ONLY valid JSON, no prose:
{
  "decision": "continue" | "stop",
  "reasoning": "1-2 sentence justification",
  "selected_proposals": ["id1", "id2"],
  "stop_reason": "only if decision=stop"
}
"""


def update_program_md(program_path: str, libro: dict):
    """Replace the <!-- AUTO-UPDATED --> block in program.md with current best results."""
    if not os.path.exists(program_path):
        return
    best_mae   = libro.data.get('best_mae')
    best_id    = libro.data.get('best_experiment_id')
    n_exp      = len(libro.data.get('experiments', []))
    best_exp   = next((e for e in libro.data['experiments'] if e['id'] == best_id), None)
    model_desc = ''
    if best_exp:
        cfg = best_exp.get('config', {})
        parts = [best_exp.get('model', '')]
        for k, v in cfg.items():
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
        model_desc = ' | '.join(parts)
    features = (best_exp or {}).get('features_pipeline', 'data/serie_selected.csv')

    new_block = (
        f"<!-- AUTO-UPDATED by El Patrón — do not edit this block manually -->\n"
        f"- **Mejor MAE**: {best_mae:.6f} ({best_id})\n"
        f"- **Modelo**: {model_desc}\n"
        f"- **Features**: {features}\n"
        f"- **Total experimentos acumulados**: {n_exp}\n"
        f"<!-- END AUTO-UPDATE -->"
    )
    with open(program_path, encoding='utf-8') as f:
        content = f.read()
    updated = re.sub(
        r'<!-- AUTO-UPDATED.*?<!-- END AUTO-UPDATE -->',
        new_block,
        content,
        flags=re.DOTALL,
    )
    if updated != content:
        with open(program_path, 'w', encoding='utf-8') as f:
            f.write(updated)


def parse_args():
    p = argparse.ArgumentParser(description='El Patrón: AI orchestration loop')
    p.add_argument('--budget',     type=int,   default=8,           help='Max experiments to run')
    p.add_argument('--paralelo',   type=int,   default=3,           help='Workers per batch')
    p.add_argument('--horizon',    type=int,   default=12)
    p.add_argument('--folds',      type=int,   default=5)
    p.add_argument('--trials',     type=int,   default=30,          help='HPO trials per model run')
    p.add_argument('--libro',      default='il_libro.json')
    p.add_argument('--paciencia',  type=int,   default=3,           help='Batches without improvement before stopping')
    p.add_argument('--min-mejora', type=float, default=0.01,        help='Min MAE improvement fraction to count')
    p.add_argument('--bootstrap',  action='store_true',             help='Run features + baseline before loop')
    p.add_argument('--program',    default='program.md',            help='Research program markdown file')
    p.add_argument('--verbose',    action='store_true')
    return p.parse_args()


def log(msg: str, nivel: str = 'info', prefix: str = ''):
    ts = datetime.now().strftime('%H:%M:%S')
    iconos = {'info': '·', 'ok': '✅', 'warn': '⚠️ ', 'error': '❌', 'decision': '🎯', 'dispatch': '🚀', 'ai': '🤖'}
    print(f"[{ts}] {prefix}{iconos.get(nivel,'·')} {msg}", flush=True)


def load_historia() -> list:
    if os.path.exists(HISTORIA_PATH):
        with open(HISTORIA_PATH) as f:
            return json.load(f)
    return []


def save_historia(historia: list):
    os.makedirs(os.path.dirname(HISTORIA_PATH), exist_ok=True)
    with open(HISTORIA_PATH, 'w') as f:
        json.dump(historia, f, indent=2)


def run_worker(script: str, args_list: list, timeout: int = 600) -> dict:
    cmd = [sys.executable, os.path.join(ROOT, script)] + args_list
    start = time.time()
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', env=env, timeout=timeout, cwd=ROOT)
        return {
            'status':  'ok' if result.returncode == 0 else 'error',
            'elapsed': time.time() - start,
            'stdout':  result.stdout or '',
            'stderr':  result.stderr or '',
        }
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'elapsed': timeout, 'stdout': '', 'stderr': 'timeout'}
    except Exception as e:
        return {'status': 'exception', 'elapsed': 0, 'stdout': '', 'stderr': str(e)}


def llamar_contabile(libro_path: str, budget_left: int, args) -> dict:
    """Run Il Contabile (AI) and return full parsed output."""
    update_status(phase="analyzing", agents_update={
        "contabile": {"status": "running", "message": f"Analyzing {libro_path} (Budget left: {budget_left})..."}
    })
    result = run_worker(
        WORKER_SCRIPTS['contabile'],
        [
            '--libro',        libro_path,
            '--output',       'agents/contabile/propuestas.json',
            '--top-k',        str(args.paralelo * 3),
            '--budget-left',  str(budget_left),
            '--historia',     'agents/patron/historia.json',
            '--program',      args.program,
        ] + (['--verbose'] if args.verbose else []),
        timeout=300,
    )
    if result['status'] != 'ok':
        log(f"Il Contabile failed: {result['stderr'][:200]}", 'error')
        update_status(agents_update={
            "contabile": {"status": "idle", "message": "Error running Il Contabile"}
        })
        return {}
    path = os.path.join(ROOT, 'agents', 'contabile', 'propuestas.json')
    with open(path) as f:
        update_status(agents_update={
            "contabile": {"status": "idle", "message": "Analysis finished. Proposals ready."}
        })
        return json.load(f)


def patron_decide(propuestas: list, libro: dict, historia: list,
                  budget_left: int, batches_sin_mejora: int, args) -> dict:
    """Ask Claude (El Patrón) which proposals to execute."""
    update_status(phase="deciding", agents_update={
        "patron": {"status": "thinking", "message": "Evaluating proposals and deciding next batch..."}
    })
    state = {
        'best_mae':            libro.data['best_mae'],
        'best_experiment_id':  libro.data['best_experiment_id'],
        'n_experiments':       len(libro.data['experiments']),
        'budget_remaining':    budget_left,
        'paralelo':            args.paralelo,
        'batches_sin_mejora':  batches_sin_mejora,
        'paciencia':           args.paciencia,
    }
    user_msg = f"""## Current swarm state
{json.dumps(state, indent=2)}

## Il Contabile's proposals (ranked by score)
{json.dumps(propuestas, indent=2)}

## Your recent decisions
{json.dumps(historia[-4:], indent=2)}

Choose up to {args.paralelo} proposals to run next, or stop the swarm.
"""
    log(f"Asking {MODEL} for dispatch decision...", 'ai')
    raw = call_claude(PATRON_SYSTEM, user_msg, max_tokens=2048)
    if args.verbose:
        print(f"  Claude: {raw[:400]}")
    
    update_status(agents_update={
        "patron": {"status": "running_loop", "message": "Decision made."}
    })
    return parse_json_response(raw)


def construir_args_worker(propuesta: dict, args) -> tuple:
    worker = propuesta['worker']
    wa     = propuesta.get('args_worker', {})
    script = WORKER_SCRIPTS.get(worker)

    if worker == 'modelos':
        args_list = [
            '--model',   wa.get('model', wa.get('model_type', 'lightgbm')),
            '--data',    wa.get('data', wa.get('features', 'data/serie_features.csv')),
            '--horizon', str(wa.get('horizon', args.horizon)),
            '--folds',   str(wa.get('folds', args.folds)),
            '--trials',  str(wa.get('trials', args.trials)),
            '--libro',   args.libro,
        ]
    elif worker == 'ensemble':
        args_list = [
            '--top-k',   str(wa.get('top_k', 3)),
            '--method',  wa.get('method', 'weighted'),
            '--horizon', str(args.horizon),
            '--folds',   str(args.folds),
            '--libro',   args.libro,
            '--data',    wa.get('data', wa.get('features', 'data/serie_features.csv')),
        ]
    elif worker == 'artigiano':
        lags = wa.get('lags', '1,2,3,6,12,24')
        if isinstance(lags, list):
            lags = ",".join(str(x) for x in lags)
        windows = wa.get('windows', wa.get('rolling_windows', '3,6,12'))
        if isinstance(windows, list):
            windows = ",".join(str(x) for x in windows)
        diffs = wa.get('diffs', wa.get('diff', '1,12'))
        if isinstance(diffs, list):
            diffs = ",".join(str(x) for x in diffs)
        fourier = wa.get('fourier', '12:4')
        if isinstance(fourier, dict):
            fourier = f"{fourier.get('period', 12)}:{fourier.get('K', 4)}"
        elif isinstance(fourier, list):
            parts = []
            for item in fourier:
                if isinstance(item, dict):
                    parts.append(f"{item.get('period', 12)}:{item.get('K', 4)}")
                else:
                    parts.append(str(item))
            fourier = ";".join(parts)

        args_list = [
            '--lags',    str(lags),
            '--windows', str(windows),
            '--diffs',   str(diffs),
            '--fourier', str(fourier),
        ]
        if wa.get('calendar') is False or wa.get('no_calendar') is True:
            args_list.append('--no-calendar')
        
        output_file = wa.get('output')
        if output_file:
            args_list += ['--output', str(output_file)]
        
        data_file = wa.get('data') or wa.get('features')
        if data_file:
            args_list += ['--data', str(data_file)]

    elif worker == 'selezionatore':
        args_list = [
            '--method', wa.get('method', 'all'),
            '--top-k',  str(wa.get('top_k', 20)),
        ]
        corr_threshold = wa.get('corr_threshold') or wa.get('corr-threshold')
        if corr_threshold:
            args_list += ['--corr-threshold', str(corr_threshold)]
        
        data_file = wa.get('data') or wa.get('features')
        if data_file:
            args_list += ['--data', str(data_file)]
            
        output_file = wa.get('output')
        if output_file:
            args_list += ['--output', str(output_file)]
    else:
        args_list = []

    return script, args_list


def ejecutar_batch(batch: list, args, batch_num: int = 1) -> list:
    import threading
    lock = threading.Lock()
    active_runs = {}

    def _run(propuesta):
        worker = propuesta['worker']
        desc = propuesta['descripcion']
        prop_id = propuesta['id']

        with lock:
            active_runs[prop_id] = (worker, desc)
            worker_msgs = {}
            for w, d in active_runs.values():
                worker_msgs.setdefault(w, []).append(d)
            
            agents_up = {}
            for w, descs in worker_msgs.items():
                agents_up[w] = {"status": "running", "message": " & ".join(descs)}
            
            update_status(phase="dispatching", agents_update={
                "patron": {"status": "running_loop", "message": f"Orchestrating batch {batch_num}..."},
                **agents_up
            })

        script, worker_args = construir_args_worker(propuesta, args)
        prefix = f"  [{propuesta['worker']}] "
        log(propuesta['descripcion'], 'dispatch', prefix=prefix)
        r = run_worker(script, worker_args, timeout=600)
        
        with lock:
            active_runs.pop(prop_id, None)
            worker_msgs = {}
            for w, d in active_runs.values():
                worker_msgs.setdefault(w, []).append(d)
            
            agents_up = {}
            if worker in worker_msgs:
                agents_up[worker] = {"status": "running", "message": " & ".join(worker_msgs[worker])}
            else:
                agents_up[worker] = {"status": "idle", "message": f"Finished: {desc}"}
            
            update_status(agents_update=agents_up)

        if r['status'] == 'ok':
            log(f"done in {r['elapsed']:.1f}s", 'ok', prefix=prefix)
            for line in r['stdout'].split('\n'):
                if any(kw in line for kw in ['MAE', '✅', '❌', 'SUPERA']):
                    print(f"    → {line.strip()}", flush=True)
        else:
            log(f"failed: {r['stderr'][:100]}", 'error', prefix=prefix)
        return propuesta, r

    with ThreadPoolExecutor(max_workers=max(1, len(batch))) as ex:
        futures = {ex.submit(_run, p): p for p in batch}
        results = [future.result() for future in as_completed(futures)]
    return results


def bootstrap(args):
    log("━━━ BOOTSTRAP: Features + Baseline ━━━")
    update_status(phase="bootstrap", agents_update={
        "patron": {"status": "running_loop", "message": "Running bootstrap phase..."},
        "libro": {"status": "idle", "message": "Ready"}
    })
    for label, script, extra in [
        ("L'Artigiano", WORKER_SCRIPTS['artigiano'],
         ['--lags', '1,2,3,6,12,24', '--windows', '3,6,12', '--diffs', '1,12', '--fourier', '12:4']),
        ("Il Selezionatore", WORKER_SCRIPTS['selezionatore'],
         ['--method', 'all', '--top-k', '20']),
        ("Seasonal Naive baseline", WORKER_SCRIPTS['modelos'],
         ['--model', 'seasonal_naive', '--horizon', str(args.horizon),
          '--folds', str(args.folds), '--libro', args.libro]),
    ]:
        log(f"{label}...", 'dispatch')
        if 'artigiano' in script:
            update_status(agents_update={"artigiano": {"status": "running", "message": "Generating initial features (lags, rolling windows, Fourier)..."}})
        elif 'selezionatore' in script:
            update_status(agents_update={
                "artigiano": {"status": "idle", "message": "Completed features"},
                "selezionatore": {"status": "running", "message": "Selecting top 20 features..."}
            })
        elif 'modelos' in script:
            update_status(agents_update={
                "selezionatore": {"status": "idle", "message": "Completed selection"},
                "modelos": {"status": "running", "message": "Training Seasonal Naive baseline..."}
            })

        r = run_worker(script, extra, timeout=120)
        
        if 'artigiano' in script:
            update_status(agents_update={"artigiano": {"status": "idle", "message": "Feature engineering complete"}})
        elif 'selezionatore' in script:
            update_status(agents_update={"selezionatore": {"status": "idle", "message": "Feature selection complete"}})
        elif 'modelos' in script:
            update_status(agents_update={
                "modelos": {"status": "idle", "message": "Baseline trained successfully"},
                "libro": {"status": "updating", "message": "Registering Seasonal Naive baseline..."}
            })
            
        log("done" if r['status'] == 'ok' else f"failed: {r['stderr'][:80]}",
            'ok' if r['status'] == 'ok' else 'error')
            
    update_status(agents_update={"libro": {"status": "idle", "message": "Baseline registered in Il Libro"}})
    print()


def main():
    print(BANNER)
    args = parse_args()
    log(f"Model: {MODEL}")
    log(f"Budget: {args.budget} | Paralelo: {args.paralelo} | Paciencia: {args.paciencia}")
    print()

    libro = IlLibro(os.path.join(ROOT, args.libro))

    if args.bootstrap or len(libro.data['experiments']) == 0:
        bootstrap(args)
        libro = IlLibro(os.path.join(ROOT, args.libro))

    mae_inicial        = libro.data['best_mae'] or float('inf')
    mae_actual         = mae_inicial
    experimentos_run   = 0
    batches_sin_mejora = 0
    batch_num          = 0
    historia           = load_historia()
    propuestas_usadas  = set()

    log(f"Starting MAE: {mae_actual}")
    print()

    while experimentos_run < args.budget:
        budget_left = args.budget - experimentos_run
        batch_num  += 1

        print(f"\n{'═'*62}")
        log(f"BATCH {batch_num} — MAE: {mae_actual:.4f} — budget left: {budget_left}")

        update_status(phase="loop", agents_update={
            "patron": {"status": "running_loop", "message": f"Starting batch {batch_num}. Budget left: {budget_left}."}
        })

        # 1. Il Contabile analyzes and proposes
        log("Consulting Il Contabile (AI)...", 'ai')
        contabile_out = llamar_contabile(args.libro, budget_left, args)
        all_propuestas = contabile_out.get('propuestas', [])
        reasoning      = contabile_out.get('reasoning', '')

        if not all_propuestas:
            log("No proposals from Il Contabile. Stopping.", 'warn')
            break

        log(f"Il Contabile reasoning: {reasoning}")

        # Filter already-used proposals
        fresh = [p for p in all_propuestas if p['id'] not in propuestas_usadas]
        if not fresh:
            log("All proposals already executed. Stopping.", 'warn')
            break

        # 2. El Patrón decides which to run
        libro = IlLibro(os.path.join(ROOT, args.libro))
        try:
            decision = patron_decide(fresh, libro, historia, budget_left, batches_sin_mejora, args)
        except Exception as e:
            log(f"El Patrón decision failed: {e} — picking top-{args.paralelo} by score", 'warn')
            decision = {
                'decision': 'continue',
                'reasoning': 'Fallback: picked top proposals by score.',
                'selected_proposals': [p['id'] for p in fresh[:args.paralelo]],
            }

        if decision.get('decision') == 'stop':
            log(f"El Patrón decided to STOP: {decision.get('stop_reason', decision.get('reasoning',''))}", 'warn')
            break

        selected_ids  = decision.get('selected_proposals', [p['id'] for p in fresh[:args.paralelo]])
        batch         = [p for p in fresh if p['id'] in selected_ids][:args.paralelo]

        # Drop malformed proposals (e.g. truncated LLM output missing required fields)
        valid_batch = []
        for p in batch:
            if not p.get('worker'):
                log(f"Skipping malformed proposal '{p.get('id','?')}' (missing 'worker' field)", 'warn')
                propuestas_usadas.add(p['id'])
                continue
            valid_batch.append(p)
        batch = valid_batch

        if not batch:
            log("No valid proposals selected. Stopping.", 'warn')
            break

        for p in batch:
            propuestas_usadas.add(p['id'])

        log(f"El Patrón decision: {decision.get('reasoning', '')}", 'decision')
        for p in batch:
            log(f"  → {p['descripcion']} [score={p.get('score',0):.3f}]")

        # 3. Dispatch batch in parallel
        selected_desc = ", ".join(p['id'] for p in batch)
        update_status(phase="dispatching", agents_update={
            "patron": {"status": "running_loop", "message": f"Dispatching batch {batch_num}: {selected_desc}"}
        })
        results = ejecutar_batch(batch, args, batch_num)
        experimentos_run += len(results)

        # 4. Check improvement
        update_status(phase="evaluating", agents_update={
            "patron": {"status": "running_loop", "message": "Evaluating results..."},
            "libro": {"status": "updating", "message": "Updating leaderboards in Il Libro..."}
        })
        libro    = IlLibro(os.path.join(ROOT, args.libro))
        nuevo_mae = libro.data['best_mae'] or mae_actual

        improved = nuevo_mae < mae_actual * (1 - args.min_mejora)
        if improved:
            pct = (mae_actual - nuevo_mae) / mae_actual * 100
            log(f"IMPROVEMENT: {mae_actual:.4f} → {nuevo_mae:.4f} ({pct:.1f}%)", 'ok')
            mae_actual = nuevo_mae
            batches_sin_mejora = 0
            update_program_md(os.path.join(ROOT, args.program), libro)
            msg = f"Batch {batch_num} done. MAE improved to {nuevo_mae:.4f}!"
        else:
            batches_sin_mejora += 1
            log(f"No significant improvement ({batches_sin_mejora}/{args.paciencia})", 'warn')
            msg = f"Batch {batch_num} done. No improvement (patience: {batches_sin_mejora}/{args.paciencia})."

        update_status(phase="loop", agents_update={
            "patron": {"status": "running_loop", "message": msg},
            "libro": {"status": "idle", "message": f"Current Best MAE: {mae_actual:.4f}"}
        })

        # 5. Log decision to history
        historia.append({
            'batch':          batch_num,
            'timestamp':      datetime.now(timezone.utc).isoformat(),
            'mae_before':     mae_actual if not improved else nuevo_mae + (mae_actual - nuevo_mae),
            'mae_after':      nuevo_mae,
            'improved':       improved,
            'selected':       [p['id'] for p in batch],
            'patron_reasoning': decision.get('reasoning', ''),
            'contabile_reasoning': reasoning,
        })
        save_historia(historia)

        if batches_sin_mejora >= args.paciencia:
            log(f"Patience exhausted ({args.paciencia} batches). Stopping.", 'warn')
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    update_status(phase="completed", agents_update={
        "patron": {"status": "idle", "message": f"Swarm complete! Final MAE: {mae_actual:.4f}"},
        "contabile": {"status": "idle", "message": "Done"},
        "artigiano": {"status": "idle", "message": "Done"},
        "selezionatore": {"status": "idle", "message": "Done"},
        "modelos": {"status": "idle", "message": "Done"},
        "ensemble": {"status": "idle", "message": "Done"},
        "libro": {"status": "idle", "message": f"Final Best MAE: {mae_actual:.4f}"}
    })
    print(f"\n{'═'*62}")
    log("SWARM COMPLETE")
    log(f"  Experiments run: {experimentos_run}/{args.budget}")
    log(f"  Batches: {batch_num}")
    log(f"  MAE start: {mae_inicial:.4f}")
    log(f"  MAE final: {mae_actual:.4f}")
    if mae_inicial < float('inf') and mae_inicial > 0:
        log(f"  Total improvement: {(mae_inicial - mae_actual) / mae_inicial * 100:.1f}%")

    IlLibro(os.path.join(ROOT, args.libro)).print_leaderboard(top_n=6)


if __name__ == '__main__':
    main()
