"""
La Mafia Dashboard — Flask server
Uso: python dashboard/app.py
Abre http://localhost:5050 en tu navegador.
Se actualiza automáticamente cada 3 segundos.
"""

import sys, os, json

# Reconfigure stdout/stderr to UTF-8 to prevent encoding crashes on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from datetime import datetime
from flask import Flask, jsonify, render_template_string

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

LIBRO_PATH      = os.path.join(ROOT, 'il_libro.json')
PROPUESTAS_PATH = os.path.join(ROOT, 'agents', 'contabile', 'propuestas.json')
STATUS_PATH     = os.path.join(ROOT, 'logs', 'status.json')

app = Flask(__name__)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


@app.route('/api/data')
def api_data():
    libro      = read_json(LIBRO_PATH) or {}
    propuestas = read_json(PROPUESTAS_PATH) or {}
    status     = read_json(STATUS_PATH) or {
        "phase": "idle",
        "agents": {
            "patron": {"status": "idle", "message": "Not running"},
            "contabile": {"status": "idle", "message": "Not running"},
            "artigiano": {"status": "idle", "message": "Not running"},
            "selezionatore": {"status": "idle", "message": "Not running"},
            "modelos": {"status": "idle", "message": "Not running"},
            "ensemble": {"status": "idle", "message": "Not running"},
            "libro": {"status": "idle", "message": "Ready"}
        }
    }

    experiments = libro.get('experiments', [])
    experiments_sorted = sorted(experiments, key=lambda e: e.get('mae_mean', 9999))

    # MAE over time (insertion order = discovery order)
    mae_timeline = [
        {
            'id':        e['id'],
            'model':     e.get('model', '?'),
            'mae':       round(e.get('mae_mean', 0), 4),
            'beats':     e.get('beats_baseline', False),
            'timestamp': e.get('timestamp', ''),
        }
        for e in experiments
    ]

    baseline_mae = next(
        (e['mae_mean'] for e in experiments if e.get('model') == 'seasonal_naive'),
        None
    )

    return jsonify({
        'best_mae':       libro.get('best_mae'),
        'best_id':        libro.get('best_experiment_id'),
        'baseline_mae':   baseline_mae,
        'n_experiments':  len(experiments),
        'leaderboard':    experiments_sorted[:10],
        'mae_timeline':   mae_timeline,
        'propuestas':     propuestas.get('propuestas', [])[:6],
        'updated_at':     datetime.now().strftime('%H:%M:%S'),
        'status':         status,
    })


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>La Mafia — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:      #0d0d0d;
    --surface: #161616;
    --card:    #1e1e1e;
    --border:  #2a2a2a;
    --gold:    #c9a84c;
    --green:   #4caf77;
    --red:     #cf5050;
    --muted:   #666;
    --text:    #e8e8e8;
    --sub:     #999;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Menlo', 'Consolas', monospace; font-size: 13px; }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { font-size: 18px; color: var(--gold); letter-spacing: 2px; }
  header .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .pulse { display: inline-block; width: 8px; height: 8px; background: var(--green);
           border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 20px 24px 12px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; }
  .stat-card .label { color: var(--sub); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .value { font-size: 28px; font-weight: bold; margin-top: 4px; color: var(--gold); }
  .stat-card .hint  { color: var(--muted); font-size: 10px; margin-top: 2px; }

  .panels { display: grid; grid-template-columns: 1.1fr 1fr; gap: 12px; padding: 0 24px 12px; }
  .panel  { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--gold); margin-bottom: 12px; }

  .leaderboard { width: 100%; border-collapse: collapse; }
  .leaderboard th { color: var(--muted); font-size: 10px; text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .leaderboard td { padding: 7px 8px; border-bottom: 1px solid #1a1a1a; font-size: 12px; }
  .leaderboard tr:last-child td { border-bottom: none; }
  .leaderboard tr.best td { color: var(--gold); }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
  .badge.ok  { background: #1a3a28; color: var(--green); }
  .badge.no  { background: #3a1a1a; color: var(--red); }

  .chart-wrap { padding: 0 24px 16px; }
  .chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .chart-card h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--gold); margin-bottom: 12px; }
  canvas { max-height: 180px; }

  .propuestas { padding: 0 24px 24px; }
  .prop-grid  { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .prop-card  { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .prop-card .score { color: var(--gold); font-size: 18px; font-weight: bold; }
  .prop-card .desc  { font-size: 11px; color: var(--text); margin: 6px 0 4px; line-height: 1.4; }
  .prop-card .meta  { font-size: 10px; color: var(--muted); }
  .prop-card .ev    { font-size: 10px; color: var(--sub); margin-top: 6px; font-style: italic; }
  .costo-bajo  { border-top: 2px solid #4caf77; }
  .costo-medio { border-top: 2px solid #c9a84c; }
  .costo-alto  { border-top: 2px solid #cf5050; }

  /* Diagram Panel Styles */
  .diagram-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 0 24px 12px;
  }
  .diagram-panel h2 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--gold);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .phase-badge {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 10px;
    color: var(--text);
    text-transform: uppercase;
    font-weight: bold;
    letter-spacing: 1px;
    transition: all 0.3s ease;
  }
  .phase-bootstrap { border-color: var(--gold); color: var(--gold); }
  .phase-analyzing { border-color: #3a8ee6; color: #3a8ee6; }
  .phase-deciding { border-color: #a88be6; color: #a88be6; }
  .phase-dispatching { border-color: var(--green); color: var(--green); }
  .phase-evaluating { border-color: var(--gold); color: var(--gold); }
  .phase-completed { border-color: #888; color: #888; }
  .phase-idle { border-color: var(--muted); color: var(--muted); }

  .diagram-content {
    display: flex;
    gap: 20px;
    align-items: center;
  }
  .diagram-svg-wrap {
    flex: 1.8;
    position: relative;
    border: 1px solid #1a1a1a;
    background: #090909;
    border-radius: 6px;
    padding: 10px;
  }
  .diagram-details {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 11px;
    height: 310px;
    display: flex;
    flex-direction: column;
  }
  .diagram-details h3 {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--sub);
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    flex-shrink: 0;
  }
  .agent-status-list {
    flex-grow: 1;
    overflow-y: auto;
    padding-right: 4px;
  }
  .agent-status-row {
    display: flex;
    align-items: flex-start;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a1a;
  }
  .agent-status-row:last-child {
    border-bottom: none;
  }
  .agent-status-row .name {
    font-weight: bold;
    min-width: 95px;
    color: var(--text);
    font-size: 11px;
  }
  .agent-status-row .status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 8px;
    margin-top: 5px;
    flex-shrink: 0;
    transition: all 0.3s ease;
  }
  .status-dot.idle { background: var(--muted); }
  .status-dot.thinking { background: #c9a84c; box-shadow: 0 0 8px #c9a84c; animation: pulse 2s infinite; }
  .status-dot.running { background: #4caf77; box-shadow: 0 0 8px #4caf77; animation: pulse 2s infinite; }
  .status-dot.updating { background: #3a8ee6; box-shadow: 0 0 8px #3a8ee6; animation: pulse 2s infinite; }
  .status-dot.success { background: #4caf77; }
  .status-dot.error { background: var(--red); box-shadow: 0 0 8px var(--red); }
  
  .agent-status-row .message {
    color: var(--sub);
    font-size: 11px;
    line-height: 1.3;
    word-break: break-word;
  }

  /* SVG styling */
  .node {
    cursor: pointer;
  }
  .node circle {
    fill: #141414;
    stroke: var(--border);
    stroke-width: 2px;
    transition: all 0.3s ease;
  }
  .node text {
    fill: var(--text);
    transition: all 0.3s ease;
    user-select: none;
  }
  .node.state-thinking circle {
    stroke: var(--gold);
    fill: #221d12;
    filter: drop-shadow(0px 0px 6px rgba(201,168,76,0.5));
  }
  .node.state-running circle {
    stroke: var(--green);
    fill: #112017;
    filter: drop-shadow(0px 0px 6px rgba(76,175,119,0.5));
  }
  .node.state-updating circle {
    stroke: #3a8ee6;
    fill: #111a24;
    filter: drop-shadow(0px 0px 6px rgba(58,142,230,0.5));
  }
  .node.state-error circle {
    stroke: var(--red);
    fill: #241111;
    filter: drop-shadow(0px 0px 6px rgba(207,80,80,0.5));
  }

  .link {
    stroke: #222;
    stroke-width: 1.5px;
    transition: all 0.3s ease;
  }
  .link-active {
    stroke: var(--gold);
    stroke-width: 2px;
    stroke-dasharray: 6, 6;
    animation: link-flow 1.5s linear infinite;
    filter: drop-shadow(0px 0px 3px rgba(201,168,76,0.3));
    marker-end: url(#arrow-active) !important;
  }
  @keyframes link-flow {
    to {
      stroke-dashoffset: -20;
    }
  }

  /* Responsive layout for diagram content */
  @media (max-width: 850px) {
    .diagram-content {
      flex-direction: column;
    }
    .diagram-details {
      width: 100%;
      height: 180px;
    }
  }

  footer { text-align: center; color: var(--muted); font-size: 10px; padding: 12px; border-top: 1px solid var(--border); }
  #updated { color: var(--sub); }
</style>
</head>
<body>

<header>
  <div>
    <h1>🤌 LA MAFIA — Swarm Dashboard</h1>
    <div class="sub">"Niente si muove senza il mio permesso."</div>
  </div>
  <div style="text-align:right">
    <span class="pulse"></span><span style="color:var(--green);font-size:11px">LIVE</span>
    <div style="color:var(--muted);font-size:10px;margin-top:4px">updated <span id="updated">—</span></div>
  </div>
</header>

<!-- KPI row -->
<div class="grid">
  <div class="stat-card">
    <div class="label">Best MAE</div>
    <div class="value" id="best-mae">—</div>
    <div class="hint" id="best-id"></div>
  </div>
  <div class="stat-card">
    <div class="label">Baseline MAE</div>
    <div class="value" id="baseline-mae" style="color:var(--red)">—</div>
    <div class="hint">Seasonal Naive</div>
  </div>
  <div class="stat-card">
    <div class="label">Improvement vs Baseline</div>
    <div class="value" id="mejora" style="color:var(--green)">—</div>
    <div class="hint">MAE reduction</div>
  </div>
  <div class="stat-card">
    <div class="label">Experiments</div>
    <div class="value" id="n-exp" style="color:var(--text)">—</div>
    <div class="hint">in Il Libro</div>
  </div>
</div>

<!-- Real-time Swarm Diagram Panel -->
<div class="diagram-panel">
  <h2>🕵️‍♂️ Swarm Real-Time Diagram <span id="swarm-phase-badge" class="phase-badge phase-idle">idle</span></h2>
  <div class="diagram-content">
    <div class="diagram-svg-wrap">
      <svg viewBox="0 0 800 320" style="width: 100%; height: auto; display: block;">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#222" />
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--gold)" />
          </marker>
        </defs>

        <!-- Links -->
        <path id="link-libro-contabile" class="link" d="M 400 270 Q 250 250 150 160" fill="none" marker-end="url(#arrow)" />
        <path id="link-contabile-patron" class="link" d="M 150 160 Q 250 70 400 50" fill="none" marker-end="url(#arrow)" />
        
        <path id="link-patron-artigiano" class="link" d="M 400 50 Q 520 40 650 60" fill="none" marker-end="url(#arrow)" />
        <path id="link-patron-selezionatore" class="link" d="M 400 50 Q 525 80 650 130" fill="none" marker-end="url(#arrow)" />
        <path id="link-patron-modelos" class="link" d="M 400 50 Q 515 130 650 200" fill="none" marker-end="url(#arrow)" />
        <path id="link-patron-ensemble" class="link" d="M 400 50 Q 510 170 650 270" fill="none" marker-end="url(#arrow)" />

        <path id="link-artigiano-libro" class="link" d="M 650 60 Q 540 180 400 270" fill="none" marker-end="url(#arrow)" />
        <path id="link-selezionatore-libro" class="link" d="M 650 130 Q 540 215 400 270" fill="none" marker-end="url(#arrow)" />
        <path id="link-modelos-libro" class="link" d="M 650 200 Q 540 245 400 270" fill="none" marker-end="url(#arrow)" />
        <path id="link-ensemble-libro" class="link" d="M 650 270 Q 525 285 400 270" fill="none" marker-end="url(#arrow)" />

        <!-- Nodes -->
        <g id="node-patron" class="node" transform="translate(400, 50)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="18">💼</text>
          <text text-anchor="middle" dy="36" font-size="10" font-weight="bold" fill="#fff">EL PATRÓN</text>
          <title>El Patrón (AI Orchestrator)</title>
        </g>

        <g id="node-contabile" class="node" transform="translate(150, 160)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="18">📖</text>
          <text text-anchor="middle" dy="36" font-size="10" font-weight="bold" fill="#fff">IL CONTABILE</text>
          <title>Il Contabile (AI Analyst)</title>
        </g>

        <g id="node-libro" class="node" transform="translate(400, 270)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="18">💾</text>
          <text text-anchor="middle" dy="36" font-size="10" font-weight="bold" fill="#fff">IL LIBRO</text>
          <title>Il Libro (Database)</title>
        </g>

        <g id="node-artigiano" class="node" transform="translate(650, 60)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="16">🛠️</text>
          <text text-anchor="start" dx="30" dy="4" font-size="10" font-weight="bold" fill="#ccc">L'ARTIGIANO</text>
          <title>L'Artigiano (Feature Engineer)</title>
        </g>

        <g id="node-selezionatore" class="node" transform="translate(650, 130)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="16">🔍</text>
          <text text-anchor="start" dx="30" dy="4" font-size="10" font-weight="bold" fill="#ccc">IL SELEZIONATORE</text>
          <title>Il Selezionatore (Feature Selector)</title>
        </g>

        <g id="node-modelos" class="node" transform="translate(650, 200)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="16">🤖</text>
          <text text-anchor="start" dx="30" dy="4" font-size="10" font-weight="bold" fill="#ccc">MODELOS</text>
          <title>Worker Modelos (Model Training & HPO)</title>
        </g>

        <g id="node-ensemble" class="node" transform="translate(650, 270)">
          <circle r="22" />
          <text text-anchor="middle" dy="6" font-size="16">🔗</text>
          <text text-anchor="start" dx="30" dy="4" font-size="10" font-weight="bold" fill="#ccc">ENSEMBLE</text>
          <title>Ensemble Worker (Blender/Stacker)</title>
        </g>
      </svg>
    </div>
    <div class="diagram-details">
      <h3>Agent Activities</h3>
      <div id="agent-details-list" class="agent-status-list"></div>
    </div>
  </div>
</div>

<!-- Leaderboard + chart panels -->
<div class="panels">
  <div class="panel">
    <h2>📖 Il Libro — Leaderboard</h2>
    <table class="leaderboard">
      <thead><tr>
        <th>#</th><th>Model</th><th>MAE</th><th>Beats?</th><th>Folds</th>
      </tr></thead>
      <tbody id="leaderboard-body"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>🎯 Next proposals from Il Contabile</h2>
    <div id="prop-list"></div>
  </div>
</div>

<!-- MAE timeline -->
<div class="chart-wrap">
  <div class="chart-card">
    <h2>📉 MAE per experiment (chronological order)</h2>
    <canvas id="mae-chart"></canvas>
  </div>
</div>

<footer>"El Patrón listens to Il Contabile's advice." — mafia-swarm v1.0</footer>

<script>
let chart = null;

function fmt(v) { return v != null ? v.toFixed(4) : '—'; }

function renderLeaderboard(rows, bestId) {
  const tbody = document.getElementById('leaderboard-body');
  tbody.innerHTML = rows.map((e, i) => {
    const isBest = e.id === bestId;
    const folds = (e.mae_per_fold || []).map(v => v.toFixed(1)).join(', ');
    return `<tr class="${isBest ? 'best' : ''}">
      <td>${i + 1}</td>
      <td>${e.model || e.id}</td>
      <td><b>${fmt(e.mae_mean)}</b></td>
      <td><span class="badge ${e.beats_baseline ? 'ok' : 'no'}">${e.beats_baseline ? '✅ yes' : '❌ no'}</span></td>
      <td style="color:var(--muted);font-size:10px">${folds || '—'}</td>
    </tr>`;
  }).join('');
}

function renderMini(propuestas) {
  const el = document.getElementById('prop-list');
  if (!propuestas.length) { el.innerHTML = '<div style="color:var(--muted);font-size:11px">No proposals yet — run Il Contabile.</div>'; return; }
  el.innerHTML = propuestas.map(p => `
    <div style="border-bottom:1px solid var(--border);padding:8px 0;display:flex;align-items:flex-start;gap:10px">
      <div style="min-width:42px;text-align:right">
        <span style="color:var(--gold);font-size:14px;font-weight:bold">${p.score.toFixed(2)}</span>
      </div>
      <div>
        <div style="font-size:11px;color:var(--text)">${p.descripcion}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">
          worker: <b style="color:var(--sub)">${p.worker}</b> &nbsp;|&nbsp;
          cost: <b style="color:var(--sub)">${p.costo}</b> &nbsp;|&nbsp;
          conf: ${(p.confianza * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  `).join('');
}

function renderChart(timeline, baselineMae) {
  const labels = timeline.map((e, i) => `#${i+1} ${e.model}`);
  const maes   = timeline.map(e => e.mae);
  const colors = timeline.map(e => e.beats ? '#4caf77' : '#cf5050');
  const baseline = timeline.length > 0 ? Array(timeline.length).fill(baselineMae) : [];

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = maes;
    chart.data.datasets[0].pointBackgroundColor = colors;
    chart.data.datasets[0].pointBorderColor = colors;
    if (baseline.length) chart.data.datasets[1].data = baseline;
    chart.update();
    return;
  }

  const ctx = document.getElementById('mae-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'MAE',
          data: maes,
          borderColor: '#c9a84c',
          backgroundColor: 'rgba(201,168,76,0.08)',
          pointBackgroundColor: colors,
          pointBorderColor: colors,
          pointRadius: 6,
          tension: 0.3,
          fill: true,
        },
        {
          label: 'Seasonal Naive',
          data: baseline,
          borderColor: '#cf5050',
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: '#555', font: { size: 9 }, maxRotation: 30 }, grid: { color: '#1e1e1e' } },
        y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#222' } }
      }
    }
  });
}

function updatePhaseBadge(phase) {
  const badge = document.getElementById('swarm-phase-badge');
  if (!badge) return;
  badge.textContent = phase;
  badge.className = 'phase-badge';
  badge.classList.add('phase-' + phase);
}

function renderAgentDetails(agents) {
  const el = document.getElementById('agent-details-list');
  if (!el) return;
  
  const order = ['patron', 'contabile', 'artigiano', 'selezionatore', 'modelos', 'ensemble', 'libro'];
  const nameMap = {
    'patron': 'El Patrón',
    'contabile': 'Il Contabile',
    'artigiano': 'L\'Artigiano',
    'selezionatore': 'Il Selezionatore',
    'modelos': 'Modelos',
    'ensemble': 'Ensemble',
    'libro': 'Il Libro'
  };

  el.innerHTML = order.map(k => {
    const a = agents[k] || { status: 'idle', message: k === 'libro' ? 'Ready' : 'Not running' };
    const statusClass = a.status || 'idle';
    const message = a.message || 'Idle';
    
    return `
      <div class="agent-status-row">
        <span class="status-dot ${statusClass}"></span>
        <span class="name">${nameMap[k]}</span>
        <span class="message">${message}</span>
      </div>
    `;
  }).join('');
}

function activateLink(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.add('link-active');
  }
}

function updateDiagram(statusObj) {
  const phase = statusObj.phase || 'idle';
  const agents = statusObj.agents || {};

  updatePhaseBadge(phase);
  renderAgentDetails(agents);

  // 1. Reset all nodes
  const nodes = ['patron', 'contabile', 'artigiano', 'selezionatore', 'modelos', 'ensemble', 'libro'];
  nodes.forEach(n => {
    const nodeEl = document.getElementById('node-' + n);
    if (!nodeEl) return;
    
    nodeEl.classList.remove('state-idle', 'state-thinking', 'state-running', 'state-updating', 'state-error');
    
    const agentState = agents[n] ? agents[n].status : 'idle';
    const msg = agents[n] ? agents[n].message : '';
    nodeEl.classList.add('state-' + agentState);
    
    const titleEl = nodeEl.querySelector('title');
    if (titleEl) titleEl.textContent = n.toUpperCase() + ' (' + agentState + ')\n' + msg;
  });

  // Reset all links
  const links = document.querySelectorAll('.link');
  links.forEach(l => {
    l.classList.remove('link-active');
  });

  // 2. Activate links based on phase and agent activity
  if (phase === 'analyzing') {
    activateLink('link-libro-contabile');
  } else if (phase === 'deciding') {
    activateLink('link-contabile-patron');
  } else if (phase === 'dispatching') {
    if (agents.artigiano && agents.artigiano.status === 'running') activateLink('link-patron-artigiano');
    if (agents.selezionatore && agents.selezionatore.status === 'running') activateLink('link-patron-selezionatore');
    if (agents.modelos && agents.modelos.status === 'running') activateLink('link-patron-modelos');
    if (agents.ensemble && agents.ensemble.status === 'running') activateLink('link-patron-ensemble');
  } else if (phase === 'evaluating') {
    if (agents.artigiano && agents.artigiano.status === 'running') activateLink('link-artigiano-libro');
    if (agents.selezionatore && agents.selezionatore.status === 'running') activateLink('link-selezionatore-libro');
    if (agents.modelos && agents.modelos.status === 'running') activateLink('link-modelos-libro');
    if (agents.ensemble && agents.ensemble.status === 'running') activateLink('link-ensemble-libro');
  } else if (phase === 'bootstrap') {
    if (agents.artigiano && agents.artigiano.status === 'running') {
      activateLink('link-patron-artigiano');
    } else if (agents.selezionatore && agents.selezionatore.status === 'running') {
      activateLink('link-patron-selezionatore');
    } else if (agents.modelos && agents.modelos.status === 'running') {
      activateLink('link-patron-modelos');
    }
    
    if (agents.libro && agents.libro.status === 'updating') {
      if (agents.artigiano && agents.artigiano.message && agents.artigiano.message.includes('complete')) activateLink('link-artigiano-libro');
      if (agents.selezionatore && agents.selezionatore.message && agents.selezionatore.message.includes('complete')) activateLink('link-selezionatore-libro');
      if (agents.modelos && agents.modelos.message && agents.modelos.message.includes('Baseline')) activateLink('link-modelos-libro');
    }
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/data');
    const d = await r.json();

    document.getElementById('best-mae').textContent    = fmt(d.best_mae);
    document.getElementById('best-id').textContent     = d.best_id || '';
    document.getElementById('baseline-mae').textContent = fmt(d.baseline_mae);
    document.getElementById('n-exp').textContent       = d.n_experiments;
    document.getElementById('updated').textContent     = d.updated_at;

    if (d.best_mae != null && d.baseline_mae != null && d.baseline_mae > 0) {
      const pct = ((d.baseline_mae - d.best_mae) / d.baseline_mae * 100).toFixed(1);
      document.getElementById('mejora').textContent = pct + '%';
    }

    renderLeaderboard(d.leaderboard, d.best_id);
    renderMini(d.propuestas);
    renderChart(d.mae_timeline, d.baseline_mae);
    if (d.status) {
      updateDiagram(d.status);
    }
  } catch (e) {
    console.warn('refresh error', e);
  }
}

refresh();
setInterval(refresh, 3000); // Poll every 3 seconds for smoother updates
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


if __name__ == '__main__':
    print("🤌  La Mafia Dashboard — http://localhost:5050")
    print(f"   Leyendo: {LIBRO_PATH}")
    print(f"   Propuestas: {PROPUESTAS_PATH}")
    print("   Ctrl+C para salir\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
