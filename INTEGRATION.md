# La Mafia — Integración con Ruflo + autoresearch

## Visión

La arquitectura de mafia-swarm adopta dos patrones externos complementarios:

| Herramienta | Rol en mafia-swarm | Qué aporta |
|---|---|---|
| **Ruflo** | Capa de orquestación de El Patrón | Queen-led swarm, memoria vectorial, MCP tools, federation |
| **autoresearch** | Loop interno de cada Capo (worker) | Editar → correr → medir → conservar/descartar → repetir |

---

## 1. El Karpathy Loop (autoresearch)

Karpathy's autoresearch tiene 3 archivos clave:

```
prepare.py   ← fijo, no se toca
train.py     ← el agente EDITA este archivo
program.md   ← el humano programa las instrucciones del agente
```

El loop:
```
LOOP hasta budget agotado:
  agente lee program.md
  agente edita train.py
  corre entrenamiento 5 min (budget fijo)
  mide val_bpb
  si mejora → conservar
  si empeora → descartar (revert)
  registrar en log
```

### Mapeo a mafia-swarm

```
prepare.py   →  data/serie.csv + pipeline/features.py  (fijo)
train.py     →  pipeline/backtest.py + variant.py       (el Capo EDITA)
val_bpb      →  MAE (rolling-origin, 5 folds)
program.md   →  agents/workers/{capo}/program.md        (✅ ya existen)
5 min budget →  N folds backtest (budget fijo de compute)
```

**Cada Capo es un agente Claude Code que:**
1. Lee su `program.md`
2. Edita `pipeline/variant.py` (copia de trabajo de `backtest.py`)
3. Corre backtest con budget fijo
4. Compara MAE vs mejor en Il Libro
5. Conserva si mejora, descarta si empeora (git revert)
6. Registra en Il Libro
7. Repite

Para activar el loop en un worker:
```bash
# Apuntar Claude Code al worker y dejar correr
cd mafia-swarm
claude agents/workers/modelos/program.md
# O con el loop nativo:
python agents/workers/modelos/loop.py --budget 10 --model lightgbm
```

---

## 2. Ruflo como orquestador de El Patrón

### Instalación

```bash
# Instalar Ruflo como MCP server (Node.js requerido)
npx ruflo init

# Esto configura:
# - Queen agent (El Patrón)
# - AgentDB con memoria vectorial HNSW
# - 314 MCP tools disponibles
# - Hooks de coordinación entre agentes
```

### Mapeo de componentes

| mafia-swarm | Ruflo |
|---|---|
| El Patrón | Queen agent |
| Il Libro | AgentDB (HNSW vector store) |
| Workers (subprocess) | Sub-agents coordinados por Queen |
| Cola de propuestas | Task queue de Ruflo |
| Presupuesto | Token/compute budget de Ruflo |

### Configuración de agentes en Ruflo

Ruflo usa archivos de definición de agentes. Crear `.ruflo/agents.json`:

```json
{
  "queen": {
    "name": "El Patrón",
    "role": "coordinator",
    "program": "agents/patron/program.md",
    "budget": { "max_workers": 6, "parallel": 3 }
  },
  "agents": [
    {
      "name": "L'Emissario",
      "role": "researcher",
      "program": "agents/emissario/program.md",
      "tools": ["web_search", "read_file", "write_file"]
    },
    {
      "name": "Il Contabile",
      "role": "analyzer",
      "program": "agents/contabile/program.md",
      "tools": ["read_file", "python_exec"]
    },
    {
      "name": "L'Artigiano",
      "role": "worker",
      "program": "agents/workers/artigiano/program.md",
      "tools": ["python_exec", "read_file", "write_file"]
    },
    {
      "name": "Il Selezionatore",
      "role": "worker",
      "program": "agents/workers/selezionatore/program.md",
      "tools": ["python_exec", "read_file", "write_file"]
    },
    {
      "name": "Worker Modelos",
      "role": "worker",
      "program": "agents/workers/modelos/program.md",
      "tools": ["python_exec", "read_file", "write_file"],
      "instances": 3
    },
    {
      "name": "Worker Ensemble",
      "role": "worker",
      "program": "agents/workers/ensemble/program.md",
      "tools": ["python_exec", "read_file", "write_file"]
    }
  ]
}
```

### Il Libro como AgentDB

Con Ruflo, Il Libro puede usar AgentDB (HNSW) en lugar de `il_libro.json` puro:

```python
# Guardar experimento en AgentDB (accesible por todos los agentes vía MCP)
# El vector embedding permite búsqueda semántica:
# "encuentra experimentos similares a lightgbm con lags de temporada"

from ruflo import AgentDB
db = AgentDB()
db.store(
    content=json.dumps(experiment),
    metadata={"model": "lightgbm", "mae": 26.75, "type": "experiment"},
    namespace="il_libro"
)

# L'Emissario puede buscar semánticamente:
resultados = db.search("modelos que superaron seasonal naive con features de Fourier")
```

### Lanzar el swarm con Ruflo

```bash
# Una vez instalado Ruflo:
ruflo start --queen "agents/patron/program.md" --agents ".ruflo/agents.json"

# O desde Claude Code con Ruflo MCP:
# El Patrón (Queen) orquesta automáticamente vía hooks
```

---

## 3. Arquitectura integrada final

```
npx ruflo init
       │
       ▼
┌──────────────────────────────────────────────────┐
│              RUFLO (MCP Server)                   │
│                                                   │
│  ┌─────────────┐    ┌────────────────────────┐   │
│  │  EL PATRÓN  │    │     AgentDB (HNSW)     │   │
│  │   (Queen)   │───▶│  Il Libro vectorial    │   │
│  └──────┬──────┘    └────────────────────────┘   │
│         │                                         │
│  ┌──────┴──────────────────────────┐              │
│  │      Cola de propuestas         │              │
│  │  (L'Emissario + Il Contabile)   │              │
│  └──────┬──────────────────────────┘              │
│         │ lanza en paralelo                       │
│  ┌──────┴──────────────────────────────────────┐  │
│  │           Workers (Karpathy Loop)           │  │
│  │                                             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────┐  │  │
│  │  │Artigiano │ │ Modelos  │ │  Ensemble  │  │  │
│  │  │ loop.py  │ │ loop.py  │ │  loop.py   │  │  │
│  │  │          │ │          │ │            │  │  │
│  │  │ editar   │ │  editar  │ │   editar   │  │  │
│  │  │ variant  │ │  variant │ │  variant   │  │  │
│  │  │ ↓        │ │  ↓       │ │  ↓         │  │  │
│  │  │ backtest │ │ backtest │ │  backtest  │  │  │
│  │  │ ↓        │ │  ↓       │ │  ↓         │  │  │
│  │  │ MAE ok?  │ │  MAE ok? │ │  MAE ok?   │  │  │
│  │  │ keep/rev │ │ keep/rev │ │  keep/rev  │  │  │
│  │  └──────────┘ └──────────┘ └────────────┘  │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## 4. Orden de implementación

### Sin Ruflo (ahora mismo — funcional)
```bash
# Loop estratégico: Il Contabile decide → El Patrón despacha → workers ejecutan
python agents/patron/run.py --budget 8 --paciencia 3

# Con bootstrap (primera vez, sin datos en Il Libro):
python agents/patron/run.py --budget 8 --bootstrap

# Compacto: solo 4 experimentos, parada si no mejora en 2 rondas
python agents/patron/run.py --budget 4 --paciencia 2 --min-mejora 0.02
```

### Con Karpathy Loop (próximo paso)
```bash
# Cada worker itera autónomamente
python agents/workers/modelos/loop.py --budget 10 --model lightgbm
```

### Con Ruflo (completo)
```bash
npx ruflo init
ruflo start --queen "agents/patron/program.md"
```

---

## 5. Diferencias clave vs implementación actual

| Aspecto | Ahora | Con Ruflo + autoresearch |
|---|---|---|
| Orquestación | subprocess Python | Ruflo MCP Queen |
| Memoria | `il_libro.json` (JSON plano) | AgentDB (HNSW vectorial) |
| Loop de mejora | Una pasada por worker | Karpathy Loop: N iteraciones autónomas |
| Workers | Script fijo | Agente que edita su propio código |
| Coordinación | ThreadPoolExecutor | Ruflo hooks + federation |
| Búsqueda de experimentos | Exacta (JSON) | Semántica (vector similarity) |
