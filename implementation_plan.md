# La Mafia — Plan de Implementación

## Visión general

Sistema de agent swarm jerárquico para forecasting univariado. Objetivo: minimizar MAE sobre una serie temporal usando un ciclo iterativo de propuestas → experimentación → registro → selección.

---

## Fase 0 — Setup del entorno

**Duración estimada:** 1–2 horas

- [ ] Instalar dependencias Python (`requirements.txt`)
- [ ] Verificar estructura de carpetas del proyecto
- [ ] Preparar datos: CSV con columnas `ds` (fecha) y `y` (valor)
- [ ] Configurar `il_libro.json` vacío (leaderboard inicial)
- [ ] (Opcional) Instalar Ruflo: `npx ruflo init` para orquestación MCP

---

## Fase 1 — Baselines obligatorios

**Responsable:** Worker > Modelos  
**Duración estimada:** 2–3 horas

Antes de cualquier experimento avanzado, establecer el piso de comparación:

- [ ] Implementar **Naive** (último valor observado)
- [ ] Implementar **Seasonal Naive** (mismo valor de la temporada anterior)
- [ ] Correr backtest rolling-origin con ambos
- [ ] Registrar MAE por fold en `il_libro.json`

> Regla: ningún modelo se acepta si no supera al Seasonal Naive.

---

## Fase 2 — Feature Engineering (L'Artigiano)

**Responsable:** Worker > Artigiano  
**Duración estimada:** 3–4 horas

- [ ] Lags (1, 2, 3, 7, 14, 28 según frecuencia)
- [ ] Rolling stats (media, std, min, max — ventanas múltiples)
- [ ] Diferencias (primera orden, estacional)
- [ ] Features de calendario (día semana, mes, trimestre, festivos si aplica)
- [ ] Componentes de Fourier (para estacionalidad múltiple)
- [ ] Guardar pipeline de features reproducible

---

## Fase 3 — Feature Selection (Il Selezionatore)

**Responsable:** Worker > Selezionatore  
**Duración estimada:** 2–3 horas

- [ ] Calcular importancia (permutation importance o SHAP)
- [ ] Calcular mutual information
- [ ] Eliminar features redundantes (correlación > 0.95)
- [ ] Producir conjunto reducido de features + justificación
- [ ] Registrar resultado en `il_libro.json`

---

## Fase 4 — Modelos clásicos y ML (Workers > Modelos)

**Responsable:** Worker > Modelos  
**Duración estimada:** 4–6 horas

Correr en paralelo (un worker por familia):

| Familia | Modelos |
|---------|---------|
| Clásicos | ARIMA, ETS, Theta |
| Gradient Boosting | LightGBM + lags, XGBoost + lags |
| Deep univariado | N-BEATS, PatchTST |

Para cada modelo:
- [ ] HPO básico (Optuna o grid search, ≤ 50 trials)
- [ ] Backtest rolling-origin (mínimo 5 folds)
- [ ] MAE por fold + MAE promedio
- [ ] Registrar config + resultados en `il_libro.json`

---

## Fase 5 — Ensemble (Worker > Ensemble)

**Responsable:** Worker > Ensemble  
**Duración estimada:** 1–2 horas

- [ ] Seleccionar Top-K modelos de `il_libro.json` (K = 3 por defecto)
- [ ] Probar blends: promedio simple, promedio ponderado por MAE inverso, stacking
- [ ] Backtest del ensemble
- [ ] Registrar en `il_libro.json`

---

## Fase 6 — Ciclo de mejora (El Patrón + Researchers)

**Responsable:** El Patrón coordina; L'Emissario e Il Contabile proponen

Ciclo iterativo hasta criterio de parada:

```
Il Contabile → analiza Il Libro → detecta patrones de error
L'Emissario → busca técnicas nuevas → genera propuestas
El Patrón → rankea propuestas (MAE esperado × confianza ÷ costo)
El Patrón → lanza workers según presupuesto
Workers → experimentan → actualizan Il Libro
```

**Criterio de parada sugerido:**
- Sin mejora de MAE > 1% en las últimas N propuestas, O
- Presupuesto de compute agotado

---

## Fase 7 — Evaluación final

- [ ] Seleccionar mejor modelo/ensemble de `il_libro.json`
- [ ] Generar reporte de resultados: MAE por fold, gráfico de predicciones vs real
- [ ] Documentar configuración ganadora

---

## Estructura de carpetas

```
mafia-swarm/
├── README.md
├── implementation_plan.md
├── requirements.txt
├── il_libro.json              ← leaderboard
├── data/
│   └── serie.csv              ← columnas: ds, y
├── pipeline/
│   ├── backtest.py            ← rolling-origin backtest
│   ├── features.py            ← feature engineering
│   └── evaluate.py            ← métricas (MAE, etc.)
└── agents/
    ├── patron/
    │   └── program.md
    ├── emissario/
    │   └── program.md
    ├── contabile/
    │   └── program.md
    └── workers/
        ├── artigiano/
        │   └── program.md
        ├── selezionatore/
        │   └── program.md
        ├── modelos/
        │   └── program.md
        └── ensemble/
            └── program.md
```

---

## Principios de evaluación (no negociables)

1. **Siempre rolling-origin** — nunca K-fold aleatorio
2. **Siempre comparar contra Seasonal Naive**
3. **Reportar MAE por fold**, no solo promedio
4. **Registrar toda config en `il_libro.json`** — reproducibilidad total
