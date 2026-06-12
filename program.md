# La Mafia — Forecasting Research Program

> Edit this file between sessions to guide the next research direction.
> El Patrón auto-updates the "Estado actual" section after each improvement.

## Problema

Serie temporal univariada (AirPassengers, mensual). Forecasting con horizonte=12 pasos.
Métrica objetivo: **MAE** en cross-validación rolling-origin (5 folds).
Baseline obligatorio: Seasonal Naive (MAE ≈ 36.42). Todo modelo ML debe superarlo.

---

## Estado actual

<!-- AUTO-UPDATED by El Patrón — do not edit this block manually -->
- **Mejor MAE**: 20.371494 (exp_20260610_230157_52bd02)
- **Modelo**: xgboost | n_estimators=165 | learning_rate=0.183 | max_depth=4 | subsample=0.640 | colsample_bytree=0.994
- **Features**: data/serie_features.csv
- **Total experimentos acumulados**: 5
<!-- END AUTO-UPDATE -->

---

## Workers disponibles

### modelos — entrena modelos ML con HPO (Optuna)
```json
{
  "worker": "modelos", "accion": "run_model",
  "args_worker": { "model": "xgboost|lightgbm|random_forest", "data": "data/serie_selected.csv", "trials": 30 }
}
```

### ensemble — combina top modelos del il_libro
```json
{
  "worker": "ensemble", "accion": "run_ensemble",
  "args_worker": { "top_k": 3, "method": "weighted" }
}
```

### artigiano — ingeniería de features sobre la serie
```json
{
  "worker": "artigiano", "accion": "run_features",
  "args_worker": { "lags": "1,2,3,6,12,24", "windows": "3,6,12,24,48", "diffs": "1,12", "fourier": "12:6" }
}
```

### selezionatore — selección de features
```json
{
  "worker": "selezionatore", "accion": "run_selection",
  "args_worker": { "method": "importance", "top_k": 20 }
}
```
⚠️ Usar siempre `method: importance` o `method: all`. Nunca `method: mutual_info` solo —
colapsa la selección a 4 features por correlación agresiva.

---

## Lo que ha funcionado

- **XGBoost + data/serie_selected.csv** domina todo el leaderboard (top 6 son XGBoost)
- **Zona ganadora**: depth 3–4, lr 0.15–0.18, subsample 0.60–0.70, colsample 0.88–0.99
- **Features seleccionadas > features completas**: selected (20) da MAE ~18 vs completas (47) da MAE ~20+
- **lag_12** es la feature más importante (importancia ≈ 0.73); **lag_24** segunda (≈ 0.21)
- Reducir subsample de 0.93 → 0.64 junto con depth 3→4 produjo la mayor mejora individual

---

## Lo que NO ha funcionado

- **LightGBM**: nunca supera XGBoost — MAE ≥ 25 vs XGBoost ~18 en mismas features
- **Ensembles**: siempre peor que el mejor modelo individual — MAE ~21 vs 18.26
- **Random Forest**: plateau en MAE ~30, muy lejos del líder
- **Features completas** (serie_features.csv): consistentemente peor que selected
- **method: mutual_info** en selezionatore: colapsa features de 20 → 4 (lag_12, lag_1, rolling_min_3, rolling_min_48)
- **HPO agresivo en LightGBM** (100 trials): no cierra la brecha con XGBoost

---

## Direcciones inexploradas

1. **XGBoost con objective='reg:absoluteerror'** — todos los experimentos usan MSE para entrenar pero evaluamos MAE. Alinear objetivo de entrenamiento con métrica de evaluación es la apuesta más obvia.
2. **Regularización explícita de XGBoost** — gamma, reg_alpha, reg_lambda, min_child_weight nunca se han configurado. Los folds 4–5 tienen MAE ~23–24 vs folds 1–2 en ~13–16, señal clara de overfitting temporal.
3. **Mantener lag_24 explícitamente** — el selezionatore lo elimina por correlación con lag_12, pero tiene importancia 0.21. Probar un dataset con lag_12, lag_24 y top features sin el filtro de correlación.
4. **Features de volatilidad** — rolling_std, EWMA spans sobre la serie. Los folds tardíos son más volátiles; features que capturen régimen de volatilidad podrían reducir el MAE en folds 4–5.
5. **XGBoost con subsample aún más bajo** (0.50–0.55) — la zona ganadora baja subsample cuando sube depth; explorar si continúa la tendencia.

---

## Notas del investigador

- El MAE tiene alta varianza entre folds: fold 1 ≈ 13–15, fold 5 ≈ 23–24. El patrón es consistente en todos los experimentos. Reducir este gap es el mayor lever disponible.
- La serie parece tener un cambio de régimen (crecimiento + estacionalidad amplificada hacia los folds tardíos). Features que capturen tendencia local podrían ayudar.
- Los mejores experimentos tienen en común: depth bajo, lr moderado, subsample conservador. No explorar configuraciones fuera de esta zona.
- Cuando il_libro se resetea (bootstrap desde cero), los primeros experimentos suelen alcanzar MAE ~20 — se necesitan varias rondas HPO para volver a ~18.
