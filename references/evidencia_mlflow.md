# Evidencia de MLflow — Equipo 23

## Experimento

```text
mibici-equipo23
```

Abrir la interfaz con:

```bash
cd /Users/mac/Desktop/MiBici_Proyecto
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Luego visitar http://127.0.0.1:5001.

## Evidencia requerida

Tomar y guardar en `reports/evidence/` las siguientes capturas:

1. Vista de la lista del experimento `mibici-equipo23` donde se vean los runs
   `P1-baseline`, `P2-baseline` y `P3-baseline`.
2. Detalle del run P1 con parámetros, métricas y tags.
3. Detalle del run P2 con sus métricas de clasificación.
4. Detalle del run P3 mostrando el artefacto `p3_perfiles_clusters.png`.
5. Vista de comparación entre dos o más runs.

Capturas guardadas:

- `reports/evidence/mlflow_lista_runs.png`
- `reports/evidence/mlflow_detalle_p1_candidato.png`
- `reports/evidence/mlflow_detalle_p2_candidato.png`
- `reports/evidence/mlflow_detalle_p3_artefacto.png`
- `reports/evidence/mlflow_comparacion_p1.png`

Los resultados reproducidos el 17 de octubre de 2026 fueron:

| Pregunta | Resultado |
|---|---:|
| P1 MAE | 11.6304 |
| P1 candidato MAE | 6.5447 |
| P1 candidato RMSE | 12.4127 |
| P2 accuracy | 0.6244 |
| P2 F1 | 0.7688 |
| P2 candidato accuracy | 0.8282 |
| P2 candidato F1 | 0.8499 |
| P3 silhouette | 0.4982 |

Cada run guarda `seed`, rango de fechas, tag de equipo, pregunta y commit de
Git. P2 es una etiqueta proxy basada en demanda/capacidad, no inventario real.
