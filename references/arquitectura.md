# Arquitectura propuesta

```mermaid
flowchart LR
  A[Portal abierto MiBici] --> B[data/raw]
  B --> C[ingestion.py]
  C --> D[data/processed]
  D --> E[features.py]
  E --> F[P1 regresión]
  E --> G[P2 clasificación]
  E --> H[P3 clustering]
  F --> I[MLflow Tracking]
  G --> I
  H --> I
  I --> J[reports/figures]
  D -. manifiesto SHA-256 .-> K[Versionado de datos]
```

La ejecución local usa Python, pandas y scikit-learn. MLflow conserva
parámetros, métricas, tags y artefactos; Git conserva código y documentación;
el manifiesto `references/data_manifest.csv` conserva la versión verificable de
los CSV mediante filas, tamaños y hashes SHA-256, sin incluirlos directamente
en GitHub.
