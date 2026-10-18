# Proyecto MiBici

**Equipo 23.** Propuesta reproducible de Machine Learning para las tres preguntas del proyecto
integrador: pronóstico de viajes (P1), riesgo operacional aproximado (P2) y
agrupación de estaciones (P3).

## Integrantes

- Miguel Ángel Durán Betancourt
- Juan Pablo Islas Guzmán — A01797727
- Bryan Steve Ortega Castillo
- Kevin Rogelio Rodriguez Magaña

## Estructura

```text
data/raw/          CSV originales; no se versionan con Git
data/processed/    tablas consolidadas
notebooks/         exploración y comunicación
src/mibici/        ingesta, features y entrenamiento
models/            modelos serializados
reports/figures/   gráficas y artefactos
references/        ML Canvas y arquitectura
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Datos

Descargar los CSV mensuales publicados por MiBici y colocarlos en `data/raw/`:

- `datos_abiertos_AAAA_MM.csv`
- `nomenclatura_AAAA_MM.csv`

Los nombres exactos pueden variar históricamente. La ingesta intenta `utf-8`,
`utf-8-sig`, `cp1252` y `latin1`, y normaliza los nombres de columnas.
La versión del snapshot se conserva en
[`references/data_manifest.csv`](references/data_manifest.csv), con nombre,
filas, tamaño y SHA-256 de cada archivo. Este manifiesto es el mecanismo de
versionado de datos: permite verificar que otra computadora usa exactamente
los mismos CSV sin subir los datos grandes al repositorio.

Consolidar los datos:

```bash
PYTHONPATH=.:src python -m mibici.dataset
```

## Entrenamiento y MLflow

```bash
PYTHONPATH=.:src python -m mibici.training
mlflow ui
```

El pipeline usa una partición temporal del 80/20 para P1 y P2. P2 es un proxy:
los datos abiertos no contienen inventario real; se marca riesgo alto cuando la
demanda previa dividida entre capacidad alcanza el umbral configurado en
`config.py`. Este supuesto debe validarse con el equipo.

## Flujo colaborativo

La rama principal debe permanecer protegida. Cada cambio se trabaja en una
rama `feature/<tema>`, se revisa mediante Pull Request y se integra solo
después de ejecutar el pipeline y sus validaciones. Los CSV grandes se
reconstruyen mediante la ingesta documentada y se verifican contra el
manifiesto; nunca se suben directamente al repositorio.

## Limitaciones conocidas

- No hay inventario histórico de bicicletas; P2 no representa una medición real
  de disponibilidad.
- El catálogo usado debe corresponder al periodo documentado.
- Antes de la entrega hay que agregar las URL exactas de las fuentes, el rango
  real, conteos finales y capturas de MLflow.
