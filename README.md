# mibici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Plantilla%20de%20proyecto-328F97?logo=cookiecutter" />
</a>

Proyecto de MLOps con datos del sistema de bicicleta pública del Área Metropolitana de Guadalajara, MiBici.

## Estructura del proyecto

```
├── LICENSE            <- Licencia de código abierto (si se define una)
├── Taskfile.yml       <- Archivo de tareas con comandos convenientes como `task data` o `task test`
├── README.md          <- Documento principal para desarrolladores del proyecto.
├── data
│   ├── external       <- Datos provenientes de fuentes de terceros.
│   ├── interim        <- Datos intermedios que han sido transformados.
│   ├── processed      <- Conjuntos de datos finales y canónicos para modelado.
│   └── raw            <- Datos originales e inmutables descargados.
│
├── docs               <- Proyecto predeterminado de mkdocs; consulta www.mkdocs.org para detalles.
│
├── models             <- Modelos entrenados y serializados, predicciones o resúmenes de modelos.
│
├── notebooks          <- Jupyter notebooks. La convención de nombres es un número (para ordenamiento),
│                         las iniciales del autor y una descripción breve separada por `-`, ej.
│                         `1.0-jqp-exploracion-inicial-datos`.
│
├── pyproject.toml     <- Archivo de configuración del proyecto con metadatos del paquete
│                         mibici y configuración de herramientas como ruff.
│
├── references         <- Diccionarios de datos, manuales y materiales explicativos.
│
├── reports            <- Análisis generados en HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos y figuras generadas para su uso en reportes.
│
├── requirements.txt   <- Archivo de requerimientos para reproducir el entorno (si aplica).
│
├── setup.cfg          <- Archivo de configuración para linters.
│
└── mibici             <- Código fuente del proyecto.
    │
    ├── __init__.py             <- Convierte a mibici en un módulo de Python
    │
    ├── config.py               <- Variables útiles y configuraciones del proyecto
    │
    ├── dataset.py              <- Scripts para descargar o procesar datos
    │
    ├── features.py             <- Código para generar variables (features) para modelado
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Código para inferencia con modelos entrenados          
    │   └── train.py            <- Código para entrenar modelos
    │
    └── plots.py                <- Código para generar visualizaciones
```

## Gestión y descarga del conjunto de datos

Existen dos maneras de obtener los datos para trabajar en el proyecto: **descargar los datos versionados existentes desde el almacenamiento remoto** (enfoque recomendado para colaborar) o **reconstruir y procesar el dataset desde la fuente externa**.

### 1. Descargar el dataset existente desde el almacenamiento remoto (Recomendado)

Para comenzar a trabajar de inmediato con la versión oficial de los datos sin procesar gigabytes de archivos crudos ni depender de la web de MiBici, descarga los artefactos rastreados en DVC desde el bucket de Google Cloud Storage (`gs://mibici-mlops-project`):

```bash
# 1. Autentícate con tus credenciales de Google Cloud
gcloud auth application-default login

# 2. Descarga los datos versionados exactos
uv run dvc pull
# o alternativamente con Task:
task dvc:pull
```

> **Permisos de acceso:** Tu cuenta de Google requiere permisos de lectura en el bucket (`roles/storage.objectViewer` o `roles/storage.objectUser`). El propietario del proyecto puede otorgarte acceso ejecutando:
> ```bash
> task grant-access EMAIL=tu_correo@gmail.com ROLE=roles/storage.objectViewer
> ```

---

### 2. Reconstruir o actualizar el dataset desde la fuente externa

Si necesitas reconstruir el dataset desde cero o actualizarlo con datos recién publicados en la [página de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), puedes usar DVC o ejecutar el script directamente.

#### A. Ejecución de la etapa de DVC (`dvc repro`)

```bash
uv sync
uv run dvc repro --force dataset
# o alternativamente con Task:
task data
```

**¿Qué hace exactamente este comando?**
* **No descarga** los datos desde Google Cloud Storage; ejecuta el comando `uv run python -m mibici.dataset` configurado en [dvc.yaml](dvc.yaml).
* Consulta el sitio web de MiBici, descarga los archivos CSV mensuales correspondientes a la **ventana por defecto (los últimos 3 años hasta la fecha actual)**, procesa y normaliza los viajes, une la nomenclatura de estaciones y genera:
  * `data/processed/trips.parquet` (~390 MB)
  * `data/raw/stations_nomenclature.csv` (~40 KB)
* La bandera `--force` fuerza a DVC a reejecutar la etapa ignorando el caché previo.
* **Consideraciones importantes:**
  * **Ventana móvil:** Dado que el rango por defecto se calcula a partir de `date.today()`, ejecutarlo en distintas fechas descargará nuevos datos y alterará los hashes en [dvc.lock](dvc.lock).
  * **Tiempo y recursos:** Procesar 3 años de viajes implica descargar y transformar millones de registros, lo que puede tardar varios minutos dependiendo de la conexión y CPU.

#### B. Delimitar un rango de fechas personalizado

Puedes personalizar el período de descarga utilizando las opciones `--start-date` y `--end-date` (en formato `AAAA-MM-DD`). Esto es muy útil para pruebas rápidas, análisis acotados o para evitar procesar 3 años completos:

```bash
# Ejemplo: Descargar únicamente los viajes del año 2023
uv run python -m mibici.dataset --start-date 2023-01-01 --end-date 2023-12-31

# Ejemplo: Descargar desde el 1 de enero de 2024 hasta la fecha actual
uv run python -m mibici.dataset --start-date 2024-01-01

# Ejemplo: Descargar un período corto para desarrollo y pruebas rápidas
uv run python -m mibici.dataset --start-date 2024-05-01 --end-date 2024-05-31
```

**Parámetros disponibles:**
* `--start-date`: Primer día del rango (`AAAA-MM-DD`). Por defecto: hace 3 años.
* `--end-date`: Último día del rango (`AAAA-MM-DD`). Por defecto: fecha actual.
* `--output-path`: Ruta de salida para el archivo parquet limpio (por defecto: `data/processed/trips.parquet`).
* `--nomenclature-output-path`: Ruta de salida para la nomenclatura (por defecto: `data/raw/stations_nomenclature.csv`).

---

### 3. Subir nuevos datos versionados al almacenamiento remoto (Mantenedores)

Si eres el encargado de actualizar el dataset oficial y deseas compartir la nueva versión con el equipo en Google Cloud Storage:

```bash
gcloud auth application-default login
uv run dvc push
# o alternativamente con Task:
task dvc:push
```

Posteriormente, haz commit y push del archivo `dvc.lock` actualizado a Git para que tus compañeros puedan sincronizar los nuevos datos con `uv run dvc pull`.

---

## Calidad de código y estilo (Ruff y Pre-commit Hooks)

El proyecto utiliza [Ruff](https://docs.astral.sh/ruff/) como linter y formateador de código extremadamente rápido, configurado en el archivo [pyproject.toml](file:///home/bryan/learning/mibici/pyproject.toml) (longitud máxima de línea de 99 caracteres y ordenamiento automático de imports con `isort`).

### Uso de Ruff

Puedes verificar o aplicar formato manualmente utilizando `task` o directamente `uv`:

- **Verificar formato y linteo (sin modificar archivos):**
  ```bash
  task lint
  # Equivalente a:
  # uv run ruff format --check
  # uv run ruff check
  ```

- **Aplicar formateo y autocorrección de errores:**
  ```bash
  task format
  # Equivalente a:
  # uv run ruff check --fix
  # uv run ruff format
  ```

### Git Pre-commit Hook

Para evitar subir código con errores de estilo o sintaxis, se incluye la configuración de [pre-commit](https://pre-commit.com/) en [.pre-commit-config.yaml](file:///home/bryan/learning/mibici/.pre-commit-config.yaml), la cual ejecuta automáticamente `ruff-check --fix` y `ruff-format` antes de cada `git commit`.

1. **Instalar el hook en tu entorno Git local:**
   ```bash
   task hooks:install
   # o: uv run pre-commit install
   ```

2. **Ejecutar el hook manualmente sobre todos los archivos:**
   ```bash
   task hooks:run
   # o: uv run pre-commit run --all-files
   ```

Una vez instalado, cada vez que ejecutes `git commit`, Ruff revisará y formateará los archivos modificados de manera transparente.

---

## Flujo de trabajo y contribución (GitHub Flow)

Para colaborar en este proyecto se sigue la metodología [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow), diseñada para mantener un ciclo de entrega continuo, simple y ordenado:

1. **Crear una rama (`branch`):** Parte siempre de la rama `main` actualizada con un nombre descriptivo según la convención de ramas.
2. **Realizar cambios y commits:** Haz commits frecuentes, atómicos y con mensajes claros siguiendo la convención de commits.
3. **Verificar calidad localmente:** Antes de subir tus cambios o solicitar revisión, ejecuta las herramientas de formateo y tests:
   ```bash
   task lint
   task test
   ```
4. **Abrir un Pull Request (PR):** Publica tu rama en GitHub y abre un Pull Request hacia `main`. Describe qué cambios introdujiste, capturas/resultados si aplica, y referencia los issues relacionados.
5. **Revisión de código:** Resuelve comentarios, sugerencias o verificaciones automáticas de CI.
6. **Merge y limpieza:** Una vez aprobado, el PR se fusiona a `main` y la rama de trabajo se elimina.

---

### Convención de nombres de ramas

Las ramas deben nombrarse en minúsculas, usando `kebab-case` (palabras separadas por guiones) y organizadas mediante un prefijo que describa el propósito del cambio:

```text
<tipo>/<descripcion-corta>
```

#### Prefijos disponibles:

| Prefijo | Propósito | Ejemplo |
| :--- | :--- | :--- |
| **`feat/`** | Nueva funcionalidad o módulo del proyecto | `feat/weather-feature-extraction` |
| **`fix/`** | Corrección de un bug o error en datos/código | `fix/station-id-type-mismatch` |
| **`docs/`** | Cambios exclusivamente en documentación | `docs/add-contributing-guide` |
| **`refactor/`** | Cambios en el código que no alteran la lógica ni añaden funcionalidades | `refactor/dataset-cleaning-pipeline` |
| **`test/`** | Adición o corrección de pruebas unitarias o de integración | `test/add-dataset-schema-checks` |
| **`chore/`** | Tareas de mantenimiento, dependencias o tooling | `chore/update-uv-lock` |

---

### Convención de mensajes de commits (Conventional Commits)

Adoptamos la especificación de [Conventional Commits](https://www.conventionalcommits.org/) para mantener un historial de cambios legible, uniforme y automatizable:

```text
<tipo>(<alcance-opcional>): <descripción concisa en imperativo>

[cuerpo opcional explicando el porqué o detalles del cambio]
```

#### Tipos de commits comunes:
- **`feat`**: Incorpora una nueva funcionalidad (`feat(dataset): add holiday indicator feature`).
- **`fix`**: Corrige un error o bug (`fix(dvc): set correct quota project in adc`).
- **`docs`**: Modificaciones en documentación (`docs(readme): document git workflow and branch conventions`).
- **`refactor`**: Reorganización de código sin cambiar comportamiento (`refactor(dataset): simplify date parsing loop`).
- **`test`**: Pruebas automáticas (`test(dataset): add tests for raw stations nomenclature`).
- **`chore`**: Actualizaciones de paquetes, configuraciones o tareas rutinarias (`chore(deps): update hydra-core`).

#### Buenas prácticas:
- **Modo imperativo:** Redacta el mensaje como una orden directa (ej. `add feature`, no `added feature` / `agrega validación`, no `agregando validación`).
- **Línea de título concisa:** Mantén la primera línea en menos de 72 caracteres.
- **Commits atómicos:** Cada commit debe resolver una única unidad lógica de trabajo.

--------

## Entrega del Equipo 23

Integrantes:

- Miguel Ángel Durán Betancourt — A01797974
- Juan Pablo Islas Guzmán — A01797727
- Bryan Steve Ortega Castillo — A01797846
- Kevin Rogelio Rodriguez Magaña — A01840418

El repositorio usa DVC para versionar los datos grandes. La reconstrucción
desde la fuente oficial se ejecuta con:

```bash
uv sync
uv run dvc repro --force dataset
```

Para ejecutar los experimentos base:

- **P3 (Clustering de Estaciones — ejecuta directamente sobre los datos de la ingesta):**
  ```bash
  uv run python -m mibici.modeling.train_p3
  ```

- **P1 (Regresión de Demanda Diaria):**
  ```bash
  uv run python notebooks/preparacion_p1.py
  uv run python -m mibici.modeling.train_p1
  ```

- **P2 (Clasificación de Presión Neta):**
  ```bash
  uv run python -m mibici.baseline_p2
  ```

## MLflow Projects — propuesta común de P1, P2 y P3

`MLproject` ofrece `ingest`, `train` y `search`. Cada ejecución desde la carpeta
del repositorio crea una corrida en el experimento `mibici-equipo23-mlprojects`.
El proyecto declara Python 3.13.16 y dependencias en `python_env.yaml`; el
procedimiento local recomendado usa las versiones exactas de `uv.lock`.
Los datos **no** se incluyen en Git. Para P2 se añadió la cadena fija de DVC
`p2_dataset → p2_movements → p2_examples` en `dvc.yaml`. Hay que ejecutar
`dvc repro p2_examples`, revisar el `dvc.lock` producido y subir datos mediante
`dvc push` desde una cuenta con acceso a GCS. No se ha publicado un `dvc.lock`
de esas etapas desde este entorno sin los datos originales.

```bash
cd ~/proyectos/MiBici
git pull
uv sync --locked
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
# En otra terminal: uv run --locked mlflow server --backend-store-uri sqlite:///mlflow_mibici.db --host 127.0.0.1 --port 5001
uv run --locked dvc pull
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e ingest -P question=P2
```

Si no existe `p2_examples.parquet`, pero sí los viajes corregidos y el
catálogo, `-e ingest -P question=P2 -P build=true` reconstruye movimientos y
ejemplos. Para versionarlos realmente en DVC ejecuta `uv run --locked dvc repro
p2_examples` sobre la etapa completa y comparte el `dvc.lock` y el cache
mediante `dvc push`. La etapa de ingesta fija 2023-11-01 a 2026-09-30;
evita cambiar los cortes temporales entre corridas.

```bash
# P2: baseline, árbol individual y búsqueda anidada (tres árboles + baseline).
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e train -P question=P2 -P model=baseline
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e train -P question=P2 -P model=tree -P max_depth=5 -P min_samples_leaf=100 -P class_weight=balanced
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e search -P question=P2 -P seed=42 -P sample_fraction=0.05

# P1: primero preparar los datos propios; luego ejecutar variantes.
uv run --locked python notebooks/preparacion_p1.py
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e train -P question=P1 -P model=rf -P max_depth=6 -P min_samples_leaf=8 -P n_estimators=200
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e search -P question=P1

# P3: agrupación por estaciones; model=agglomerative no permite inferencia nueva.
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e train -P question=P3 -P model=kmeans -P k=4
uv run --locked mlflow run . --env-manager local --experiment-name mibici-equipo23-mlprojects -e search -P question=P3
```

Las métricas de P2 se calculan por estación e intervalo de 30 minutos. La
etiqueta indica **presión neta estimada a partir de viajes**, no que existan
cero bicicletas o cero anclajes reales. La búsqueda usa train para ajustar,
validation para seleccionar y abre test una vez para el ganador. El criterio
es balanced accuracy de validación, siempre que el recall de cada clase de
presión sea al menos 0.10. Las curvas PR usan un histograma de 200 intervalos
y son aproximadas; la matriz de confusión y las métricas son exactas.

Para registrar la versión elegida (una vez revisadas métricas y artefactos),
usa el `run_id` impreso por cada script. El alias `champion` distingue la
mejor versión académica, **no habilita un servicio en producción**. P2 se
registra con el nombre explícito `Presion_Proxy_Academica`.

```bash
uv run --locked python -m mibici.register_project_models --question P2 --run-id RUN_ID_GANADOR
# Si MLflow 3 expone solo URI models:/m-..., agregar --model-uri models:/m-ID
```

El registro comprueba que el alias resuelva a la nueva versión y que el modelo
cargado prediga una fila sintética. P3 debe ser KMeans para aceptar `predict`.
Cada responsable debe repetir entrenamiento, elegir y registrar P1 y P3, y
adjuntar capturas de sus modelos y corridas al documento del equipo.
Las búsquedas P1/P3 ejecutan tres candidatos anidados; P1 evalúa test solo
para el elegido por MAE de validación. P3 elige por silhouette sobre el mismo
conjunto de estaciones y no pretende que esa métrica sea desempeño futuro.
