from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Configuración de P2
P2_INTERVAL_MINUTES = 30
P2_BATCH_SIZE = 200_000
P2_TRIPS_PATH = PROCESSED_DATA_DIR / "trips_fechas_corregidas.parquet"
P2_STATIONS_PATH = RAW_DATA_DIR / "stations_nomenclature_reconstruccion.csv"
P2_MOVEMENTS_PATH = INTERIM_DATA_DIR / "p2_movimientos_corregidos.parquet"

# Reglas iniciales para construir ejemplos de clasificación de P2.
P2_ACTIVITY_DAYS = 7
P2_PRESSURE_THRESHOLD = 0.30

# Periodos del dataset supervisado P2
P2_DATA_START = "2023-11-01"
P2_DATA_END = "2026-10-01"  # Exclusivo: incluye hasta septiembre.
P2_TRAIN_END = "2026-01-01"
P2_VALIDATION_END = "2026-07-01"
P2_EXAMPLES_PATH = PROCESSED_DATA_DIR / "p2_examples.parquet"
P2_EXAMPLES_SUMMARY_PATH = INTERIM_DATA_DIR / "p2_examples_summary.json"
