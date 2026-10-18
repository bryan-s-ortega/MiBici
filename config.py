"""Configuración central del proyecto MiBici."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
DATA_MANIFEST = ROOT / "references" / "data_manifest.csv"

SEED = 42
TEAM_ID = "equipo23"
START_DATE = "2025-01-01"
END_DATE = "2026-09-30"
RISK_CAPACITY_RATIO = 0.80
MLFLOW_EXPERIMENT = f"mibici-{TEAM_ID}"
