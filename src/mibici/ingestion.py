"""Lectura, normalización y consolidación de archivos abiertos de MiBici."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR


def normalize_column(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = [normalize_column(column) for column in frame.columns]
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise UnicodeError(f"No se pudo decodificar {path}: {last_error}")


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    travel_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    station_files = sorted(raw_dir.glob("nomenclatura_*.csv"))
    if not travel_files:
        raise FileNotFoundError(
            f"No hay viajes en {raw_dir}. Coloca archivos datos_abiertos_AAAA_MM.csv."
        )
    if not station_files:
        raise FileNotFoundError(
            f"No hay catálogo en {raw_dir}. Coloca nomenclatura_AAAA_MM.csv."
        )
    trips = pd.concat((read_csv_robust(path) for path in travel_files), ignore_index=True)
    stations = read_csv_robust(station_files[-1])
    return trips, stations


def save_consolidated(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> None:
    trips, stations = load_raw(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trips.to_csv(output_dir / "viajes_consolidados.csv", index=False)
    stations.to_csv(output_dir / "estaciones.csv", index=False)
    print(f"Guardados {len(trips):,} viajes y {len(stations):,} estaciones.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    save_consolidated(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()

