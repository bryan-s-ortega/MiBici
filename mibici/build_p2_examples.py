"""Construye ejemplos temporales de clasificación para P2."""

from collections import Counter
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici.config import (
    P2_ACTIVITY_DAYS,
    P2_DATA_START,
    P2_DATA_END,
    P2_TRAIN_END,
    P2_VALIDATION_END,
    P2_EXAMPLES_PATH,
    P2_EXAMPLES_SUMMARY_PATH,
    P2_INTERVAL_MINUTES,
    P2_MOVEMENTS_PATH,
    P2_PRESSURE_THRESHOLD,
    P2_STATIONS_PATH,
)
from mibici.features import build_p2_station_examples


def assign_split(examples):
    """Asigna periodos completos a entrenamiento, validación o prueba."""
    train_end = pd.Timestamp(P2_TRAIN_END)
    validation_end = pd.Timestamp(P2_VALIDATION_END)
    data_end = pd.Timestamp(P2_DATA_END)

    examples = examples.copy()
    examples["split"] = ""

    train = examples["target_end"].le(train_end)
    validation = (
        examples["prediction_time"].ge(train_end)
        & examples["target_end"].le(validation_end)
    )
    test = (
        examples["prediction_time"].ge(validation_end)
        & examples["target_end"].le(data_end)
    )

    examples.loc[train, "split"] = "train"
    examples.loc[validation, "split"] = "validation"
    examples.loc[test, "split"] = "test"

    if examples["split"].eq("").any():
        raise ValueError("Hay ventanas objetivo que cruzan límites de partición.")

    return examples


def main():
    stations = pd.read_csv(P2_STATIONS_PATH)
    if stations["id"].isna().any() or stations["id"].duplicated().any():
        raise ValueError("El catálogo tiene identificadores inválidos.")

    stations["dpcapacity"] = pd.to_numeric(
        stations["dpcapacity"], errors="coerce"
    )
    stations = stations.loc[
        stations["dpcapacity"].notna() & stations["dpcapacity"].gt(0)
    ].sort_values("id")

    P2_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = P2_EXAMPLES_PATH.with_suffix(".tmp.parquet")
    writer = None
    counts = Counter()
    station_counts = Counter()
    total = 0

    try:
        for station in stations.itertuples(index=False):
            # Solo cargamos los movimientos de esta estación.
            movements = pd.read_parquet(
                P2_MOVEMENTS_PATH,
                filters=[("station_id", "==", int(station.id))],
                columns=[
                    "station_id", "interval_start", "departures", "arrivals"
                ],
            )
            if movements.empty:
                continue

            examples = build_p2_station_examples(
                movements,
                capacity=float(station.dpcapacity),
                start=P2_DATA_START,
                end=P2_DATA_END,
            )
            if examples.empty:
                continue

            examples = assign_split(examples)

            # Resumen de clases para entrenamiento y validación.
            # La prueba se reserva; aquí solo contamos sus filas y estaciones.
            for split, part in examples.groupby("split"):
                counts[(split, "total")] += len(part)
                station_counts[split] += 1
                if split != "test":
                    for label, quantity in part["target"].value_counts().items():
                        counts[(split, label)] += int(quantity)

            table = pa.Table.from_pandas(examples, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
            total += len(examples)

            print(
                f"Estación {station.id}: {len(examples):,} ejemplos. "
                f"Acumulado: {total:,}",
                flush=True,
            )

        if writer is None:
            raise ValueError("No se generaron ejemplos.")
        writer.close()
        writer = None
        temporary.replace(P2_EXAMPLES_PATH)

    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    summary = {
        "total_examples": total,
        "interval_minutes": P2_INTERVAL_MINUTES,
        "activity_days": P2_ACTIVITY_DAYS,
        "pressure_threshold": P2_PRESSURE_THRESHOLD,
        "period_start": P2_DATA_START,
        "period_end_exclusive": P2_DATA_END,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        item = {
            "rows": counts[(split, "total")],
            "stations": station_counts[split],
        }
        if split != "test":
            item["classes"] = {
                label: counts[(split, label)]
                for label in (
                    "sin_presion_neta_elevada",
                    "presion_sin_bicicletas",
                    "presion_sin_anclajes",
                )
            }
        summary["splits"][split] = item

    P2_EXAMPLES_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    P2_EXAMPLES_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nRESUMEN:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDataset guardado en: {P2_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
