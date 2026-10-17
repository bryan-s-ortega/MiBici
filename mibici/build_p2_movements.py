"""Construye movimientos observados por estación e intervalo para P2."""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici.config import (
    INTERIM_DATA_DIR,
    P2_BATCH_SIZE,
    P2_INTERVAL_MINUTES,
    P2_MOVEMENTS_PATH,
    P2_STATIONS_PATH,
    P2_TRIPS_PATH,
)
from mibici.features import aggregate_p2_movements


def main():
    """Deduplica, agrega por lotes y consolida sin alterar los datos fuente."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(P2_STATIONS_PATH)
    if stations["id"].isna().any() or stations["id"].duplicated().any():
        raise ValueError("El catálogo contiene IDs faltantes o duplicados.")

    capacity = pd.to_numeric(stations["dpcapacity"], errors="coerce")
    valid_ids = set(
        stations.loc[capacity.notna() & capacity.gt(0), "id"].astype(int)
    )

    # Solo necesitamos recordar las versiones de los IDs repetidos.
    # Así evitamos crear un diccionario con los 13 millones de viajes.
    id_table = pq.read_table(P2_TRIPS_PATH, columns=["trip_id"])
    if id_table.column("trip_id").null_count:
        raise ValueError("Hay trip_id faltantes; revisar antes de continuar.")

    values = id_table.column("trip_id").combine_chunks().to_numpy()
    unique_ids, counts = np.unique(values, return_counts=True)
    repeated_ids = set(unique_ids[counts > 1].tolist())
    del id_table, values, unique_ids, counts

    seen_repeated = {}
    report = Counter()
    columns = [
        "trip_id", "started_at", "ended_at",
        "origin_station_id", "destination_station_id",
    ]
    source = pq.ParquetFile(P2_TRIPS_PATH)

    # Los archivos temporales se eliminan al terminar o si ocurre un error.
    with TemporaryDirectory(prefix="p2_", dir=INTERIM_DATA_DIR) as folder:
        temporary = Path(folder)

        for number, batch in enumerate(
            source.iter_batches(batch_size=P2_BATCH_SIZE, columns=columns), 1
        ):
            data = batch.to_pandas().reset_index(drop=True)
            report["source_rows"] += len(data)
            keep = np.ones(len(data), dtype=bool)

            # Comprobamos que las copias sigan siendo equivalentes.
            candidates = data.loc[data["trip_id"].isin(repeated_ids)]
            for row in candidates.itertuples(index=True, name=None):
                index, trip_id, *movement = row
                signature = tuple(None if pd.isna(x) else x for x in movement)

                if trip_id in seen_repeated:
                    if signature != seen_repeated[trip_id]:
                        raise ValueError(
                            f"Movimiento contradictorio en trip_id={trip_id}"
                        )
                    keep[index] = False
                    report["duplicate_copies_removed"] += 1
                else:
                    seen_repeated[trip_id] = signature

            data = data.loc[keep].copy()
            for column in ("started_at", "ended_at"):
                data[column] = pd.to_datetime(data[column], errors="raise")

            valid_time = (
                data["started_at"].notna()
                & data["ended_at"].notna()
                & data["ended_at"].gt(data["started_at"])
            )
            report["unique_rows_invalid_time"] += int((~valid_time).sum())
            report["unique_rows_valid_time"] += int(valid_time.sum())

            # Contamos extremos excluidos después de deduplicar y validar tiempo.
            report["invalid_origin_events"] += int(
                (valid_time & ~data["origin_station_id"].isin(valid_ids)).sum()
            )
            report["invalid_destination_events"] += int(
                (valid_time & ~data["destination_station_id"].isin(valid_ids)).sum()
            )

            movements = aggregate_p2_movements(data, valid_ids)
            report["departure_events"] += int(movements["departures"].sum())
            report["arrival_events"] += int(movements["arrivals"].sum())

            # Guardamos pequeños agregados, particionados por mes del evento.
            if not movements.empty:
                months = movements["interval_start"].dt.strftime("%Y-%m")
                for month, part in movements.groupby(months):
                    destination = temporary / month
                    destination.mkdir(exist_ok=True)
                    part.to_parquet(
                        destination / f"batch_{number:04d}.parquet", index=False
                    )

            print(f"Lote {number}: {report['source_rows']:,} viajes revisados",
                  flush=True)

        # Consolidación mensual: una fila por estación e intervalo.
        output = temporary / "movimientos_finales.parquet"
        writer = None
        try:
            for month_folder in sorted(temporary.iterdir()):
                if not month_folder.is_dir():
                    continue

                parts = [
                    pd.read_parquet(path)
                    for path in sorted(month_folder.glob("*.parquet"))
                ]
                month_data = pd.concat(parts, ignore_index=True)
                consolidated = (
                    month_data.groupby(
                        ["station_id", "interval_start"], as_index=False
                    )[["departures", "arrivals"]]
                    .sum()
                    .sort_values(["station_id", "interval_start"])
                )
                consolidated["station_id"] = consolidated["station_id"].astype(
                    "int64"
                )
                table = pa.Table.from_pandas(consolidated, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                report["station_intervals_with_events"] += len(consolidated)
                print(f"Mes consolidado: {month_folder.name}", flush=True)
                del parts, month_data, consolidated, table
        finally:
            if writer is not None:
                writer.close()

        if not output.exists():
            raise ValueError("No se generaron movimientos válidos.")

        # El resultado anterior solo se reemplaza cuando termina el proceso.
        output.replace(P2_MOVEMENTS_PATH)

    summary = dict(report)
    summary["interval_minutes"] = P2_INTERVAL_MINUTES
    summary["repeated_trip_ids"] = len(repeated_ids)
    report_path = INTERIM_DATA_DIR / "p2_movimientos_resumen.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nRESUMEN:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nArchivo generado: {P2_MOVEMENTS_PATH}")


if __name__ == "__main__":
    main()
