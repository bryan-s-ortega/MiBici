"""Comprueba identificadores de viaje antes de agregar movimientos de P2."""

import json

import numpy as np
import pyarrow.parquet as pq

from mibici.config import INTERIM_DATA_DIR, P2_TRIPS_PATH


def audit_trip_ids():
    """Detecta identificadores faltantes y repetidos sin modificar los viajes."""
    # Leemos únicamente el identificador, no todas las columnas del dataset.
    table = pq.read_table(P2_TRIPS_PATH, columns=["trip_id"])
    column = table.column("trip_id")
    missing = column.null_count

    # Ignoramos los nulos solo para contar los identificadores conocidos.
    ids = column.drop_null().combine_chunks().to_numpy()
    unique_ids, counts = np.unique(ids, return_counts=True)
    repeated = counts > 1

    report = {
        "total_rows": table.num_rows,
        "missing_trip_ids": missing,
        "unique_non_null_trip_ids": int(len(unique_ids)),
        "repeated_trip_ids": int(repeated.sum()),
        "rows_with_repeated_ids": int(counts[repeated].sum()),
        "extra_rows_beyond_first": int((counts[repeated] - 1).sum()),
        "repeated_id_examples": [
            int(value) for value in unique_ids[repeated][:10]
        ],
    }

    # El diagnóstico se guarda junto a los datos intermedios locales.
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = INTERIM_DATA_DIR / "p2_auditoria_ids.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nDiagnóstico guardado en: {output}")
    return report


if __name__ == "__main__":
    audit_trip_ids()


def audit_repeated_movements():
    """Distingue copias equivalentes para P2 de identificadores contradictorios."""
    import pandas as pd

    identifiers = pq.read_table(P2_TRIPS_PATH, columns=["trip_id"])
    values = identifiers.column("trip_id").drop_null().combine_chunks().to_numpy()
    unique_ids, counts = np.unique(values, return_counts=True)
    repeated_ids = unique_ids[counts > 1]
    del identifiers, values, unique_ids, counts

    columns = [
        "trip_id",
        "started_at",
        "ended_at",
        "origin_station_id",
        "destination_station_id",
    ]
    parts = []

    # Recorremos por lotes y conservamos solo las filas con ID repetido.
    from mibici.config import P2_BATCH_SIZE

    source = pq.ParquetFile(P2_TRIPS_PATH)
    for batch in source.iter_batches(
        batch_size=P2_BATCH_SIZE, columns=columns
    ):
        frame = batch.to_pandas()
        selected = frame.loc[frame["trip_id"].isin(repeated_ids)]
        if not selected.empty:
            parts.append(selected)

    if parts:
        repeated_rows = pd.concat(parts, ignore_index=True)

        # Colapsamos únicamente versiones iguales en los campos de P2.
        versions = repeated_rows.drop_duplicates(subset=columns)
        versions_per_id = versions.groupby("trip_id").size()

        conflicting_ids = versions_per_id[versions_per_id > 1].index
        equivalent_ids = versions_per_id[versions_per_id == 1].index

        report = {
            "repeated_ids_equivalent_for_p2": int(len(equivalent_ids)),
            "repeated_ids_with_conflicting_movements": int(len(conflicting_ids)),
            "conflicting_id_examples": [
                int(value) for value in conflicting_ids[:10]
            ],
        }

        if len(conflicting_ids):
            print("\nEJEMPLOS DE MOVIMIENTOS CONTRADICTORIOS:")
            examples = repeated_rows.loc[
                repeated_rows["trip_id"].isin(conflicting_ids[:5])
            ].sort_values(["trip_id", "started_at"])
            print(examples.to_string(index=False))
    else:
        report = {
            "repeated_ids_equivalent_for_p2": 0,
            "repeated_ids_with_conflicting_movements": 0,
            "conflicting_id_examples": [],
        }

    output = INTERIM_DATA_DIR / "p2_auditoria_duplicados.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nCOMPARACIÓN DE IDENTIFICADORES REPETIDOS:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    audit_repeated_movements()
