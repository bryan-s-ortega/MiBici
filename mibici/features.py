"""Transformaciones reutilizables para el proyecto MiBici."""

import pandas as pd

from mibici.config import P2_INTERVAL_MINUTES


def aggregate_p2_movements(
    trips: pd.DataFrame,
    valid_station_ids: set[int],
    interval_minutes: int = P2_INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estación e intervalo.

    Excluye viajes con fechas faltantes o duración no positiva.
    Cada extremo se valida por separado para conservar movimientos
    de estaciones válidas aunque el otro extremo no sea modelable.

    Esta función produce conteos observados, no predictores históricos.
    Los intervalos sin eventos no se generan aquí.
    """
    if interval_minutes <= 0:
        raise ValueError("El intervalo debe ser positivo.")

    columns = [
        "started_at",
        "ended_at",
        "origin_station_id",
        "destination_station_id",
    ]
    data = trips.loc[:, columns].copy()

    for column in ("started_at", "ended_at"):
        data[column] = pd.to_datetime(data[column], errors="raise")

    # Regla inicial documentada: conservar solo duración positiva.
    valid_time = (
        data["started_at"].notna()
        & data["ended_at"].notna()
        & (data["ended_at"] > data["started_at"])
    )
    data = data.loc[valid_time]
    counts = []

    # Una salida usa la hora de inicio; una llegada, la hora de fin.
    for station_column, time_column, count_name in (
        ("origin_station_id", "started_at", "departures"),
        ("destination_station_id", "ended_at", "arrivals"),
    ):
        valid_side = data[station_column].isin(valid_station_ids)
        events = data.loc[valid_side, [station_column, time_column]].copy()

        events = events.rename(columns={station_column: "station_id"})
        events["interval_start"] = events[time_column].dt.floor(
            f"{interval_minutes}min"
        )

        grouped = events.groupby(
            ["station_id", "interval_start"]
        ).size().rename(count_name)
        counts.append(grouped)

    # Unimos salidas y llegadas del mismo intervalo.
    result = pd.concat(counts, axis=1).fillna(0)
    result = result.astype({"departures": "int64", "arrivals": "int64"})

    return result.reset_index().sort_values(
        ["station_id", "interval_start"], ignore_index=True
    )


def build_p2_station_examples(
    movements: pd.DataFrame,
    capacity: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Construye predictores históricos y etiqueta para una sola estación.

    start es inclusivo y end exclusivo. Solo deben abarcar cobertura conocida.
    Los ceros representan ausencia de movimientos válidos registrados,
    no disponibilidad real ni confirmación de que la estación estaba abierta.
    """
    import numpy as np

    from mibici.config import P2_ACTIVITY_DAYS, P2_PRESSURE_THRESHOLD

    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("La capacidad debe ser positiva y finita.")
    if movements["station_id"].nunique() != 1:
        raise ValueError("La función requiere exactamente una estación.")
    if movements["interval_start"].duplicated().any():
        raise ValueError("Hay intervalos duplicados para esta estación.")

    station_id = int(movements["station_id"].iloc[0])
    step = pd.Timedelta(minutes=P2_INTERVAL_MINUTES)
    start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)

    if (
        start_time >= end_time
        or start_time != start_time.floor(f"{P2_INTERVAL_MINUTES}min")
        or end_time != end_time.floor(f"{P2_INTERVAL_MINUTES}min")
    ):
        raise ValueError("Los límites deben ser válidos y alineados al intervalo.")

    calendar = pd.date_range(
        start_time,
        end_time,
        freq=step,
        inclusive="left",
    )

    # Cada fila representa el intervalo [t, t + 30 minutos).
    grid = (
        movements.set_index("interval_start")[["departures", "arrivals"]]
        .reindex(calendar, fill_value=0)
        .astype("int64")
    )

    # shift(1) excluye los movimientos del intervalo que vamos a predecir.
    previous = grid.shift(1)
    examples = pd.DataFrame(index=calendar)
    examples["departures_prev_30m"] = previous["departures"]
    examples["arrivals_prev_30m"] = previous["arrivals"]

    two_hours = 120 // P2_INTERVAL_MINUTES
    for column in ("departures", "arrivals"):
        examples[f"{column}_prev_2h"] = previous[column].rolling(
            two_hours, min_periods=two_hours
        ).sum()

    activity_windows = P2_ACTIVITY_DAYS * 24 * 60 // P2_INTERVAL_MINUTES
    historical_activity = grid.sum(axis=1).shift(1).rolling(
        activity_windows, min_periods=activity_windows
    ).sum()

    examples["hour"] = calendar.hour
    examples["minute"] = calendar.minute
    examples["day_of_week"] = calendar.dayofweek
    examples["capacity"] = float(capacity)

    # Estas cantidades futuras se usan exclusivamente para crear la etiqueta.
    pressure = (grid["departures"] - grid["arrivals"]) / capacity
    examples["target"] = np.select(
        [
            pressure >= P2_PRESSURE_THRESHOLD,
            pressure <= -P2_PRESSURE_THRESHOLD,
        ],
        ["presion_sin_bicicletas", "presion_sin_anclajes"],
        default="sin_presion_neta_elevada",
    )

    # Exigimos historia completa y actividad previa, nunca actividad futura.
    eligible = historical_activity.notna() & historical_activity.gt(0)
    examples = examples.loc[eligible].copy()

    examples.insert(0, "station_id", station_id)
    examples.index.name = "prediction_time"
    examples = examples.reset_index()
    examples["target_end"] = examples["prediction_time"] + step
    return examples
