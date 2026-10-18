"""Feature engineering común para P1, P2 y P3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RISK_CAPACITY_RATIO
from mibici.ingestion import normalize_column


DATE_COLUMNS = (
    "fecha_inicio",
    "fecha_hora_inicio",
    "start_time",
    "inicio",
    "inicio_del_viaje",
)
START_STATION_COLUMNS = (
    "id_estacion_inicio",
    "estacion_inicio",
    "start_station_id",
    "origen",
    "origen_id",
)


def daily_station_features(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    risk_ratio: float = RISK_CAPACITY_RATIO,
) -> pd.DataFrame:
    trips = trips.copy()
    stations = stations.copy()
    trips.columns = [normalize_column(column) for column in trips.columns]
    stations.columns = [normalize_column(column) for column in stations.columns]
    date_col = next((column for column in DATE_COLUMNS if column in trips), None)
    station_col = next((column for column in START_STATION_COLUMNS if column in trips), None)
    if date_col is None or station_col is None:
        raise KeyError(f"No se encontraron fecha y estación de inicio: {trips.columns.tolist()}")

    base = trips[[date_col, station_col]].rename(
        columns={date_col: "fecha_hora", station_col: "estacion_id"}
    )
    base["fecha"] = pd.to_datetime(base["fecha_hora"], errors="coerce").dt.floor("D")
    base["estacion_id"] = base["estacion_id"].astype("string")
    base = base.dropna(subset=["fecha", "estacion_id"])
    daily = (
        base.groupby(["fecha", "estacion_id"], as_index=False)
        .size()
        .rename(columns={"size": "viajes_iniciados"})
        .sort_values(["estacion_id", "fecha"])
    )
    grouped = daily.groupby("estacion_id")["viajes_iniciados"]
    daily["viajes_lag_1"] = grouped.shift(1)
    daily["viajes_lag_7"] = grouped.shift(7)
    daily["media_7_dias"] = grouped.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    daily["objetivo_viajes_siguiente"] = grouped.shift(-1)
    daily["dia_semana"] = daily["fecha"].dt.dayofweek
    daily["mes"] = daily["fecha"].dt.month

    capacity_col = next(
        (
            column
            for column in ("capacidad", "capacity", "capacidad_estacion", "dpcapacity")
            if column in stations
        ),
        None,
    )
    station_id_col = next(
        (column for column in ("id_estacion", "estacion_id", "station_id", "id") if column in stations),
        None,
    )
    if capacity_col and station_id_col:
        station_columns = [station_id_col, capacity_col]
        for coordinate in ("latitude", "longitude", "latitud", "longitud"):
            if coordinate in stations and coordinate not in station_columns:
                station_columns.append(coordinate)
        capacity = stations[station_columns].rename(
            columns={
                station_id_col: "estacion_id",
                capacity_col: "capacidad",
                "latitud": "latitude",
                "longitud": "longitude",
            }
        )
        capacity["estacion_id"] = capacity["estacion_id"].astype("string")
        daily = daily.merge(capacity, on="estacion_id", how="left")
        for coordinate in ("latitude", "longitude"):
            if coordinate not in daily:
                daily[coordinate] = pd.NA
        daily["riesgo_alto"] = (
            daily["viajes_lag_1"] / daily["capacidad"] >= risk_ratio
        ).astype("Int64")
        daily["riesgo_alto_siguiente"] = (
            daily["objetivo_viajes_siguiente"] / daily["capacidad"] >= risk_ratio
        ).astype("Int64")
    else:
        daily["capacidad"] = pd.NA
        daily["latitude"] = pd.NA
        daily["longitude"] = pd.NA
        daily["riesgo_alto"] = pd.NA
        daily["riesgo_alto_siguiente"] = pd.NA
    return daily


def station_features(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.groupby("estacion_id").agg(
        viajes_promedio=("viajes_iniciados", "mean"),
        viajes_desviacion=("viajes_iniciados", "std"),
        dias_observados=("fecha", "nunique"),
        capacidad=("capacidad", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.fillna(0)
