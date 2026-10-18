from datetime import datetime, timedelta

import pandas as pd

from mibici.features import daily_station_features
from mibici.ingestion import normalize_column
from mibici.training import train_p1, train_p2, train_p3


def make_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for offset in range(12):
        date = datetime(2026, 1, 1) + timedelta(days=offset)
        for station, count in (("A", 6), ("B", 3), ("C", 2)):
            rows.extend(
                {"Fecha Inicio": date, "ID Estacion Inicio": station} for _ in range(count)
            )
    return pd.DataFrame(rows), pd.DataFrame(
        {"ID Estacion": ["A", "B", "C"], "Capacidad": [10, 10, 10]}
    )


def test_normalize_column() -> None:
    assert normalize_column("Fecha/Hora Inicio") == "fecha_hora_inicio"


def test_features_and_baselines(tmp_path) -> None:
    trips, stations = make_data()
    daily = daily_station_features(trips, stations)
    assert {
        "viajes_lag_1",
        "objetivo_viajes_siguiente",
        "riesgo_alto",
        "riesgo_alto_siguiente",
    } <= set(daily)
    assert train_p1(daily)["n_test"] > 0
    assert train_p2(daily)["n_test"] > 0
    assert train_p3(daily, figures_dir=tmp_path)["n_stations"] == 3
