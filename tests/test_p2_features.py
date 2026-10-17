"""Pruebas de separación temporal y construcción de etiquetas de P2."""

import pandas as pd

from mibici.features import build_p2_station_examples


def sample_movements():
    """Ocho días de una estación ficticia con movimientos conocidos."""
    calendar = pd.date_range(
        "2026-01-01",
        "2026-01-09",
        freq="30min",
        inclusive="left",
    )
    return pd.DataFrame({
        "station_id": 2,
        "interval_start": calendar,
        "departures": 1,
        "arrivals": 1,
    })


def build(data):
    return build_p2_station_examples(
        data,
        capacity=10,
        start="2026-01-01",
        end="2026-01-09",
    ).set_index("prediction_time")


def test_future_changes_label_but_not_present_predictors():
    data = sample_movements()
    moment = pd.Timestamp("2026-01-08 10:00")
    original = build(data)

    changed_data = data.copy()
    # Cambiamos movimientos desde el instante de predicción en adelante.
    future = changed_data["interval_start"].ge(moment)
    changed_data.loc[future, "departures"] = 9
    changed_data.loc[future, "arrivals"] = 0
    changed = build(changed_data)

    predictors = [
        "departures_prev_30m",
        "arrivals_prev_30m",
        "departures_prev_2h",
        "arrivals_prev_2h",
        "hour",
        "minute",
        "day_of_week",
        "capacity",
    ]
    pd.testing.assert_series_equal(
        original.loc[moment, predictors],
        changed.loc[moment, predictors],
    )
    assert original.loc[moment, "target"] == "sin_presion_neta_elevada"
    assert changed.loc[moment, "target"] == "presion_sin_bicicletas"


def test_missing_interval_is_not_replaced_by_older_interval():
    data = sample_movements()
    # Quitamos las 09:30: según el supuesto del baseline, cuenta como cero.
    data = data.loc[
        data["interval_start"].ne(pd.Timestamp("2026-01-08 09:30"))
    ]
    result = build(data)
    row = result.loc[pd.Timestamp("2026-01-08 10:00")]

    assert row["departures_prev_30m"] == 0
    assert row["arrivals_prev_30m"] == 0
    assert row["departures_prev_2h"] == 3
    assert row["arrivals_prev_2h"] == 3


def test_history_and_target_boundaries():
    result = build(sample_movements())

    # Exigimos siete días completos de historia.
    assert result.index.min() == pd.Timestamp("2026-01-08")
    assert result["target_end"].max() == pd.Timestamp("2026-01-09")
    assert (
        result["target_end"] - result.index == pd.Timedelta(minutes=30)
    ).all()


def test_pressure_threshold_in_both_directions():
    data = sample_movements()
    first = pd.Timestamp("2026-01-08 10:00")
    second = pd.Timestamp("2026-01-08 10:30")

    # Tres movimientos netos / diez anclajes = 30%.
    data.loc[data["interval_start"].eq(first), ["departures", "arrivals"]] = [3, 0]
    data.loc[data["interval_start"].eq(second), ["departures", "arrivals"]] = [0, 3]
    result = build(data)

    assert result.loc[first, "target"] == "presion_sin_bicicletas"
    assert result.loc[second, "target"] == "presion_sin_anclajes"


def test_inactive_station_is_not_eligible():
    data = sample_movements()
    data[["departures", "arrivals"]] = 0

    assert build(data).empty
