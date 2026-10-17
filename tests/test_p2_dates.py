"""Comprueba que la ingesta no intercambie día y mes."""

import pandas as pd
import pytest

from mibici.dataset import parse_trip_dates


def test_september_is_not_january():
    result = parse_trip_dates(pd.Series([
        "01/09/2024 0:00",
        "13/09/2024 10:15:30",
        "2024-09-01 00:00:00",
    ]))

    assert result.iloc[0] == pd.Timestamp("2024-09-01 00:00:00")
    assert result.iloc[1] == pd.Timestamp("2024-09-13 10:15:30")
    assert result.iloc[2] == pd.Timestamp("2024-09-01 00:00:00")


def test_trip_can_cross_month_boundary():
    result = parse_trip_dates(pd.Series([
        "30/09/2024 23:55",
        "01/10/2024 0:10",
    ]))

    assert result.iloc[1] - result.iloc[0] == pd.Timedelta(minutes=15)


@pytest.mark.parametrize("value", [
    "31/02/2024 10:00",
    "09/30/2024 10:00",
    "fecha desconocida",
    None,
])
def test_invalid_or_unsupported_dates_are_rejected(value):
    with pytest.raises(ValueError):
        parse_trip_dates(pd.Series([value]))
