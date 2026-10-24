"""Verifica el contrato del proyecto y que P2 no consulte test al seleccionar."""

import mlflow
from mlflow.projects import load_project
import pandas as pd
import pytest

from mibici.modeling import project_p1, project_p2, train_p2, train_p3
from mibici.modeling.project_p1 import split_data


def test_entries_exist():
    project = load_project(".")
    for name in ("ingest", "train", "search"):
        assert project.get_entry_point(name) is not None


def test_p1_temporal_splits_do_not_overlap():
    frame = pd.DataFrame({"fecha": pd.date_range("2025-01-01", periods=10)})
    train, validation, test = split_data(frame)
    assert list(map(len, (train, validation, test))) == [6, 2, 2]
    assert train.fecha.max() < validation.fecha.min() < test.fecha.min()


def test_p2_eligibility_requires_both_pressure_recall():
    assert not project_p2.eligible({
        "val_recall_presion_sin_bicicletas": 0.0,
        "val_recall_presion_sin_anclajes": 1.0,
    })


@pytest.mark.filterwarnings("ignore:The specified dataset source")
def test_p2_run_logs_model_and_metrics(tmp_path, monkeypatch):
    path = tmp_path / "examples.parquet"
    rows = []
    for split in ("train", "validation", "test"):
        for index, target in enumerate(project_p2.LABELS):
            for row in range(12):
                rows.append({**{column: float(index * 5 + row % 2)
                                for column in project_p2.FEATURES},
                             "target": target, "split": split})
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(project_p2, "P2_EXAMPLES_PATH", path)
    monkeypatch.setattr(train_p2, "P2_EXAMPLES_PATH", path)
    mlflow.set_tracking_uri("sqlite:///" + str(tmp_path / "mlflow.db"))
    mlflow.set_experiment("test-project-p2")
    try:
        result = project_p2.run_p2(
            model="tree", seed=42, sample_fraction=1.0,
            max_depth=3, min_samples_leaf=1, class_weight="balanced")
        assert result["metrics"]["val_balanced_accuracy"] == 1.0
        assert "test_balanced_accuracy" not in mlflow.get_run(result["run_id"]).data.metrics
        model = mlflow.sklearn.load_model(result["model_uri"])
        assert len(model.predict(pd.DataFrame(rows).iloc[:1][project_p2.FEATURES])) == 1
    finally:
        mlflow.end_run()


def test_p3_logged_pipeline_accepts_raw_features(tmp_path, monkeypatch):
    trips_path = tmp_path / "trips.parquet"
    stations_path = tmp_path / "stations.csv"
    pd.DataFrame({"started_at": pd.date_range("2026-01-01", periods=6),
                  "origin_station_id": [1, 2, 3, 4, 5, 6],
                  "destination_station_id": [2, 3, 4, 5, 6, 1]}).to_parquet(trips_path)
    pd.DataFrame({"id": [1, 2, 3, 4, 5, 6]}).to_csv(stations_path, index=False)
    columns = list(dict.fromkeys(train_p3.BASELINE_FEATURES + train_p3.CANDIDATE_FEATURES))
    values = {column: [float(i + 1 + j * .1) for i in range(6)]
              for j, column in enumerate(columns)}
    values.update({"latitude": [20.67 + i * .01 for i in range(6)],
                   "longitude": [-103.35 - i * .01 for i in range(6)],
                   "capacidad": [15 + i for i in range(6)],
                   "nombre": [f"estacion {i}" for i in range(6)]})
    features = pd.DataFrame(values)
    monkeypatch.setattr(train_p3, "P3_TRIPS_PATH", trips_path)
    monkeypatch.setattr(train_p3, "P3_STATIONS_PATH", stations_path)
    monkeypatch.setattr(train_p3, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(train_p3, "extract_p3_station_features", lambda trips, nom: features)
    uri = "sqlite:///" + str(tmp_path / "mlflow_p3.db")
    train_p3.train_p3(model_type="candidato", algorithm="kmeans", k=2,
                      tracking_uri=uri, experiment="test-project-p3", sweep_k=False)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("test-project-p3")
    assert client.search_runs([exp.experiment_id])
    models = client.search_logged_models(experiment_ids=[exp.experiment_id])
    model = mlflow.sklearn.load_model(models[0].model_uri)
    assert "scale" in model.named_steps
    assert len(model.predict(features[train_p3.CANDIDATE_FEATURES].head(1))) == 1


def test_p1_search_evaluates_only_winner_on_test(tmp_path, monkeypatch):
    path = tmp_path / "demand.csv"
    frame = pd.DataFrame({"fecha": pd.date_range("2025-01-01", periods=100)})
    frame["dia_semana"] = frame.fecha.dt.dayofweek
    frame["mes"] = frame.fecha.dt.month
    frame["fin_semana"] = (frame.dia_semana >= 5).astype(int)
    frame["lag_1"] = 20 + frame.dia_semana
    frame["lag_7"] = 20 + frame.dia_semana
    frame["viajes"] = 20 + frame.dia_semana + frame.fin_semana
    frame.to_csv(path, index=False)
    monkeypatch.setattr(project_p1, "P1_DATA_PATH", path)
    mlflow.set_tracking_uri("sqlite:///" + str(tmp_path / "mlflow_p1.db"))
    mlflow.set_experiment("test-project-p1")
    winner = project_p1.search_p1(42)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-project-p1")
    children = [run for run in client.search_runs([experiment.experiment_id])
                if run.data.tags.get("model_family") == "rf"]
    assert len(children) == 3
    assert {run.info.run_id for run in children if "test_mae" in run.data.metrics} == {winner}
