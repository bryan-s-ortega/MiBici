"""Corridas comparables de demanda diaria, con corte temporal 60/20/20."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici.config import P1_DATA_PATH, PROJ_ROOT
from mibici.modeling.train_p1 import FEATURES, file_hash


def evaluate(y_true, y_pred, prefix="val"):
    return {f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_r2": float(r2_score(y_true, y_pred))}


def split_data(frame):
    """Mismos cortes para todas las variantes, prueba reservada."""
    frame = frame.sort_values("fecha").reset_index(drop=True)
    first, second = int(len(frame) * .6), int(len(frame) * .8)
    if first < 2 or second == first or second == len(frame):
        raise ValueError("P1 necesita datos suficientes para tres periodos")
    return frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]


def run_p1(*, model, seed, n_estimators, max_depth, min_samples_leaf):
    if n_estimators < 1 or max_depth < 1 or min_samples_leaf < 1:
        raise ValueError("Hiperparámetros P1 deben ser positivos")
    df = pd.read_csv(P1_DATA_PATH, parse_dates=["fecha"])
    if df[FEATURES + ["viajes", "fecha"]].isna().any().any():
        raise ValueError("Datos P1 incompletos")
    train, val, test = split_data(df)
    X_train, y_train = train[FEATURES], train["viajes"]
    X_val, y_val = val[FEATURES], val["viajes"]
    params = {"model": model, "seed": seed, "n_estimators": n_estimators,
              "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
              "dataset_sha256": file_hash(P1_DATA_PATH), "split": "60/20/20 chronological"}
    with mlflow.start_run(run_name=f"P1_{model}_d{max_depth}_leaf{min_samples_leaf}",
                          nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"question": "P1", "model_family": model,
                         "stage_of_work": "selection", "team": "23",
                         "commit": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()})
        mlflow.log_params(params)
        mlflow.log_input(mlflow.data.from_pandas(
            train[FEATURES + ["viajes"]], source=str(P1_DATA_PATH),
            targets="viajes", name="p1_demanda_diaria"), context="training")
        if model == "baseline":
            predictions = val["lag_7"].to_numpy()
            estimator = None
        else:
            estimator = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=1)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_val)
        metrics = evaluate(y_val, predictions)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "resumen.json").write_text(json.dumps({
                "metricas_validacion": metrics, "filas": {"train": len(train),
                "validation": len(val), "test_reservado": len(test)},
                "cortes": [str(val["fecha"].min()), str(test["fecha"].min())],
            }, indent=2), encoding="utf-8")
            fig, ax = plt.subplots()
            ax.scatter(y_val, predictions, s=8, alpha=.5)
            ax.set(xlabel="Demanda real", ylabel="Demanda predicha", title="P1 validación")
            fig.tight_layout()
            fig.savefig(folder / "real_vs_predicho.png")
            plt.close(fig)
            mlflow.log_artifacts(str(folder), artifact_path="evidencia")
        if estimator is not None:
            mlflow.sklearn.log_model(
                estimator, name="modelo", input_example=X_val.head(5),
                signature=infer_signature(X_val.head(5), estimator.predict(X_val.head(5))),
                serialization_format="cloudpickle")
        print(json.dumps({"run_id": run.info.run_id, "validation": metrics}, indent=2))
        return run.info.run_id


def search_p1(seed: int):
    """Tres variantes RF y lag7; se consulta test solo después de seleccionar."""
    choices = [(100, 4, 5), (200, 6, 8), (300, 9, 12)]
    client = MlflowClient()
    with mlflow.start_run(run_name="P1_busqueda", nested=mlflow.active_run() is not None):
        mlflow.set_tags({"question": "P1", "model_family": "search",
                         "stage_of_work": "hyperparameter_search", "team": "23"})
        baseline_id = run_p1(model="baseline", seed=seed, n_estimators=100,
                             max_depth=4, min_samples_leaf=5)
        candidates = []
        for n, depth, leaf in choices:
            run_id = run_p1(model="rf", seed=seed, n_estimators=n,
                            max_depth=depth, min_samples_leaf=leaf)
            mae = client.get_run(run_id).data.metrics["val_mae"]
            candidates.append((mae, run_id))
        _, winner = min(candidates)
        mlflow.log_param("winner_run_id", winner)
        mlflow.log_param("baseline_run_id", baseline_id)
        # No reentrena: evalúa exactamente el modelo versionado del run ganador.
        frame = pd.read_csv(P1_DATA_PATH, parse_dates=["fecha"])
        _, _, test = split_data(frame)
        model = mlflow.sklearn.load_model(f"runs:/{winner}/modelo")
        test_metrics = evaluate(test["viajes"], model.predict(test[FEATURES]), "test")
        for key, value in test_metrics.items():
            client.log_metric(winner, key, value)
        client.set_tag(winner, "stage_of_work", "selected_and_tested")
        print(json.dumps({"winner_run_id": winner, "test": test_metrics}, indent=2))
        return winner
