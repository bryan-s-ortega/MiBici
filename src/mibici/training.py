"""Entrenamiento de baselines P1, P2 y P3 con registro opcional en MLflow."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_MANIFEST,
    END_DATE,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    ROOT,
    SEED,
    START_DATE,
    TEAM_ID,
)
from mibici.features import daily_station_features, station_features
from mibici.ingestion import load_raw


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "sin-git"


def data_version() -> str:
    """Return the SHA-256 of the committed manifest for the raw data snapshot."""
    if not DATA_MANIFEST.exists():
        return "manifest-no-encontrado"
    return hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest()


def train_p1(daily: pd.DataFrame) -> dict[str, float]:
    data = daily.dropna(subset=["objetivo_viajes_siguiente", "viajes_lag_1"]).copy()
    cutoff = data["fecha"].quantile(0.8)
    train, test = data[data["fecha"] < cutoff], data[data["fecha"] >= cutoff]
    means = train.groupby("estacion_id")["objetivo_viajes_siguiente"].mean()
    prediction = test["estacion_id"].map(means).fillna(train["objetivo_viajes_siguiente"].mean())
    mae = mean_absolute_error(test["objetivo_viajes_siguiente"], prediction)
    return {"mae": float(mae), "n_train": float(len(train)), "n_test": float(len(test))}


def train_p1_candidate(daily: pd.DataFrame) -> dict[str, float]:
    data = daily.dropna(
        subset=[
            "objetivo_viajes_siguiente",
            "viajes_lag_1",
            "viajes_lag_7",
            "media_7_dias",
        ]
    )
    cutoff = data["fecha"].quantile(0.8)
    train, test = data[data["fecha"] < cutoff], data[data["fecha"] >= cutoff]
    columns = ["viajes_lag_1", "viajes_lag_7", "media_7_dias", "dia_semana", "mes"]
    model = HistGradientBoostingRegressor(random_state=SEED, max_iter=100)
    model.fit(train[columns], train["objetivo_viajes_siguiente"])
    prediction = model.predict(test[columns])
    return {
        "mae": float(mean_absolute_error(test["objetivo_viajes_siguiente"], prediction)),
        "rmse": float(
            mean_squared_error(test["objetivo_viajes_siguiente"], prediction) ** 0.5
        ),
        "n_train": float(len(train)),
        "n_test": float(len(test)),
    }


def train_p2(daily: pd.DataFrame) -> dict[str, float]:
    data = daily.dropna(subset=["riesgo_alto_siguiente"]).copy()
    if data.empty:
        return {"accuracy": float("nan"), "f1": float("nan"), "n_test": 0.0}
    cutoff = data["fecha"].quantile(0.8)
    train, test = data[data["fecha"] < cutoff], data[data["fecha"] >= cutoff]
    majority = int(train["riesgo_alto_siguiente"].mode().iloc[0])
    prediction = np.full(len(test), majority)
    return {
        "accuracy": float(accuracy_score(test["riesgo_alto_siguiente"], prediction)),
        "f1": float(f1_score(test["riesgo_alto_siguiente"], prediction, zero_division=0)),
        "n_test": float(len(test)),
    }


def train_p2_candidate(daily: pd.DataFrame) -> dict[str, float]:
    data = daily.dropna(
        subset=["riesgo_alto_siguiente", "viajes_lag_1", "viajes_lag_7", "media_7_dias"]
    )
    cutoff = data["fecha"].quantile(0.8)
    train, test = data[data["fecha"] < cutoff], data[data["fecha"] >= cutoff]
    columns = ["viajes_lag_1", "viajes_lag_7", "media_7_dias", "dia_semana", "mes"]
    model = LogisticRegression(class_weight="balanced", max_iter=300, random_state=SEED)
    model.fit(train[columns], train["riesgo_alto_siguiente"])
    prediction = model.predict(test[columns])
    return {
        "accuracy": float(accuracy_score(test["riesgo_alto_siguiente"], prediction)),
        "f1": float(f1_score(test["riesgo_alto_siguiente"], prediction, zero_division=0)),
        "n_test": float(len(test)),
    }


def train_p3(daily: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> dict[str, float]:
    features = station_features(daily)
    if len(features) < 3:
        return {"silhouette": float("nan"), "n_stations": float(len(features))}
    matrix = StandardScaler().fit_transform(features)
    k = min(4, len(features) - 1)
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(matrix)
    score = silhouette_score(matrix, labels)
    features = features.assign(cluster=labels)
    figures_dir.mkdir(parents=True, exist_ok=True)
    features.groupby("cluster").mean(numeric_only=True).plot(kind="bar", figsize=(9, 5))
    plt.tight_layout()
    plt.savefig(figures_dir / "p3_perfiles_clusters.png", dpi=150)
    plt.close()
    return {"silhouette": float(score), "n_stations": float(len(features)), "n_clusters": float(k)}


def write_metrics_artifact(question: str, metrics: dict[str, float]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{question.lower()}_baseline_metrics.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return path


def log_run(question: str, metrics: dict[str, float], artifact: Path | None = None) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow no está instalado; resultados calculados sin registro.")
        return
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{question}-baseline"):
        mlflow.log_params(
            {
                "seed": SEED,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "data_version": data_version(),
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if np.isfinite(value)})
        mlflow.set_tags({"team": TEAM_ID, "question": question, "git_commit": git_commit()})
        if artifact and artifact.exists():
            mlflow.log_artifact(str(artifact))


def run_pipeline(raw_dir: Path) -> None:
    trips, stations = load_raw(raw_dir)
    daily = daily_station_features(trips, stations)
    results = {
        "P1": train_p1(daily),
        "P1-candidato": train_p1_candidate(daily),
        "P2": train_p2(daily),
        "P2-candidato": train_p2_candidate(daily),
        "P3": train_p3(daily),
    }
    for question, metrics in results.items():
        artifact = write_metrics_artifact(question, metrics)
        if question == "P3":
            artifact = FIGURES_DIR / "p3_perfiles_clusters.png"
        log_run(question, metrics, artifact)
        print(question, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    run_pipeline(args.raw_dir) if args.raw_dir else run_pipeline(Path("data/raw"))


if __name__ == "__main__":
    main()
