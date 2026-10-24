"""Experimentos P2: presión neta como proxy, nunca disponibilidad física observada."""

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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier

from mibici.config import P2_EXAMPLES_PATH, PROJ_ROOT
from mibici.modeling.train_p2 import FEATURES, LABELS, file_hash, read_batches, sample_training


def metrics_from_matrix(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    """Calcula las métricas sin materializar millones de predicciones."""
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    if not np.all(support):
        raise ValueError("Hay clases sin ejemplos en la partición")
    tp = np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=(precision + recall) > 0)
    metrics = {
        f"{prefix}_accuracy": float(tp.sum() / matrix.sum()),
        f"{prefix}_balanced_accuracy": float(recall.mean()),
        f"{prefix}_f1_macro": float(f1.mean()),
    }
    for i, label in enumerate(LABELS):
        metrics[f"{prefix}_precision_{label}"] = float(precision[i])
        metrics[f"{prefix}_recall_{label}"] = float(recall[i])
    return metrics


def evaluate_stream(model, split: str):
    """Confusión exacta y curva PR por histograma de probabilidad de 200 bins."""
    matrix = np.zeros((3, 3), dtype=np.int64)
    positives = np.zeros((2, 200), dtype=np.int64)
    negatives = np.zeros((2, 200), dtype=np.int64)
    model_labels = list(model.classes_)
    indices = [model_labels.index(label) for label in LABELS]
    for frame in read_batches(split):
        actual = frame["target"].to_numpy()
        predicted = model.predict(frame[FEATURES])
        actual_ix = np.select([actual == label for label in LABELS], [0, 1, 2])
        pred_ix = np.select([predicted == label for label in LABELS], [0, 1, 2])
        np.add.at(matrix, (actual_ix, pred_ix), 1)
        probabilities = model.predict_proba(frame[FEATURES])[:, indices]
        for target_ix in (1, 2):
            bins = np.minimum((probabilities[:, target_ix] * 200).astype(int), 199)
            positives[target_ix - 1] += np.bincount(
                bins[actual_ix == target_ix], minlength=200
            )
            negatives[target_ix - 1] += np.bincount(
                bins[actual_ix != target_ix], minlength=200
            )
    return matrix, positives, negatives


def eligible(metrics: dict[str, float]) -> bool:
    """Evita elegir por accuracy un clasificador que ignore una clase de presión."""
    return all(metrics[f"val_recall_{label}"] >= 0.1 for label in LABELS[1:])


def make_artifacts(folder: Path, split: str, matrix, pos, neg, metrics) -> None:
    (folder / f"metricas_{split}.json").write_text(
        json.dumps({"metricas": metrics, "clases": LABELS, "matriz": matrix.tolist()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 6))
    ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Sin bicis", "Sin anclajes"]).plot(
        ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(folder / f"matriz_{split}.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ix, ax in enumerate(axes):
        # Una observación ponderada por bin y por clase. La curva es aproximada;
        # la matriz y las métricas de arriba son exactas sobre la partición.
        counts = np.r_[pos[ix], neg[ix]]
        scores = np.tile((np.arange(200) + .5) / 200, 2)
        binary = np.r_[np.ones(200), np.zeros(200)]
        precision, recall, _ = precision_recall_curve(
            binary, scores, sample_weight=counts)
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", title=LABELS[ix + 1],
               xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(folder / f"curvas_pr_{split}_aproximadas.png", dpi=120)
    plt.close(fig)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
    ).strip()


def fit_log_run(sample, train_rows: int, digest: str, *, model: str, seed: int,
                max_depth: int, min_samples_leaf: int, class_weight: str):
    """Una corrida hija (o individual) que registra modelo, linaje y validación."""
    params = {"model": model, "seed": seed, "max_depth": max_depth,
              "min_samples_leaf": min_samples_leaf, "class_weight": class_weight}
    with mlflow.start_run(run_name=f"P2_{model}_d{max_depth}_{class_weight}",
                          nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"question": "P2", "model_family": model,
                         "stage_of_work": "selection", "team": "23",
                         "commit": _commit(), "label": "proxy_presion_neta_no_disponibilidad"})
        mlflow.log_params({**params, "train_rows_available": train_rows,
                           "train_rows_sampled": len(sample), "dataset_sha256": digest,
                           "split": "train<2026-01; val<2026-07; test>=2026-07"})
        mlflow.log_input(mlflow.data.from_pandas(
            sample[FEATURES + ["target"]].head(1000),
            source=str(P2_EXAMPLES_PATH), targets="target", name="p2_ejemplos_muestra_esquema"
        ), context="training")
        if model == "baseline":
            estimator = DummyClassifier(strategy="most_frequent")
        else:
            estimator = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                class_weight=None if class_weight == "none" else "balanced",
                random_state=seed)
        estimator.fit(sample[FEATURES], sample["target"])
        if set(estimator.classes_) != set(LABELS):
            raise ValueError("Faltan clases al entrenar")
        matrix, pos, neg = evaluate_stream(estimator, "validation")
        metrics = metrics_from_matrix(matrix, "val")
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "configuracion.json").write_text(
                json.dumps(params, indent=2), encoding="utf-8")
            (folder / "manifiesto.json").write_text(json.dumps({
                "archivo": str(P2_EXAMPLES_PATH),
                "sha256": digest, "filas_train_disponibles": train_rows,
                "filas_train_muestra": len(sample), "filas_validacion": int(matrix.sum()),
                "target": "proxy de presión neta, no inventario observado",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            make_artifacts(folder, "validation", matrix, pos, neg, metrics)
            mlflow.log_artifacts(str(folder), artifact_path="evidencia")
        model_info = mlflow.sklearn.log_model(
            sk_model=estimator, name="modelo",
            signature=infer_signature(sample[FEATURES].head(5),
                                      estimator.predict(sample[FEATURES].head(5))),
            input_example=sample[FEATURES].head(5),
            serialization_format="cloudpickle")
        print(json.dumps({"run_id": run.info.run_id, "metrics": metrics}, indent=2))
        return {"run_id": run.info.run_id, "metrics": metrics,
                "model_uri": model_info.model_uri, "estimator": estimator}


def run_p2(*, model: str, seed: int, sample_fraction: float, max_depth: int,
           min_samples_leaf: int, class_weight: str):
    if not 0 < sample_fraction <= 1 or max_depth < 1 or min_samples_leaf < 1:
        raise ValueError("Parámetros de muestreo/árbol inválidos")
    sample, train_rows = sample_training(sample_fraction, seed)
    return fit_log_run(sample, train_rows, file_hash(P2_EXAMPLES_PATH),
                       model=model, seed=seed, max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf, class_weight=class_weight)


def search_p2(*, seed: int, sample_fraction: float):
    """Tres candidatos y baseline; prueba una sola vez tras escoger con validación."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction debe estar en (0, 1]")
    sample, train_rows = sample_training(sample_fraction, seed)
    digest = file_hash(P2_EXAMPLES_PATH)
    choices = [("baseline", 1, 100, "none"),
               ("tree", 3, 100, "balanced"),
               ("tree", 5, 100, "balanced"),
               ("tree", 7, 200, "balanced")]
    with mlflow.start_run(run_name="P2_busqueda", nested=mlflow.active_run() is not None):
        mlflow.set_tags({"question": "P2", "model_family": "search",
                         "stage_of_work": "hyperparameter_search", "team": "23",
                         "commit": _commit()})
        runs = [fit_log_run(sample, train_rows, digest, model=m, seed=seed,
                            max_depth=d, min_samples_leaf=l, class_weight=w)
                for m, d, l, w in choices]
        candidates = [run for run in runs if run["metrics"]["val_recall_" + LABELS[1]] >= .1
                      and eligible(run["metrics"])]
        if not candidates:
            raise ValueError("Ningún candidato reconoce ambas clases de presión")
        winner = max(candidates, key=lambda run: run["metrics"]["val_balanced_accuracy"])
        mlflow.log_param("winner_run_id", winner["run_id"])
        mlflow.log_metric("best_val_balanced_accuracy",
                          winner["metrics"]["val_balanced_accuracy"])
        # Se abre test solo después de la elección, sin reoptimizar umbrales.
        matrix, pos, neg = evaluate_stream(winner["estimator"], "test")
        metrics = metrics_from_matrix(matrix, "test")
        client = MlflowClient()
        client.set_tag(winner["run_id"], "stage_of_work", "selected_and_tested")
        for key, value in metrics.items():
            client.log_metric(winner["run_id"], key, value)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_artifacts(folder, "test", matrix, pos, neg, metrics)
            for artifact in folder.iterdir():
                client.log_artifact(winner["run_id"], str(artifact), artifact_path="evidencia")
        print(json.dumps({"winner_run_id": winner["run_id"],
                          "model_uri": winner["model_uri"],
                          "validation": winner["metrics"], "test": metrics}, indent=2))
        return winner["run_id"]
