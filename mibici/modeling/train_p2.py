"""Entrena un árbol de P2 y lo evalúa por bloques en toda la validación."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from mibici.config import P2_EXAMPLES_PATH, PROJ_ROOT


FEATURES = [
    "departures_prev_30m", "arrivals_prev_30m",
    "departures_prev_2h", "arrivals_prev_2h",
    "hour", "minute", "day_of_week", "capacity",
]
LABELS = [
    "sin_presion_neta_elevada",
    "presion_sin_bicicletas",
    "presion_sin_anclajes",
]


def parse_args():
    """Permite cambiar la configuración sin editar el código."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--class-weight", choices=["balanced", "none"],
                        default="balanced")
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment", default="mibici-equipo23")
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction debe estar entre 0 y 1.")
    if args.max_depth < 1 or args.min_samples_leaf < 1:
        parser.error("La profundidad y el mínimo por hoja deben ser positivos.")
    return args


def read_batches(split):
    """Lee únicamente la partición pedida, con memoria limitada."""
    scanner = ds.dataset(str(P2_EXAMPLES_PATH), format="parquet").scanner(
        columns=FEATURES + ["target"],
        filter=ds.field("split") == split,
        batch_size=100_000,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if not frame["target"].isin(LABELS).all():
            raise ValueError("Hay etiquetas desconocidas o faltantes.")
        frame[FEATURES] = frame[FEATURES].astype("float64")
        if not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError("Hay variables faltantes o no finitas.")
        yield frame


def sample_training(fraction, seed):
    """Cada fila tiene la misma probabilidad de entrar en la muestra."""
    rng = np.random.default_rng(seed)
    pieces = []
    total = 0
    for frame in read_batches("train"):
        total += len(frame)
        selected = rng.random(len(frame)) < fraction
        if selected.any():
            pieces.append(frame.loc[selected].copy())

    if not pieces:
        raise ValueError("La muestra quedó vacía.")

    sample = pd.concat(pieces, ignore_index=True)
    if set(sample["target"]) != set(LABELS):
        raise ValueError("La muestra no contiene las tres clases.")

    return sample, total


def evaluate(model):
    """Acumula una matriz de confusión sin guardar millones de predicciones."""
    matrix = np.zeros((3, 3), dtype=np.int64)
    for frame in read_batches("validation"):
        predictions = model.predict(frame[FEATURES])
        matrix += confusion_matrix(
            frame["target"], predictions, labels=LABELS
        )

    if (matrix.sum(axis=1) == 0).any():
        raise ValueError("Faltan clases en la validación.")

    # Cada celda representa un par (clase real, predicha) y su frecuencia.
    # Calcular métricas con esas frecuencias equivale a usar todas las filas.
    actual = np.repeat(np.arange(3), 3)
    predicted = np.tile(np.arange(3), 3)
    weights = matrix.ravel()
    precision, recall, _, _ = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2],
        sample_weight=weights, zero_division=0,
    )
    metrics = {
        "val_accuracy": float(accuracy_score(
            actual, predicted, sample_weight=weights)),
        "val_balanced_accuracy": float(balanced_accuracy_score(
            actual, predicted, sample_weight=weights)),
        "val_f1_macro": float(f1_score(
            actual, predicted, labels=[0, 1, 2], average="macro",
            sample_weight=weights, zero_division=0)),
    }
    for i, label in enumerate(LABELS):
        metrics[f"val_precision_{label}"] = float(precision[i])
        metrics[f"val_recall_{label}"] = float(recall[i])
    return matrix, metrics


def file_hash(path):
    """Identifica exactamente el archivo utilizado."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    name = f"P2_arbol_d{args.max_depth}_{args.class_weight}"

    with mlflow.start_run(run_name=name) as run:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        ).strip()
        mlflow.set_tags({
            "equipo": "23", "pregunta": "P2",
            "target_tipo": "proxy_presion_neta",
            "git_commit": commit,
            "working_tree_dirty": str(bool(dirty)).lower(),
        })
        mlflow.log_params(vars(args))
        mlflow.log_params({
            "modelo": "DecisionTreeClassifier",
            "muestreo": "Bernoulli uniforme solo en train",
            "batch_size": 100_000,
            "test_evaluado": False,
            "dataset_sha256": file_hash(P2_EXAMPLES_PATH),
        })

        print("Seleccionando muestra de entrenamiento...", flush=True)
        sample, total = sample_training(args.sample_fraction, args.seed)
        counts = {
            label: int((sample["target"] == label).sum())
            for label in LABELS
        }
        print("Filas seleccionadas:", len(sample), "de", total)
        print("Clases en la muestra:", counts, flush=True)
        mlflow.log_params({
            "train_rows_available": total,
            "train_rows_used": len(sample),
        })

        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=None if args.class_weight == "none" else "balanced",
            random_state=args.seed,
        )
        print("Entrenando árbol...", flush=True)
        model.fit(sample[FEATURES], sample["target"])

        print("Evaluando toda la validación...", flush=True)
        matrix, metrics = evaluate(model)
        mlflow.log_metrics(metrics)
        mlflow.log_param("validation_rows", int(matrix.sum()))

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = {
                "metrics": metrics,
                "labels": LABELS,
                "confusion_matrix": matrix.tolist(),
                "sample_class_counts": counts,
                "features": FEATURES,
                "feature_importances": model.feature_importances_.tolist(),
            }
            (folder / "resultados.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (folder / "reglas_arbol.txt").write_text(
                export_text(model, feature_names=FEATURES),
                encoding="utf-8",
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            ConfusionMatrixDisplay(
                matrix,
                display_labels=["Sin presión alta", "Sin bicicletas", "Sin anclajes"],
            ).plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title("P2 — Árbol — Validación completa")
            fig.tight_layout()
            fig.savefig(folder / "matriz_confusion.png", dpi=150)
            plt.close(fig)
            mlflow.log_artifacts(str(folder), artifact_path="evaluacion")

        # Guardamos el modelo entrenado y un ejemplo de sus entradas.
        print("Guardando modelo en MLflow...", flush=True)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo",
            input_example=sample[FEATURES].head(5),
            serialization_format="cloudpickle",
        )

        for relative in (
            "mibici/modeling/train_p2.py", "mibici/config.py",
            "mibici/features.py", "mibici/build_p2_examples.py", "uv.lock",
        ):
            mlflow.log_artifact(
                str(PROJ_ROOT / relative), artifact_path="codigo"
            )

        print("\nRESULTADOS:")
        print(json.dumps(metrics, indent=2))
        print("\nCORRIDA COMPLETADA:", run.info.run_id)


if __name__ == "__main__":
    main()
