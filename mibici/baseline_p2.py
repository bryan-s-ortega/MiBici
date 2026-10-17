"""Baseline mayoritario de P2: entrenamiento y evaluación sin cargar todo el dataset."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")  # Guarda gráficos sin necesitar una ventana.
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import ConfusionMatrixDisplay

from mibici.config import P2_EXAMPLES_PATH, PROJ_ROOT


LABELS = [
    "sin_presion_neta_elevada",
    "presion_sin_bicicletas",
    "presion_sin_anclajes",
]


def count_classes(path: Path, split: str) -> np.ndarray:
    """Cuenta todas las etiquetas de una partición mediante bloques pequeños."""
    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(
        columns=["target"],
        filter=ds.field("split") == split,
        batch_size=200_000,
    )

    counts = Counter()
    for batch in scanner.to_batches():
        counts.update(batch.column(0).to_pylist())

    unknown = set(counts).difference(LABELS)
    if unknown:
        raise ValueError(f"Etiquetas desconocidas o faltantes: {unknown}")

    result = np.array([counts[label] for label in LABELS], dtype=np.int64)
    if (result == 0).any():
        raise ValueError(f"Hay clases sin ejemplos en {split}: {counts}")

    return result


def evaluate(train_counts: np.ndarray, val_counts: np.ndarray):
    """Elige la clase con entrenamiento y evalúa exactamente en validación."""
    majority_index = int(train_counts.argmax())

    # Filas = clase real. Columnas = clase predicha.
    # Como siempre predecimos lo mismo, solo una columna tendrá valores.
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    matrix[:, majority_index] = val_counts

    true_positives = matrix.diagonal().astype(float)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    recall = true_positives / actual
    precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros(len(LABELS)),
        where=predicted != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(LABELS)),
        where=(precision + recall) != 0,
    )

    metrics = {
        "val_accuracy": float(true_positives.sum() / matrix.sum()),
        "val_balanced_accuracy": float(recall.mean()),
        "val_f1_macro": float(f1.mean()),
    }
    for label, score in zip(LABELS, recall):
        metrics[f"val_recall_{label}"] = float(score)

    return LABELS[majority_index], matrix, metrics


def sha256_file(path: Path) -> str:
    """Obtiene una huella del archivo para identificar la versión evaluada."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("mibici-equipo23")

    with mlflow.start_run(run_name="P2_baseline_mayoritario") as run:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        ).strip()

        mlflow.set_tags({
            "equipo": "23",
            "pregunta": "P2",
            "target_tipo": "proxy_presion_neta",
            "git_commit": commit,
            "working_tree_dirty": str(bool(status)).lower(),
        })

        print("Contando clases de entrenamiento...", flush=True)
        train_counts = count_classes(P2_EXAMPLES_PATH, "train")

        print("Contando clases de validación...", flush=True)
        val_counts = count_classes(P2_EXAMPLES_PATH, "validation")

        majority, matrix, metrics = evaluate(train_counts, val_counts)

        print("Calculando huella del dataset...", flush=True)
        mlflow.log_params({
            "modelo": "regla_clase_mayoritaria",
            "clase_predicha": majority,
            "train_rows": int(train_counts.sum()),
            "validation_rows": int(val_counts.sum()),
            "dataset_sha256": sha256_file(P2_EXAMPLES_PATH),
            "aleatoriedad": "ninguna",
            "test_evaluado": False,
        })
        mlflow.log_metrics(metrics)

        # Guardamos la regla completa y su evaluación como evidencia.
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = {
                "regla": "Predecir siempre la clase mayoritaria del entrenamiento",
                "clase_predicha": majority,
                "orden_clases": LABELS,
                "train_counts": train_counts.tolist(),
                "validation_counts": val_counts.tolist(),
                "confusion_matrix": matrix.tolist(),
                "metrics": metrics,
            }
            report_path = folder / "baseline_resultados.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path), artifact_path="evaluacion")

            fig, ax = plt.subplots(figsize=(9, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Sin presión alta", "Sin bicicletas", "Sin anclajes"],
            ).plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title("P2 — Baseline mayoritario — Validación")
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")
            fig.tight_layout()
            image_path = folder / "matriz_confusion.png"
            fig.savefig(image_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(image_path), artifact_path="evaluacion")

        # Conservamos el código utilizado, incluso si aún no está en un commit.
        for relative in (
            "mibici/baseline_p2.py",
            "mibici/config.py",
            "mibici/features.py",
            "mibici/build_p2_examples.py",
            "uv.lock",
        ):
            path = PROJ_ROOT / relative
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="codigo")

        print("\nClase predicha:", majority)
        print(json.dumps(metrics, indent=2))
        print("\nCORRIDA COMPLETADA:", run.info.run_id)


if __name__ == "__main__":
    main()
