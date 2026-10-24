"""Entrada común para las tres preguntas; no inventa ni sustituye datos faltantes."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from mibici.config import (
    P1_DATA_PATH,
    P2_EXAMPLES_PATH,
    P2_STATIONS_PATH,
    P2_TRIPS_PATH,
    P3_STATIONS_PATH,
    P3_TRIPS_PATH,
)

EXPERIMENT = "mibici-equipo23-mlprojects"
INPUTS = {
    "P1": (P1_DATA_PATH,),
    "P2": (P2_EXAMPLES_PATH,),
    "P3": (P3_TRIPS_PATH, P3_STATIONS_PATH),
}


def check_inputs(question: str) -> None:
    missing = [str(path) for path in INPUTS[question] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan datos de {question}: {', '.join(missing)}. "
            "Ejecuta 'dvc pull' o la preparación documentada en README; "
            "nunca se genera un dataset alternativo silenciosamente."
        )


def ingest(question: str, build: str) -> None:
    """Valida insumos existentes o construye explícitamente los ejemplos P2."""
    if build not in {"true", "false"}:
        raise ValueError("--build debe ser true o false")
    if build == "true":
        if question != "P2":
            raise ValueError("Construcción integrada solo disponible para P2")
        for path in (P2_TRIPS_PATH, P2_STATIONS_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Falta el insumo corregido de P2: {path}")
        from mibici.build_p2_examples import main as build_examples
        from mibici.build_p2_movements import main as build_movements

        build_movements()
        build_examples()
    check_inputs(question)
    for path in INPUTS[question]:
        print(f"{question} OK: {path} ({Path(path).stat().st_size:,} bytes)")


def train(args: argparse.Namespace) -> None:
    check_inputs(args.question)
    # El proyecto recibe MLFLOW_TRACKING_URI del comando mlflow run. Sin ese
    # valor MLflow utiliza el backend local configurado por el usuario.
    mlflow.set_experiment(EXPERIMENT)
    if args.question == "P2":
        if args.model not in {"tree", "baseline"}:
            raise ValueError("P2 admite tree o baseline")
        from mibici.modeling.project_p2 import run_p2

        run_p2(
            model=args.model,
            seed=args.seed,
            sample_fraction=args.sample_fraction,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
        )
    elif args.question == "P1":
        if args.model not in {"rf", "baseline"}:
            raise ValueError("P1 admite rf o baseline")
        from mibici.modeling.project_p1 import run_p1

        run_p1(
            model=args.model,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
    else:
        if args.model not in {"kmeans", "agglomerative"}:
            raise ValueError("P3 admite kmeans o agglomerative")
        from mibici.modeling.train_p3 import train_p3

        train_p3(
            model_type="candidato",
            algorithm=args.model,
            k=args.k,
            seed=args.seed,
            tracking_uri=mlflow.get_tracking_uri(),
            experiment=EXPERIMENT,
            sweep_k=False,
        )


def search(args: argparse.Namespace) -> None:
    check_inputs(args.question)
    mlflow.set_experiment(EXPERIMENT)
    if args.question == "P2":
        from mibici.modeling.project_p2 import search_p2

        search_p2(seed=args.seed, sample_fraction=args.sample_fraction)
    elif args.question == "P1":
        from mibici.modeling.project_p1 import search_p1

        search_p1(seed=args.seed)
    else:
        from mibici.modeling.train_p3 import train_p3

        with mlflow.start_run(run_name="P3_busqueda",
                              nested=mlflow.active_run() is not None):
            mlflow.set_tags({"question": "P3", "model_family": "search",
                             "stage_of_work": "hyperparameter_search", "team": "23"})
            results = [train_p3(
                model_type="candidato", algorithm="kmeans", k=k, seed=args.seed,
                tracking_uri=mlflow.get_tracking_uri(), experiment=EXPERIMENT,
                sweep_k=False) for k in (3, 4, 5)]
            winner = max(results, key=lambda item: item["silhouette_score"])
            mlflow.log_param("winner_run_id", winner["run_id"])
            mlflow.log_metric("best_silhouette", winner["silhouette_score"])
            mlflow.tracking.MlflowClient().set_tag(
                winner["run_id"], "stage_of_work", "selected_unsupervised")
            print(f"P3 mejor run por silhouette: {winner['run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="entry", required=True)
    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument("--question", choices=INPUTS, default="P2")
    ingest_parser.add_argument("--build", choices=["true", "false"], default="false")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--question", choices=INPUTS, default="P2")
    train_parser.add_argument("--model", default="tree")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-depth", type=int, default=5)
    train_parser.add_argument("--min-samples-leaf", type=int, default=100)
    train_parser.add_argument("--sample-fraction", type=float, default=0.05)
    train_parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    train_parser.add_argument("--n-estimators", type=int, default=200)
    train_parser.add_argument("--k", type=int, default=4)
    search_parser = sub.add_parser("search")
    search_parser.add_argument("--question", choices=INPUTS, default="P2")
    search_parser.add_argument("--seed", type=int, default=42)
    search_parser.add_argument("--sample-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.entry == "ingest":
        ingest(args.question, args.build)
    elif args.entry == "train":
        train(args)
    else:
        search(args)


if __name__ == "__main__":
    main()
