"""Entrenamiento y registro en MLflow para P3 (Clustering de Estaciones)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    FIGURES_DIR,
    P3_DEFAULT_K,
    P3_EXPERIMENT_NAME,
    P3_RANDOM_STATE,
    P3_STATIONS_PATH,
    P3_TRIPS_PATH,
    PROJ_ROOT,
)
from mibici.features import extract_p3_station_features

# Conjuntos de variables evaluados
BASELINE_FEATURES = [
    "viajes_promedio_diario",
    "viajes_desviacion_diaria",
    "capacidad",
    "latitude",
    "longitude",
]

CANDIDATE_FEATURES = [
    "viajes_promedio_diario",
    "pct_pico_matutino",
    "pct_mediodia",
    "pct_pico_vespertino",
    "pct_fin_semana",
    "ratio_flujo_neto",
    "capacidad",
    "latitude",
    "longitude",
]

CLUSTER_NAMES = {
    0: "Polo Laboral / Atractor Vespertino",
    1: "Residencial Emisor / Matutino",
    2: "Corredor Mixto / Ocio y Fin de Semana",
    3: "Estaciones Periféricas / Baja Intensidad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=P3_DEFAULT_K, help="Número de clusters")
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "agglomerative"],
        default="kmeans",
        help="Algoritmo de clustering",
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "candidato"],
        default="candidato",
        help="Tipo de modelo y conjunto de features",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5001",
        help="URI del servidor de tracking de MLflow",
    )
    parser.add_argument(
        "--experiment",
        default=P3_EXPERIMENT_NAME,
        help="Nombre del experimento de MLflow",
    )
    parser.add_argument("--seed", type=int, default=P3_RANDOM_STATE, help="Semilla aleatoria")
    parser.add_argument(
        "--sweep-k",
        action="store_true",
        default=True,
        help="Generar análisis de sensibilidad del codo y silueta para k=2..8",
    )
    return parser.parse_args()


def check_tracking_uri(uri: str) -> str:
    """Verifica si la URI HTTP responde; en caso contrario, utiliza SQLite local."""
    if uri.startswith(("http://", "https://")):
        try:
            with urlopen(uri, timeout=1.5):
                return uri
        except (URLError, TimeoutError, OSError):
            local_db = f"sqlite:///{PROJ_ROOT}/mlruns.db"
            print(f"[Aviso] Servidor HTTP {uri} no disponible. Usando backend local: {local_db}")
            return local_db
    return uri


def file_hash(path: Path) -> str:
    """Calcula el hash SHA-256 para trazabilidad del archivo de datos."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sweep_k_metrics(
    X_scaled: np.ndarray, seed: int, max_k: int = 8
) -> tuple[list[int], list[float], list[float]]:
    """Evalúa inercia y silhouette para k desde 2 hasta max_k."""
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, km.labels_)))
    return k_range, inertias, silhouettes


def plot_k_evaluation(
    k_range: list[int],
    inertias: list[float],
    silhouettes: list[float],
    chosen_k: int,
    output_path: Path,
) -> None:
    """Genera gráfico del Método del Codo y Silhouette Score."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_range, inertias, marker="o", color="#1E40AF", linewidth=2)
    ax1.axvline(chosen_k, color="#DC2626", linestyle="--", label=f"k seleccionado = {chosen_k}")
    ax1.set_title("Método del Codo (Inercia / WCSS)", fontweight="bold")
    ax1.set_xlabel("Número de Clusters (k)")
    ax1.set_ylabel("Inercia Intra-Cluster")
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(k_range, silhouettes, marker="s", color="#00953B", linewidth=2)
    ax2.axvline(chosen_k, color="#DC2626", linestyle="--", label=f"k seleccionado = {chosen_k}")
    ax2.set_title("Coeficiente de Silueta Promedio", fontweight="bold")
    ax2.set_xlabel("Número de Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xticks(k_range)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_spatial_clusters(df_clusters: pd.DataFrame, output_path: Path) -> None:
    """Genera mapa de dispersión espacial de estaciones coloreadas por cluster."""
    valid_coords = df_clusters[
        (df_clusters["latitude"] > 20.4)
        & (df_clusters["latitude"] < 20.9)
        & (df_clusters["longitude"] > -103.6)
        & (df_clusters["longitude"] < -103.2)
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for c in sorted(valid_coords["cluster"].unique()):
        sub = valid_coords[valid_coords["cluster"] == c]
        label = CLUSTER_NAMES.get(c, f"Cluster {c}")
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=sub["capacidad"] * 3.5,
            color=cmap(c),
            alpha=0.8,
            edgecolors="#222222",
            linewidth=0.5,
            label=f"C{c}: {label} ({len(sub)} est.)",
        )

    ax.set_title(
        "Distribución Geoespacial de Clusters de Estaciones MiBici (AMG)\n"
        "(Tamaño del punto proporcional a la capacidad de anclajes)",
        fontweight="bold",
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_profiles(
    features_df: pd.DataFrame, feature_cols: list[str], output_path: Path
) -> None:
    """Genera gráfico de barras de los promedios estandarizados por cluster."""
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features_df[feature_cols]),
        columns=feature_cols,
        index=features_df.index,
    )
    scaled["cluster"] = features_df["cluster"]

    cluster_means = scaled.groupby("cluster").mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_means.T.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")

    ax.set_title(
        "Perfil Multivariado Estandarizado de Clusters de Estaciones (P3)", fontweight="bold"
    )
    ax.set_xlabel("Variable")
    ax.set_ylabel("Valor Estandarizado (Z-score)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    labels = [f"C{c}: {CLUSTER_NAMES.get(c, f'Cluster {c}')}" for c in cluster_means.index]
    ax.legend(labels, loc="upper right", frameon=True, fontsize=8.5)
    plt.xticks(rotation=40, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_p3(
    model_type: str = "candidato",
    algorithm: str = "kmeans",
    k: int = P3_DEFAULT_K,
    seed: int = P3_RANDOM_STATE,
    tracking_uri: str = "http://127.0.0.1:5001",
    experiment: str = P3_EXPERIMENT_NAME,
    sweep_k: bool = True,
) -> dict[str, float]:
    """Ejecuta el pipeline completo de clustering para P3 y registra en MLflow."""
    uri = check_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    # 1. Carga de datos crudos/procesados
    print("[1/5] Cargando dataset de viajes y catálogo de estaciones...", flush=True)
    trips = pd.read_parquet(
        P3_TRIPS_PATH,
        columns=["started_at", "origin_station_id", "destination_station_id"],
    )
    nom = pd.read_csv(P3_STATIONS_PATH)

    # 2. Extracción de características
    print("[2/5] Extrayendo características por estación con mibici.features...", flush=True)
    features_df = extract_p3_station_features(trips, nom)
    print(f"      Estaciones procesadas: {len(features_df)}", flush=True)

    # Selección de variables según modelo
    feature_cols = BASELINE_FEATURES if model_type == "baseline" else CANDIDATE_FEATURES
    X = features_df[feature_cols].copy()

    # Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Ajuste del modelo
    print(f"[3/5] Entrenando {algorithm.upper()} con k={k} ({model_type})...", flush=True)
    if algorithm == "kmeans":
        cluster = KMeans(n_clusters=k, random_state=seed, n_init=10)
    elif algorithm == "agglomerative":
        cluster = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")

    # El pipeline recibe variables sin escalar también al inferir: evita la
    # discrepancia entre el ejemplo de entrada y el estimador serializado.
    model = Pipeline([("scale", StandardScaler()), ("cluster", cluster)])
    labels = model.fit_predict(X)
    features_df["cluster"] = labels

    # 4. Cálculo de métricas
    sil_score = float(silhouette_score(X_scaled, labels))
    db_score = float(davies_bouldin_score(X_scaled, labels))
    ch_score = float(calinski_harabasz_score(X_scaled, labels))
    inertia = float(cluster.inertia_) if algorithm == "kmeans" else float("nan")

    metrics = {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "n_stations": float(len(features_df)),
        "n_clusters": float(k),
    }
    if not np.isnan(inertia):
        metrics["inertia"] = inertia

    # 5. Registro en MLflow
    run_name = f"P3_{algorithm}_{model_type}_k{k}"
    print(f"[4/5] Registrando ejecución en MLflow ({run_name})...", flush=True)

    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None) as run:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        ).strip()

        mlflow.set_tags(
            {
                "equipo": "23",
                "pregunta": "P3",
                "tipo_tarea": "Clustering",
                "tipo_modelo": model_type,
                "target_tipo": "segmentacion_estaciones",
                "git_commit": commit,
                "working_tree_dirty": str(bool(dirty)).lower(),
                "question": "P3",
                "model_family": algorithm,
                "stage_of_work": "selection",
                "team": "23",
                "commit": commit,
            }
        )

        mlflow.log_params(
            {
                "algoritmo": algorithm,
                "k": k,
                "tipo_modelo": model_type,
                "seed": seed,
                "num_features": len(feature_cols),
                "features_usadas": ",".join(feature_cols),
                "data_sha256": file_hash(P3_TRIPS_PATH),
                "nomenclature_sha256": file_hash(P3_STATIONS_PATH),
                "periodo_inicio": str(trips["started_at"].min()),
                "periodo_fin": str(trips["started_at"].max()),
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_input(
            mlflow.data.from_pandas(
                X.assign(station_key=features_df.index.astype(str)),
                source=str(P3_TRIPS_PATH), name="p3_perfiles_estacion"
            ), context="training")

        # Generar artefactos visuales y reportes
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Artefacto 1: Mapa espacial
            map_file = temp_path / "p3_mapa_espacial_clusters.png"
            plot_spatial_clusters(features_df, map_file)

            # Artefacto 2: Perfiles multivariados
            profile_file = temp_path / "p3_perfiles_clusters.png"
            plot_cluster_profiles(features_df, feature_cols, profile_file)

            # Artefacto 3: Curvas de codo y silueta
            if sweep_k and algorithm == "kmeans":
                k_range, inertias, silhouettes = sweep_k_metrics(X_scaled, seed, max_k=8)
                k_file = temp_path / "p3_seleccion_k.png"
                plot_k_evaluation(k_range, inertias, silhouettes, k, k_file)

            # Artefacto 4: CSV de asignación de estaciones
            cluster_csv = temp_path / "p3_estaciones_clustering.csv"
            export_cols = ["nombre", "capacidad", "latitude", "longitude", "cluster"] + [
                c for c in feature_cols if c not in ["latitude", "longitude", "capacidad"]
            ]
            features_df[export_cols].to_csv(cluster_csv)

            # Artefacto 5: Resumen JSON
            summary_file = temp_path / "resumen_p3.json"
            counts = features_df["cluster"].value_counts().to_dict()
            summary = {
                "metricas": metrics,
                "distribucion_clusters": {
                    f"Cluster_{c} ({CLUSTER_NAMES.get(c, 'N/A')})": int(cnt)
                    for c, cnt in counts.items()
                },
                "variables": feature_cols,
            }
            summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            (temp_path / "configuracion.json").write_text(json.dumps({
                "model_type": model_type, "algorithm": algorithm, "k": k,
                "seed": seed, "features": feature_cols,
            }, indent=2), encoding="utf-8")
            (temp_path / "manifiesto.json").write_text(json.dumps({
                "trips_sha256": file_hash(P3_TRIPS_PATH),
                "stations_sha256": file_hash(P3_STATIONS_PATH),
                "stations_count": len(features_df),
                "note": "Clustering descriptivo; no existe train/validation/test supervisado",
            }, indent=2), encoding="utf-8")

            # Guardar copia fija en reports/figures/
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            for f in temp_path.glob("*"):
                (FIGURES_DIR / f.name).write_bytes(f.read_bytes())

            mlflow.log_artifacts(str(temp_path), artifact_path="artefactos_p3")

        # Registro del modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo_p3",
            input_example=X.head(5),
            signature=(infer_signature(X.head(5), model.predict(X.head(5)))
                       if algorithm == "kmeans" else infer_signature(X.head(5))),
            serialization_format="cloudpickle",
        )

        # Registro del código fuente
        for rel_file in (
            "mibici/modeling/train_p3.py",
            "mibici/features.py",
            "mibici/config.py",
        ):
            target_code = PROJ_ROOT / rel_file
            if target_code.exists():
                mlflow.log_artifact(str(target_code), artifact_path="codigo")

    print("[5/5] Corrida completada exitosamente!")
    print(f"      Run ID: {run.info.run_id}")
    print(f"      Silhouette Score:    {sil_score:.4f}")
    print(f"      Davies-Bouldin:      {db_score:.4f}")
    print(f"      Calinski-Harabasz:   {ch_score:.1f}")
    return {**metrics, "run_id": run.info.run_id}


def main() -> None:
    args = parse_args()
    train_p3(
        model_type=args.model_type,
        algorithm=args.algorithm,
        k=args.k,
        seed=args.seed,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        sweep_k=args.sweep_k,
    )


if __name__ == "__main__":
    main()
