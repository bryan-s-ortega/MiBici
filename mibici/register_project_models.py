"""Registra una corrida seleccionada y comprueba carga e inferencia del alias."""

from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from mibici.project_cli import EXPERIMENT

NAMES = {"P1": "MiBici_Equipo23_P1_Demanda",
         "P2": "MiBici_Equipo23_P2_Presion_Proxy_Academica",
         "P3": "MiBici_Equipo23_P3_Estaciones"}


def register(question: str, run_id: str, model_uri: str | None = None):
    """Solo registra modelos de la pregunta correcta y con métricas existentes."""
    client = MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get("question") != question:
        raise ValueError("La pregunta del run no coincide con la solicitada")
    if question == "P2" and not {
        "test_balanced_accuracy", "test_recall_presion_sin_bicicletas",
        "test_recall_presion_sin_anclajes"
    }.issubset(run.data.metrics):
        raise ValueError("P2 requiere selección y evaluación final en test")
    if question == "P3" and run.data.tags.get("model_family") != "kmeans":
        raise ValueError("AgglomerativeClustering no implementa predict para nuevas estaciones")
    name = NAMES[question]
    path = "modelo_p3" if question == "P3" else "modelo"
    uri = model_uri or f"runs:/{run_id}/{path}"
    version = mlflow.register_model(model_uri=uri, name=name)
    client.set_model_version_tag(name, version.version, "academic_review", "required")
    if question == "P2":
        client.set_model_version_tag(name, version.version, "target", "presion_proxy_no_inventario")
    client.set_registered_model_alias(name, "champion", version.version)
    alias_uri = f"models:/{name}@champion"
    model = mlflow.sklearn.load_model(alias_uri)
    columns = list(model.feature_names_in_)
    example = pd.DataFrame([{column: 15.0 if column == "capacity" else 0.0
                             for column in columns}])
    predictions = model.predict(example)
    if len(predictions) != 1:
        raise RuntimeError("El modelo registrado no respondió a una fila")
    actual = client.get_model_version_by_alias(name, "champion")
    if str(actual.version) != str(version.version):
        raise RuntimeError("El alias no señala la versión registrada")
    result = {"question": question, "name": name, "version": version.version,
              "alias": "champion", "source_run": run_id,
              "load_uri": alias_uri, "smoke_prediction": str(predictions[0]),
              "academic_review_required": True}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", choices=NAMES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-uri", default=None,
                        help="URI exacta de Logged Models si no funciona runs:/run_id/name")
    args = parser.parse_args()
    mlflow.set_experiment(EXPERIMENT)
    register(args.question, args.run_id, args.model_uri)


if __name__ == "__main__":
    main()
