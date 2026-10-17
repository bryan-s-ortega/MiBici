
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Configuración
DATA_PATH = Path("data/processed/p1_demanda_diaria.csv")
MODEL_PATH = Path("models/p1_random_forest.pkl")

FEATURES = [
    "dia_semana",
    "mes",
    "fin_semana",
    "lag_1",
    "lag_7",
]

# Cargar datos
df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# División cronológica
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[FEATURES]
y_train = train["viajes"]

X_test = test[FEATURES]
y_test = test["viajes"]

print("Entrenamiento:", len(train), "registros")
print("Prueba:", len(test), "registros")
print("Periodo de prueba:", test["fecha"].min(), "a", test["fecha"].max())

# Configurar MLflow local
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MiBici_P1_Regresion")


def evaluar(y_real, y_pred):
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "r2": r2_score(y_real, y_pred),
    }


# Experimento 1: Baseline semanal
with mlflow.start_run(run_name="baseline_lag7"):
    pred_baseline = test["lag_7"].to_numpy()

    metricas = evaluar(y_test, pred_baseline)

    mlflow.log_param("modelo", "baseline_lag7")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metrics(metricas)

    print("\nBASELINE:", metricas)


# Experimento 2: Random Forest
with mlflow.start_run(run_name="random_forest_p1"):
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X_train, y_train)
    pred_rf = modelo.predict(X_test)

    metricas = evaluar(y_test, pred_rf)

    mlflow.log_params({
        "modelo": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 3,
        "random_state": 42,
        "test_size": 0.2,
    })

    mlflow.log_metrics(metricas)
    mlflow.sklearn.log_model(
    modelo,
    name="modelo_p1",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    print("\nRANDOM FOREST:", metricas)

print("\nEntrenamiento finalizado.")
