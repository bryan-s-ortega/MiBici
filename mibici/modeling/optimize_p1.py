
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "p1_demanda_diaria.csv"

# MLflow
mlflow.set_tracking_uri((ROOT / "mlflow.db").as_uri().replace("file:///", "sqlite:///"))
mlflow.set_experiment("MiBici_P1_Optimizacion")

# Cargar datos
df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

FEATURES = ["dia_semana", "mes", "fin_semana", "lag_1", "lag_7"]
TARGET = "viajes"

# Separación cronológica: 60% entrenamiento,
# 20% validación y 20% prueba.
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train = train[FEATURES]
y_train = train[TARGET]
X_val = val[FEATURES]
y_val = val[TARGET]

print("Entrenamiento:", len(train))
print("Validación:", len(val))
print("Prueba reservada:", len(test))

# Configuraciones a comparar
configuraciones = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 7, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 3},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
]

resultados = []

for i, params in enumerate(configuraciones, start=1):

    modelo = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    with mlflow.start_run(run_name=f"rf_p1_config_{i}"):
        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("split", "60_train_20_val_20_test")
        mlflow.log_metric("val_mae", float(mae))
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))

    resultados.append({
        "configuracion": i,
        **params,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2
    })

resultados_df = pd.DataFrame(resultados)
resultados_df = resultados_df.sort_values("val_mae")

print("\n=== RESULTADOS DE VALIDACIÓN ===")
print(resultados_df.to_string(index=False))

print("\n=== MEJOR CONFIGURACIÓN ===")
print(resultados_df.iloc[0].to_string())
