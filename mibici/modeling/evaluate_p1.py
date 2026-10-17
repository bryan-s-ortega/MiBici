
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "p1_demanda_diaria.csv"
MODEL_PATH = ROOT / "models" / "p1_random_forest_optimizado.pkl"

df = pd.read_csv(DATA, parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

features = ["dia_semana", "mes", "fin_semana", "lag_1", "lag_7"]

# División cronológica 80% / 20%
limite = int(len(df) * 0.8)
train = df.iloc[:limite]
test = df.iloc[limite:]

modelo = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

modelo.fit(train[features], train["viajes"])
predicciones = modelo.predict(test[features])

mae = mean_absolute_error(test["viajes"], predicciones)
rmse = np.sqrt(mean_squared_error(test["viajes"], predicciones))
r2 = r2_score(test["viajes"], predicciones)

print("\n=== EVALUACIÓN FINAL P1 ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# Guardar modelo
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, MODEL_PATH)

# Guardar predicciones para graficar
resultados = pd.DataFrame({
    "fecha": test["fecha"].to_numpy(),
    "viajes_reales": test["viajes"].to_numpy(),
    "viajes_predichos": predicciones
})

resultados.to_csv(
    ROOT / "data" / "processed" / "p1_predicciones.csv",
    index=False
)

# Registrar evaluación final en MLflow
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("MiBici_P1_Regresion")

with mlflow.start_run(run_name="random_forest_p1_optimizado"):
    mlflow.log_params({
        "n_estimators": 500,
        "max_depth": "None",
        "min_samples_leaf": 5,
        "random_state": 42
    })
    mlflow.log_metrics({
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    })
    mlflow.sklearn.log_model(
        modelo,
        name="modelo_p1_optimizado",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("\nModelo y predicciones guardados correctamente.")
