
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

modelo = joblib.load(
    ROOT / "models" / "p1_random_forest_optimizado.pkl"
)

variables = [
    "dia_semana",
    "mes",
    "fin_semana",
    "lag_1",
    "lag_7"
]

importancia = pd.DataFrame({
    "variable": variables,
    "importancia": modelo.feature_importances_
}).sort_values("importancia", ascending=False)

print("\n=== IMPORTANCIA DE VARIABLES P1 ===")
print(importancia.to_string(index=False))

# Guardar resultados
reports = ROOT / "reports"
figures = reports / "figures"
figures.mkdir(parents=True, exist_ok=True)

importancia.to_csv(
    reports / "p1_importancia_variables.csv",
    index=False
)

# Crear gráfica
plt.figure(figsize=(9, 5))
plt.barh(
    importancia["variable"][::-1],
    importancia["importancia"][::-1]
)
plt.xlabel("Importancia relativa")
plt.title("MiBici P1 - Importancia de variables")
plt.tight_layout()

plt.savefig(
    figures / "p1_importancia_variables.png",
    dpi=150
)

print("\nGráfica guardada correctamente.")
