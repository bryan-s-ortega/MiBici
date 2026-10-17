
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "p1_predicciones.csv"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["fecha"])
df = df.sort_values("fecha")

# Gráfica 1: Demanda real vs. predicha
plt.figure(figsize=(13, 5))
plt.plot(df["fecha"], df["viajes_reales"],
         label="Viajes reales", linewidth=1.8)
plt.plot(df["fecha"], df["viajes_predichos"],
         label="Predicción Random Forest", linewidth=1.8)

plt.title("MiBici P1 - Demanda diaria real vs. predicha")
plt.xlabel("Fecha")
plt.ylabel("Número de viajes")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES / "p1_real_vs_predicho.png", dpi=150)
plt.show()

# Gráfica 2: Errores de predicción
df["error"] = df["viajes_reales"] - df["viajes_predichos"]

plt.figure(figsize=(12, 4))
plt.scatter(df["fecha"], df["error"], alpha=0.7)
plt.axhline(0, linestyle="--")
plt.title("MiBici P1 - Errores de predicción")
plt.xlabel("Fecha")
plt.ylabel("Error (viajes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES / "p1_errores.png", dpi=150)
plt.show()

print("Gráficas guardadas en reports/figures/")
