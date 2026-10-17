
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/trips.parquet")
OUTPUT_PATH = Path("data/processed/p1_demanda_diaria.csv")

# Cargar datos
df = pd.read_parquet(DATA_PATH, columns=["started_at"])
df["fecha"] = df["started_at"].dt.normalize()

# Calcular demanda diaria
demanda = (
    df.groupby("fecha")
    .size()
    .rename("viajes")
    .sort_index()
)

# Crear calendario completo
calendario = pd.date_range(
    demanda.index.min(),
    demanda.index.max(),
    freq="D"
)

demanda = demanda.reindex(calendario)
demanda.index.name = "fecha"

# Detectar días sin información
faltantes = demanda[demanda.isna()]

print("Días sin registros:", len(faltantes))
print("Fechas faltantes:", faltantes.index.tolist())

# Variables temporales
demanda = demanda.reset_index()
demanda["dia_semana"] = demanda["fecha"].dt.dayofweek
demanda["mes"] = demanda["fecha"].dt.month
demanda["fin_semana"] = (
    demanda["dia_semana"] >= 5
).astype(int)

# Rezagos basados en días calendario
demanda["lag_1"] = demanda["viajes"].shift(1)
demanda["lag_7"] = demanda["viajes"].shift(7)

# Eliminar filas sin objetivo o rezagos válidos
demanda = demanda.dropna(
    subset=["viajes", "lag_1", "lag_7"]
).copy()

# Guardar dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
demanda.to_csv(OUTPUT_PATH, index=False)

print("\n=== DATASET P1 ===")
print(demanda.head())

print("\nDimensiones:", demanda.shape)
print("Fecha inicial:", demanda["fecha"].min())
print("Fecha final:", demanda["fecha"].max())
print("Archivo generado:", OUTPUT_PATH)
