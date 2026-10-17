
import pandas as pd

# Cargar dataset procesado de MiBici
df = pd.read_parquet("data/processed/trips.parquet")

# Dimensiones
print("\n=== DIMENSIONES ===")
print(df.shape)

# Nombres de columnas
print("\n=== COLUMNAS ===")
print(df.columns.tolist())

# Primeros registros
print("\n=== PRIMEROS 5 REGISTROS ===")
print(df.head())

# Tipos de datos
print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

# Valores faltantes
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

# Resumen general
print("\n=== RESUMEN ESTADISTICO ===")
print(df.describe(include="all"))
