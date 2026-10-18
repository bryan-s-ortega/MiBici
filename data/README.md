# Datos

Los CSV originales se guardan en `data/raw/` y están excluidos de Git por su
tamaño. La ingesta consolida los archivos en `data/processed/`; esas salidas
también se excluyen para evitar duplicar cientos de megabytes en el repositorio.

Para reconstruir la tabla procesada:

```bash
PYTHONPATH=src python -m mibici.ingestion --raw-dir data/raw --output-dir data/processed
```

El conjunto usado para los resultados iniciales contiene 21 meses de viajes
(enero de 2025 a septiembre de 2026), aproximadamente 663 MB en bruto,
7,894,097 viajes consolidados y 484 estaciones en el catálogo.
