# Fuentes de datos y uso

**Equipo:** 23

- MiBici, **Datos abiertos**: https://www.mibici.net/en/open-data/
- Catálogo de estaciones (ejemplo mensual): https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_08.csv

Antes de congelar la entrega se debe anotar en este archivo la URL exacta de
cada mes descargado, la fecha de descarga y el conteo de filas obtenido. Los
archivos originales se mantienen fuera de Git y se reconstruyen con la ingesta
documentada; sus hashes están congelados en `data_manifest.csv`.

## Descarga actual

Se descargaron del portal oficial los viajes de **enero de 2025 a septiembre
de 2026** (21 archivos) y `nomenclatura_2026_09.csv`. El tamaño local de los
CSV de viajes es aproximadamente 663 MB. La descarga se realizó el 17 de
octubre de 2026. Las rutas exactas y sus identificadores están en el HTML del
portal; los archivos locales conservan el nombre mensual oficial.

La consolidación actual contiene **7,894,097 viajes** y **484 estaciones**.

Los identificadores de usuario, cuando existan, no se usan como features ni se
publican en tablas o gráficas. Se conservan únicamente las variables necesarias
para agregación operacional.
