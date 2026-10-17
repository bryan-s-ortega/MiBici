# P2: diagnóstico inicial y decisiones de preparación

Responsable: Miguel Ángel Duran Betancourt.

## Pregunta de negocio

¿Una estación estará en riesgo de quedarse sin bicicletas o sin
anclajes disponibles en un periodo futuro?

Los archivos no contienen inventario instantáneo ni movimientos de
rebalanceo. Por tanto, se propone predecir presión de flujo neto como
aproximación al riesgo, no disponibilidad real.

## Datos inspeccionados

- Archivo de viajes: data/processed/trips.parquet.
- Catálogo: data/raw/stations_nomenclature.csv.
- Registros de viajes: 13,313,152.
- Filas del catálogo: 484.
- Identificadores duplicados en el catálogo: 0.
- Estaciones distintas observadas en viajes: 473.
- Primer inicio: 2023-10-15 23:37:28.
- Último inicio: 2026-09-30 23:59:48.
- Última llegada: 2026-10-01 00:14:52.
- Sin valores faltantes en fechas e identificadores de origen y destino.
- Viajes con duración cero: 34,353.
- Viajes con duración negativa: 163.
- Estaciones observadas ausentes del catálogo: 405 y 407.
- Capacidades faltantes o no positivas en el catálogo: 4.
- Viajes con origen sin capacidad válida: 1,960.
- Viajes con destino sin capacidad válida: 2,123.
- Viajes con al menos un extremo sin capacidad válida: 3,954.

Los conteos de problemas pueden solaparse; no deben sumarse para
calcular el total de registros excluidos.

Estos resultados proceden del diagnóstico ejecutado localmente.
Todavía no son métricas de un modelo.

## Reglas propuestas para el primer baseline

1. Conservar intactos los archivos descargados con DVC.
2. Excluir de los movimientos los viajes con duración negativa.
3. Excluir inicialmente los viajes con duración cero. Su causa no está
   confirmada; esta exclusión es un supuesto que debe documentarse.
4. Evaluar la capacidad por extremo: conservar la salida o llegada de
   una estación válida aunque el otro extremo no tenga capacidad.
5. No generar predicciones para estaciones sin capacidad válida.
6. Usar started_at para las salidas y ended_at para las llegadas.
7. No utilizar user_id, gender ni birth_year como predictores de P2.
8. Registrar por separado los conteos de exclusión y sus solapamientos.
9. Revisar duplicados de trip_id antes de la agregación final.

## Formulación inicial, pendiente de validar con el equipo

- Unidad: estación al inicio de un intervalo de 30 minutos.
- Horizonte: los siguientes 30 minutos.
- Balance futuro: salidas futuras menos llegadas futuras.
- Presión normalizada: balance futuro dividido por capacidad.
- Presión hacia falta de bicicletas: presión >= 0.30.
- Presión hacia falta de anclajes: presión <= -0.30.
- Sin presión neta elevada: valores entre ambos umbrales.

El umbral de 0.30 es provisional, no oficial de MiBici.
Las etiquetas se construyen con el futuro; los predictores solo pueden
utilizar información disponible al momento de la predicción.
La presión neta no identifica todos los episodios de estación vacía o llena.

## Partición temporal propuesta

- Entrenamiento: noviembre de 2023 a diciembre de 2025.
- Validación: enero a junio de 2026.
- Prueba: julio a septiembre de 2026.

Octubre de 2023 es parcial y se propone excluirlo del modelado.
Las ventanas objetivo deben quedar completamente dentro de su conjunto.
La partición debe coordinarse con el equipo antes de entrenar.

## Limitaciones y verificaciones pendientes

- La capacidad del catálogo puede diferir de la capacidad histórica.
- Un intervalo sin viajes no demuestra que la estación estaba operativa.
- Los conteos mensuales no prueban cobertura completa de todos los días.
- Debe definirse una política de actividad de estaciones sin usar el futuro.
- Las llegadas de octubre de 2026 no constituyen cobertura completa de octubre.
- Revisar duplicados, cobertura diaria y distribución de las clases.
- Vincular los resultados definitivos al commit y a la versión DVC utilizados.
