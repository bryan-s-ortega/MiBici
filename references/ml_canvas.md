# ML Canvas — MiBici (Equipo 23)

**Integrantes:** Miguel Ángel Durán Betancourt; Juan Pablo Islas Guzmán
(A01797727); Bryan Steve Ortega Castillo; Kevin Rogelio Rodriguez Magaña.

## Problema

El operador necesita anticipar demanda, priorizar estaciones con posible
saturación y entender grupos de estaciones para planear rebalanceo y expansión.

## Usuarios y decisiones

| Pregunta | Usuario | Decisión |
|---|---|---|
| P1: demanda futura | Operador | Asignar flota y personal por estación |
| P2: riesgo | Equipo de rebalanceo | Priorizar inspección o traslado |
| P3: grupos | Planeación urbana | Detectar perfiles y zonas con patrones similares |

## Datos y unidad

Viajes mensuales de MiBici, agregados a estación-día para P1/P2 y estación-
periodo para P3. Se usarán fechas, estación de origen, destino, duración,
hora/día y capacidad/coordenadas del catálogo.

## Objetivos y métricas

- P1: viajes iniciados el siguiente día; baseline de media histórica; MAE y RMSE.
- P2: proxy binario de riesgo del siguiente día por demanda/capacidad; baseline
  de clase mayoritaria; precision, recall y F1, priorizando recall.
- P3: grupos de estaciones; K-Means; silhouette score y revisión geográfica.

## Riesgos y supuestos

P2 no observa inventario real de bicicletas. El umbral de capacidad es un
supuesto operacional y no una etiqueta observada. Se evita fuga temporal con
lags y particiones cronológicas. Los identificadores de usuario se excluyen de
las features y no se publican.
