# Fase 5 — Modelo de probabilidad de impago (`prob_impago`)

Código: `src/ml/modelo_impago.py`. Corre como el paso `modelo_prob_impago`
del DAG `bronze_ingest`, después de `clean_silver_loans` y
`asignar_prestamos`.

Salidas:

- `data/gold/prob_impago_cliente`: un renglón por cliente (`cc_num`,
  `loan_id`, `prob_impago`).
- `data/gold/modelo_impago_metricas.json`: métricas y coeficientes de la
  corrida.

## Cómo funciona

1. **Entrena** con los 255,347 préstamos de Loan Default (Silver), usando su
   etiqueta real de impago (`is_default`, 11.6% de impago).
2. **Evalúa** con un 20% de préstamos que el modelo no vio al entrenar
   (división aleatoria con semilla fija = 42, reproducible).
3. **Calcula** la probabilidad de impago de cada uno de los 924 clientes con
   las características del préstamo que le asignó la llave sintética
   (`docs/fase4_llave_sintetica_prestamos.md`).

El modelo se reentrena en cada corrida (tarda segundos) y no se guarda en
disco, para evitar el error de escritura de Spark en Docker/Windows con
muchos archivos chiquitos. Sus coeficientes y métricas sí quedan en el JSON.

## Variables

Todas vienen del préstamo (fuente Loan Default), respetando las decisiones
de `docs/fase4_decisiones_kpis.md`:

- Numéricas: edad, `ingreso_mensual_fuente` (`Income` / 12), monto, score
  (`CreditScore`), meses empleado, líneas de crédito, tasa de interés, plazo,
  **DTI calculado** (la misma fórmula que el KPI `ratio_endeudamiento`), y si
  tiene hipoteca, dependientes o aval.
- Categóricas (one-hot): escolaridad, tipo de empleo, estado civil y
  propósito del préstamo.
- **No se usan:** el ingreso simulado (es para capacidad de ahorro) ni el
  `DTIRatio` del dataset (no documentado; además empeora un poco el modelo).

Todas las variables se estandarizan antes de entrenar, para que los
coeficientes se puedan comparar entre sí.

## Por qué regresión logística

El catálogo de KPIs de la Fase 1 proponía XGBoost/LightGBM. Se compararon
con la misma división de datos:

| Modelo | AUC en prueba |
|---|---|
| Regresión logística | 0.750 |
| Gradient boosting (HistGradientBoosting, mismo tipo que LightGBM) | 0.751 |

Dan prácticamente el mismo resultado, así que se eligió la regresión
logística porque **es explicable**: cada variable tiene un coeficiente que
dice si sube o baja el riesgo y cuánto. En un contexto bancario eso importa
más que una diferencia de 0.001 de AUC. Además corre con Spark MLlib, que ya
está en la imagen de Docker, sin agregar dependencias.

## Resultados (corrida del 2 oct 2026)

| Métrica | Valor |
|---|---|
| Préstamos para entrenar / probar | 204,087 / 51,260 |
| AUC en prueba | 0.749 |
| AUC en entrenamiento | 0.753 (casi igual: no hay sobreajuste) |
| Tasa de impago real en prueba | 11.60% |
| Tasa de impago promedio que predice el modelo | 11.59% (bien calibrado) |

**AUC 0.75** quiere decir que, si se toma al azar un préstamo que cayó en
impago y uno que no, el modelo le da más riesgo al que sí cayó 3 de cada 4
veces (0.5 sería adivinar). Es una señal moderada, consistente con lo que
ya se vio del dataset: es sintético y ninguna variable explica el impago
por sí sola.

## Qué pesa más (coeficientes)

Positivo = sube el riesgo; negativo = lo baja. Las 10 de mayor efecto:

| Variable | Coeficiente | Lectura |
|---|---|---|
| Edad | −0.60 | Más joven, más riesgo |
| Tasa de interés | +0.45 | Más tasa, más riesgo |
| Meses empleado | −0.34 | Menos antigüedad laboral, más riesgo |
| DTI | +0.26 | Más endeudado, más riesgo |
| Ingreso mensual | −0.18 | Menos ingreso, más riesgo |
| Monto del préstamo | +0.17 | Préstamo más grande, más riesgo |
| Tiene aval | −0.14 | Con aval, menos riesgo |
| Plazo | +0.14 | Plazo más largo, más riesgo |
| Tiene dependientes | −0.13 | Con dependientes, menos riesgo |
| Score de crédito | −0.12 | Mejor score, menos riesgo |

Todas van en la dirección que se esperaría de un banco, menos
"dependientes", que es un efecto pequeño del dataset sintético. El score
pesa poco porque en este dataset casi no se relaciona con el impago
(13.1% a 10.2% de impago entre el cuartil de score más bajo y el más alto).

## Probabilidad de impago de los 924 clientes

| Mín | p10 | Mediana | p90 | Máx |
|---|---|---|---|---|
| 0.6% | 2.7% | 8.7% | 25.9% | 84.6% |

65 clientes tienen 30% o más; 8 tienen 50% o más.

Casos representativos (los mismos de `docs/fase4_decisiones_kpis.md`):

| | A: gasto bajo | B: típico | C: gasto alto | D: ahorro negativo | E: DTI > 1 |
|---|---|---|---|---|---|
| DTI | 0.19 | 0.66 | 0.69 | 0.66 | 1.78 |
| `prob_impago` | 2.9% | 14.7% | 2.8% | 5.6% | 16.2% |

## Limitaciones

- **No es la probabilidad de un cliente real.** Es la probabilidad del
  préstamo asignado con la llave sintética (Opción A).
- **No se relaciona con la capacidad de ahorro.** El ahorro usa el ingreso
  simulado y el modelo usa el de la fuente, como se decidió. Por eso pueden
  verse casos como el D: ahorro negativo pero riesgo bajo. Incluso, los
  clientes que más ahorran tienden a tener un poco más de riesgo, porque
  gastan menos y la llave les asigna préstamos de gente con menos ingreso.
  Conviene presentarlos como dos lecturas distintas del cliente.
- **Señal moderada** (AUC 0.75) por la naturaleza sintética del dataset.

## Pendiente

- Revisar con Eduardo las métricas, los coeficientes y si este nivel de
  explicabilidad es suficiente.
- Definir cortes de riesgo (por ejemplo bajo / medio / alto) para el
  dashboard ejecutivo.
