# Fase 4 — Llave sintética clientes ↔ préstamos (Loan Default)

Decisión del 25 sep 2026, dentro de la **Opción A** acordada con Eduardo el
11 sep: a cada cliente del dataset de transacciones se le asigna, de forma
controlada y documentada, un préstamo del dataset **Loan Default Prediction**
(Kaggle, `nikhil1e9/loan-default`). El préstamo asignado **no es de esa
persona**: es una construcción para poder calcular `ratio_endeudamiento` y,
más adelante, entrenar el modelo de `prob_impago`.

## El dataset

`data/raw/Loan_default.csv` — 255,347 préstamos, 18 columnas, sin nulos ni
IDs duplicados, 11.6% en impago (`Default = 1`). Montos en USD; `Income` es
**anual**.

Revisión al ingerirlo (25 sep 2026):

- **La etiqueta de impago tiene lógica**: caen más en impago los clientes
  más jóvenes, con menos ingreso, con tasa de interés más alta, desempleados
  o sin aval. La señal es moderada (ninguna variable por sí sola la explica),
  suficiente para entrenar un modelo, no para uno perfecto.
- **Es un dataset sintético**: casi todas las columnas tienen distribución
  uniforme, las categorías están repartidas casi exactamente igual y las
  variables no se relacionan entre sí (por ejemplo, ingreso y score tienen
  correlación ~0).
- **`DTIRatio` no cuadra con `LoanAmount / Income`** (correlación ~0): son
  dos medidas independientes. Por eso `ratio_endeudamiento` se calcula en
  las dos formas (ver abajo).

## Flujo en el pipeline

```
data/raw/Loan_default.csv
  -> ingest_loan_default    (Bronze: data/bronze/loan_default)
  -> clean_silver_loans     (Silver: data/silver/loan_default, con validación)
  -> asignar_prestamos      (Gold: data/gold/asignacion_prestamos)  <- también usa gold_perfil_cliente
  -> calcular_kpis          (Gold: data/gold/kpis_cliente)
```

## Regla de asignación: "por parecido"

1. Se ordena a los clientes por su **gasto mensual real con tarjeta**
   (`gasto_promedio_mensual`) y a los préstamos por el **ingreso** de su
   titular (`income_anual`). Los empates se rompen por `cc_num` / `loan_id`.
2. Cada cliente recibe un percentil a la mitad de su escalón:
   `percentil = (posición - 0.5) / número_de_clientes`.
3. Recibe el préstamo que está en ese mismo percentil de ingreso:
   `posición_préstamo = piso(percentil × número_de_préstamos) + 1`.

Propiedades, verificadas con pruebas automáticas y sobre los datos reales:

- **Reproducible**: mismos datos de entrada → misma asignación siempre.
- **Uno a uno**: ningún préstamo se repite (924 clientes → 924 préstamos
  distintos).
- **Coherente**: quien más gasta recibe a alguien que más gana
  (correlación gasto–ingreso asignado: 0.945).
- **No sesga el impago**: los préstamos asignados tienen 10.9% de impago,
  parecido al 11.6% del dataset completo.

Limitación: solo se empareja por gasto ↔ ingreso. Las demás
características del préstamo (edad, score, monto, si pagó o no) llegan tal
cual vienen con ese registro.

## Uso de los datos del préstamo asignado

Decidido con Eduardo (oct 2026), detalle en `docs/fase4_decisiones_kpis.md`:

- `score_credito_fuente`: el score del préstamo asignado reemplaza al score
  simulado.
- `ingreso_mensual_fuente` (`Income` / 12): se usa para el DTI y el modelo
  de impago. La capacidad de ahorro sigue usando el ingreso simulado.
- `ratio_endeudamiento`: DTI calculado con el pago mensual del préstamo
  (monto, tasa y plazo) entre `ingreso_mensual_fuente`. El `DTIRatio` del
  dataset no se usa.

Las alternativas que se evaluaron antes de decidir (ahorro con el ingreso
del préstamo, `DTIRatio` del dataset, monto / ingreso anual) se quitaron del
código para que no se mezclen; sus resultados quedan en el historial de Git
(commit `28d3d58`).
