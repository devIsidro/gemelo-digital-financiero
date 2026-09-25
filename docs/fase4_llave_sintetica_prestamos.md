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

## Resultados sobre los datos reales (924 clientes)

| KPI | Mediana | Nota |
|---|---|---|
| `capacidad_ahorro` (ingreso simulado con fórmula) | 60% | 14 clientes negativos |
| `capacidad_ahorro_ingreso_prestamo` | 9% | 183 clientes negativos |
| `ratio_endeudamiento_total` (préstamo / ingreso anual) | 1.52 | rango 0.04 – 14.5 |
| `ratio_endeudamiento_dti` (`DTIRatio`) | 0.50 | rango 0.10 – 0.90 |
| Score de crédito del préstamo asignado | 568 | rango 300 – 849 |

Por qué salen 183 clientes con ahorro negativo usando el ingreso del
préstamo: el ingreso de Loan Default está repartido de forma uniforme entre
1,250 y 12,500 USD al mes, y el gasto con tarjeta no. En los extremos no
cuadran. De los 183 negativos, 69 están en el 10% que menos gasta y 114 en
el 20% que más gasta (el 10% más alto sale negativo completo: gasta entre
~11,800 y ~22,000 USD al mes, y el ingreso máximo del dataset es 12,500).
En el 70% de en medio no
hay ningún negativo; su ahorro mediano va de 6% a 13% según el grupo.

## Preguntas para Eduardo

1. **¿Cómo medimos `ratio_endeudamiento`?**
   - Préstamo total / ingreso anual (la fórmula del catálogo de KPIs), o
   - `DTIRatio`: qué parte del ingreso mensual se va en pagar deudas (como lo
     mide normalmente un banco).
2. **¿Qué ingreso usamos para `capacidad_ahorro`?**
   - El simulado con fórmula (mediana 60%, varía más entre clientes, pero
     ingresos altos: ~14,500 USD/mes), o
   - El del préstamo asignado (mediana 9%, más realista, pero los extremos
     salen negativos por la forma de los datos).
3. **¿Reemplazamos el score simulado por el del préstamo?** El simulado deja
   a 906 de 924 clientes en 850; el del préstamo va de 300 a 849.
