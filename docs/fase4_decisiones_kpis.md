# Fase 4 — Decisiones sobre score, ingreso y endeudamiento

Registro de las decisiones tomadas con Eduardo (BBVA) entre el 27 sep y el
1 oct 2026, y de cómo quedaron implementadas en `src/spark/gold_kpis.py`
(tabla `data/gold/kpis_cliente`).

| # | Decisión | Resultado |
|---|---|---|
| 1 | Score de crédito | Se usa el score del préstamo asignado (`score_credito_fuente`). Se descarta el simulado. |
| 2 | Ingreso | Dos ingresos separados: simulado para ahorro y flujo; el de la fuente para DTI y modelo de impago. |
| 3 | Endeudamiento | DTI, con fórmula propia documentada abajo (todo mensual). |
| 4 | Moneda | USD, la moneda original de los datos. |

## 1. Score de crédito

**Se usa:** `score_credito_fuente`, el `CreditScore` del préstamo que la llave
sintética le asigna a cada cliente (`docs/fase4_llave_sintetica_prestamos.md`).

Es un score **que viene de la fuente** (dataset Loan Default Prediction de
Kaggle). No lo calcula ningún modelo de este proyecto y no debe presentarse
como tal.

**Alternativa descartada: score simulado con fórmula.** La primera versión
de la Opción A calculaba un score por cliente
(`850 − penalización por fraude + bonificación por número de transacciones
± variación`, ver `docs/fase4_datos_simulados_ingreso_credito.md`). Se
descartó por tres razones:

1. **No discriminaba.** Sobre los 924 clientes reales, 906 quedaban
   exactamente en 850: la bonificación (+40) casi siempre superaba a la
   variación (±30) y el tope los cortaba a todos en el máximo.
2. **Mezclaba fraude con riesgo crediticio.** Penalizaba a los clientes con
   transacciones marcadas como fraude, pero en el dataset de transacciones
   `is_fraud` significa que la tarjeta fue usada por un tercero: la regla
   castigaba a la víctima.
3. **No es compatible con el modelo de impago.** El modelo se entrena con
   los préstamos de Loan Default, donde el score es una variable. Aplicarlo
   con un score de otra fórmula daría predicciones no confiables.

Se evaluó también corregir la fórmula quitando la parte de fraude, pero las
únicas señales restantes (número de transacciones y una variación
generada) hacían el score prácticamente arbitrario.

El código del score simulado se eliminó de `simular_perfil_financiero.py`;
queda en el historial de Git (commit `28d3d58` y anteriores).

## 2. Ingreso: dos conceptos que no se mezclan

| Columna | Qué es | De dónde sale | Se usa para | No usar para |
|---|---|---|---|---|
| `ingreso_mensual_simulado` | Ingreso mensual simulado con fórmula (Opción A) | `simular_perfil_financiero.py`: tramo por tamaño de ciudad ± variación + 0.3 × gasto | `capacidad_ahorro`, `flujo_efectivo_mensual` | DTI, modelo de impago |
| `ingreso_mensual_fuente` | Ingreso del titular del préstamo asignado | `Income` de Loan Default (anual) / 12 | `ratio_endeudamiento` (DTI), modelo de impago | Ahorro, flujo |

Por qué dos: el ahorro necesita un ingreso coherente con el gasto real de
cada cliente (el simulado se construye a partir de él), y el modelo de
impago necesita el mismo tipo de ingreso con el que se entrena (el de la
fuente va de 1,250 a 12,500 USD/mes; el simulado, de ~10,000 a ~35,000).

## 3. Ratio de endeudamiento (DTI)

```
ratio_endeudamiento = pago_mensual_prestamo / ingreso_mensual_fuente
```

**Numerador — `pago_mensual_prestamo`:** pago mensual fijo del préstamo
asignado, con la fórmula estándar de un crédito amortizable:

```
pago_mensual_prestamo = monto × r / (1 − (1 + r)^(−n))
  monto = LoanAmount        (USD)
  r     = InterestRate / 100 / 12   (tasa mensual)
  n     = LoanTerm          (meses)
```

Incluye capital e intereses de ese préstamo. **No incluye** otras deudas
del cliente (tarjetas, otros créditos, hipoteca): el dataset no trae sus
saldos ni pagos, solo `NumCreditLines` y `HasMortgage` como indicadores.

**Denominador — `ingreso_mensual_fuente`:** `Income / 12`. `Income` es el
ingreso anual del titular del préstamo. El dataset no especifica si es bruto
o neto (antes o después de impuestos); se usa tal como viene.

**Periodicidad:** numerador y denominador son **mensuales** y en USD.

**Por qué no se usa el `DTIRatio` que trae el dataset:** Loan Default no
documenta cómo se calculó, y no se puede reproducir con sus propias
columnas (correlación ~0 con pago mensual / ingreso mensual del mismo
registro). Además, el calculado separa mejor el riesgo:

| Cuartil (de menor a mayor) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Impago según DTI calculado | 7.8% | 9.3% | 11.2% | 18.2% |
| Impago según `DTIRatio` del dataset | 10.7% | 11.5% | 12.0% | 12.3% |

(Sobre los 255,347 préstamos.)

**Alternativa descartada: monto total / ingreso anual** (la fórmula
preliminar del catálogo de KPIs de la Fase 1). No considera plazo ni tasa:
dos personas con la misma deuda tienen pagos mensuales muy distintos según
esas condiciones, y lo que determina si alguien puede pagar es el pago de
cada mes.

**Valores mayores a 1.** 330 de 924 clientes tienen DTI > 1 (su pago
mensual supera su ingreso mensual) y 135 tienen DTI > 2. No se recortan:
es una característica del dataset, que es sintético y genera los montos
de préstamo (5,000 a 250,000 USD) sin relación con el ingreso del titular.
En los 255,347 préstamos pasa en el 36%.

## Distribución de las variables (924 clientes)

| Variable | Mín | p10 | Mediana | p90 | Máx |
|---|---|---|---|---|---|
| Gasto mensual real (USD) | 1,156 | 2,248 | 6,167 | 11,838 | 21,956 |
| `ingreso_mensual_simulado` (USD) | 10,244 | 11,933 | 14,559 | 22,206 | 34,599 |
| `ingreso_mensual_fuente` (USD) | 1,256 | 2,377 | 6,872 | 11,371 | 12,493 |
| `score_credito_fuente` | 300 | 350 | 568 | 791 | 849 |
| `pago_mensual_prestamo` (USD) | 183 | 1,242 | 4,452 | 11,657 | 23,060 |
| `ratio_endeudamiento` (DTI) | 0.02 | 0.18 | 0.68 | 2.49 | 14.00 |
| `capacidad_ahorro` | −22% | 27% | 60% | 84% | 94% |

## Casos representativos

| | A: gasto bajo (p10) | B: típico (mediana) | C: gasto alto (p90) | D: ahorro negativo | E: DTI > 1 |
|---|---|---|---|---|---|
| Gasto mensual real | 2,246 | 6,188 | 11,847 | 21,956 | 4,115 |
| Ingreso simulado | 12,586 | 14,122 | 13,512 | 17,975 | 10,982 |
| **Capacidad de ahorro** | **82%** | **56%** | **12%** | **−22%** | **63%** |
| Ingreso de la fuente | 2,373 | 6,878 | 11,375 | 12,493 | 4,439 |
| Préstamo (monto / tasa / plazo) | 21,190 / 10.2% / 60 m | 134,850 / 12.8% / 36 m | 238,239 / 11.4% / 36 m | 95,111 / 7.2% / 12 m | 175,940 / 7.3% / 24 m |
| Pago mensual | 452 | 4,529 | 7,839 | 8,240 | 7,897 |
| **DTI** | **0.19** | **0.66** | **0.69** | **0.66** | **1.78** |
| Score de la fuente | 318 | 302 | 424 | 406 | 376 |

Montos en USD. Se ve la separación entre los dos ingresos: el caso E ahorra
63% según su ingreso simulado, pero el préstamo asignado le pide 1.78 veces
su ingreso de la fuente. Son dos lecturas distintas del mismo cliente, por
eso no se mezclan.

## Pendiente

Revisión con Eduardo de esta distribución y de los casos antes de cerrar
la parte de KPIs. Material preparado en
`docs/fase5_revision_distribuciones.md` (gráficas y explicación por caso).
