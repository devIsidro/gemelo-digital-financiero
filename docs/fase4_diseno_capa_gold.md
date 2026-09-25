# Fase 4 — Diseño de la Capa Gold (Semana 13)

Primer paso de la Fase 4: antes de escribir el job de PySpark que arma Gold,
revisar qué de lo prometido en `config/kpis.yaml` (Fase 1) se puede construir
de verdad con los datos que hoy existen en Silver, y qué no.

## Qué hay hoy en Silver

Solo un dataset ingerido y limpio: **Credit Card Transactions Fraud
Detection**. Columnas disponibles: `trans_num`, `cc_num`, `amt`, `category`,
`merchant`, `trans_date_trans_time`, `is_fraud`, `city_pop`, `year`,
`month`, `day`.

No hay (todavía) ningún dataset con ingreso, deuda o historial crediticio.

## Estado real de los 5 KPIs catalogados en Fase 1

| KPI | Se puede calcular hoy | Por qué |
|---|---|---|
| `tasa_exito_ingesta` | Sí | Viene de logs de Airflow, ya cubierto por el Panel Operativo (Fase 3) |
| `flujo_efectivo_proyectado` | Parcial | Se puede ver *gasto* transaccional por cliente/mes, pero sin datos de ingreso no es un flujo de efectivo real, solo la mitad de la ecuación |
| `prob_impago` | No | Necesita el dataset de Loan Default Prediction (ingresos, historial crediticio) — nunca se ingirió |
| `capacidad_ahorro` | No | Necesita el dataset de Financial Transactions (Expenses & Income) — nunca se ingirió |
| `ratio_endeudamiento` | No | Necesita datos de deuda/ingreso anual — no existen en el dataset actual |

**El problema no es solo "faltan datasets".** Los datasets sugeridos en el
documento oficial (`Path Data Engineering Tecmilenio`, sección 5) —
Credit Card Fraud Detection, Loan Default Prediction, Personal Finance/
Banking Datasets— son poblaciones de Kaggle independientes entre sí, sin
un identificador de cliente compartido. El propio documento ya lo anticipa
en esa sección:

> "La selección de las fuentes de datos está abierta al estudiante. Deberá
> buscar, proponer y construir las conexiones lógicas entre múltiples
> datasets públicos para dar vida al 'Gemelo Digital Financiero'."

Y en la sección 10 confirma que el modelo de riesgo completo sí necesita
esa unión (ingreso/historial + comportamiento transaccional):
`P_impago = f(Ingresos, Historial, Balance Transaccional)`.

**Decisión (reunión con Eduardo, 11 sep 2026): Opción A.** Se construye una
llave sintética/controlada — no aleatoria — para unir Loan Default e Income
al dataset de transacciones, documentando explícitamente que es una
simulación (coherente con el "problema de negocio simulado" del proyecto),
en vez de buscar datasets con cliente real compartido (Opción B, riesgo de
tiempo) o recortar el alcance de KPIs (Opción C).

## Primera tabla Gold propuesta (construible ya, sin datos nuevos)

`gold_perfil_cliente` — un renglón por `cc_num`, agregado desde Silver:

- `gasto_total`
- `gasto_promedio_mensual` (gasto total / `meses_activo`)
- `meses_activo` (días entre primera y última transacción / 30.44, mínimo 1)
- `num_transacciones`
- `categoria_principal` (la de mayor gasto)
- `pct_transacciones_fraude`
- `primera_transaccion`, `ultima_transaccion`
- `city_pop`

Todos los montos en USD (moneda original del dataset).

Esto sí alimenta `tasa_exito_ingesta` y la mitad "gasto" de
`flujo_efectivo_proyectado`, y es la base para el modelo predictivo de
riesgo más adelante (Fase 5) aunque todavía no tengamos la señal de ingreso.

## Estado actual de Gold (25 sep 2026)

El DAG `bronze_ingest` ya construye Gold después de validar Silver, y
también ingiere y limpia el dataset Loan Default en paralelo:

| Tabla Gold | Job | Contenido |
|---|---|---|
| `data/gold/perfil_cliente` | `gold_transactions.py` | Gasto real por cliente |
| `data/gold/perfil_financiero_simulado` | `simular_perfil_financiero.py` | Ingreso y score SIMULADOS con fórmula (Opción A) |
| `data/gold/asignacion_prestamos` | `asignar_prestamos.py` | Llave sintética: un préstamo de Loan Default por cliente |
| `data/gold/kpis_cliente` | `gold_kpis.py` | Todo lo anterior unido + KPIs |

| KPI | Estado |
|---|---|
| `capacidad_ahorro` | Calculado con dos ingresos (simulado y del préstamo); falta elegir con Eduardo |
| `flujo_efectivo_proyectado` | Base calculada (`flujo_efectivo_mensual`); falta la parte Monte Carlo |
| `ratio_endeudamiento` | Calculado en dos formas (total y DTI); falta elegir con Eduardo |
| `prob_impago` | Pendiente: modelo entrenado con las etiquetas de Loan Default |

Detalle de la llave sintética, resultados y preguntas abiertas:
`docs/fase4_llave_sintetica_prestamos.md`.

## Siguiente paso

1. Revisar con Eduardo las 3 preguntas de
   `docs/fase4_llave_sintetica_prestamos.md` y dejar una sola versión de
   cada KPI.
2. Entrenar el modelo de `prob_impago` con los 255,347 préstamos y
   aplicarlo a cada cliente con las características de su préstamo
   asignado.
