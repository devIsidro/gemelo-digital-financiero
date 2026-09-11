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
- `gasto_promedio_mensual`
- `num_transacciones`
- `categoria_principal` (la de mayor gasto)
- `pct_transacciones_fraude`
- `ultima_transaccion`

Esto sí alimenta `tasa_exito_ingesta` y la mitad "gasto" de
`flujo_efectivo_proyectado`, y es la base para el modelo predictivo de
riesgo más adelante (Fase 5) aunque todavía no tengamos la señal de ingreso.

## Siguiente paso

Ya existe `src/spark/gold_transactions.py`, que construye
`gold_perfil_cliente` sobre Silver (sin datasets adicionales), siguiendo el
mismo patrón de `silver_transactions.py`. Con la Opción A decidida, lo que
sigue es:

1. Diseñar y documentar cómo se genera la llave sintética/controlada que
   simula el vínculo entre un `cc_num` de transacciones y un registro de
   Loan Default / Income (con sus reglas y supuestos explícitos, para que
   quede claro que es una simulación y no un dato real).
2. Ingerir los datasets de Loan Default Prediction e Income a Bronze.
3. Extender la Capa Gold para unir esa información simulada con
   `gold_perfil_cliente` y así poder calcular `prob_impago`,
   `capacidad_ahorro` y `ratio_endeudamiento`.
