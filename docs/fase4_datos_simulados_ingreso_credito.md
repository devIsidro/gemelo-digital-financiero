# Fase 4 — Reglas para simular ingreso e historial crediticio (Opción A)

Este documento define **cómo** se va a construir el ingreso mensual y el
historial crediticio simulados por cliente, que es lo que decidió Eduardo
en la reunión del 11 de septiembre (ver `fase4_diseno_capa_gold.md`). Estas
reglas ya están implementadas en `src/spark/simular_perfil_financiero.py` y
probadas sobre el Silver real completo (924 clientes).

**Moneda (decisión del 25 sep 2026): todo en USD.** El dataset de
transacciones es de clientes de EE.UU. y sus montos vienen en dólares. Para
que ingreso y gasto se puedan comparar directo (por ejemplo en
`capacidad_ahorro`), el ingreso simulado también se expresa en USD.

## Por qué "controlado" y no al azar

La palabra clave de la decisión de Eduardo fue "controlado y documentado",
no aleatorio. Eso significa dos cosas:

- **Reproducible:** el mismo cliente (`cc_num`) siempre debe obtener el
  mismo ingreso e historial simulado, corrida tras corrida. Esto se logra
  usando el propio `cc_num` como semilla de una función pseudoaleatoria
  (no una semilla global fija ni un número random distinto cada vez).
- **Basado en una fórmula documentada**, no en valores inventados a mano.
  Cualquiera que lea este documento puede reproducir el cálculo.

## Señales reales que ya tenemos por cliente

De `gold_perfil_cliente` (construido en `gold_transactions.py`, sin datos
nuevos):

- `gasto_total` y `gasto_promedio_mensual` (gasto total / meses de
  actividad del cliente, mínimo 1 mes)
- `num_transacciones`
- `categoria_principal`
- `pct_transacciones_fraude`
- `city_pop` (población de la ciudad del cliente, viene desde Bronze)

## Regla propuesta: ingreso mensual simulado

```
ingreso_base(city_pop)      = tramo por tamaño de ciudad (ciudad grande = base más alta)
ingreso_variable(cc_num)    = variación pseudoaleatoria, semilla = hash(cc_num), rango ±20%
ingreso_mensual_simulado    = clamp(
                                 ingreso_base * (1 + ingreso_variable)
                                 + factor_gasto * gasto_promedio_mensual,
                                 minimo, maximo
                               )
```

- `city_pop` se agrupa en 3-4 tramos (ciudad pequeña / mediana / grande) con
  un ingreso base distinto por tramo — es la única señal "geográfica" que
  tenemos y sirve como ancla razonable.
- `factor_gasto` es un número pequeño y documentado (ej. 0.3) que hace que
  clientes que gastan más simulen un ingreso algo mayor, dándole coherencia
  interna al dataset simulado (un cliente no puede simular un ingreso muy
  bajo y un gasto mensual muy alto).
- `clamp(..., minimo, maximo)` evita valores absurdos (ingresos negativos o
  desproporcionados) — los límites se documentan explícitamente (ej. entre
  $8,000 y $120,000 USD/mes). Tramos base: ciudad pequeña (< 50,000 hab.)
  12,000 USD, mediana (< 300,000 hab.) 18,000 USD, grande 26,000 USD.

## Regla propuesta: historial crediticio simulado

```
score_base                  = 850 - penalizacion_fraude - penalizacion_antiguedad
penalizacion_fraude         = pct_transacciones_fraude * factor_fraude
penalizacion_antiguedad     = bonificación si num_transacciones es alto (más historial)
score_final_simulado        = clamp(score_base + variacion(cc_num), 300, 850)
categoria_credito           = "Bueno" / "Regular" / "Malo" según rangos de score_final_simulado
```

- Usa el mismo rango que un score de crédito real (300–850) para que sea
  interpretable, pero el valor en sí es simulado.
- `pct_transacciones_fraude` penaliza el score — es la única señal de
  "riesgo" real que tenemos hoy.
- `num_transacciones` alto se usa como proxy de "cliente con historial",
  no de buen comportamiento — se documenta esa diferencia.

## Advertencia que debe quedar escrita en el código y en el catálogo de KPIs

Estos valores **no representan el ingreso o historial real de ninguna
persona**. Son una construcción documentada para poder ejercitar el
pipeline completo (Bronze → Silver → Gold → KPIs de riesgo) dentro del
marco de "problema de negocio simulado" del proyecto. Cualquier KPI que
dependa de estos campos (`prob_impago`, `capacidad_ahorro`,
`ratio_endeudamiento`) debe llevar esta misma nota.

## Resultado sobre los datos reales (25 sep 2026)

Corrido sobre el Silver completo (924 clientes, 555,719 transacciones):

- Gasto mensual real con tarjeta: mediana ~6,170 USD.
- Ingreso mensual simulado: mediana ~14,560 USD.
- `capacidad_ahorro` (en `src/spark/gold_kpis.py`): mediana 60%; 14
  clientes gastan más de lo que ganan (ahorro negativo).
- Score simulado: 906 de 924 clientes topan exactamente en 850 (908 en
  "Bueno"). La bonificación por número de transacciones (+40) casi siempre
  es mayor que la variación (±30), así que casi todos se pasan de 850 y el
  límite los corta. **Pendiente de revisar con Eduardo** — así como está,
  el score casi no distingue entre clientes.

Nota sobre el gasto mensual: la primera versión usaba `promedio por compra
× 30`, que suponía una compra al día y subestimaba el gasto real ~3 veces.
Se corrigió a gasto total / meses de actividad.

## Siguiente paso

Ya se integró el dataset Loan Default con una llave sintética
(`docs/fase4_llave_sintetica_prestamos.md`). Su ingreso y su score pueden
reemplazar a los simulados de este documento; queda pendiente decidirlo con
Eduardo (preguntas 2 y 3 de ese documento).
