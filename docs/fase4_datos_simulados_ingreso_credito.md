# Fase 4 — Reglas para simular ingreso e historial crediticio (Opción A)

Este documento define **cómo** se va a construir el ingreso mensual y el
historial crediticio simulados por cliente, que es lo que decidió Eduardo
en la reunión del 11 de septiembre (ver `fase4_diseno_capa_gold.md`). Es un
documento de diseño — todavía no hay código, solo las reglas.

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

- `gasto_total` y `gasto_promedio_mensual_aprox`
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
                                 + factor_gasto * gasto_promedio_mensual_aprox,
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
  $8,000 y $120,000 MXN/mes).

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

## Siguiente paso

Con estas reglas ya escritas, el siguiente paso (código) es:

1. Escribir el script que genera esta tabla simulada por cliente
   (ingreso + historial), aplicando exactamente estas fórmulas.
2. Ingerirla a Bronze como un dataset más (mismo patrón que
   `ingest_fraud_transactions.py`).
3. Unirla a `gold_perfil_cliente` y calcular los 3 KPIs pendientes.

Esto no se ha empezado todavía — este documento es solo el diseño, listo
para revisarlo con Eduardo o para retomarlo como el próximo bloque de
código.
