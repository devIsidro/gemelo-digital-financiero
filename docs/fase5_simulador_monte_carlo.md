# Fase 5 — Simulador financiero Monte Carlo

Entregable de la semana 19 del plan ("Simulador de escenarios —
implementar Monte Carlo") y del "Simulador Financiero Estocástico" del
documento del proyecto. Completa el último KPI del catálogo que faltaba,
`flujo_efectivo_proyectado` (`config/kpis.yaml`).

- Código: `src/ml/simulador.py` (motor, sin Spark) y pruebas en
  `tests/ml/test_simulador.py`.
- En el pipeline: paso 11 del DAG, `simulador_monte_carlo`, después de
  `calcular_kpis`. Tarda unos 20 segundos para los 924 clientes.
- En el dashboard ejecutivo: simulador interactivo en la pestaña
  «Perfil 360» y comparación de escenarios en «Portafolio».

## Qué hace

En lugar de dar un solo número ("este cliente va a ahorrar X"), genera
**1,000 futuros posibles** del dinero de cada cliente, mes a mes durante
**36 meses**, y los resume en bandas: el caso típico (mediana), el 50% de
los casos más probables y el 90%.

Para cada mes `t` (mes 1 = enero 2021, el siguiente al último dato):

```
ingreso_t = ingreso_mensual_simulado × (1 + aumento)^(t/12) × tiene_ingreso_t
gasto_t   = gasto_base × (1 + inflación)^(t/12) × factor_diciembre_t × variación_t
flujo_t   = ingreso_t + utilidad_negocio_t − gasto_t − pago_crédito_t
ahorro_t  = ahorro_(t−1) + flujo_t        (ahorro_0 = 0)
```

- **gasto_base y variación** salen del gasto real con tarjeta de cada
  cliente (Silver), de julio a noviembre de 2020. La variación de cada mes
  es aleatoria (lognormal con media 1) con el coeficiente de variación del
  propio cliente: un cliente con gasto muy irregular tiene bandas más
  anchas.
- **Diciembre** se trata aparte: en los datos el gasto de diciembre es
  **1.79 veces** el de un mes normal (calculado en cada corrida). Por eso
  las gráficas tienen un escalón en los meses 12, 24 y 36.
- **tiene_ingreso_t**: cada mes hay una probabilidad pequeña de perder el
  ingreso por un tiempo (desempleo, enfermedad). Es la principal fuente de
  "estrés" del modelo.
- **ingreso_mensual_simulado** es el de la Opción A, el mismo de
  `capacidad_ahorro`. No se usa el ingreso de la fuente de préstamos, como
  se decidió (decisión 2 de `docs/fase4_decisiones_kpis.md`).

## Supuestos

| Supuesto | Valor | De dónde sale |
|---|---|---|
| Trayectorias por cliente y escenario | 1,000 | Con 10,000 la mediana cambia menos de 2% (ver Validación) |
| Horizonte | 36 meses | Documento del proyecto: proyecciones a 12, 24 o 36 meses |
| Gasto base y variación | Por cliente, de julio a noviembre de 2020 | Gasto real de Silver (mediana: 5,302 USD; variación típica 25%) |
| Variación mínima y máxima | 5% y 80% | Evita que un cliente con pocos datos quede sin incertidumbre o con una exagerada |
| Factor de diciembre | 1.79 | Calculado de los datos en cada corrida |
| Inflación del gasto | 3% anual | Supuesto académico; ajustable |
| Aumento del ingreso | 3% anual | Igual a la inflación: el ingreso real se mantiene |
| Probabilidad de perder el ingreso | 0.4% al mes (≈ 4.7% al año) | Supuesto académico; ajustable |
| Meses sin ingreso, en promedio | 3 | Supuesto académico; ajustable |
| Ahorro inicial | 0 | El dataset no trae saldos de cuentas |
| Meta del fondo de emergencia | 6 meses de gasto | Regla común de finanzas personales |

Todos los valores están en un solo lugar (`Supuestos` en
`src/ml/simulador.py`) y se guardan con cada corrida en
`data/gold/simulacion_parametros.json`.

## Escenarios

| Escenario | Qué pasa | Parámetros por defecto |
|---|---|---|
| Sin cambios | El cliente sigue igual | — |
| Crédito hipotecario | Desde el mes 1 paga un crédito a tasa fija | Monto de 3 años de ingreso, 10% anual, 20 años (pago ≈ 35% del ingreso) |
| Cambio de empleo | En el mes 6 cambia de trabajo: unos meses sin ingreso y luego un sueldo distinto | 1 mes sin ingreso en promedio (máximo 4); el sueldo nuevo cambia entre −10% y +25% |
| Abrir un negocio | Invierte en un negocio que empieza a dar utilidad en el mes 4 y madura en un año; puede cerrar | Inversión de 6 meses de ingreso; utilidad esperada de 25% del ingreso (desviación 30%); 30% de probabilidad de cerrar en 2 años |

En el dashboard se pueden mover los parámetros principales de cada
escenario y la simulación se recalcula al momento. El enganche del crédito
y la inversión del negocio se suponen cubiertos con ahorros previos (que el
dataset no tiene); para el negocio se reporta aparte en cuántos meses la
utilidad paga la inversión.

## Resultados por cliente

Tablas en `data/gold/` (las escribe el paso 11 del DAG):

| Tabla | Contenido |
|---|---|
| `simulacion_insumos.parquet` | Ingreso, gasto base y variación de cada cliente |
| `simulacion_resumen.parquet` | Por cliente y escenario: flujo de efectivo y ahorro a 1, 3, 6, 12, 24 y 36 meses (percentiles 5, 50 y 95), probabilidad de números rojos, capacidad de ahorro del primer año, meses para el fondo de emergencia y, en el negocio, probabilidad de recuperar la inversión |
| `simulacion_trayectorias.parquet` | Bandas mes a mes del ahorro (percentiles 5, 25, 50, 75 y 95) |
| `simulacion_parametros.json` | Supuestos y parámetros con que se corrió |

**`flujo_efectivo_proyectado`** (KPI del catálogo) = columnas `flujo_m1_*`,
`flujo_m3_*` y `flujo_m6_*`: el flujo del mes 1, 3 y 6 con su banda de
confianza.

**Números rojos**: el ahorro acumulado baja de cero en algún momento de los
36 meses, es decir, el cliente gastaría más de lo que tiene. Un cliente está
**en riesgo** en un escenario si eso pasa en al menos 10% de sus futuros
posibles.

## Resultados del portafolio (924 clientes)

| Escenario | Clientes en riesgo de números rojos | Ahorro a 12 meses (mediana) | Capacidad de ahorro, año 1 | Flujo del mes 6 (mediana) |
|---|---|---|---|---|
| Sin cambios | 6.8% | 105,610 | 62% | 9,351 |
| Crédito hipotecario | 35.4% | 49,179 | 28% | 4,604 |
| Cambio de empleo | 12.9% | 97,929 | 61% | −4,172 |
| Abrir un negocio | 7.4% | 115,653 | 69% | 10,119 |

Montos en USD. Lo que se ve:

- **El crédito hipotecario es el escenario de más estrés**: el riesgo pasa
  de 7% a 35% de los clientes. Depende sobre todo de cuánto ahorra hoy el
  cliente: con el crédito quedan en riesgo el 99% de los que ahorran menos
  de 35% de su ingreso, contra el 3% de los que ahorran más de 65%.
- **Cambio de empleo**: el golpe es de corto plazo (el flujo del mes 6 es
  negativo por los meses sin ingreso), pero a 36 meses la mediana termina
  un poco arriba porque el sueldo nuevo sube en promedio.
- **Abrir un negocio**: en promedio mejora el ahorro, pero con la mayor
  dispersión de todos (las bandas más anchas). Para el cliente típico, la
  probabilidad de recuperar la inversión en 36 meses es de 40% con los
  supuestos por defecto.
- **Sin cambios**, la mayoría está bien porque el ingreso simulado es alto
  frente al gasto real (capacidad de ahorro mediana de 62%).

Una cosa que puede llamar la atención: los clientes de **riesgo de impago
alto** son los que menos quedan en riesgo de números rojos (12% contra 41%
de los de riesgo bajo, en el escenario del crédito). Es la misma situación
de la revisión de distribuciones: el riesgo de impago usa el ingreso de la
fuente y el simulador el ingreso simulado, que sale del gasto real. Son dos
lecturas distintas del cliente y por eso no se mezclan.

### Ejemplo: Cliente 0065

| Escenario | Riesgo de números rojos | Ahorro a 12 meses (mediana) | 90% de los casos | Fondo de emergencia |
|---|---|---|---|---|
| Sin cambios | 7.6% | 84,477 | 51,898 a 102,929 | 4 meses |
| Crédito hipotecario | 28.3% | 34,342 | 3,015 a 53,026 | 9 meses |
| Cambio de empleo | 6.4% | 75,972 | 35,442 a 106,293 | 4 meses |
| Abrir un negocio | 6.4% | 93,282 | 58,018 a 123,393 | 4 meses |

Con el crédito su pago sería de unos 4,116 USD (35% de su ingreso) y su
riesgo de números rojos casi se cuadruplica.

## Validación

- **Cálculo a mano**: sin incertidumbre (inflación, variación y riesgo de
  perder el ingreso en cero), el simulador da exactamente
  `ingreso − gasto` cada mes. Hay una prueba automática.
- **Convergencia**: en 40 clientes al azar, 1,000 trayectorias contra
  10,000 dan una mediana del ahorro a 12 meses que difiere 0.1% a 1.6%
  (según el escenario) y una probabilidad de números rojos que difiere 0.2
  puntos (máximo 4 puntos).
- **Reproducible**: cada cliente y escenario tiene su semilla fija. El job
  del DAG y el simulador del dashboard dan exactamente los mismos números,
  y dos corridas seguidas dan el mismo resultado.
- **Reglas que el paso del DAG revisa** antes de guardar: todos los
  clientes y escenarios simulados, sin percentiles vacíos y con las bandas
  en orden. Si algo falla, el paso se marca en rojo.
- 10 pruebas automáticas del motor (fórmula del crédito, cambio de empleo,
  negocio, diciembre, insumos y bandas). Una de ellas corre 32 casos
  extremos: los controles del dashboard en sus límites, un cliente que gasta
  más de lo que gana, ingreso cero y gasto cero.
- Prueba del dashboard clic por clic con 7 clientes extremos, los 4
  escenarios y cada control en su mínimo y su máximo: sin errores. Si
  todavía no se corre el paso del DAG, el dashboard muestra un aviso en
  lugar de fallar.

## Limitaciones

- El ingreso es simulado (Opción A) y constante salvo por los eventos del
  escenario; el dataset no trae ingresos reales de estos clientes.
- El gasto es solo el de la tarjeta; no incluye renta, servicios u otros
  pagos que el dataset no tiene. Por eso la capacidad de ahorro sale alta.
- Solo hay datos de 6 meses de gasto, así que la única estacionalidad que
  se modela es la de diciembre.
- No hay saldos iniciales: "ahorro acumulado" es lo que se junta desde el
  mes 1.
- Los supuestos de inflación, desempleo y de cada escenario son académicos
  y se pueden ajustar.

## Decisiones con Eduardo (reunión del 16 oct 2026)

Se presentó el simulador en vivo en el dashboard ejecutivo. Eduardo dio su
visto bueno y aprobó los supuestos tal como estaban:

1. **Supuestos generales:** se quedan la inflación de 3% anual, la
   probabilidad de 0.4% al mes de perder el ingreso y los 3 meses sin
   ingreso en promedio.
2. **Parámetros de los escenarios:** se quedan los valores por defecto,
   incluido el crédito hipotecario de 3 años de ingreso a 10% anual y 20
   años.
3. **"En riesgo de números rojos":** se queda el umbral de 10% de los
   futuros posibles.
