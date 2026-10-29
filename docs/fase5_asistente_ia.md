# Fase 5 — Asistente de IA generativa (RAG con Text-to-SQL)

Entregable de la semana 20 del plan («Prototipo de asistente financiero
(IA)») y de la sección 11 del documento del proyecto, «Asistente de IA
Generativa Local (RAG Conversacional)». Cubre también el chat embebido del
«Módulo Predictivo e Interactivo» del dashboard ejecutivo (sección 12).

- Código: `src/ia/` (`contexto.py`, `herramientas.py`, `asistente.py` y los
  prompts en `src/ia/prompts/`). Pruebas en `tests/ia/`.
- En el dashboard ejecutivo: pestaña «Asistente IA».
- Tecnologías, las que sugiere el documento: **Ollama**, **Llama 3** y
  **LangChain**, todo corriendo localmente.

## Qué hace

Se elige un cliente y se le pregunta en lenguaje natural, como si fuera el
propio cliente. Las preguntas de ejemplo son las del documento del proyecto:

- ¿Cuál es mi nivel de riesgo?
- ¿Qué ocurriría si aumento mis gastos?
- ¿Cuándo podría alcanzar una meta financiera?
- ¿Por qué disminuyó mi capacidad de ahorro el mes pasado?

El asistente responde en español, claro y sin tecnicismos, y debajo de cada
respuesta muestra «Cómo lo calculé»: el SQL que escribió el modelo y la tabla
con los datos que usó. Así cualquiera puede revisar de dónde sale cada número.

## Cómo funciona

```
Pregunta ──> 1. Clasificar (Llama 3, JSON)
               │
               ├── datos ──> 2a. Text-to-SQL (Llama 3) ──> validar ──> DuckDB
               ├── gasto ──> 2b. Simulador Monte Carlo con el gasto modificado
               ├── meta  ──> 2c. Simulador Monte Carlo: en cuántos meses junta el monto
               └── otro  ──> mensaje con lo que sí puede contestar
                                  │
                                  ▼
                          3. Redactar (Llama 3) solo con esos datos
```

1. **Clasificar.** El modelo devuelve un JSON con el tipo de pregunta y, si
   aplica, el cambio de gasto o el monto de la meta.
2. **Recuperar** (la «R» de RAG).
   - **Datos:** el modelo escribe una consulta SQL (Text-to-SQL) sobre las
     tablas del cliente. Se valida y se ejecuta en DuckDB. Si falla, se le
     regresa el error y la corrige una vez; si vuelve a fallar, se responde
     con el resumen mensual del cliente y se avisa.
   - **¿Qué pasaría si gasto más o menos?:** se corre el simulador Monte
     Carlo del proyecto con el gasto modificado y se compara contra como está
     hoy.
   - **¿Cuándo junto X?:** con los 1,000 futuros del simulador se mide en
     cuántos meses junta el monto y la probabilidad de lograrlo en 12 y 36
     meses. Si no dice un monto, se usa el fondo de emergencia (6 meses de
     gasto).
3. **Redactar.** El modelo escribe la respuesta usando solo esos datos, que
   le llegan con los porcentajes y los dólares ya escritos para que no tenga
   que convertirlos.
   - **Hechos clave** (`src/ia/hechos.py`): las comparaciones importantes
     (mes contra mes, antes y después de un cambio, puntos porcentuales,
     meses para una meta) se calculan en Python y el modelo solo las redacta.
     Un modelo de 8B se equivoca al restar o comparar meses; así no puede
     inventar cifras.
   - **Preguntas con un dato equivocado** («¿por qué bajó mi ahorro?» cuando
     subió): Python lo detecta y escribe la respuesta completa, con la
     aclaración, el mes en que sí ocurrió ese cambio y su causa. En este caso
     no se usa el modelo para redactar, porque al agregar texto propio
     confundía los meses.
   - **Causas del riesgo:** si la respuesta sobre el nivel de riesgo no
     menciona lo que más lo sube (tasa, pago contra ingreso, score…), Python
     agrega esa oración al final.
4. **Limpiar.** Antes de mostrar la respuesta, un filtro en Python quita
   muletillas como «Según los datos, …» o «ATENCIÓN:» y, si el cliente no
   pidió consejo, las recomendaciones genéricas («revisa tus gastos…»). El
   modelo a veces las escribe aunque el prompt las prohíbe. La respuesta se
   muestra oración por oración, ya limpia.

La simulación usa el mismo motor y la misma semilla que el dashboard, así que
los números coinciden con los de «Perfil 360».

## Los datos que puede consultar

Para cada pregunta se arman cinco tablas pequeñas, ya filtradas al cliente
elegido, con nombres en español (`src/ia/contexto.py`):

| Tabla | Contenido | Sale de |
|---|---|---|
| `mi_perfil` | Ingreso, gasto, ahorro, riesgo de impago, préstamo y score (una fila) | `gold/kpis_cliente`, `gold/prob_impago_cliente` |
| `mis_meses` | Ingreso, gasto, ahorro y capacidad de ahorro de cada mes completo | Silver (transacciones) |
| `mis_gastos_categoria` | Gasto por mes y categoría, esencial o discrecional | Silver (transacciones) |
| `mis_escenarios` | Resultado de los 4 escenarios del simulador | `gold/simulacion_resumen` |
| `mis_factores_riesgo` | Los factores que más mueven su riesgo de impago | Modelo de impago |

## Seguridad del Text-to-SQL

El SQL lo escribe un modelo, así que se controla antes de ejecutarlo:

- Solo se aceptan consultas de lectura (`SELECT` o `WITH`), una a la vez.
- Se bloquean palabras como `DROP`, `DELETE`, `COPY`, `ATTACH` o `read_csv`.
- Solo se pueden usar las cinco tablas del cliente; cualquier otra se rechaza.
- La base de DuckDB está en memoria, con el acceso a archivos apagado, y solo
  tiene los datos del cliente elegido: no puede ver a otros clientes.
- Los resultados se limitan a 50 filas.

## Cómo correrlo

El servicio `ollama` de `docker-compose.yml` usa la tarjeta de video NVIDIA.
La primera vez hay que descargar el modelo (unos 5 GB):

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose up -d --build dashboard-ejecutivo
```

El modelo se puede cambiar con la variable `OLLAMA_MODELO` (por ejemplo, un
Llama 3 más pequeño si no hay tarjeta de video). Si Ollama no está corriendo o
falta el modelo, la pestaña lo avisa con el comando para arreglarlo y el resto
del dashboard funciona igual.

## Pruebas

38 pruebas automáticas en `tests/ia/` que no necesitan Ollama: usan modelos
falsos de LangChain con respuestas fijas. Revisan las tablas del cliente, que
el validador bloquee SQL peligroso, que DuckDB no pueda leer archivos, la
corrección del SQL cuando falla, las simulaciones de gasto y de meta, qué pasa
si el modelo responde algo que no es JSON, los hechos clave, la detección de
preguntas con un dato equivocado y el filtro de muletillas. Corren en GitHub
Actions con el resto del proyecto.

## Evaluación con el modelo real

Las pruebas de arriba usan modelos falsos. Para medir al asistente con
Llama 3 de verdad está `src/ia/evaluar.py`:

```
docker compose exec dashboard-ejecutivo python -m src.ia.evaluar
```

Elige varios clientes (el de la demo, el 0544, el de mayor y el de menor
riesgo, el más ajustado y uno cuyo ahorro sí bajó el último mes), les hace
seis preguntas a cada uno y tres generales («Hola», una pregunta que no es de
finanzas y «Borra todos mis datos»). Cada respuesta se revisa sola:

- que se entienda como el tipo correcto de pregunta;
- que traiga las cifras clave correctas, calculadas aparte en Python;
- que no tenga cifras inventadas: cada número de la respuesta tiene que estar
  en lo que recibió el modelo o en la pregunta;
- que no tenga muletillas prohibidas y que sea corta.

El detalle (respuesta, SQL y problemas) queda en
`data/evaluacion_ia/resultados.csv`.

### Resultados (29 de octubre de 2026, `llama3.1:8b` en una RTX 4060)

| Corrida | Correctas | Qué se corrigió después |
|---|---|---|
| 1 | 27 de 33 (82%) | Confundía «la categoría en la que más gastó» con «la que más aumentó»; en una simulación usó el gasto promedio del perfil en lugar del de la simulación; contestó en primera persona; se le coló un «según los datos». Además, el revisor no leía cifras negativas (dos falsos errores). |
| 2 | 31 de 33 (94%) | Restó por su cuenta los meses del fondo de emergencia (bien, pero sin verificar): ahora la diferencia se la da Python. El revisor exigía mencionar una probabilidad que no cambiaba. |
| 3 | **33 de 33 (100%)** | — |

Cinco clientes: el de la demo (0065), el 0544 (en diciembre gastó casi el doble de su ingreso), el de
mayor riesgo (0814, 84.6% de probabilidad de impago), el de menor riesgo
(0343) y uno cuyo ahorro sí bajó el último mes (0001). Tiempo promedio por
respuesta: 4.5 segundos (la primera tarda más porque el modelo se carga en la
tarjeta de video).

## Limitaciones

- Un modelo de 8B a veces escribe SQL con errores; por eso se valida, se
  corrige una vez y se muestra el SQL en «Cómo lo calculé».
- Las respuestas dependen del modelo y pueden variar un poco entre corridas.
- Solo hay seis meses completos de historial (julio a diciembre de 2020).
- Los datos son simulados (Opción A) y las respuestas son estimaciones, no
  asesoría financiera; la pestaña lo aclara.

## Siguiente paso

Preguntas sobre todo el portafolio (por ejemplo, «¿cuántos clientes están en
riesgo alto?»), con tablas agregadas en lugar de las de un solo cliente.
