# Fase 6 — Observabilidad y hardening

Semanas 21 y 22 del brief: «Observabilidad (dashboard de monitoreo, métricas
operativas, SLA, alertas)» y «Hardening (plataforma estable, optimización,
pruebas, corrección de errores)». Además, la demo final pide «demostrar la
consistencia de los datos en todas las capas del Lakehouse»; eso también se
mide aquí.

Código en `src/observabilidad/`; se ve en el **Panel Operativo**
(http://localhost:8501).

## Qué se agregó

```
DAG bronze_ingest
  pasos 1 a 11 ──(si uno falla)──> alerta inmediata (on_failure_callback)
        │
        ▼
  paso 12: revisar_slas  (corre siempre, aunque algo haya fallado)
        ├── consistencia Bronze → Silver → Gold
        ├── 6 SLA
        └── alertas ──> data/observabilidad/alertas.jsonl ──> Panel Operativo
                   └──> correo (si está configurado)
```

### 1. Acuerdos de servicio (SLA)

| SLA | Objetivo | Cómo se mide |
|---|---|---|
| Duración del pipeline | ≤ 15 min | Duración de la corrida (lo típico es ~5 min) |
| Frescura de los datos | ≤ 26 h desde la última carga | Fin de la última corrida exitosa (el DAG es diario, 2 h de margen) |
| Tasa de éxito | ≥ 90% | Corridas exitosas / terminadas, últimas 30 |
| Calidad de datos | 100% | Las 11 reglas de negocio de Silver (6 de transacciones, 5 de préstamos) |
| Consistencia entre capas | 100% | Las 7 revisiones de abajo |
| Asistente de IA | ≥ 95% correctas y ≤ 10 s | Última evaluación con el modelo real (`src/ia/evaluar.py`) |

Los objetivos están en `src/observabilidad/slas.py` (`OBJETIVOS`). Un SLA sin
datos para medirlo (por ejemplo, antes de la primera corrida) se muestra en
gris y no genera alerta.

### 2. Consistencia entre capas

Que nada se pierda ni se invente al pasar de una capa a otra:

| Revisión | Qué comprueba |
|---|---|
| Transacciones: Bronze → Silver | Ninguna fila de Silver es nueva, no hay llaves repetidas y lo descartado es ≤ 1% |
| Préstamos: Bronze → Silver | Lo mismo para los préstamos |
| Clientes: Silver → Gold | Los mismos clientes, una sola vez, en el perfil, los KPIs y la probabilidad de impago |
| Gasto: Silver → Gold | El gasto y el número de compras de cada cliente cuadran al centavo |
| Préstamo asignado | Cada cliente tiene un préstamo distinto que existe en Silver |
| Probabilidad de impago | Todas entre 0 y 1, sin vacíos |
| Simulador Monte Carlo | Cada cliente tiene sus 4 escenarios |

Con los datos actuales: 555,719 transacciones en Bronze y en Silver, 924
clientes en todas las tablas de Gold y $38,562,903.11 de gasto total igual en
Silver y en Gold.

### 3. Alertas

- **Paso que falla:** cuando un paso del DAG falla después de sus reintentos,
  se manda una alerta crítica en ese momento, con el error y el enlace a su
  registro en Airflow.
- **SLA que no se cumple:** el paso `revisar_slas` manda una alerta por cada
  SLA fuera de objetivo. Calidad, consistencia y frescura son críticas; las
  demás, advertencias.
- **Dónde llegan:** siempre al Panel Operativo («Alertas recientes») y, si se
  configura, por correo (un solo correo por revisión con todas las alertas).
- Si el correo falla, la alerta igual queda en el panel con el motivo; una
  alerta nunca tumba el DAG.

**Configurar el correo** (opcional), en el archivo `.env`, que no se sube a
GitHub:

```
ALERTAS_SMTP_USUARIO=cuenta@gmail.com
ALERTAS_SMTP_CONTRASENA=contraseña-de-aplicación
ALERTAS_CORREO_PARA=quien-recibe@correo.com
```

Con Gmail se usa una **contraseña de aplicación** (requiere verificación en dos
pasos), no la contraseña normal. Para probar:

```
docker compose up -d
docker compose exec dashboard python -m src.observabilidad.revisar --correo-prueba
```

### 4. Monitoreo del asistente de IA

El panel muestra el resultado de la última evaluación del asistente (respuestas
correctas y tiempo promedio) y si el modelo local de Ollama está activo.

### 5. Hardening

- **Healthchecks** en Docker para el scheduler de Airflow, los dos dashboards y
  Ollama (antes solo los tenían Postgres, MinIO y el webserver de Airflow).
- **Reinicio automático** (`restart: unless-stopped`) de los servicios que
  deben estar siempre arriba.
- **Arranque en orden:** el webserver y el scheduler esperan a que
  `airflow-init` termine bien (antes podían arrancar antes de que existiera la
  base de Airflow), y el Panel Operativo espera a que Postgres esté listo.
- **El último paso refleja las fallas:** como `revisar_slas` corre aunque algo
  falle, si hubo pasos fallidos se marca en rojo al final; si no, Airflow
  daría la corrida por exitosa.
- **Pruebas:** 16 pruebas nuevas en `tests/observabilidad/` (consistencia con
  un Lakehouse chiquito, cada SLA, alertas, correo con un servidor SMTP falso).

## Revisar desde la terminal

```
docker compose exec dashboard python -m src.observabilidad.revisar
```

Imprime los SLA y la consistencia, guarda el resultado en
`data/observabilidad/estado.json` y termina con código 0 si todo se cumple
(2 si algo no). Con `--notificar` además registra y manda las alertas.

## Prueba de punta a punta

1. `docker compose up -d --build` y esperar a que `docker compose ps` muestre
   los servicios como *healthy*.
2. Correr el DAG `bronze_ingest` desde Airflow (http://localhost:8080).
3. Al terminar (~6 min), los 12 pasos en verde y, en el Panel Operativo, los
   SLA y la consistencia entre capas en verde.
4. `python -m src.observabilidad.revisar` (comando de arriba) termina con 0.
